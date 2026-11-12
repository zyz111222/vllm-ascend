# HiSparse on Ascend

HiSparse keeps active sparse-attention KV rows in NPU Resident/Hot storage and
refills selected missing rows from Host memory. Core owns HMA allocation,
Coordinator admission/eviction, request lifetimes, prefetch and mirror completion.
The Ascend backend supplies residency resolution, transfers and the existing SFA
Attention interface. Attention core arithmetic is unchanged.

## Supported configurations

- Experimental sparse RoPE MLA/SFA with Model Runner V2.
- BF16 KV and native Ascend 950 C8: E4M3FN main/Indexer values with FP32 scales.
  Main and Indexer cache dtypes are independently selectable.
- Private Host pools per tensor-parallel rank and Core eager Host mirroring.
- Ascend 950 with a SIMT-capable CANN 9.2 toolchain. A2/A3 are not newly qualified.
- TopK must satisfy the existing SFA requirement of divisibility by 128.

C4, TurboQuant, INT8-format C8, context parallelism and external P/D transport
are outside this backend's current support.

## Build the resolver

Use the paired merged vLLM version in `.github/vllm-main-verified.commit` and
install vllm-ascend normally in the matching Torch/torch-npu/CANN environment.
The standard CMake build compiles `csrc/hisparse/hisparse_resolver.cpp` using
CANN's ASC integration when the Ascend 950 SIMT headers are available. It links
the kernel into `vllm_ascend_C` and registers
`torch.ops._C_ascend.hisparse_resolve_residency` with the other Ascend operators;
there is no separate resolver build or library-loading step.

An unsupported capacity or missing extension raises an error before execution;
there is no Triton resolver fallback. Other build targets retain their existing
operators without registering this resolver.

## Enable HiSparse

Set `VLLM_USE_V2_MODEL_RUNNER=1` and add these options to the compatible model's
normal serving command:

```bash
--block-size 128 \
--kv-cache-dtype bfloat16 \
--attention-config '{"indexer_kv_dtype":"bf16","hisparse_config":{"eager_host_mirror":true}}' \
--kv-transfer-config '{"kv_connector":"HiSparseConnector","kv_role":"kv_both","kv_connector_extra_config":{"host_pool_gib":4}}'
```

For native A5 C8, replace the two dtype options with:

```bash
--kv-cache-dtype fp8 \
--attention-config '{"indexer_kv_dtype":"fp8","hisparse_config":{"eager_host_mirror":true}}'
```

| Parameter | Meaning |
| --- | --- |
| `hisparse_config` | Enables Core HiSparse for the supported sparse Attention backend. |
| `eager_host_mirror` | Keep `true` for this foundation; Core maintains the Host KV mirror. |
| `host_pool_gib` | Host pool size per rank. Size it for the workload; the example reserves 4 GiB per rank. |
| `device_buffer_size` | Per-request Hot capacity from Core's HiSparse configuration. It must cover the selected verification-row union. |
| `kv_cache_dtype` / `indexer_kv_dtype` | Select main and Indexer KV formats independently from model-weight quantization. |

## DeepSeek V4.1 shared compressed sources (development)

The local V4.1 integration follows upstream [#65905](https://github.com/vllm-project/vllm/pull/65905)
and [#65970](https://github.com/vllm-project/vllm/pull/65970). It requires the paired
Core compressed-page patch as well as this Ascend branch. The initial target is
Ascend 950, Model Runner V2, PCP1/DCP1, using the model's existing 320-byte MXFP4
main KV, 68-byte indexer rows, and 544-byte SWA rows. This path does not use the
legacy sparse-offload fusion operators or the SM90 split-page CUDA kernels.

Add these options to the normal V4.1 serving command (keep its weight quantization
and Engram configuration unchanged):

```bash
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_KV_CACHE_LAYOUT=BLHNC
# Additional vllm serve arguments:
--block-size 128 \
--attention-config '{"hisparse_config":{"device_buffer_size":4096,"eager_host_mirror":true}}' \
--kv-transfer-config '{"kv_connector":"HiSparseConnector","kv_role":"kv_both","kv_connector_extra_config":{"host_pool_gib":64,"dsv41_sources":[20]}}'
```

`dsv41_sources` selects main-KV owners from `[2,8,14,20]`, defaulting to `[20]`.
Indexer K, folded index K, SWA, compressor state, and unselected main sources stay
on device. `[2,20]` and all four owners are also exposed for capacity/transfer
comparisons. Reduced backbones must explicitly select owners they contain.

C2 keeps 64 physical rows per logical 128-token page; C1 keeps 128. Only complete
C2 groups are mirrored. Replay skip-update and graph padding masks are retained.
The five index groups reading source 20 resolve sequentially: each group waits
for the previous attention read before it can evict Hot rows. With DSpark5,
4096 Hot rows cover the per-group worst-case union of `6*512` rows. The output
workspace retains all five groups' mappings. Nonresident Prefill stages historical
source pages once per forward and shares that binding across its consumer groups.

CPU layout, byte-address, write-mask, and allocator regression tests are available.
The NPU tests cover real mapped-host refills, C1/C2, five independent groups,
single/verification queries, physical-index attention, and graph replay. Device
and full-model qualification must be recorded before claiming this development
path is validated. No DSV4.1 throughput benefit has yet been established.

A large logical offload volume does not guarantee a smaller physical slab. The
current shared-pool stride is still bounded by the compressor-state group; report
layout/padding changes separately from offload and compare identical serving
configurations under measured mean TPOT limits.

## Capacity and correctness notes

The resolver preserves the complete MTP query-row union before choosing LRU
victims, and produces the refill plan on device. Hot capacity must fit both the
int16 LRU representation (at most 32768 rows) and the compiled target's UB budget.
These checks do not change the configured TopK or discard selected tokens.

Native C8 uses the existing Ascend E4M3FN values, BF16 RoPE and FP32 scales.
Weight quantization and KV-cache quantization remain separate choices. HiSparse
preserves the shared-slab strides and uses the existing quantizer and Attention
operations. No upstream vLLM source change or unmerged Core PR is required.

Zero misses remove cold-row refills; eager Host mirroring and residency
bookkeeping still run. Size Host memory per rank and compare matched workloads
when assessing performance. Prefill, mixed batches and Decode retain current
request mappings, padding and completion-before-reuse rules.

## Validation and performance

Ascend 950 eager/Graph components and TP8/MTP3 full-model checks were recorded
at the source revisions below. The default-direct C8 Prolog write path still
requires fresh device and model qualification. A historical matched C8 test
with two rounds per arm, fixed inputs and 8K outputs measured these mean SSE
TPOT increases:

| Input / concurrency | Reported residency | ON versus OFF |
| --- | --- | --- |
| 16K / C8 | Zero misses | +1.03% |
| 32K / C8 | Misses present | +10.01% |

These measurements used merged Core `39dbf518` (including #63801), Ascend
`5a02c50`, Python 3.12 and CANN 9.2.1 on the same machine. Inputs, outputs and
MTP counters matched. N2 does not establish statistical significance or dataset
accuracy. See [PR #20926](https://github.com/vllm-project/vllm-ascend/pull/20926)
for exact configuration, commands and separate capacity qualifications.

Formal performance runs exclude profiling. Reported miss and H2D counters are
asynchronous software observations, not physical DMA measurements. Qualification
on one workload does not establish dataset accuracy or support for other devices.
