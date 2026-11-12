# DSV4.1 decoder KV offload validation branch

This is a development branch for Ascend 950DT/A5, MRV2, PCP1/DCP1.
It is paired with the same `decoder-kvoffload-dsv41` branch in
`zyz111222/vllm`. Core base: `24420b7a341af37a9bc3186e6fc6062512e0aa58`;
Ascend base: `f1007cd14ac0636621c64e3af6596ce76f93dd96`.
Both branches are required. No PR or device/performance qualification is implied.

## Checkout and dependencies

Clone the two repositories as sibling directories named `vllm` and `vllm-ascend`:

```bash
git clone -b decoder-kvoffload-dsv41 https://github.com/zyz111222/vllm.git
git clone -b decoder-kvoffload-dsv41 https://github.com/zyz111222/vllm-ascend.git
```

Use an existing working DSV4.1 Ascend environment with Python 3.12,
Torch/torch-npu 2.10 and CANN 9.2.1, plus the matching triton-ascend.
Use `uv` and a task venv. Install/build both checkouts using their normal
editable-install instructions. Rebuild Ascend for the target 950 device;
do not reuse binaries from another CANN/Python/architecture combination.
The extension must register `hisparse_resolve_residency`,
`hisparse_resolver_scratch_capacity`, `swap_blocks_batch`, and
`kv_compress_epilog_v2`. Install its CANN custom vendor payload as well:
the existing KV compression epilog needs that payload, not only the Torch library.
No compiled libraries, credentials, weights, or machine logs are committed here.

The example scripts use the current venv Python and discover the sibling
checkouts. Provide this machine's model path, IP and NIC explicitly.

## Validation status

Before packaging: 22 DSV4.1 CPU tests and 12 focused allocator regressions passed.
18 new NPU cases were collected but not executed; two additional packed-row
refill cases were prepared. NPU, full-model, DSpark5, and throughput qualification
are still pending because the source machine was occupied. Ruff checks, formatting,
Python syntax and whitespace checks passed. Full pre-commit/format CI was attempted
but stopped while installing actionlint dependencies from proxy.golang.org; full
CI is not claimed. AI assistance was
used to prepare the implementation; review the changes before production use.

CPU and device checks, from the Ascend checkout:

```bash
export PYTHONPATH="$(realpath ../vllm):$PWD"
export DSV41_TEST_MODEL=/path/to/DeepSeek-V4.1-Flash
python -m pytest --noconftest -q tests/ut/models/test_deepseek_v41_hisparse.py
# Run only on available devices:
python -m pytest --noconftest -x -v tests/e2e/pull_request/one_card/test_dsv41_hisparse_attention.py
python -m pytest --noconftest -q tests/e2e/pull_request/one_card/test_hisparse_ops.py -k 'compact_refill and 320'
```

## Serving and benchmarking

Run the layout probe first. The serving example targets a single eight-card
machine, DP8/TP1 by default (`--dp 1 --tp 8` is a separate comparison topology).
Each DP rank has a local API port. `--dry-run` only prepares manifests.
Remove it to launch after the device tests pass and the cards are free.

```bash
python examples/decoder_kvoffload/inspect_decoder_kvoffload.py \
  --model /path/to/DeepSeek-V4.1-Flash --sources 20 --out /path/to/results/probe.json
python examples/decoder_kvoffload/serve_decoder_kvoffload.py \
  --mode on --model /path/to/DeepSeek-V4.1-Flash \
  --ip YOUR_MACHINE_IP --nic YOUR_NIC --sources 20 \
  --dry-run --run-dir /path/to/results/on-serve
```

Keep the same settings for OFF and ON, including DSpark, graph mode, Engram CPU
offload, memory budget, batch limits and topology. `--eager` and
`--speculative-tokens 0` are available for initial diagnosis. Default DSpark is 5.
Selected main-KV owners default to `[20]`; `--sources 2,20` or
`--sources 2,8,14,20` select other policies. Other KV resources stay on NPU.

After all endpoints are healthy and `launch.json` says `running`, benchmark:

```bash
python examples/decoder_kvoffload/bench_decoder_kvoffload.py \
  --mode on --launch-manifest /path/to/results/on-serve/launch.json \
  --input-tokens 131072 --output-tokens 1024 --cache-ratio 0.9 \
  --concurrency 64 --out /path/to/results/on-c64
python examples/decoder_kvoffload/summarize_decoder_kvoffload.py \
  --run-dir /path/to/results
```

N is fixed at `4*C`; prefix warmup is separate. Trial/session cache namespaces
prevent cross-run full-prompt cache hits. Sweep OFF and ON beyond the initial
concurrency point. Compare their highest measured valid output throughput under
mean request TPOT <=20/50 ms, with P99 reported separately. Whole-run throughput
includes Prefill and is not a pure Decode-window metric.

## Important behavior

Prefill writes new KV to Resident and mirrors it to Host. Core may allocate Hot
and unpin clean Resident pages during chunked Prefill. Prefill attention reads
Resident directly, or stages the source's full valid history, reusing that staging
across consumers in the same forward; it does not use Decode's TopK/LRU Hot path.
Decode refills selected missing rows into Hot. Source 20 has five index groups;
they resolve and consume in order, with separate output mappings and a per-group
speculative union. C2 keeps 64 physical rows per logical 128-token page.

The inherited C25 fix is in Core: allocate Hot first, reclaim only safe clean
Resident pages, then allocate the next Prefill chunk. It allowed 25 requests to
finish in the earlier GLM run, with three remaining capacity preemptions; this is
not a DSV4.1 or C25 no-preemption performance qualification.

The current shared-pool stride is still dominated by compressor state and page
alignment (462848 to 409600 bytes in the inspected A5 layout). Do not equate
logical KV offload volume with physical capacity savings or promise a throughput
gain before device measurements.
