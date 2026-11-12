# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""A5 compressed attention must preserve output when HiSparse remaps top-k."""

from dataclasses import replace

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.attention.dsa_v41 import AscendDSAV41Metadata, DeepseekV41LayerMetadata
from vllm_ascend.models.deepseek_v41.mixed_quant_attention import (
    build_smla_metadata,
    qsmla,
    write_attention_cache,
)
from vllm_ascend.ops.triton.build_window_indices import build_window_indices_triton


@pytest.mark.parametrize("ratio", [1, 2])
@pytest.mark.parametrize("queries", [1, 6])
@pytest.mark.parametrize("capture", [False, True])
def test_mxfp4_physical_indices_match_logical_paged_attention(ratio, queries, capture):
    from vllm_ascend.hisparse.backend import AscendHiSparseBackend
    from vllm_ascend.utils import bootstrap_custom_op_env

    bootstrap_custom_op_env()
    if not torch.npu.is_available() or not torch.npu.get_device_name(0).startswith("Ascend950"):
        pytest.skip("DSV4.1 mixed-quant HiSparse requires Ascend 950")
    torch.npu.set_device(0)
    AscendHiSparseBackend().check_available()
    torch.manual_seed(52)
    device = "npu:0"
    rows, block = queries * 2, 128 // ratio
    query = torch.randn((rows, 64, 512), dtype=torch.bfloat16, device=device)
    window = torch.empty((3, 128, 1, 544), dtype=torch.uint8, device=device)
    compressed = torch.empty((4, block, 1, 320), dtype=torch.uint8, device=device)
    for cache, kind in [(window, "win"), (compressed, "cmp")]:
        count = cache.shape[0] * cache.shape[1]
        write_attention_cache(
            cache,
            torch.arange(count, dtype=torch.int64, device=device),
            torch.randn((count, 512), dtype=torch.bfloat16, device=device),
            kind=kind,
        )
    starts = torch.tensor([0, queries, rows], dtype=torch.int32, device=device)
    seq_lens = torch.full((2,), 64 + queries, dtype=torch.int32, device=device)
    positions = torch.arange(64, 64 + queries, device=device).repeat(2)
    win_indices, win_lengths = build_window_indices_triton(positions, 128)
    swa = AscendDSAV41Metadata(
        block_table=torch.tensor([[1], [2]], dtype=torch.int32, device=device),
        query_start_loc=starts,
        seq_lens=seq_lens,
        slot_mapping=torch.empty((0, 2), dtype=torch.int32, device=device),
        compress_ratio=1,
        storage_block_size=128,
        is_compressor_state=False,
        positions=positions,
        num_reqs=2,
        num_actual_reqs=2,
        num_actual_tokens=rows,
        ori_sparse_indices=win_indices,
        ori_topk_length=win_lengths,
        smla_metadata=build_smla_metadata(win_lengths, starts),
    )
    logical_table = torch.tensor([[1], [3]], dtype=torch.int32, device=device)
    main = replace(
        swa, block_table=logical_table, compress_ratio=ratio, storage_block_size=block, cache_seq_lens=seq_lens // ratio
    )
    metadata = DeepseekV41LayerMetadata(attention=main, swa=swa, compressor=None, indexer=None)
    logical = torch.full((rows, 512), -1, dtype=torch.int32, device=device)
    logical[:, :16] = torch.arange(16, dtype=torch.int32, device=device)
    physical = logical.clone()
    request_ids = torch.arange(2, device=device).repeat_interleave(queries)
    physical[:, :16] += logical_table[request_ids, 0, None] * block
    remapped = replace(
        main,
        block_table=torch.arange(4, dtype=torch.int32, device=device).expand(2, -1).contiguous(),
        cache_seq_lens=torch.full_like(seq_lens, 4 * block),
        cmp_residual=None,
    )
    remapped_metadata = replace(metadata, attention=remapped)
    lengths = torch.full((rows, 1), 16, dtype=torch.int32, device=device)
    sinks = torch.zeros(64, dtype=torch.float32, device=device)

    def run(meta, indices):
        return qsmla(
            query,
            window,
            compressed,
            meta,
            indices,
            window_size=128,
            sinks=sinks,
            softmax_scale=512**-0.5,
            compressed_lengths=lengths,
        )

    run(remapped_metadata, physical)
    graph = None
    try:
        if capture:
            torch.npu.synchronize()
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                actual = run(remapped_metadata, physical)
            query.neg_()
            for _ in range(3):
                graph.replay()
        else:
            actual = run(remapped_metadata, physical)
        expected = run(metadata, logical)
        torch.npu.synchronize()
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, expected, atol=0.01, rtol=0.01)
    finally:
        torch.npu.synchronize()
        if graph is not None:
            graph.reset()


@pytest.mark.parametrize("ratio", [1, 2])
@pytest.mark.parametrize("queries", [1, 6])
@pytest.mark.parametrize("capture", [False, True])
def test_shared_source_refills_five_index_groups_from_real_host(ratio, queries, capture):
    from vllm.v1.hisparse.runtime import (
        HiSparseCacheHandle,
        HiSparseRuntime,
        ResolvedHiSparseConfig,
        initialize_hisparse_runtime_buffers,
    )

    from vllm_ascend.hisparse.backend import AscendHiSparseBackend
    from vllm_ascend.hisparse.host_pool import allocate_mapped_host
    from vllm_ascend.models.deepseek_v41.hisparse import (
        prepare_source_attention,
        prepare_source_batch,
        resolve_attention,
    )

    if not torch.npu.is_available() or not torch.npu.get_device_name(0).startswith("Ascend950"):
        pytest.skip("DSV4.1 HiSparse requires mapped Host on Ascend 950")
    torch.npu.set_device(0)
    backend = AscendHiSparseBackend()
    backend.check_available()
    device = torch.device("npu:0")
    rows, block = 2 * queries, 128 // ratio
    hot_size, source_rows = 4096, 8192
    hot_pages = hot_size // block
    blocks = 1 + 2 * hot_pages
    stride = block * 320 * 2
    raw = torch.empty(blocks * stride, dtype=torch.int8, device=device)
    runtime = HiSparseRuntime(
        ResolvedHiSparseConfig(top_k=512, device_buffer_size=hot_size, max_union_rows=queries * 512),
        max_num_reqs=2,
        row_width=320,
        kv_dtype=torch.uint8,
        device=device,
        max_swap_rows=rows * 5,
        storage_block_size=block,
    )
    handle = HiSparseCacheHandle(runtime)
    resident_table = torch.zeros((2, source_rows // block), dtype=torch.int32, device=device)
    handle.bind_cache(
        raw,
        byte_offset=0,
        block_stride=stride,
        num_blocks=blocks,
        block_size=128,
        storage_block_size=block,
        block_table=resident_table,
        slot_mapping=torch.full((rows,), -1, dtype=torch.int64, device=device),
    )
    hot_table = torch.arange(1, blocks, dtype=torch.int32, device=device).view(2, -1)
    runtime.bind_hot_cache(
        raw,
        byte_offset=0,
        block_stride=stride,
        num_blocks=blocks,
        block_size=128,
        storage_block_size=block,
        block_table=hot_table,
    )
    host = allocate_mapped_host((block + source_rows * 2, 320), torch.uint8)
    host.copy_((torch.arange(host.numel(), dtype=torch.int64) % 251).to(torch.uint8).view_as(host))
    runtime.bind_source_cache(host, registered_host_pool=host)
    initialize_hisparse_runtime_buffers([handle], max_num_reqs=2)
    runtime.request_state_indices.copy_(torch.tensor([1, 0], dtype=torch.int32, device=device))
    handle.source_block_table = torch.arange(1, 1 + source_rows * 2 // block, dtype=torch.int32, device=device).view(
        2, -1
    )
    prepare_source_attention(handle)
    starts = torch.tensor([0, queries, rows], dtype=torch.int32, device=device)
    req_ids = torch.arange(2, dtype=torch.int32, device=device).repeat_interleave(queries)
    metadata = AscendDSAV41Metadata(
        block_table=handle.source_block_table,
        query_start_loc=starts,
        seq_lens=torch.full((2,), source_rows * ratio, dtype=torch.int32, device=device),
        cache_seq_lens=torch.full((2,), source_rows, dtype=torch.int32, device=device),
        slot_mapping=torch.empty((0, 2), dtype=torch.int32, device=device),
        compress_ratio=ratio,
        storage_block_size=block,
        logical_block_size=128,
        is_compressor_state=False,
        num_reqs=2,
        num_actual_reqs=2,
        num_actual_tokens=rows,
        num_input_tokens=rows,
        num_decode_tokens=rows,
        max_query_len=queries,
        req_id_per_token=req_ids,
    )
    selections = []
    for group in range(5):
        # Disjoint per-query selections exercise the verification-row union;
        # successive index groups evict earlier groups after their reads finish.
        logical = (torch.arange(rows * 512, dtype=torch.int32).view(rows, 512) + group * queries * 512) % source_rows
        selections.append(logical.to(device))
    expected = [
        host[(block + req_ids.cpu().long()[:, None] * source_rows + topk.cpu().long())].clone() for topk in selections
    ]

    def forward():
        prepare_source_batch(handle, metadata)
        outputs = []
        for selected in selections:
            cache, _, physical, _ = resolve_attention(handle, metadata, selected, new_index_group=True)
            # Read before the following index group is allowed to evict slots.
            outputs.append(cache.view(-1, 320)[physical.long()].clone())
            follower = resolve_attention(handle, metadata, selected, new_index_group=False)
            assert follower[2] is physical
        return outputs

    graph = None
    try:
        forward()
        torch.npu.synchronize()
        if capture:
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                actual = forward()
            host.bitwise_xor_(127)
            expected = [x.bitwise_xor(127) for x in expected]
            runtime.reset_hot_state()
            graph.replay()
        else:
            runtime.reset_hot_state()
            actual = forward()
        torch.npu.synchronize()
        for got, want in zip(actual, expected, strict=True):
            torch.testing.assert_close(got.cpu(), want, rtol=0, atol=0)
    finally:
        torch.npu.synchronize()
        if graph is not None:
            graph.reset()
        host._hisparse_registration.close()


@pytest.mark.parametrize("capture", [False, True])
def test_request_mapping_refresh_uses_persistent_graph_storage(capture):
    from vllm_ascend.models.deepseek_v41.hisparse import build_request_ids

    if not torch.npu.is_available():
        pytest.skip("Requires an NPU")
    torch.npu.set_device(0)
    rows = torch.arange(8, dtype=torch.int32, device="npu:0")
    starts = torch.tensor([0, 2, 6, 8], dtype=torch.int32, device="npu:0")
    output = torch.empty(8, dtype=torch.int32, device="npu:0")
    address = output.data_ptr()
    build_request_ids(rows, starts, output)
    graph = None
    try:
        if capture:
            torch.npu.synchronize()
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                build_request_ids(rows, starts, output)
        starts.copy_(torch.tensor([0, 3, 3, 5], dtype=torch.int32, device="npu:0"))
        if graph is None:
            build_request_ids(rows, starts, output)
        else:
            graph.replay()
        torch.npu.synchronize()
        assert output.data_ptr() == address
        assert output.cpu().tolist() == [0, 0, 0, 2, 2, 3, 3, 3]
    finally:
        torch.npu.synchronize()
        if graph is not None:
            graph.reset()
