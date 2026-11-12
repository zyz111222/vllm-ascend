# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""CPU storage contracts for compressed shared-source HiSparse (no NPU init)."""

import os
from copy import copy
from pathlib import Path

import numpy as np
import pytest
import torch
from vllm.config import (
    AttentionConfig,
    CacheConfig,
    KVTransferConfig,
    ModelConfig,
    SchedulerConfig,
    VllmConfig,
)
from vllm.distributed.kv_transfer.kv_connector.v1.hisparse.worker import compressed_row_mirror_spans
from vllm.v1.hisparse.layout import get_hisparse_kv_cache_config
from vllm.v1.hisparse.runtime import PagedCacheView, _get_max_swap_rows
from vllm.v1.kv_cache_interface import HiSparseHotSpec, HiSparseResidentSpec, UniformTypeKVCacheSpecs
from vllm.v1.worker.utils import allocate_kv_cache

from vllm_ascend.attention.dsa_v41 import AscendDSAV41Metadata
from vllm_ascend.core.kv_cache_interface import AscendCircularBufferSpec
from vllm_ascend.models.deepseek_v41.cache_config import (
    make_folded_index_cache_spec,
    make_index_cache_spec,
    make_mla_cache_spec,
    make_swa_cache_spec,
)
from vllm_ascend.models.deepseek_v41.hisparse import (
    cache_groups,
    offloaded_sources,
    prepare_cache_views,
    resident_write_slots,
)


@pytest.fixture(scope="module")
def runtime():
    model = Path(os.environ.get("DSV41_TEST_MODEL", "/home/weights/DeepSeek-V4.1-Flash-1017"))
    if not (model / "config.json").is_file():
        pytest.skip("set DSV41_TEST_MODEL to a local V4.1 config directory")
    config = VllmConfig(
        model_config=ModelConfig(
            model=str(model), tokenizer_mode="deepseek_v41", trust_remote_code=True, max_model_len=1048576
        ),
        cache_config=CacheConfig(block_size=128),
        attention_config=AttentionConfig(hisparse_config={"device_buffer_size": 4096}),
        scheduler_config=SchedulerConfig(
            is_encoder_decoder=False, max_model_len=1048576, max_num_seqs=8, max_num_batched_tokens=4096
        ),
        kv_transfer_config=KVTransferConfig(
            kv_connector="HiSparseConnector", kv_role="kv_both", kv_connector_extra_config={"host_pool_gib": 1}
        ),
    )
    # Same public resolved-layout field set by the engine's worker RPC.
    config.cache_config.kv_cache_layout = "BLHNC"
    return config


@pytest.fixture
def specs():
    result = {}
    for layer, ratio in [(2, 2), (8, 2), (14, 2), (20, 1)]:
        name = f"model.layers.{layer}.self_attn"
        result[name + ".long_kv_cache"] = make_mla_cache_spec(block_size=128, head_size=512, compress_ratio=ratio)
        result[name + ".indexer.k_cache"] = make_index_cache_spec(block_size=128, head_size=128, compress_ratio=ratio)
        if ratio == 2:
            result[name + ".compressor.state_cache"] = AscendCircularBufferSpec(
                block_size=32, num_kv_heads=1, head_size=1024, dtype=torch.float32, head_size_v=0
            )
    result["model.layers.20.self_attn.indexer.k_cache_folded"] = make_folded_index_cache_spec(block_size=128)
    for layer in range(40):
        result[f"model.layers.{layer}.self_attn.swa_cache"] = make_swa_cache_spec(
            block_size=128, window_size=128, head_size=512, dtype=torch.bfloat16, cache_dtype="auto"
        )
    return result


@pytest.mark.parametrize("owners", [[20], [2, 20], [2, 8, 14, 20]])
def test_selective_sources_preserve_all_cache_roles(runtime, specs, owners):
    runtime.kv_transfer_config.kv_connector_extra_config["dsv41_sources"] = owners
    groups = cache_groups(runtime, specs)
    assert {int(n.split(".layers.")[1].split(".")[0]) for n in groups[0].layer_names} == set(owners)
    listed = [
        n
        for g in groups
        for n in g.layer_names
        if not isinstance(g.kv_cache_spec, (HiSparseHotSpec, HiSparseResidentSpec))
    ]
    assert set(listed) == set(specs)
    assert len(listed) == len(specs)
    for group in groups:
        if isinstance(group.kv_cache_spec, HiSparseHotSpec):
            ratio = specs[group.layer_names[0].removesuffix(".hisparse_hot")].tokens_per_state
            assert group.kv_cache_spec.storage_block_size == 128 // ratio
            assert group.kv_cache_spec.blocks_per_request == 4096 // (128 // ratio)


@pytest.mark.parametrize("owners", [[20], [2, 20], [2, 8, 14, 20]])
def test_real_cpu_allocation_keeps_indexer_key_scale_and_state_geometry(runtime, specs, owners):
    runtime.kv_transfer_config.kv_connector_extra_config["dsv41_sources"] = owners
    groups = cache_groups(runtime, specs)
    cfg = get_hisparse_kv_cache_config(runtime, groups, 8 * 1024**2, 1024**3)
    device_cfg = copy(cfg)
    device_cfg.kv_cache_tensors = [t for t in cfg.kv_cache_tensors if not t.host_resident]
    caches = allocate_kv_cache(device_cfg, torch.device("cpu"), runtime.cache_config.get_resolved_kv_cache_layout())
    # The host source is allocated/bound by Core; this test exercises the same
    # per-page indexer and circular-state views over the bounded device slab.
    device_cfg.kv_cache_groups = [g for g in groups if not g.host_resident]
    prepare_cache_views(device_cfg, caches)
    key, scale = caches["model.layers.2.self_attn.indexer.k_cache"]
    assert key.shape[1:] == (64, 1, 64)
    assert scale.shape[1:] == (64, 1, 4)
    key[1].fill_(17)
    scale[1].fill_(29)
    assert torch.all(key[1] == 17) and torch.all(scale[1] == 29)
    assert scale.data_ptr() - key.data_ptr() == 64 * 64
    state = caches["model.layers.2.self_attn.compressor.state_cache"]
    assert state.shape[1:] == (32, 1, 1024)
    assert state.dtype == torch.float32


@pytest.mark.parametrize("ratio", [1, 2])
def test_compressed_mirror_span_matches_complete_token_groups(ratio):
    # Includes partial start/end groups, a full page, and an unfinished row.
    starts = np.array([0, 1, 2, 127, 128, 129], dtype=np.int64)
    counts = np.array([1, 1, 1, 1, 128, 7], dtype=np.int64)
    blocks, offsets, dest, amount = compressed_row_mirror_spans(
        starts + 256, starts, counts, logical_block_size=128, storage_block_size=128 // ratio
    )
    expected = []
    for start, count in zip(starts, counts, strict=True):
        for token in range(start, start + count):
            if (token + 1) % ratio == 0:
                expected.append((int((token + 256) // ratio), int(token // ratio)))
    actual = []
    for block, offset, target, count in zip(blocks, offsets, dest, amount, strict=True):
        for row in range(int(count)):
            actual.append((int(block * (128 // ratio) + offset + row), int(target + row)))
    assert actual == expected


def test_shared_source_workspace_covers_five_index_groups(runtime):
    # Eight requests, one query each, five independent consumers of source 20.
    assert _get_max_swap_rows(runtime) == 8 * 5


@pytest.mark.parametrize("physical_rows", [64, 128])
def test_paged_hot_view_preserves_physical_addresses(physical_rows):
    raw = torch.empty(3 * 409600, dtype=torch.int8)
    view = PagedCacheView.bind(
        raw,
        dtype=torch.uint8,
        row_width=320,
        byte_offset=40960,
        block_stride=409600,
        num_blocks=3,
        block_size=physical_rows,
    )
    view.cache[1, 3].fill_(73)
    dense_row = (409600 // 320) + 3
    assert torch.all(view.attention_cache.view(-1, 320)[dense_row] == 73)
    assert view.attention_block_stride == 1280


@pytest.mark.parametrize("owners", [[], [21], ["20"]])
def test_invalid_source_selection_is_rejected(runtime, owners):
    runtime.kv_transfer_config.kv_connector_extra_config["dsv41_sources"] = owners
    with pytest.raises(ValueError, match="KV owners"):
        offloaded_sources(runtime)


@pytest.mark.parametrize("ratio", [1, 2])
def test_resident_writes_exclude_padding_and_incomplete_c2(ratio):
    metadata = AscendDSAV41Metadata(
        block_table=torch.zeros((2, 1), dtype=torch.int32),
        query_start_loc=torch.tensor([0, 3, 3], dtype=torch.int32),
        seq_lens=torch.tensor([7, 0], dtype=torch.int32),
        slot_mapping=torch.empty((4, 2), dtype=torch.int32),
        compress_ratio=ratio,
        storage_block_size=128 // ratio,
        is_compressor_state=False,
        positions=torch.tensor([4, 5, 6, 7]),
        num_actual_tokens=3,
        num_input_tokens=4,
        num_actual_reqs=1,
    )
    result = resident_write_slots(torch.tensor([260, 261, 262, 263]), metadata)
    assert result.tolist() == ([260, 261, 262, -1] if ratio == 1 else [-1, 130, -1, -1])


def test_replay_skip_update_masks_resident_writes():
    metadata = AscendDSAV41Metadata(
        block_table=torch.zeros((1, 1), dtype=torch.int32),
        query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
        seq_lens=torch.tensor([6], dtype=torch.int32),
        slot_mapping=torch.full((2, 2), -1, dtype=torch.int32),
        flat_slot_mapping=torch.full((2,), -1, dtype=torch.int64),
        compress_ratio=2,
        storage_block_size=64,
        is_compressor_state=False,
        positions=torch.tensor([4, 5]),
        num_actual_tokens=2,
        num_input_tokens=2,
        num_actual_reqs=1,
    )
    assert resident_write_slots(torch.tensor([260, 261]), metadata).tolist() == [-1, -1]


@pytest.mark.parametrize("owners", [[20], [2, 20], [2, 8, 14, 20]])
def test_engine_global_planning_keeps_host_budget_and_converges_ranks(runtime, specs, owners):
    from vllm.v1.core.kv_cache_planning import (
        get_global_kv_cache_groups,
        get_kv_cache_configs,
    )

    from vllm_ascend.models.deepseek_v41.hisparse import resolve_specs

    runtime.kv_transfer_config.kv_connector_extra_config["dsv41_sources"] = owners
    resolved = resolve_specs(runtime, specs)
    groups = get_global_kv_cache_groups(runtime, [resolved, resolved])
    configs = get_kv_cache_configs(runtime, [resolved, resolved], [30 * 2**30, 31 * 2**30], groups)
    assert configs[0].num_blocks == configs[1].num_blocks
    assert configs[0].hisparse_host_num_blocks == configs[1].hisparse_host_num_blocks
    for cfg in configs:
        host = [t for t in cfg.kv_cache_tensors if t.host_resident]
        device = [t for t in cfg.kv_cache_tensors if not t.host_resident]
        assert host and device
        assert len({t.size for t in device}) == 1
        per_layer = {
            name: (
                g.kv_cache_spec.kv_cache_specs[name]
                if isinstance(g.kv_cache_spec, UniformTypeKVCacheSpecs)
                else g.kv_cache_spec
            )
            for g in cfg.kv_cache_groups
            for name in g.layer_names
        }
        assert (
            max(
                t.offset
                + t.layer_stride * (len(t.layers) - 1)
                + t.block_stride * (cfg.num_blocks - 1)
                + per_layer[t.layers[-1]].page_size_bytes
                for t in device
            )
            <= device[0].size
        )


def test_reduced_backbone_never_groups_target_swa_with_dspark_draft(runtime, specs):
    from dataclasses import replace

    runtime.kv_transfer_config.kv_connector_extra_config["dsv41_sources"] = [20]
    for layer in (37, 38, 39):
        del specs[f"model.layers.{layer}.self_attn.swa_cache"]
    target = specs["model.layers.0.self_attn.swa_cache"]
    for layer in range(3):
        specs[f"model.mtp.{layer}.self_attn.swa_cache"] = replace(target, is_draft=True)
    for group in cache_groups(runtime, specs):
        if any(name.endswith(".swa_cache") for name in group.layer_names):
            assert len({specs[name].is_draft for name in group.layer_names}) == 1


def test_request_ids_keep_storage_across_changed_graph_batches():
    from vllm_ascend.models.deepseek_v41.hisparse import build_request_ids

    output = torch.empty(8, dtype=torch.int32)
    address = output.data_ptr()
    rows = torch.arange(8)
    result = build_request_ids(rows, torch.tensor([0, 2, 6, 8]), output)
    assert result is output and output.data_ptr() == address
    assert output.tolist() == [0, 0, 1, 1, 1, 1, 2, 2]
    result = build_request_ids(rows, torch.tensor([0, 3, 3, 5]), output)
    assert result is output and output.data_ptr() == address
    assert output.tolist() == [0, 0, 0, 2, 2, 3, 3, 3]
