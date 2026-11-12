"""Inspect the actual Ascend DSV4.1 cache layout without loading weights."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def inspect(model, sources=(20,), hot_rows=4096):
    import torch
    from vllm.config import (
        AttentionConfig,
        CacheConfig,
        KVTransferConfig,
        ModelConfig,
        SchedulerConfig,
        VllmConfig,
    )
    from vllm.v1.hisparse.layout import get_hisparse_kv_cache_config

    from vllm_ascend.ascend_config import SparseKVOffloadConfig
    from vllm_ascend.core.kv_cache_interface import AscendCircularBufferSpec
    from vllm_ascend.models.deepseek_v41.cache_config import (
        STATE_RING_ROWS,
        get_layer_tuples,
        make_folded_index_cache_spec,
        make_index_cache_spec,
        make_mla_cache_spec,
        make_swa_cache_spec,
        uses_a5_packed_cache,
    )
    from vllm_ascend.models.deepseek_v41.hisparse import cache_groups

    config_path = model / "config.json"
    config = json.loads(config_path.read_text())
    text = config["text_config"]
    block = 128
    specs = {}
    main_names = []
    for layer in text["kv_source_layer_ids"]:
        prefix = f"model.layers.{layer}.self_attn"
        ratio = text["compress_ratios"][layer]
        main_names.append(prefix + ".long_kv_cache")
        specs[main_names[-1]] = make_mla_cache_spec(block_size=block, head_size=text["head_dim"], compress_ratio=ratio)
        specs[prefix + ".indexer.k_cache"] = make_index_cache_spec(
            block_size=block, head_size=text["index_head_dim"], compress_ratio=ratio
        )
        if ratio == 2:
            specs[prefix + ".compressor.state_cache"] = AscendCircularBufferSpec(
                block_size=STATE_RING_ROWS,
                num_kv_heads=1,
                head_size=2 * text["head_dim"],
                dtype=torch.float32,
                head_size_v=0,
            )
        if layer == text["candidate_source_layer_id"] and uses_a5_packed_cache():
            specs[prefix + ".indexer.k_cache_folded"] = make_folded_index_cache_spec(block_size=block)
    for layer in range(text["num_hidden_layers"]):
        specs[f"model.layers.{layer}.self_attn.swa_cache"] = make_swa_cache_spec(
            block_size=block,
            window_size=text["sliding_window"],
            head_size=text["head_dim"],
            dtype=torch.bfloat16,
            cache_dtype="auto",
        )
    for layer in range(text["num_nextn_predict_layers"]):
        specs[f"model.mtp.{layer}.self_attn.swa_cache"] = make_swa_cache_spec(
            block_size=block,
            window_size=text["sliding_window"],
            head_size=text["head_dim"],
            dtype=torch.bfloat16,
            cache_dtype="auto",
        )
    pages, tuples = get_layer_tuples(specs)
    runtime = VllmConfig(
        model_config=ModelConfig(
            model=str(model),
            tokenizer_mode="deepseek_v41",
            trust_remote_code=True,
            max_model_len=1048576,
        ),
        attention_config=AttentionConfig(hisparse_config={"device_buffer_size": hot_rows, "eager_host_mirror": True}),
        cache_config=CacheConfig(block_size=128),
        scheduler_config=SchedulerConfig(
            is_encoder_decoder=False,
            max_model_len=1048576,
            max_num_seqs=64,
            max_num_batched_tokens=4096,
        ),
        kv_transfer_config=KVTransferConfig(
            kv_connector="HiSparseConnector",
            kv_role="kv_both",
            kv_connector_extra_config={
                "host_pool_gib": 64,
                "dsv41_sources": list(sources),
            },
        ),
    )
    runtime.cache_config.kv_cache_layout = "BLHNC"
    probes = {}
    try:
        groups = cache_groups(runtime, specs)
        planned = get_hisparse_kv_cache_config(runtime, groups, 30 * 2**30, 64 * 2**30)
        probes["hisparse"] = {
            "layout_created": True,
            "offloaded_sources": list(sources),
            "device_buffer_size": hot_rows,
            "device_block_stride": planned.kv_cache_tensors[-1].block_stride,
            "device_num_blocks_at_30gib": planned.num_blocks,
            "host_num_blocks_at_64gib": planned.hisparse_host_num_blocks,
            "groups": [
                {
                    "role": g.role.value,
                    "layers": g.layer_names,
                    "page_bytes": g.kv_cache_spec.page_size_bytes,
                    "storage_rows": getattr(g.kv_cache_spec, "storage_block_size", None),
                }
                for g in groups
            ],
        }
    except (ValueError, AssertionError, NotImplementedError) as error:
        probes["hisparse"] = {"layout_created": False, "error": str(error)}
    try:
        SparseKVOffloadConfig(enabled=True)._validate_preconditions(runtime)
        probes["legacy_sparse"] = {"accepted": True}
    except (ValueError, AssertionError, NotImplementedError) as error:
        probes["legacy_sparse"] = {"accepted": False, "error": str(error)}
    logical_main = sum(specs[name].unpadded_page_size_bytes for name in main_names) / block
    return {
        "model": str(model),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "text_config": text,
        "hardware": {
            "npu": subprocess.run(["npu-smi", "info"], capture_output=True, text=True, check=True).stdout,
            "topology": subprocess.run(
                ["npu-smi", "info", "-t", "topo"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout,
            "meminfo": Path("/proc/meminfo").read_text(),
        },
        "a5_packed": uses_a5_packed_cache(),
        "npu_initialized": torch.npu.is_initialized(),
        "probes": probes,
        "specs": {
            name: {
                "type": type(spec).__name__,
                "page_bytes": spec.unpadded_page_size_bytes,
                "tokens_per_state": spec.tokens_per_state,
            }
            for name, spec in specs.items()
        },
        "physical_tuple_pages": pages,
        "physical_tuples": tuples,
        "physical_pool_bytes_per_logical_block": sum(pages),
        "logical_main_bytes_per_token": logical_main,
        "logical_main_gib_by_context": {
            str(n): n * logical_main / 2**30 for n in [32768, 131072, 262144, 524288, 983040]
        },
        "source_root": str(Path(__file__).resolve().parents[2].parent),
        "note": "Layout/probe evidence only. No DSV4.1 ON inference or throughput is claimed.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sources", default="20")
    parser.add_argument("--hot-rows", type=int, default=4096)
    args = parser.parse_args()
    result = inspect(args.model, [int(x) for x in args.sources.split(",")], args.hot_rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(
        json.dumps(
            {
                k: result[k]
                for k in [
                    "probes",
                    "npu_initialized",
                    "physical_tuple_pages",
                    "logical_main_bytes_per_token",
                    "physical_pool_bytes_per_logical_block",
                ]
            },
            ensure_ascii=False,
            indent=2,
        )
    )
