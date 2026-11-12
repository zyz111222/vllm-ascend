"""Prepare or run isolated DSV4.1 baseline / HiSparse comparison endpoints."""

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


def commands(args):
    task = Path(__file__).resolve().parents[2].parent
    core, ascend = task / "vllm", task / "vllm-ascend"
    base_env = os.environ.copy()
    base_env.update(
        {
            "PYTHONPATH": f"{core}:{ascend}",
            "VLLM_USE_V2_MODEL_RUNNER": "1",
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
            "HCCL_IF_IP": args.ip,
            "VLLM_HOST_IP": args.ip,
            "GLOO_SOCKET_IFNAME": args.nic,
            "TP_SOCKET_IFNAME": args.nic,
            "HCCL_SOCKET_IFNAME": args.nic,
            "HCCL_OP_EXPANSION_MODE": "CCU_SCHED",
            "HCCL_BUFFSIZE": "1024",
            "ASCEND_LOCAL_COMM_RES": '{"version":"1.3"}',
            "OMP_NUM_THREADS": "10",
        }
    )
    result = []
    for rank in range(args.dp):
        env = base_env.copy()
        env["ASCEND_RT_VISIBLE_DEVICES"] = ",".join(str(rank * args.tp + j) for j in range(args.tp))
        env["HCCL_HOST_SOCKET_PORT_RANGE"] = f"{61000 + rank * 100}-{61099 + rank * 100}"
        env["HCCL_NPU_SOCKET_PORT_RANGE"] = env["HCCL_HOST_SOCKET_PORT_RANGE"]
        cmd = [
            sys.executable,
            "-m",
            "vllm.entrypoints.cli.main",
            "serve",
            args.model,
            "--host",
            "127.0.0.1",
            "--port",
            str(args.port + rank),
            "--served-model-name",
            "dsv41",
            "--data-parallel-size",
            str(args.dp),
            "--data-parallel-size-local",
            "1",
            "--data-parallel-rank",
            str(rank),
            "--data-parallel-address",
            args.ip,
            "--data-parallel-rpc-port",
            str(args.rpc_port),
            "--tensor-parallel-size",
            str(args.tp),
            "--enable-expert-parallel",
            "--max-model-len",
            str(args.max_model_len),
            "--max-num-batched-tokens",
            str(args.batch_tokens),
            "--max-num-seqs",
            str(args.seqs),
            "--gpu-memory-utilization",
            str(args.memory_utilization),
            "--block-size",
            "128",
            "--quantization",
            "deepseek_v4_fp8",
            "--dtype",
            "bfloat16",
            "--seed",
            "42",
            "--engram-config",
            '{"cpu_offload":true,"dp_shared_memory":true}',
            "--tokenizer-mode",
            "deepseek_v41",
            "--trust-remote-code",
            "--enable-prefix-caching",
            "--swa-bounded-replay",
            "--async-scheduling",
            "--safetensors-load-strategy",
            "lazy",
            "--compilation-config",
            json.dumps({"cudagraph_mode": "NONE" if args.eager else "FULL_DECODE_ONLY"}),
            "--additional-config",
            '{"enable_cpu_binding":true,"enable_flashcomm1":false,"multistream_overlap_shared_expert":true}',
        ]
        if args.mode == "on":
            env["VLLM_KV_CACHE_LAYOUT"] = "BLHNC"
            cmd += [
                "--attention-config",
                json.dumps(
                    {
                        "hisparse_config": {
                            "device_buffer_size": args.hot_rows,
                            "eager_host_mirror": True,
                        }
                    }
                ),
                "--kv-transfer-config",
                json.dumps(
                    {
                        "kv_connector": "HiSparseConnector",
                        "kv_role": "kv_both",
                        "kv_connector_extra_config": {
                            "host_pool_gib": args.host_pool_gib,
                            "dsv41_sources": args.sources,
                        },
                    }
                ),
            ]
        if args.speculative_tokens:
            cmd += [
                "--speculative-config",
                json.dumps(
                    {
                        "method": "dspark",
                        "num_speculative_tokens": args.speculative_tokens,
                        "enforce_eager": True,
                    }
                ),
            ]
        result.append((cmd, env))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["off", "on"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dp", type=int, default=8)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--port", type=int, default=37941)
    parser.add_argument("--rpc-port", type=int, default=38141)
    parser.add_argument("--ip", required=True)
    parser.add_argument("--nic", required=True)
    parser.add_argument("--max-model-len", type=int, default=1048576)
    parser.add_argument("--batch-tokens", type=int, default=4096)
    parser.add_argument("--seqs", type=int, default=64)
    parser.add_argument("--memory-utilization", type=float, default=0.90)
    parser.add_argument("--speculative-tokens", type=int, default=5)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--sources", type=lambda s: [int(x) for x in s.split(",")], default=[20])
    parser.add_argument("--hot-rows", type=int, default=4096)
    parser.add_argument("--host-pool-gib", type=float, default=64)
    parser.add_argument("--eager", action="store_true")
    args = parser.parse_args()
    if args.dp * args.tp != 8:
        parser.error("This preparation targets the 77 server's eight devices.")
    if args.speculative_tokens < 0:
        parser.error("Speculative token count must be nonnegative; zero disables DSpark.")
    if not args.sources or not set(args.sources) <= {2, 8, 14, 20}:
        parser.error("sources must select KV owners 2,8,14,20")
    if args.hot_rows < (args.speculative_tokens + 1) * 512:
        parser.error("hot-rows must cover the speculative query union")
    prepared = commands(args)
    args.run_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "mode": args.mode,
        "settings": {
            "batch_tokens": args.batch_tokens,
            "max_model_len": args.max_model_len,
            "seqs": args.seqs,
            "memory_utilization": args.memory_utilization,
            "speculative_tokens": args.speculative_tokens,
            "eager": args.eager,
            "sources": args.sources,
            "hot_rows": args.hot_rows,
        },
        "status": "prepared_unvalidated",
        "topology": {"dp": args.dp, "tp": args.tp},
        "model": args.model,
        "endpoints": [f"http://127.0.0.1:{args.port + rank}" for rank in range(args.dp)],
        "commands": [
            {
                "argv": cmd,
                "env": {k: v for k, v in env.items() if k not in os.environ or os.environ[k] != v},
            }
            for cmd, env in prepared
        ],
        "upstream_references": [65905, 65970],
        "note": "No claimed DSV4.1 HiSparse qualification. Engram CPU offload is fixed on both arms.",
    }
    task = Path(__file__).resolve().parents[2].parent
    manifest["source"] = {}
    for name in ["vllm", "vllm-ascend"]:
        repo = task / name
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        diff = subprocess.run(["git", "diff", "--binary"], cwd=repo, capture_output=True, check=True).stdout
        for extra in subprocess.check_output(
            ["git", "ls-files", "--others", "--exclude-standard"], cwd=repo, text=True
        ).splitlines():
            if Path(extra).suffix in {".py", ".md", ".yaml"}:
                addition = subprocess.run(
                    ["git", "diff", "--binary", "--no-index", "--", "/dev/null", extra],
                    cwd=repo,
                    capture_output=True,
                )
                if addition.returncode != 1:
                    raise RuntimeError(f"Cannot snapshot {extra}: {addition.stderr!r}")
                diff += addition.stdout
        (args.run_dir / (name + ".patch")).write_bytes(diff)
        manifest["source"][name] = {
            "path": str(repo),
            "head": head,
            "diff_sha256": hashlib.sha256(diff).hexdigest(),
        }
    native = task / "vllm-ascend/vllm_ascend/vllm_ascend_C.cpython-312-aarch64-linux-gnu.so"
    manifest["native_sha256"] = hashlib.sha256(native.read_bytes()).hexdigest()
    vendor = task / "vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer"
    manifest["vendor_sha256"] = hashlib.sha256(
        json.dumps(
            {
                str(p.relative_to(vendor)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(vendor.rglob("*"))
                if p.is_file()
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    manifest["model_config_sha256"] = hashlib.sha256((Path(args.model) / "config.json").read_bytes()).hexdigest()
    (args.run_dir / "launch.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    if args.dry_run:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return
    if args.mode == "on":
        probe = args.run_dir / "compatibility.json"
        subprocess.run(
            [
                prepared[0][0][0],
                str(Path(__file__).with_name("inspect_decoder_kvoffload.py")),
                "--model",
                args.model,
                "--out",
                str(probe),
                "--sources",
                ",".join(map(str, args.sources)),
                "--hot-rows",
                str(args.hot_rows),
            ],
            env=prepared[0][1],
            check=True,
        )
        if not json.loads(probe.read_text())["probes"]["hisparse"]["layout_created"]:
            raise SystemExit("DSV4.1 HiSparse layout preflight failed; inspect compatibility.json. No server started.")
    npu = subprocess.run(["npu-smi", "info"], capture_output=True, text=True, check=True).stdout
    if npu.count("No running processes found in NPU") != 8:
        raise SystemExit("77 is occupied; existing services are left running.")
    processes = []
    try:
        for rank, (cmd, env) in enumerate(prepared):
            with (args.run_dir / f"rank{rank}.log").open("wb") as log:
                process = subprocess.Popen(
                    cmd,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            processes.append(process)
        (args.run_dir / "owned-processes.json").write_text(json.dumps([p.pid for p in processes]))
        deadline = time.monotonic() + 1800
        pending = set(manifest["endpoints"])
        while pending:
            if any(p.poll() is not None for p in processes):
                raise RuntimeError("A rank exited before readiness; inspect rank logs.")
            for endpoint in list(pending):
                try:
                    with urllib.request.urlopen(endpoint + "/health", timeout=2) as response:
                        if response.status == 200:
                            pending.remove(endpoint)
                except OSError:
                    pass
            if time.monotonic() > deadline:
                raise TimeoutError("DSV4.1 startup did not become healthy within 1800s")
            if pending:
                time.sleep(2)
        manifest["status"] = "running"
        (args.run_dir / "launch.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
        print("All endpoints healthy; launch.json is ready for the benchmark.", flush=True)
        while all(p.poll() is None for p in processes):
            time.sleep(2)
        raise RuntimeError("A rank exited; inspect rank logs.")
    except KeyboardInterrupt:
        pass
    finally:
        for process in processes:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        for process in processes:
            try:
                process.wait(timeout=40)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        manifest["status"] = "stopped"
        (args.run_dir / "launch.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
