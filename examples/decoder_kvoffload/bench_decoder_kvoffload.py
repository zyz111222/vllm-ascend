"""Measure fixed-length DSV4.1 requests with N=4C and per-session prefixes."""

import argparse
import asyncio
import hashlib
import json
import random
import statistics
import time
import uuid
from pathlib import Path


def tokens(count, seed):
    rng = random.Random(seed)
    return [rng.randrange(1000, 100000) for _ in range(count)]


def percentile(values, q):
    values = sorted(values)
    position = (len(values) - 1) * q
    lo, hi = int(position), min(int(position) + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (position - lo)


def metric(raw, name):
    values = [
        float(line.rsplit(" ", 1)[1])
        for line in raw.splitlines()
        if line.startswith(name + "{") or line.startswith(name + " ")
    ]
    return sum(values) if values else None


async def execute(args):
    import aiohttp

    launch_manifest = json.loads(args.launch_manifest.read_text())
    args.out.mkdir(parents=True, exist_ok=False)
    prefix_count = int(args.input_tokens * args.cache_ratio) // 128 * 128
    prefixes = [tokens(prefix_count, args.seed + i) for i in range(args.concurrency)]
    request_count = 4 * args.concurrency
    cache_namespace = uuid.uuid4().hex
    endpoints = [x.rstrip("/") for x in args.endpoints.split(",")]
    manifest = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    manifest["out"] = str(args.out)
    manifest.update(
        {
            "requests": request_count,
            "prefix_tokens": prefix_count,
            "cache_namespace": cache_namespace,
            "mode": args.mode,
            "endpoints": endpoints,
        }
    )
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))

    def prepare(index, warm=False):
        slot = index % args.concurrency
        prompt = (
            prefixes[slot]
            if warm
            else prefixes[slot] + tokens(args.input_tokens - prefix_count, args.seed + 100000 + index)
        )
        count = 1 if warm else args.output_tokens
        body = {
            "model": "dsv41",
            "prompt": prompt,
            "max_tokens": count,
            "min_tokens": count,
            "ignore_eos": True,
            "temperature": 0,
            "seed": args.seed,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        # Keep each trial independent, including repeated concurrency sweeps.
        # Warmup and the four turns share only their session namespace.
        body["cache_salt"] = f"{cache_namespace}-{slot if prefix_count else index}"
        row = {
            "request": index,
            "endpoint": endpoints[slot % len(endpoints)],
            "warmup": warm,
            "prompt_sha256": hashlib.sha256(json.dumps(prompt).encode()).hexdigest(),
            "actual_input_tokens": len(prompt),
            "success": False,
        }
        return json.dumps(body, separators=(",", ":")).encode(), row

    # Generate and encode every request before measurement so long prompts do
    # not block the event loop while Decode responses are being timed.
    payloads = [prepare(i) for i in range(request_count)]
    warm_payloads = [prepare(i, True) for i in range(args.concurrency)] if prefix_count else []
    semaphore = asyncio.Semaphore(args.concurrency)
    timeout = aiohttp.ClientTimeout(total=args.timeout, sock_read=args.timeout)
    async with aiohttp.ClientSession(timeout=timeout, read_bufsize=1024**2) as session:

        async def snapshots(label):
            result = {}
            for endpoint in endpoints:
                async with session.get(endpoint + "/metrics") as response:
                    response.raise_for_status()
                    result[endpoint] = await response.text()
            (args.out / (label + "-metrics.json")).write_text(json.dumps(result))
            return result

        async def request(index, warm=False):
            payload, original = (warm_payloads if warm else payloads)[index]
            row = original.copy()
            endpoint = row["endpoint"]
            count = 1 if warm else args.output_tokens
            async with semaphore:
                start = time.perf_counter()
                first = None
                try:
                    async with session.post(
                        endpoint + "/v1/completions",
                        data=payload,
                        headers={"Content-Type": "application/json"},
                    ) as response:
                        row["http_status"] = response.status
                        if response.status != 200:
                            raise RuntimeError((await response.text())[:2000])
                        async for raw in response.content:
                            line = raw.decode().strip()
                            if not line.startswith("data:"):
                                continue
                            payload = line[5:].strip()
                            if payload == "[DONE]":
                                break
                            data = json.loads(payload)
                            if data.get("error"):
                                raise RuntimeError(str(data["error"]))
                            for choice in data.get("choices", []):
                                if first is None and (choice.get("text") or choice.get("token_ids")):
                                    first = time.perf_counter()
                            if data.get("usage") is not None:
                                row["usage"] = data["usage"]
                    end = time.perf_counter()
                    usage = row.get("usage", {})
                    if (
                        first is None
                        or usage.get("prompt_tokens") != row["actual_input_tokens"]
                        or usage.get("completion_tokens") != count
                    ):
                        raise ValueError("Missing first token or actual token-length mismatch")
                    row.update(
                        {
                            "success": True,
                            "ttft_s": first - start,
                            "latency_s": end - start,
                            "mean_tpot_ms": 1000 * (end - first) / (count - 1) if count > 1 else None,
                        }
                    )
                except (
                    aiohttp.ClientError,
                    TimeoutError,
                    ValueError,
                    RuntimeError,
                ) as error:
                    row["error"] = str(error)
                    row["latency_s"] = time.perf_counter() - start
            return row

        if prefix_count:
            warm = await asyncio.gather(*(request(i, True) for i in range(args.concurrency)))
            (args.out / "warmup.json").write_text(json.dumps(warm, indent=2))
            if not all(x["success"] for x in warm):
                raise RuntimeError("Prefix warmup failed; scored requests were not started")
        before = await snapshots("before")
        start = time.perf_counter()

        async def conversation(slot):
            return [await request(slot + turn * args.concurrency) for turn in range(4)]

        groups = await asyncio.gather(*(conversation(slot) for slot in range(args.concurrency)))
        rows = [row for group in groups for row in group]
        elapsed = time.perf_counter() - start
        after = await snapshots("after")
    (args.out / "requests.json").write_text(json.dumps(rows, indent=2))
    good = [x for x in rows if x["success"]]

    def delta(name):
        totals = []
        for endpoint in endpoints:
            a, b = metric(before[endpoint], name), metric(after[endpoint], name)
            if a is None or b is None:
                return None
            totals.append(b - a)
        return sum(totals)

    hits, queries = (
        delta("vllm:prefix_cache_hits_total"),
        delta("vllm:prefix_cache_queries_total"),
    )
    cache_ratio = hits / queries if hits is not None and queries else None
    means = [x["mean_tpot_ms"] for x in good]
    mean = statistics.mean(means) if means else None
    valid = len(good) == request_count and cache_ratio is not None and abs(cache_ratio - args.cache_ratio) <= 0.01
    h2d = delta("vllm:hisparse_host_to_device_bytes_total")
    if args.mode == "on" and h2d is None:
        valid = False
    result = {
        "mode": args.mode,
        "launch_manifest": str(args.launch_manifest.resolve()),
        "topology": launch_manifest.get("topology"),
        "settings": launch_manifest.get("settings"),
        "source": launch_manifest.get("source"),
        "native_sha256": launch_manifest.get("native_sha256"),
        "vendor_sha256": launch_manifest.get("vendor_sha256"),
        "model_config_sha256": launch_manifest.get("model_config_sha256"),
        "requested_cache_ratio": args.cache_ratio,
        "cache_namespace": cache_namespace,
        "concurrency": args.concurrency,
        "requests": request_count,
        "completed": len(good),
        "failed": request_count - len(good),
        "input_tokens": args.input_tokens,
        "output_tokens": args.output_tokens,
        "measured_cache_ratio": cache_ratio,
        "prefix_hits": hits,
        "prefix_queries": queries,
        "duration_s": elapsed,
        "output_tokens_s": len(good) * args.output_tokens / elapsed,
        "requests_s": len(good) / elapsed,
        "mean_tpot_ms": mean,
        "p99_request_tpot_ms": percentile(means, 0.99) if means else None,
        "mean_ttft_s": statistics.mean(x["ttft_s"] for x in good) if good else None,
        "preemptions_api": delta("vllm:num_preemptions_total"),
        "valid": valid,
        "hisparse_h2d_bytes_software_estimate": h2d,
        "hisparse_misses_software_estimate": delta("vllm:hisparse_cache_misses_total"),
        "mean_tpot_le20ms": valid and mean <= 20,
        "mean_tpot_le50ms": valid and mean <= 50,
        "note": (
            "Whole-run throughput; not a pure Decode-window metric. API preemption counters may lag before first token."
        ),
    }
    (args.out / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return 0 if valid else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["off", "on"], required=True)
    parser.add_argument("--launch-manifest", type=Path, required=True)
    parser.add_argument(
        "--endpoints",
        help="Comma-separated rank APIs; defaults to launch.json endpoints",
    )
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--input-tokens", type=int, default=131072)
    parser.add_argument("--output-tokens", type=int, default=1024)
    parser.add_argument("--cache-ratio", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=float, default=3600)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.launch_manifest.read_text())
    if manifest["mode"] != args.mode or manifest.get("status") != "running":
        parser.error("Use a manifest from an actually launched matching service.")
    if args.endpoints is None:
        args.endpoints = ",".join(manifest["endpoints"])
    if args.concurrency < 1 or not 0 <= args.cache_ratio < 1 or args.output_tokens < 2:
        parser.error("Require C>=1, 0<=cache ratio<1 and output tokens>=2.")
    if args.input_tokens + args.output_tokens > 1048576:
        parser.error("Input plus output exceeds this checkpoint's 1M context limit.")
    return asyncio.run(execute(args))


if __name__ == "__main__":
    raise SystemExit(main())
