"""Report scanned throughput maxima under mean TPOT SLOs, preserving conditions."""

import argparse
import json
from pathlib import Path


def summarize(run_dir):
    rows = []
    for path in sorted(run_dir.rglob("result.json")):
        row = json.loads(path.read_text())
        row["evidence"] = str(path.relative_to(run_dir))
        rows.append(row)
    groups = {}
    for row in rows:
        # Smoke rows do not enter formal performance maxima.
        if row["output_tokens"] != 1024:
            continue
        settings = row.get("settings") or {}
        condition = {
            k: row.get(k)
            for k in [
                "input_tokens",
                "output_tokens",
                "requested_cache_ratio",
                "topology",
                "model_config_sha256",
                "source",
                "native_sha256",
                "vendor_sha256",
            ]
        }
        condition["settings"] = settings
        key = json.dumps(condition, sort_keys=True)
        groups.setdefault(key, []).append(row)
    lines = [
        "# DSV4.1 decoder KV offload results",
        "",
        "Measured output tokens / whole-run wall time; N=4*C. Prefix warmup is excluded.",
        "Maxima below apply only to the measured concurrency grid. P99 TPOT is reported separately.",
        "",
    ]
    summary = []
    if not groups:
        lines += ["No formal 1K-output performance results are available.", ""]
    for key, cases in groups.items():
        condition = json.loads(key)
        lines += [
            f"Conditions: `{key}`",
            "",
            "|Mean TPOT limit|OFF C / token/s / TPOT ms|ON C / token/s / TPOT ms|Throughput gain|",
            "|---|---|---|---|",
        ]
        for slo in [20, 50]:
            selected = {}
            for mode in ["off", "on"]:
                eligible = [
                    r
                    for r in cases
                    if r["mode"] == mode and r["valid"] and r["mean_tpot_ms"] is not None and r["mean_tpot_ms"] <= slo
                ]
                selected[mode] = max(eligible, key=lambda r: r["output_tokens_s"]) if eligible else None
            off, on = selected["off"], selected["on"]
            gain = (on["output_tokens_s"] / off["output_tokens_s"] - 1) if off and on else None

            def cell(row):
                return (
                    f"{row['concurrency']} / {row['output_tokens_s']:.2f} / {row['mean_tpot_ms']:.3f}"
                    if row
                    else "No valid point"
                )

            lines.append(
                f"|{slo}ms|{cell(off)}|{cell(on)}|{gain:.1%}|"
                if gain is not None
                else f"|{slo}ms|{cell(off)}|{cell(on)}|Pending|"
            )
            summary.append(
                {
                    "conditions": condition,
                    "slo_ms": slo,
                    "selected": selected,
                    "throughput_gain": gain,
                }
            )
        lines += [
            "",
            "|Mode|C|Valid|Output token/s|Mean TPOT ms|P99 request TPOT ms|Cache hit|H2D estimate B|Evidence|",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for row in sorted(cases, key=lambda r: (r["mode"], r["concurrency"])):
            lines.append(
                "|"
                + "|".join(
                    str(row.get(k))
                    for k in [
                        "mode",
                        "concurrency",
                        "valid",
                        "output_tokens_s",
                        "mean_tpot_ms",
                        "p99_request_tpot_ms",
                        "measured_cache_ratio",
                        "hisparse_h2d_bytes_software_estimate",
                        "evidence",
                    ]
                )
                + "|"
            )
        lines += [
            "",
            "Same-concurrency TPOT changes (separate from the SLO maxima):",
            "",
        ]
        for c in sorted({r["concurrency"] for r in cases}):
            paired = {r["mode"]: r for r in cases if r["concurrency"] == c and r["valid"]}
            if paired.keys() >= {"off", "on"}:
                off, on = paired["off"], paired["on"]
                lines.append(
                    f"- C={c}: throughput {on['output_tokens_s'] / off['output_tokens_s'] - 1:+.1%}; "
                    f"mean TPOT {on['mean_tpot_ms'] / off['mean_tpot_ms'] - 1:+.1%}."
                )
        lines.append("")
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    (run_dir / "summary.md").write_text("\n".join(lines))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    summarize(args.run_dir)
    print(args.run_dir / "summary.md")
