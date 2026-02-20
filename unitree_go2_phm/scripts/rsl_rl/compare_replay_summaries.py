from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(path: str) -> dict:
    p = Path(path)
    with p.open("r") as f:
        return json.load(f)


def _mean(summary: dict, key: str) -> float:
    agg = summary.get("aggregate", {})
    if key not in agg:
        return float("nan")
    return float(agg[key].get("mean", float("nan")))


def main():
    parser = argparse.ArgumentParser(description="Compare two evaluate_replay summary.json outputs.")
    parser.add_argument("--a", type=str, required=True, help="Summary JSON path for condition A (e.g. governor OFF)")
    parser.add_argument("--b", type=str, required=True, help="Summary JSON path for condition B (e.g. governor ON)")
    args = parser.parse_args()

    a = _load(args.a)
    b = _load(args.b)

    metrics = [
        "yaw_mae_exec",
        "temp_max_c",
        "vpack_min_v",
        "vcell_min_v",
        "energy_j",
        "mean_power_w",
        "time_temp_over_warn_s",
        "mean_scale_lin",
        "time_scale_lt_0p9_s",
        "completed_rate",
    ]

    print("=" * 86)
    print(f"{'Metric':<24} | {'A':>16} | {'B':>16} | {'Delta(B-A)':>16}")
    print("-" * 86)
    for key in metrics:
        a_val = _mean(a, key) if key != "completed_rate" else float(a.get("aggregate", {}).get(key, float("nan")))
        b_val = _mean(b, key) if key != "completed_rate" else float(b.get("aggregate", {}).get(key, float("nan")))
        delta = b_val - a_val
        print(f"{key:<24} | {a_val:>16.6f} | {b_val:>16.6f} | {delta:>16.6f}")
    print("=" * 86)


if __name__ == "__main__":
    main()
