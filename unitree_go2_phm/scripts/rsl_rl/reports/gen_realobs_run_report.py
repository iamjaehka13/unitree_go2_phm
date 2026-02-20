#!/usr/bin/env python3
"""Generate comparison report (summary JSON + plots + markdown) for RealObs runs.

Usage example:
  python3 gen_realobs_run_report.py \
    --root unitree_go2_phm/scripts/rsl_rl/logs/rsl_rl/unitree_go2_realobs \
    --run std_safe_s42=2026-02-19_10-53-22_realobs_single_motor_3k_std_safe_s42_perf \
    --run std_safe_s43=2026-02-20_00-00-00_realobs_single_motor_3k_std_safe_s43_perf \
    --run std_safe_s44=2026-02-20_00-00-00_realobs_single_motor_3k_std_safe_s44_perf \
    --out_dir unitree_go2_phm/scripts/rsl_rl/reports/figures_2026_02_20 \
    --out_md unitree_go2_phm/scripts/rsl_rl/reports/2026-02-20_realobs_growth_journal.md \
    --title "2026-02-20 RealObs Growth Journal"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

TAGS = {
    "reward": "Train/mean_reward",
    "ep_len": "Train/mean_episode_length",
    "time_out": "Episode_Termination/time_out",
    "base_contact": "Episode_Termination/base_contact",
    "bad_orientation": "Episode_Termination/bad_orientation",
    "noise_std": "Policy/mean_noise_std",
    "err_xy": "Metrics/base_velocity/error_vel_xy",
    "err_yaw": "Metrics/base_velocity/error_vel_yaw",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate RealObs multi-run report from TensorBoard events.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("unitree_go2_phm/scripts/rsl_rl/logs/rsl_rl/unitree_go2_realobs"),
        help="Root directory containing run folders.",
    )
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Run spec in the form label=run_folder_name_or_abs_path. Repeatable.",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        required=True,
        help="Output directory for summary.json and figures.",
    )
    parser.add_argument(
        "--out_md",
        type=Path,
        required=True,
        help="Output markdown summary path.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="RealObs Growth Journal",
        help="Report markdown title.",
    )
    parser.add_argument("--final_iter", type=int, default=2999, help="Iteration index for final snapshot metrics.")
    parser.add_argument("--late_lo", type=int, default=2600, help="Late phase start iteration for slope.")
    parser.add_argument("--late_hi", type=int, default=2999, help="Late phase end iteration for slope.")
    parser.add_argument("--boundary_a", type=int, default=1000, help="Boundary A for pre/post delta window.")
    parser.add_argument("--boundary_b", type=int, default=2000, help="Boundary B for pre/post delta window.")
    parser.add_argument("--window", type=int, default=20, help="Window size for pre/post mean around boundaries.")
    return parser.parse_args()


def parse_runs(args: argparse.Namespace) -> list[tuple[str, Path]]:
    runs: list[tuple[str, Path]] = []
    for item in args.run:
        if "=" not in item:
            raise ValueError(f"Invalid --run format: {item} (expected label=path)")
        label, raw_path = item.split("=", 1)
        p = Path(raw_path)
        run_dir = p if p.is_absolute() else args.root / p
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")
        runs.append((label.strip(), run_dir))
    if len(runs) < 2:
        raise ValueError("At least two --run entries are required for comparison.")
    return runs


def read_scalars(event_file: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    acc = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
    acc.Reload()
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for key, tag in TAGS.items():
        if tag not in acc.Tags().get("scalars", []):
            continue
        events = acc.Scalars(tag)
        steps = np.array([e.step for e in events], dtype=np.int64)
        vals = np.array([e.value for e in events], dtype=np.float64)
        uniq: dict[int, float] = {}
        for s, v in zip(steps, vals):
            uniq[int(s)] = float(v)
        xs = np.array(sorted(uniq.keys()), dtype=np.int64)
        ys = np.array([uniq[int(x)] for x in xs], dtype=np.float64)
        out[key] = (xs, ys)
    return out


def val_at(series: tuple[np.ndarray, np.ndarray], step: int) -> float:
    xs, ys = series
    if len(xs) == 0:
        return float("nan")
    idx = np.where(xs <= step)[0]
    if len(idx) == 0:
        return float(ys[0])
    return float(ys[idx[-1]])


def window_mean(series: tuple[np.ndarray, np.ndarray], lo: int, hi: int) -> float:
    xs, ys = series
    m = (xs >= lo) & (xs <= hi)
    if not np.any(m):
        return float("nan")
    return float(np.mean(ys[m]))


def slope(series: tuple[np.ndarray, np.ndarray], lo: int, hi: int) -> float:
    xs, ys = series
    m = (xs >= lo) & (xs <= hi)
    if np.sum(m) < 2:
        return float("nan")
    x = xs[m].astype(np.float64)
    y = ys[m].astype(np.float64)
    a, _ = np.polyfit(x, y, 1)
    return float(a)


def event_file_of(run_dir: Path) -> Path:
    files = sorted(run_dir.glob("events.out.tfevents.*"))
    if not files:
        raise FileNotFoundError(f"No TensorBoard event file in {run_dir}")
    return files[0]


def main() -> int:
    args = parse_args()
    runs = parse_runs(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)

    all_data: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    summary: dict[str, dict] = {"runs": {}, "meta": {}}

    for label, run_dir in runs:
        data = read_scalars(event_file_of(run_dir))
        all_data[label] = data
        s: dict[str, float] = {}
        for key in ("reward", "ep_len", "time_out", "base_contact", "bad_orientation", "noise_std", "err_xy", "err_yaw"):
            if key in data:
                s[key] = val_at(data[key], args.final_iter)
        if "time_out" in data:
            s[f"time_out_slope_{args.late_lo}_{args.late_hi}"] = slope(data["time_out"], args.late_lo, args.late_hi)
        if "bad_orientation" in data:
            s[f"bad_orientation_slope_{args.late_lo}_{args.late_hi}"] = slope(
                data["bad_orientation"], args.late_lo, args.late_hi
            )
        for boundary in (args.boundary_a, args.boundary_b):
            for key in ("noise_std", "time_out", "base_contact", "bad_orientation"):
                if key not in data:
                    continue
                pre = window_mean(data[key], boundary - args.window, boundary - 1)
                post = window_mean(data[key], boundary, boundary + args.window)
                s[f"{key}_delta_{boundary}"] = post - pre
        summary["runs"][label] = s

    summary["meta"] = {
        "final_iter": args.final_iter,
        "late_window": [args.late_lo, args.late_hi],
        "boundaries": [args.boundary_a, args.boundary_b],
        "window": args.window,
        "runs": [{"label": label, "run_dir": str(run_dir)} for label, run_dir in runs],
    }

    # Figure 1: full trends
    fig, axs = plt.subplots(5, 1, figsize=(12, 18), sharex=True)
    plot_keys = ["reward", "ep_len", "time_out", "base_contact", "bad_orientation"]
    plot_titles = [
        "Train/mean_reward",
        "Train/mean_episode_length",
        "Episode_Termination/time_out",
        "Episode_Termination/base_contact",
        "Episode_Termination/bad_orientation",
    ]
    for ax, key, title in zip(axs, plot_keys, plot_titles):
        for label, _ in runs:
            if key not in all_data[label]:
                continue
            x, y = all_data[label][key]
            ax.plot(x, y, label=label)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.axvline(args.boundary_a, color="gray", linestyle="--", alpha=0.5)
        ax.axvline(args.boundary_b, color="gray", linestyle="--", alpha=0.5)
    axs[-1].set_xlabel("iteration")
    axs[0].legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(args.out_dir / "tb_full_trends.png", dpi=160)
    plt.close(fig)

    # Figure 2: boundary zoom
    fig, axs = plt.subplots(2, 2, figsize=(14, 9), sharex=False)
    windows = [
        (args.boundary_a - 50, args.boundary_a + 50),
        (args.boundary_b - 50, args.boundary_b + 50),
    ]
    for j, (lo, hi) in enumerate(windows):
        ax_top = axs[0, j]
        for label, _ in runs:
            if "noise_std" not in all_data[label]:
                continue
            x, y = all_data[label]["noise_std"]
            m = (x >= lo) & (x <= hi)
            ax_top.plot(x[m], y[m], label=label)
        ax_top.set_title(f"Policy/mean_noise_std ({lo}-{hi})")
        ax_top.grid(True, alpha=0.3)

        ax_bottom = axs[1, j]
        for label, _ in runs:
            if "time_out" in all_data[label]:
                x, y = all_data[label]["time_out"]
                m = (x >= lo) & (x <= hi)
                ax_bottom.plot(x[m], y[m], label=f"{label}:time_out")
            if "bad_orientation" in all_data[label]:
                x, y = all_data[label]["bad_orientation"]
                m = (x >= lo) & (x <= hi)
                ax_bottom.plot(x[m], y[m], linestyle="--", alpha=0.8, label=f"{label}:bad_orient")
        ax_bottom.set_title(f"time_out / bad_orientation ({lo}-{hi})")
        ax_bottom.grid(True, alpha=0.3)
    axs[0, 0].legend(fontsize=8)
    axs[1, 0].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(args.out_dir / "tb_boundary_zoom.png", dpi=160)
    plt.close(fig)

    # Figure 3: late phase
    fig, axs = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    late_keys = ["reward", "time_out", "base_contact", "bad_orientation"]
    for ax, key in zip(axs.ravel(), late_keys):
        for label, _ in runs:
            if key not in all_data[label]:
                continue
            x, y = all_data[label][key]
            m = x >= args.late_lo
            ax.plot(x[m], y[m], label=label)
        ax.set_title(f"{TAGS[key]} ({args.late_lo}+)")
        ax.grid(True, alpha=0.3)
    axs[0, 0].legend(fontsize=8)
    axs[-1, 0].set_xlabel("iteration")
    axs[-1, 1].set_xlabel("iteration")
    fig.tight_layout()
    fig.savefig(args.out_dir / "tb_late_phase.png", dpi=160)
    plt.close(fig)

    with (args.out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    md: list[str] = []
    md.append(f"# {args.title}")
    md.append("")
    md.append("## 실험 구성")
    for label, run_dir in runs:
        md.append(f"- {label}: `{run_dir.name}`")
    md.append("")
    md.append(f"## {args.final_iter} 최종 비교")
    for label, _ in runs:
        s = summary["runs"][label]
        md.append(f"- {label}")
        md.append(
            f"  - reward {s.get('reward', float('nan')):.4f}, ep_len {s.get('ep_len', float('nan')):.2f}, "
            f"time_out {s.get('time_out', float('nan')):.4f}, base_contact {s.get('base_contact', float('nan')):.4f}, "
            f"bad_orientation {s.get('bad_orientation', float('nan')):.4f}"
        )
        md.append(
            f"  - noise_std {s.get('noise_std', float('nan')):.4f}, err_xy {s.get('err_xy', float('nan')):.4f}, "
            f"err_yaw {s.get('err_yaw', float('nan')):.4f}"
        )
    md.append("")
    md.append(f"## 후반 드리프트 slope ({args.late_lo}~{args.late_hi})")
    for label, _ in runs:
        s = summary["runs"][label]
        md.append(
            f"- {label}: "
            f"bad_orientation_slope={s.get(f'bad_orientation_slope_{args.late_lo}_{args.late_hi}', float('nan')):+.6f}/iter, "
            f"time_out_slope={s.get(f'time_out_slope_{args.late_lo}_{args.late_hi}', float('nan')):+.6f}/iter"
        )
    md.append("")
    md.append("## 경계 변화량(평균, post-pre)")
    for label, _ in runs:
        s = summary["runs"][label]
        md.append(f"- {label}")
        for boundary in (args.boundary_a, args.boundary_b):
            md.append(
                f"  - iter{boundary}: "
                f"noise_std {s.get(f'noise_std_delta_{boundary}', float('nan')):+.4f}, "
                f"time_out {s.get(f'time_out_delta_{boundary}', float('nan')):+.4f}, "
                f"base_contact {s.get(f'base_contact_delta_{boundary}', float('nan')):+.4f}, "
                f"bad_orientation {s.get(f'bad_orientation_delta_{boundary}', float('nan')):+.4f}"
            )
    md.append("")
    md.append("## 산출물")
    md.append(f"- summary: `{args.out_dir / 'summary.json'}`")
    md.append(f"- figure: `{args.out_dir / 'tb_full_trends.png'}`")
    md.append(f"- figure: `{args.out_dir / 'tb_boundary_zoom.png'}`")
    md.append(f"- figure: `{args.out_dir / 'tb_late_phase.png'}`")

    args.out_md.write_text("\n".join(md), encoding="utf-8")
    print(f"[OK] wrote {args.out_dir / 'summary.json'}")
    print(f"[OK] wrote {args.out_md}")
    print(f"[OK] figures -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

