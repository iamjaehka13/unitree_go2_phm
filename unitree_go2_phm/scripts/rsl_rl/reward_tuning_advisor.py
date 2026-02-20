#!/usr/bin/env python3
"""
Reward tuning advisor for Unitree Go2 PHM experiments.

This script is analysis-only:
- It never edits config files.
- It only prints recommendation deltas and gate decisions.

Modes:
1) recommend: suggest manual reward-weight deltas from eval outputs.
2) gate: compare baseline vs candidate eval and output KEEP/ROLLBACK.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
from pathlib import Path
from typing import Any


DEFAULT_SCENARIO_WEIGHTS = {
    "fresh": 0.10,
    "used": 0.20,
    "aged": 0.30,
    "critical": 0.40,
}


def _parse_float(x: Any, default: float = float("nan")) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _parse_scenario_weights(spec: str) -> dict[str, float]:
    """
    Parse: "fresh:0.1,used:0.2,aged:0.3,critical:0.4"
    """
    if not spec.strip():
        return dict(DEFAULT_SCENARIO_WEIGHTS)
    out: dict[str, float] = {}
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Invalid scenario weight item: '{item}'")
        k, v = item.split(":", 1)
        out[k.strip()] = float(v.strip())
    return out


def _load_eval_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Eval JSON must be an object: {path}")
    return data


def _weighted_metric(eval_data: dict, metric_key: str, scenario_weights: dict[str, float]) -> float:
    weighted_sum = 0.0
    weight_sum = 0.0
    for scenario, w in scenario_weights.items():
        if scenario not in eval_data:
            continue
        entry = eval_data[scenario].get(metric_key)
        if isinstance(entry, dict):
            val = _parse_float(entry.get("mean"))
        else:
            val = _parse_float(entry)
        if math.isnan(val):
            continue
        weighted_sum += float(w) * val
        weight_sum += float(w)
    if weight_sum <= 0.0:
        return float("nan")
    return weighted_sum / weight_sum


def _scenario_metric(eval_data: dict, scenario: str, metric_key: str) -> float:
    if scenario not in eval_data:
        return float("nan")
    entry = eval_data[scenario].get(metric_key)
    if isinstance(entry, dict):
        return _parse_float(entry.get("mean"))
    return _parse_float(entry)


def _has_metric(eval_data: dict, metric_key: str) -> bool:
    for v in eval_data.values():
        if isinstance(v, dict) and metric_key in v:
            return True
    return False


def _resolve_temp_metric_key(eval_data: dict) -> str:
    # Prefer explicit semantics keys when present; fallback to legacy key.
    for key in ("final_max_temp_case_proxy", "final_max_temp_coil_hotspot", "final_max_temp"):
        if _has_metric(eval_data, key):
            return key
    return "final_max_temp"


def _load_reward_share_csv(path: Path) -> dict[str, dict[str, float]]:
    """
    Returns: {scenario: {term: share_abs_mean}}
    """
    out: dict[str, dict[str, float]] = {}
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            scenario = str(row.get("scenario", "")).strip()
            term = str(row.get("term", "")).strip()
            share = _parse_float(row.get("share_abs_mean"))
            if not scenario or not term or math.isnan(share):
                continue
            out.setdefault(scenario, {})[term] = share
    return out


def _weighted_term_share(
    term: str,
    scenario_weights: dict[str, float],
    share_csv_data: dict[str, dict[str, float]] | None,
    eval_data: dict,
) -> float:
    weighted_sum = 0.0
    weight_sum = 0.0
    for scenario, w in scenario_weights.items():
        val = float("nan")
        if share_csv_data is not None:
            val = _parse_float(share_csv_data.get(scenario, {}).get(term))
        if math.isnan(val):
            key = f"reward_share_{term}"
            if scenario in eval_data and key in eval_data[scenario]:
                item = eval_data[scenario][key]
                if isinstance(item, dict):
                    val = _parse_float(item.get("mean"))
                else:
                    val = _parse_float(item)
        if math.isnan(val):
            continue
        weighted_sum += float(w) * val
        weight_sum += float(w)
    if weight_sum <= 0.0:
        return float("nan")
    return weighted_sum / weight_sum


def _extract_reward_weights_from_cfg(cfg_path: Path, rewards_class: str | None) -> dict[str, float]:
    source = cfg_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(cfg_path))

    class_nodes: list[ast.ClassDef] = [n for n in tree.body if isinstance(n, ast.ClassDef)]
    target_class: ast.ClassDef | None = None

    if rewards_class:
        for cls in class_nodes:
            if cls.name == rewards_class:
                target_class = cls
                break
        if target_class is None:
            raise ValueError(f"Rewards class '{rewards_class}' not found in {cfg_path}")
    else:
        for cls in class_nodes:
            if cls.name.endswith("RewardsCfg"):
                target_class = cls
                break
        if target_class is None:
            raise ValueError(f"No '*RewardsCfg' class found in {cfg_path}")

    out: dict[str, float] = {}
    for node in target_class.body:
        if not isinstance(node, ast.Assign):
            continue
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            continue
        term_name = node.targets[0].id
        if not isinstance(node.value, ast.Call):
            continue

        func = node.value.func
        is_rewterm = (
            (isinstance(func, ast.Name) and func.id == "RewTerm")
            or (isinstance(func, ast.Attribute) and func.attr == "RewTerm")
        )
        if not is_rewterm:
            continue

        weight = None
        for kw in node.value.keywords:
            if kw.arg == "weight":
                try:
                    val = ast.literal_eval(kw.value)
                except Exception:
                    val = None
                if isinstance(val, (int, float)):
                    weight = float(val)
                break
        if weight is None:
            continue
        out[term_name] = weight
    return out


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _recommend_mode(args: argparse.Namespace) -> int:
    scenario_weights = _parse_scenario_weights(args.scenario_weights)
    eval_data = _load_eval_json(Path(args.eval_json))
    share_csv_data = _load_reward_share_csv(Path(args.reward_csv)) if args.reward_csv else None
    weights = _extract_reward_weights_from_cfg(Path(args.env_cfg), args.rewards_class)

    max_step = float(args.max_step_change)
    if max_step <= 0.0 or max_step > 0.15:
        raise ValueError("--max_step_change must be in (0, 0.15].")
    down_step = min(float(args.down_step_change), max_step)

    survival = _weighted_metric(eval_data, "survived", scenario_weights)
    tracking = _weighted_metric(eval_data, "mean_tracking_error_xy", scenario_weights)
    mean_power = _weighted_metric(eval_data, "mean_power", scenario_weights)
    temp_metric_key = _resolve_temp_metric_key(eval_data)
    max_temp = _weighted_metric(eval_data, temp_metric_key, scenario_weights)
    max_sat = _weighted_metric(eval_data, "max_saturation", scenario_weights)
    critical_survival = _scenario_metric(eval_data, "critical", "survived")

    share_track_lin = _weighted_term_share("track_lin_vel_xy", scenario_weights, share_csv_data, eval_data)
    share_track_ang = _weighted_term_share("track_ang_vel_z", scenario_weights, share_csv_data, eval_data)
    share_energy = _weighted_term_share("energy_efficiency", scenario_weights, share_csv_data, eval_data)
    share_thermal = _weighted_term_share("thermal_safety", scenario_weights, share_csv_data, eval_data)
    share_sat = _weighted_term_share("saturation_prevention", scenario_weights, share_csv_data, eval_data)

    deltas: dict[str, float] = {}
    reasons: dict[str, list[str]] = {}

    def bump(term: str, delta: float, reason: str):
        if term not in weights:
            return
        prev = deltas.get(term, 0.0)
        new_delta = _clamp(prev + delta, -max_step, max_step)
        if abs(new_delta) < 1e-6:
            return
        deltas[term] = new_delta
        reasons.setdefault(term, []).append(reason)

    poor_walking = (
        (not math.isnan(survival) and survival < float(args.survival_floor))
        or (not math.isnan(tracking) and tracking > float(args.tracking_ceiling))
        or (not math.isnan(critical_survival) and critical_survival < float(args.critical_survival_floor))
    )

    if poor_walking:
        bump("track_lin_vel_xy", +max_step, "Walking priority: survival/tracking below floor.")
        bump("track_ang_vel_z", +min(0.08, max_step), "Walking priority: yaw tracking support.")
        if not math.isnan(share_thermal) and share_thermal > float(args.max_thermal_share_when_unstable):
            bump("thermal_safety", -down_step, "Reduce thermal dominance while walking is unstable.")
        if not math.isnan(share_energy) and share_energy > float(args.max_energy_share_when_unstable):
            bump("energy_efficiency", -down_step, "Reduce energy dominance while walking is unstable.")
    else:
        # Thermal safety
        if not math.isnan(max_temp) and max_temp > float(args.temp_high_threshold):
            if math.isnan(share_thermal) or share_thermal < float(args.target_thermal_share):
                bump("thermal_safety", +max_step, "Temp high and thermal share low.")
            else:
                bump("thermal_safety", +down_step, "Temp high.")

        # Energy efficiency
        if not math.isnan(mean_power) and mean_power > float(args.power_high_threshold):
            if math.isnan(share_energy) or share_energy < float(args.target_energy_share):
                bump("energy_efficiency", +max_step, "Power high and energy share low.")
            else:
                bump("energy_efficiency", +down_step, "Power high.")

        # Saturation prevention
        if not math.isnan(max_sat) and max_sat > float(args.saturation_high_threshold):
            if math.isnan(share_sat) or share_sat < float(args.target_saturation_share):
                bump("saturation_prevention", +min(0.08, max_step), "Saturation high and saturation share low.")
            else:
                bump("saturation_prevention", +down_step, "Saturation high.")

        # If walking already very strong, allow small trade for PHM efficiency.
        if (
            not math.isnan(survival)
            and survival > float(args.survival_relax_threshold)
            and not math.isnan(tracking)
            and tracking < float(args.tracking_relax_threshold)
            and not math.isnan(share_track_lin)
            and share_track_lin > float(args.max_track_lin_share)
        ):
            bump("track_lin_vel_xy", -down_step, "Walking margin is high; rebalance toward PHM terms.")

    # Keep output sparse and safe: top-N by absolute delta only.
    ranked_terms = sorted(deltas.keys(), key=lambda k: abs(deltas[k]), reverse=True)
    ranked_terms = ranked_terms[: int(args.max_terms_per_round)]

    print("\n=== Reward Tuning Advisor (recommend) ===")
    print(f"eval_json: {args.eval_json}")
    if args.reward_csv:
        print(f"reward_csv: {args.reward_csv}")
    print(f"env_cfg: {args.env_cfg}")
    print(f"rewards_class: {args.rewards_class or '(auto)'}")
    print("")
    print("[Current weighted metrics]")
    print(
        "survival={:.4f}, tracking_xy={:.4f}, mean_power={:.3f}, "
        "max_temp({})={:.3f}, max_saturation={:.3f}".format(
            survival,
            tracking,
            mean_power,
            temp_metric_key,
            max_temp,
            max_sat,
        )
    )
    print("[Current weighted reward shares]")
    print(f"track_lin={share_track_lin:.4f}, track_ang={share_track_ang:.4f}, energy={share_energy:.4f}, thermal={share_thermal:.4f}, saturation={share_sat:.4f}")

    if len(ranked_terms) == 0:
        print("\nNo safe recommendation generated. Keep current weights for this round.")
        return 0

    print("\n[Recommended manual edits]  (no auto-edit)")
    print("term, old_weight, delta_pct, new_weight")
    output_rows: list[dict[str, Any]] = []
    for term in ranked_terms:
        old_w = weights[term]
        delta = deltas[term]
        new_w = old_w * (1.0 + delta)
        print(f"{term}, {old_w:.6f}, {delta*100.0:+.1f}%, {new_w:.6f}")
        for r in reasons.get(term, []):
            print(f"  - {r}")
        output_rows.append(
            {
                "term": term,
                "old_weight": old_w,
                "delta_pct": delta * 100.0,
                "new_weight": new_w,
                "reasons": reasons.get(term, []),
            }
        )

    print("\n[Policy]")
    print(f"- Change limit per term is capped at +/-{max_step*100.0:.1f}% in this script.")
    print("- Apply only 1~2 top edits, retrain, then run gate mode.")
    print("- If gate fails: rollback to previous weights.")

    if args.output_json:
        payload = {
            "mode": "recommend",
            "inputs": {
                "eval_json": str(args.eval_json),
                "reward_csv": str(args.reward_csv) if args.reward_csv else None,
                "env_cfg": str(args.env_cfg),
                "rewards_class": args.rewards_class,
                "scenario_weights": scenario_weights,
            },
            "metrics": {
                "survival": survival,
                "tracking_xy": tracking,
                "mean_power": mean_power,
                "max_temp": max_temp,
                "max_temp_metric_key": temp_metric_key,
                "max_saturation": max_sat,
            },
            "shares": {
                "track_lin_vel_xy": share_track_lin,
                "track_ang_vel_z": share_track_ang,
                "energy_efficiency": share_energy,
                "thermal_safety": share_thermal,
                "saturation_prevention": share_sat,
            },
            "recommendations": output_rows,
        }
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nSaved recommendation JSON: {out_path}")

    return 0


def _gate_mode(args: argparse.Namespace) -> int:
    scenario_weights = _parse_scenario_weights(args.scenario_weights)
    base = _load_eval_json(Path(args.baseline_eval_json))
    cand = _load_eval_json(Path(args.candidate_eval_json))

    base_survival = _weighted_metric(base, "survived", scenario_weights)
    cand_survival = _weighted_metric(cand, "survived", scenario_weights)
    base_track = _weighted_metric(base, "mean_tracking_error_xy", scenario_weights)
    cand_track = _weighted_metric(cand, "mean_tracking_error_xy", scenario_weights)
    base_power = _weighted_metric(base, "mean_power", scenario_weights)
    cand_power = _weighted_metric(cand, "mean_power", scenario_weights)

    base_critical_survival = _scenario_metric(base, "critical", "survived")
    cand_critical_survival = _scenario_metric(cand, "critical", "survived")

    surv_ok = cand_survival >= (base_survival - float(args.max_survival_drop_abs))
    track_ok = cand_track <= (base_track + float(args.max_tracking_increase_abs))
    power_ok = cand_power <= (base_power * (1.0 + float(args.max_power_increase_ratio)))
    critical_surv_ok = cand_critical_survival >= (base_critical_survival - float(args.max_critical_survival_drop_abs))

    keep = surv_ok and track_ok and power_ok and critical_surv_ok

    print("\n=== Reward Tuning Advisor (gate) ===")
    print(f"baseline:  {args.baseline_eval_json}")
    print(f"candidate: {args.candidate_eval_json}")
    print("")
    print("[Weighted metrics]")
    print(f"survival: baseline={base_survival:.4f}, candidate={cand_survival:.4f}, limit_drop={float(args.max_survival_drop_abs):.4f}")
    print(f"tracking: baseline={base_track:.4f}, candidate={cand_track:.4f}, limit_inc={float(args.max_tracking_increase_abs):.4f}")
    print(f"power:    baseline={base_power:.4f}, candidate={cand_power:.4f}, limit_inc_ratio={float(args.max_power_increase_ratio):.4f}")
    print("[Critical scenario]")
    print(
        f"critical survival: baseline={base_critical_survival:.4f}, "
        f"candidate={cand_critical_survival:.4f}, "
        f"limit_drop={float(args.max_critical_survival_drop_abs):.4f}"
    )

    print("")
    print("[Gate checks]")
    print(f"survival_ok={surv_ok}")
    print(f"tracking_ok={track_ok}")
    print(f"power_ok={power_ok}")
    print(f"critical_survival_ok={critical_surv_ok}")

    decision = "KEEP" if keep else "ROLLBACK"
    print(f"\nDecision: {decision}")
    if not keep:
        print("- Recommendation: rollback previous weight set and try smaller deltas (<=10%).")

    if args.output_json:
        payload = {
            "mode": "gate",
            "decision": decision,
            "checks": {
                "survival_ok": surv_ok,
                "tracking_ok": track_ok,
                "power_ok": power_ok,
                "critical_survival_ok": critical_surv_ok,
            },
            "baseline": {
                "survival": base_survival,
                "tracking_xy": base_track,
                "mean_power": base_power,
                "critical_survival": base_critical_survival,
            },
            "candidate": {
                "survival": cand_survival,
                "tracking_xy": cand_track,
                "mean_power": cand_power,
                "critical_survival": cand_critical_survival,
            },
        }
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Saved gate JSON: {out_path}")

    return 0 if keep else 2


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Reward tuning advisor (recommend/gate).")
    sub = p.add_subparsers(dest="mode", required=True)

    rec = sub.add_parser("recommend", help="Suggest manual reward-weight deltas.")
    rec.add_argument("--eval_json", type=str, required=True, help="Path to eval_*.json from evaluate.py")
    rec.add_argument(
        "--reward_csv",
        type=str,
        default=None,
        help="Optional path to reward_breakdown_*.csv from evaluate.py",
    )
    rec.add_argument(
        "--env_cfg",
        type=str,
        required=True,
        help="Reward config file path (e.g., unitree_go2_phm_env_cfg.py)",
    )
    rec.add_argument(
        "--rewards_class",
        type=str,
        default=None,
        help="Rewards class name (e.g., RewardsCfg or RealObsRewardsCfg). Default: first *RewardsCfg class.",
    )
    rec.add_argument(
        "--scenario_weights",
        type=str,
        default="fresh:0.1,used:0.2,aged:0.3,critical:0.4",
        help="Scenario weighted-average spec.",
    )
    rec.add_argument("--max_step_change", type=float, default=0.10, help="Max per-term relative change (<=0.15).")
    rec.add_argument("--down_step_change", type=float, default=0.05, help="Small opposite-direction step.")
    rec.add_argument("--max_terms_per_round", type=int, default=2, help="Max number of terms to change per round.")

    rec.add_argument("--survival_floor", type=float, default=0.90)
    rec.add_argument("--critical_survival_floor", type=float, default=0.70)
    rec.add_argument("--tracking_ceiling", type=float, default=0.20)
    rec.add_argument("--power_high_threshold", type=float, default=300.0)
    rec.add_argument("--temp_high_threshold", type=float, default=67.0)
    rec.add_argument("--saturation_high_threshold", type=float, default=0.92)

    rec.add_argument("--target_energy_share", type=float, default=0.05)
    rec.add_argument("--target_thermal_share", type=float, default=0.08)
    rec.add_argument("--target_saturation_share", type=float, default=0.03)
    rec.add_argument("--max_track_lin_share", type=float, default=0.65)
    rec.add_argument("--max_thermal_share_when_unstable", type=float, default=0.18)
    rec.add_argument("--max_energy_share_when_unstable", type=float, default=0.15)
    rec.add_argument("--survival_relax_threshold", type=float, default=0.97)
    rec.add_argument("--tracking_relax_threshold", type=float, default=0.10)
    rec.add_argument("--output_json", type=str, default=None)

    gate = sub.add_parser("gate", help="Compare baseline vs candidate and output KEEP/ROLLBACK.")
    gate.add_argument("--baseline_eval_json", type=str, required=True)
    gate.add_argument("--candidate_eval_json", type=str, required=True)
    gate.add_argument(
        "--scenario_weights",
        type=str,
        default="fresh:0.1,used:0.2,aged:0.3,critical:0.4",
        help="Scenario weighted-average spec.",
    )
    gate.add_argument("--max_survival_drop_abs", type=float, default=0.02)
    gate.add_argument("--max_tracking_increase_abs", type=float, default=0.01)
    gate.add_argument("--max_power_increase_ratio", type=float, default=0.05)
    gate.add_argument("--max_critical_survival_drop_abs", type=float, default=0.03)
    gate.add_argument("--output_json", type=str, default=None)

    return p


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    if args.mode == "recommend":
        return _recommend_mode(args)
    if args.mode == "gate":
        return _gate_mode(args)
    raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    raise SystemExit(main())
