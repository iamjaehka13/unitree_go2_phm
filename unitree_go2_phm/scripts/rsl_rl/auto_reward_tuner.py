#!/usr/bin/env python3
from __future__ import annotations

"""
Auto reward tuner (unattended loop):

  train -> evaluate -> gate -> recommend -> auto-apply -> next round

Design goals:
- 2-day unattended execution with rollback safety.
- Keep per-term reward change bounded (default <=10%).
- Never apply more than a small number of terms per round.
"""

import argparse
import ast
import glob
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _now_stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _format_weight(v: float) -> str:
    s = f"{float(v):.8f}".rstrip("0").rstrip(".")
    if s in ("", "-0"):
        s = "0"
    if "e" not in s and "E" not in s and "." not in s:
        s += ".0"
    return s


def _run_cmd(cmd: list[str], cwd: Path, log_path: Path, check: bool = True) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n[RUN] {' '.join(cmd)}")
    print(f"[LOG] {log_path}")
    with log_path.open("w", encoding="utf-8") as f:
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            f.write(line)
    rc = proc.wait()
    if check and rc != 0:
        raise RuntimeError(f"Command failed (rc={rc}): {' '.join(cmd)}")
    return rc


def _latest_file(path_glob: str) -> Path | None:
    matches = [Path(p) for p in sorted(glob.glob(path_glob))]
    if len(matches) == 0:
        return None
    matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0]


def _extract_reward_weights_from_cfg(cfg_path: Path, rewards_class: str) -> dict[str, float]:
    source = cfg_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(cfg_path))
    class_nodes = [n for n in tree.body if isinstance(n, ast.ClassDef)]

    target_class: ast.ClassDef | None = None
    for cls in class_nodes:
        if cls.name == rewards_class:
            target_class = cls
            break
    if target_class is None:
        raise ValueError(f"Rewards class '{rewards_class}' not found in {cfg_path}")

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


def _find_class_block(text: str, class_name: str) -> tuple[int, int]:
    class_pat = re.compile(rf"^([ \t]*)class[ \t]+{re.escape(class_name)}\b.*:\s*$", re.MULTILINE)
    m = class_pat.search(text)
    if m is None:
        raise ValueError(f"Class '{class_name}' not found.")
    start = m.start()
    indent = len(m.group(1))
    end = len(text)
    for m2 in re.finditer(r"^([ \t]*)class[ \t]+\w+\b.*:\s*$", text, re.MULTILINE):
        if m2.start() <= start:
            continue
        if len(m2.group(1)) <= indent:
            end = m2.start()
            break
    return start, end


def _apply_weight_updates_to_cfg(
    cfg_path: Path,
    rewards_class: str,
    updates: dict[str, float],
) -> None:
    text = cfg_path.read_text(encoding="utf-8")
    start, end = _find_class_block(text, rewards_class)
    head = text[:start]
    block = text[start:end]
    tail = text[end:]

    for term, new_weight in updates.items():
        pat = re.compile(
            rf"(^[ \t]*{re.escape(term)}[ \t]*=[ \t]*RewTerm\([\s\S]*?\bweight[ \t]*=[ \t]*)"
            rf"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)",
            re.MULTILINE,
        )
        repl = r"\1" + _format_weight(float(new_weight))
        block_new, n = pat.subn(repl, block, count=1)
        if n != 1:
            raise ValueError(f"Failed to update term '{term}' in class '{rewards_class}'.")
        block = block_new

    cfg_path.write_text(head + block + tail, encoding="utf-8")


def _experiment_name_for_task(task: str) -> str:
    t = task.strip()
    if t == "Unitree-Go2-RealObs-v1":
        return "unitree_go2_realobs"
    if t == "Unitree-Go2-Baseline-v1":
        return "unitree_go2_baseline"
    if t == "Unitree-Go2-BaselineTuned-v1":
        return "unitree_go2_baseline_tuned"
    if t == "Unitree-Go2-Phm-v1":
        return "unitree_go2_phm_strategic"
    raise ValueError(f"Unsupported task for auto tuner: {task}")


def _find_train_run_dir(logs_root: Path, run_name: str, newer_than_ts: float | None = None) -> Path:
    cands = [p for p in logs_root.glob(f"*_{run_name}") if p.is_dir()]
    if newer_than_ts is not None:
        cands = [p for p in cands if p.stat().st_mtime >= float(newer_than_ts)]
    if len(cands) == 0:
        if newer_than_ts is None:
            raise FileNotFoundError(f"No train run directory found for run_name='{run_name}' under {logs_root}")
        raise FileNotFoundError(
            f"No fresh train run directory found for run_name='{run_name}' "
            f"(mtime >= {float(newer_than_ts):.3f}) under {logs_root}"
        )
    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0]


def _pick_best_checkpoint(run_dir: Path, newer_than_ts: float | None = None) -> Path:
    models = list(run_dir.glob("model_*.pt"))
    if newer_than_ts is not None:
        models = [p for p in models if p.stat().st_mtime >= float(newer_than_ts)]
    if len(models) == 0:
        if newer_than_ts is None:
            raise FileNotFoundError(f"No model_*.pt found under {run_dir}")
        raise FileNotFoundError(
            f"No fresh model_*.pt found under {run_dir} "
            f"(mtime >= {float(newer_than_ts):.3f})"
        )

    def _iter_of(p: Path) -> int:
        m = re.search(r"model_(\d+)\.pt$", p.name)
        return int(m.group(1)) if m else -1

    models.sort(key=_iter_of, reverse=True)
    return models[0]


def _assert_log_no_traceback(log_path: Path, stage: str) -> None:
    if not log_path.exists():
        raise FileNotFoundError(f"{stage} log not found: {log_path}")
    txt = log_path.read_text(encoding="utf-8", errors="replace")
    if "Traceback (most recent call last):" in txt or "Error executing job with overrides" in txt:
        raise RuntimeError(f"{stage} failed. See log: {log_path}")


@dataclass
class RoundResult:
    round_idx: int
    decision: str
    eval_json: str
    gate_json: str | None
    recommendation_json: str | None
    applied_updates: dict[str, float]
    train_run_dir: str
    checkpoint: str


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Auto reward tuner (train/eval/gate/recommend/apply loop).")
    p.add_argument("--task", type=str, default="Unitree-Go2-RealObs-v1")
    p.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
    p.add_argument("--train_num_envs", type=int, default=4096)
    p.add_argument("--train_iterations", type=int, default=1200)
    p.add_argument("--eval_num_envs", type=int, default=512)
    p.add_argument("--eval_num_episodes", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="")
    p.add_argument("--headless", action="store_true", default=True)
    p.add_argument("--max_rounds", type=int, default=20)
    p.add_argument("--max_hours", type=float, default=48.0)
    p.add_argument("--output_root", type=str, default="./auto_reward_tuning")
    p.add_argument(
        "--reward_cfg",
        type=str,
        default="../../source/unitree_go2_phm/unitree_go2_phm/tasks/manager_based/unitree_go2_phm/unitree_go2_realobs_env_cfg.py",
    )
    p.add_argument("--rewards_class", type=str, default="RealObsRewardsCfg")

    # advisor safety bounds
    p.add_argument("--max_step_change", type=float, default=0.10)
    p.add_argument("--down_step_change", type=float, default=0.05)
    p.add_argument("--max_terms_per_round", type=int, default=2)
    p.add_argument("--auto_apply_top_k", type=int, default=2)
    p.add_argument(
        "--allowed_terms",
        type=str,
        default="energy_efficiency,thermal_safety,saturation_prevention",
        help="Comma-separated reward terms allowed for auto-apply.",
    )
    p.add_argument("--scenario_weights", type=str, default="fresh:0.1,used:0.2,aged:0.3,critical:0.4")

    # gate thresholds
    p.add_argument("--max_survival_drop_abs", type=float, default=0.02)
    p.add_argument("--max_tracking_increase_abs", type=float, default=0.01)
    p.add_argument("--max_power_increase_ratio", type=float, default=0.05)
    p.add_argument("--max_critical_survival_drop_abs", type=float, default=0.03)
    p.add_argument("--stop_on_rollback", action="store_true", default=False)
    return p


def main() -> int:
    args = _build_parser().parse_args()
    if not (0.0 < float(args.max_step_change) <= 0.15):
        raise ValueError("--max_step_change must be in (0, 0.15].")
    if int(args.auto_apply_top_k) <= 0:
        raise ValueError("--auto_apply_top_k must be >= 1.")
    allowed_terms = {x.strip() for x in str(args.allowed_terms).split(",") if x.strip() != ""}
    if len(allowed_terms) == 0:
        raise ValueError("--allowed_terms must contain at least one term.")

    script_dir = Path(__file__).resolve().parent
    reward_cfg = (script_dir / args.reward_cfg).resolve()
    if not reward_cfg.exists():
        raise FileNotFoundError(f"Reward cfg not found: {reward_cfg}")

    run_root = (script_dir / args.output_root / f"run_{_now_stamp()}").resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Auto tuner output root: {run_root}")
    print(f"[INFO] Auto-apply allowed terms: {sorted(allowed_terms)}")

    backup_cfg = run_root / "reward_cfg_backup.py"
    backup_cfg.write_text(reward_cfg.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"[INFO] Reward cfg backup: {backup_cfg}")

    logs_root = script_dir / "logs" / "rsl_rl" / _experiment_name_for_task(args.task)
    logs_root.mkdir(parents=True, exist_ok=True)

    accepted_eval_json: Path | None = None
    accepted_weights = _extract_reward_weights_from_cfg(reward_cfg, args.rewards_class)
    accepted_weights_path = run_root / "accepted_weights_round_000.json"
    accepted_weights_path.write_text(json.dumps(accepted_weights, indent=2), encoding="utf-8")
    history: list[RoundResult] = []

    start_ts = time.time()
    for round_idx in range(1, int(args.max_rounds) + 1):
        elapsed_h = (time.time() - start_ts) / 3600.0
        if elapsed_h >= float(args.max_hours):
            print(f"[INFO] Time budget reached ({elapsed_h:.2f}h >= {float(args.max_hours):.2f}h). Stop.")
            break

        print("\n" + "=" * 80)
        print(f"[ROUND {round_idx:03d}] start (elapsed={elapsed_h:.2f}h)")
        print("=" * 80)

        round_dir = run_root / f"round_{round_idx:03d}"
        round_dir.mkdir(parents=True, exist_ok=True)
        run_name = f"auto_r{round_idx:03d}"
        round_start_ts = time.time()
        # Snapshot reward config used for THIS round training.
        (round_dir / "reward_cfg_before_train.py").write_text(
            reward_cfg.read_text(encoding="utf-8"), encoding="utf-8"
        )

        # 1) Train
        train_cmd = [
            "python3",
            "train.py",
            "--task",
            str(args.task),
            "--agent",
            str(args.agent),
            "--num_envs",
            str(int(args.train_num_envs)),
            "--max_iterations",
            str(int(args.train_iterations)),
            "--seed",
            str(int(args.seed)),
            "--run_name",
            run_name,
        ]
        if str(args.device).strip() != "":
            train_cmd += ["--device", str(args.device)]
        if bool(args.headless):
            train_cmd += ["--headless"]
        train_log = round_dir / "train.log"
        _run_cmd(train_cmd, cwd=script_dir, log_path=train_log, check=True)
        _assert_log_no_traceback(train_log, stage="train")

        train_run_dir = _find_train_run_dir(logs_root=logs_root, run_name=run_name, newer_than_ts=round_start_ts)
        ckpt = _pick_best_checkpoint(train_run_dir, newer_than_ts=round_start_ts)
        print(f"[INFO] Train run dir: {train_run_dir}")
        print(f"[INFO] Picked checkpoint: {ckpt}")

        # 2) Evaluate
        eval_out = round_dir / "eval"
        eval_out.mkdir(parents=True, exist_ok=True)
        eval_cmd = [
            "python3",
            "evaluate.py",
            "--task",
            str(args.task),
            "--agent",
            str(args.agent),
            "--checkpoint",
            str(ckpt),
            "--num_envs",
            str(int(args.eval_num_envs)),
            "--num_episodes",
            str(int(args.eval_num_episodes)),
            "--seed",
            str(int(args.seed)),
            "--output_dir",
            str(eval_out),
            # Auto-tuning explores reward settings; do not enforce paper fixed-fault protocol here.
            "--no-paper-protocol-strict",
        ]
        if str(args.device).strip() != "":
            eval_cmd += ["--device", str(args.device)]
        if bool(args.headless):
            eval_cmd += ["--headless"]
        eval_log = round_dir / "evaluate.log"
        _run_cmd(eval_cmd, cwd=script_dir, log_path=eval_log, check=True)
        _assert_log_no_traceback(eval_log, stage="evaluate")

        eval_json = _latest_file(str(eval_out / "eval_*.json"))
        reward_csv = _latest_file(str(eval_out / "reward_breakdown_*.csv"))
        if eval_json is None:
            raise FileNotFoundError(f"No eval_*.json generated under {eval_out}")
        print(f"[INFO] eval_json: {eval_json}")

        # 3) Gate
        decision = "KEEP"
        gate_json: Path | None = None
        if accepted_eval_json is not None:
            gate_json = round_dir / "gate.json"
            gate_cmd = [
                "python3",
                "reward_tuning_advisor.py",
                "gate",
                "--baseline_eval_json",
                str(accepted_eval_json),
                "--candidate_eval_json",
                str(eval_json),
                "--scenario_weights",
                str(args.scenario_weights),
                "--max_survival_drop_abs",
                str(float(args.max_survival_drop_abs)),
                "--max_tracking_increase_abs",
                str(float(args.max_tracking_increase_abs)),
                "--max_power_increase_ratio",
                str(float(args.max_power_increase_ratio)),
                "--max_critical_survival_drop_abs",
                str(float(args.max_critical_survival_drop_abs)),
                "--output_json",
                str(gate_json),
            ]
            rc = _run_cmd(gate_cmd, cwd=script_dir, log_path=round_dir / "gate.log", check=False)
            if gate_json.exists():
                gate_data = json.loads(gate_json.read_text(encoding="utf-8"))
                decision = str(gate_data.get("decision", "KEEP")).upper()
            else:
                decision = "KEEP" if rc == 0 else "ROLLBACK"

            if decision == "ROLLBACK":
                print("[WARN] Gate decision = ROLLBACK. Restoring accepted reward weights.")
                _apply_weight_updates_to_cfg(reward_cfg, args.rewards_class, accepted_weights)
                if bool(args.stop_on_rollback):
                    history.append(
                        RoundResult(
                            round_idx=round_idx,
                            decision=decision,
                            eval_json=str(eval_json),
                            gate_json=str(gate_json) if gate_json is not None else None,
                            recommendation_json=None,
                            applied_updates={},
                            train_run_dir=str(train_run_dir),
                            checkpoint=str(ckpt),
                        )
                    )
                    break
            else:
                accepted_eval_json = eval_json
                accepted_weights = _extract_reward_weights_from_cfg(reward_cfg, args.rewards_class)
                (run_root / f"accepted_weights_round_{round_idx:03d}.json").write_text(
                    json.dumps(accepted_weights, indent=2), encoding="utf-8"
                )
        else:
            accepted_eval_json = eval_json
            accepted_weights = _extract_reward_weights_from_cfg(reward_cfg, args.rewards_class)
            (run_root / f"accepted_weights_round_{round_idx:03d}.json").write_text(
                json.dumps(accepted_weights, indent=2), encoding="utf-8"
            )

        # 4) Recommend (use accepted eval as base)
        base_eval_for_rec = accepted_eval_json
        if base_eval_for_rec is None:
            raise RuntimeError("Internal error: accepted_eval_json is None before recommend.")

        rec_json = round_dir / "recommendation.json"
        rec_cmd = [
            "python3",
            "reward_tuning_advisor.py",
            "recommend",
            "--eval_json",
            str(base_eval_for_rec),
            "--env_cfg",
            str(reward_cfg),
            "--rewards_class",
            str(args.rewards_class),
            "--scenario_weights",
            str(args.scenario_weights),
            "--max_step_change",
            str(float(args.max_step_change)),
            "--down_step_change",
            str(float(args.down_step_change)),
            "--max_terms_per_round",
            str(int(args.max_terms_per_round)),
            "--output_json",
            str(rec_json),
        ]
        if reward_csv is not None and decision == "KEEP":
            rec_cmd += ["--reward_csv", str(reward_csv)]
        _run_cmd(rec_cmd, cwd=script_dir, log_path=round_dir / "recommend.log", check=True)

        rec_data = json.loads(rec_json.read_text(encoding="utf-8")) if rec_json.exists() else {}
        recs = list(rec_data.get("recommendations", []))
        recs_allowed = [r for r in recs if str(r.get("term", "")) in allowed_terms]
        if len(recs) == 0:
            print("[INFO] No recommendation generated. Stop loop.")
            history.append(
                RoundResult(
                    round_idx=round_idx,
                    decision=decision,
                    eval_json=str(eval_json),
                    gate_json=str(gate_json) if gate_json is not None else None,
                    recommendation_json=str(rec_json),
                    applied_updates={},
                    train_run_dir=str(train_run_dir),
                    checkpoint=str(ckpt),
                )
            )
            break
        if len(recs_allowed) == 0:
            print("[INFO] Recommendations exist but none are in allowed_terms. Stop loop.")
            history.append(
                RoundResult(
                    round_idx=round_idx,
                    decision=decision,
                    eval_json=str(eval_json),
                    gate_json=str(gate_json) if gate_json is not None else None,
                    recommendation_json=str(rec_json),
                    applied_updates={},
                    train_run_dir=str(train_run_dir),
                    checkpoint=str(ckpt),
                )
            )
            break

        # 5) Auto-apply top-k recommendations for next round
        apply_k = min(int(args.auto_apply_top_k), len(recs_allowed))
        apply_updates: dict[str, float] = {}
        for item in recs_allowed[:apply_k]:
            term = str(item["term"])
            new_weight = float(item["new_weight"])
            apply_updates[term] = new_weight

        print(f"[INFO] Auto-apply top-{apply_k} updates: {apply_updates}")
        _apply_weight_updates_to_cfg(reward_cfg, args.rewards_class, apply_updates)
        current_weights = _extract_reward_weights_from_cfg(reward_cfg, args.rewards_class)
        (round_dir / "applied_updates.json").write_text(json.dumps(apply_updates, indent=2), encoding="utf-8")
        (round_dir / "weights_after_apply.json").write_text(json.dumps(current_weights, indent=2), encoding="utf-8")
        (round_dir / "reward_cfg_after_apply.py").write_text(
            reward_cfg.read_text(encoding="utf-8"), encoding="utf-8"
        )

        history.append(
            RoundResult(
                round_idx=round_idx,
                decision=decision,
                eval_json=str(eval_json),
                gate_json=str(gate_json) if gate_json is not None else None,
                recommendation_json=str(rec_json),
                applied_updates=apply_updates,
                train_run_dir=str(train_run_dir),
                checkpoint=str(ckpt),
            )
        )

    summary = {
        "timestamp": _now_stamp(),
        "task": args.task,
        "reward_cfg": str(reward_cfg),
        "rewards_class": args.rewards_class,
        "max_hours": float(args.max_hours),
        "max_rounds": int(args.max_rounds),
        "train_num_envs": int(args.train_num_envs),
        "train_iterations": int(args.train_iterations),
        "eval_num_envs": int(args.eval_num_envs),
        "eval_num_episodes": int(args.eval_num_episodes),
        "history": [r.__dict__ for r in history],
    }
    (run_root / "auto_tuning_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    final_weights = _extract_reward_weights_from_cfg(reward_cfg, args.rewards_class)
    (run_root / "final_reward_weights.json").write_text(json.dumps(final_weights, indent=2), encoding="utf-8")
    (run_root / "final_reward_cfg.py").write_text(reward_cfg.read_text(encoding="utf-8"), encoding="utf-8")
    print("\n[DONE] Auto reward tuning finished.")
    print(f"[DONE] Summary: {run_root / 'auto_tuning_summary.json'}")
    print(f"[DONE] Final reward weights: {run_root / 'final_reward_weights.json'}")
    print(f"[DONE] Final reward cfg snapshot: {run_root / 'final_reward_cfg.py'}")
    print(f"[DONE] Reward cfg (current): {reward_cfg}")
    print(f"[DONE] Original backup: {backup_cfg}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
