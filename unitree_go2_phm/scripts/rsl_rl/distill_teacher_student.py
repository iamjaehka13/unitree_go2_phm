"""Teacher-student action distillation for Unitree Go2 PHM.

Teacher:
    - Privileged policy trained on Unitree-Go2-Phm-v1.
Student:
    - Real-observable policy for Unitree-Go2-RealObs-v1.

This script performs online behavioral distillation:
    min || a_student(o_student) - a_teacher(o_teacher) ||^2

After distillation pretraining, continue student RL training with:
    python train.py --task Unitree-Go2-RealObs-v1 --resume ...
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Teacher-student distillation for Unitree Go2 PHM.")
parser.add_argument("--teacher_task", type=str, default="Unitree-Go2-Phm-v1")
parser.add_argument("--student_task", type=str, default="Unitree-Go2-RealObs-v1")
parser.add_argument("--teacher_agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--student_agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--teacher_checkpoint", type=str, required=True, help="Path to teacher checkpoint (.pt)")
parser.add_argument(
    "--student_init_checkpoint",
    type=str,
    default=None,
    help="Optional student checkpoint to warm-start distillation.",
)
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--num_updates", type=int, default=2000, help="Number of distillation updates.")
parser.add_argument(
    "--steps_per_update",
    type=int,
    default=24,
    help="Environment steps per update (teacher rollout horizon).",
)
parser.add_argument("--distill_lr", type=float, default=3.0e-4)
parser.add_argument("--weight_decay", type=float, default=0.0)
parser.add_argument("--grad_clip", type=float, default=1.0)
parser.add_argument(
    "--dagger_start_beta",
    type=float,
    default=1.0,
    help="Teacher action mixing weight at the start (1.0 = pure teacher rollout).",
)
parser.add_argument(
    "--dagger_end_beta",
    type=float,
    default=0.2,
    help="Teacher action mixing weight at the end (lower -> more student exposure).",
)
parser.add_argument(
    "--fixed_risk_factor",
    type=float,
    default=1.0,
    help="Force risk_factor command to a fixed value for both teacher/student during distillation.",
)
parser.add_argument(
    "--align_student_dynamics",
    type=str,
    default="none",
    choices=["none", "brownout", "thermal", "all"],
    help=(
        "Distillation-only dynamics alignment mode. "
        "'none' keeps student task semantics intact; "
        "'all' copies teacher brownout+thermal settings into student."
    ),
)
parser.add_argument("--save_interval", type=int, default=100)
parser.add_argument(
    "--output_root",
    type=str,
    default="logs/rsl_rl/unitree_go2_realobs",
    help="Root output directory for distillation runs.",
)
parser.add_argument("--run_name", type=str, default="")

# simulator flags
parser.add_argument("--disable_fabric", action="store_true", default=False)

# append AppLauncher args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# clear out sys.argv for Hydra/Omniverse
sys.argv = [sys.argv[0]] + hydra_args

# launch app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym

from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

import isaaclab_tasks  # noqa: F401
import unitree_go2_phm.tasks  # noqa: F401


def _now_stamp() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def _get_policy_module(runner: OnPolicyRunner):
    try:
        return runner.alg.policy
    except AttributeError:
        return runner.alg.actor_critic


def _student_action_with_grad(policy_nn, obs: torch.Tensor) -> torch.Tensor:
    if hasattr(policy_nn, "act_inference"):
        return policy_nn.act_inference(obs)
    if hasattr(policy_nn, "act"):
        out = policy_nn.act(obs)
        if isinstance(out, tuple):
            return out[0]
        return out
    raise RuntimeError("Unsupported student policy module: missing act_inference/act.")


def _safe_reset_recurrent(policy_nn, dones: torch.Tensor):
    if hasattr(policy_nn, "reset"):
        policy_nn.reset(dones)


def _obs_from_reset_output(reset_output):
    """Normalize Gymnasium reset output across API versions."""
    if isinstance(reset_output, tuple):
        return reset_output[0]
    return reset_output


def _sync_named_command(teacher_base_env, student_base_env, command_name: str):
    try:
        t = teacher_base_env.command_manager.get_command(command_name)
        s = student_base_env.command_manager.get_command(command_name)
    except Exception:
        return
    if t is None or s is None:
        return
    if not isinstance(t, torch.Tensor) or not isinstance(s, torch.Tensor):
        return
    if t.shape != s.shape:
        return
    s[:] = t


def _set_named_command_constant(base_env, command_name: str, value: float):
    try:
        cmd = base_env.command_manager.get_command(command_name)
    except Exception:
        return
    if cmd is None:
        return
    if not isinstance(cmd, torch.Tensor):
        return
    cmd[...] = float(value)


def _sync_commands(teacher_base_env, student_base_env):
    # Keep command-conditioned observations aligned across teacher/student envs.
    _sync_named_command(teacher_base_env, student_base_env, "base_velocity")


def _runner_save(runner: OnPolicyRunner, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        runner.save(str(path))
    except TypeError:
        runner.save(str(path), {})


def _resolve_checkpoint(path: str) -> str:
    p = Path(path)
    if p.exists():
        return str(p.resolve())
    return path


def _scheduled_beta(update_idx: int, total_updates: int, start_beta: float, end_beta: float) -> float:
    if total_updates <= 1:
        return float(max(0.0, min(1.0, start_beta)))
    progress = float(update_idx - 1) / float(total_updates - 1)
    beta = float(start_beta) + progress * (float(end_beta) - float(start_beta))
    return float(max(0.0, min(1.0, beta)))


def _align_student_dynamics_with_teacher(
    teacher_env_cfg,
    student_env_cfg,
    mode: str,
) -> dict[str, Any]:
    """
    Distillation should isolate observation-gap, not dynamics-gap.
    Optionally align selected dynamics/safety knobs for teacher-student transition matching.
    """
    mode_norm = str(mode).strip().lower()
    info: dict[str, Any] = {
        "mode": mode_norm,
        "brownout_applied": False,
        "thermal_applied": False,
        "brownout_before": {},
        "brownout_after": {},
        "thermal_before": None,
        "thermal_after": None,
    }

    if mode_norm in ("brownout", "all"):
        for name in ("brownout_voltage_source", "brownout_enter_v", "brownout_recover_v"):
            if not hasattr(student_env_cfg, name):
                continue
            info["brownout_before"][name] = getattr(student_env_cfg, name)
            if hasattr(teacher_env_cfg, name):
                setattr(student_env_cfg, name, getattr(teacher_env_cfg, name))
            info["brownout_after"][name] = getattr(student_env_cfg, name)
        info["brownout_applied"] = True

    teacher_terms = getattr(teacher_env_cfg, "terminations", None)
    student_terms = getattr(student_env_cfg, "terminations", None)
    if teacher_terms is None or student_terms is None:
        return info

    teacher_thermal = getattr(teacher_terms, "thermal_failure", None)
    student_thermal = getattr(student_terms, "thermal_failure", None)
    if teacher_thermal is None or student_thermal is None:
        return info

    teacher_params = getattr(teacher_thermal, "params", None)
    if mode_norm in ("thermal", "all"):
        info["thermal_before"] = dict(student_thermal.params) if isinstance(getattr(student_thermal, "params", None), dict) else None
    if mode_norm in ("thermal", "all") and isinstance(teacher_params, dict):
        student_thermal.params = dict(teacher_params)
        info["thermal_applied"] = True
        info["thermal_after"] = dict(student_thermal.params)
    return info


def main():
    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args_cli.seed)

    run_name = _now_stamp()
    if args_cli.run_name:
        run_name += f"_{args_cli.run_name}"
    run_dir = Path(args_cli.output_root) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # 1) Parse env cfgs
    teacher_env_cfg = parse_env_cfg(
        args_cli.teacher_task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    student_env_cfg = parse_env_cfg(
        args_cli.student_task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    teacher_env_cfg.seed = int(args_cli.seed)
    student_env_cfg.seed = int(args_cli.seed)
    alignment_info = _align_student_dynamics_with_teacher(
        teacher_env_cfg,
        student_env_cfg,
        mode=args_cli.align_student_dynamics,
    )
    print(
        "[INFO] Student dynamics alignment: "
        f"mode={alignment_info['mode']} "
        f"(brownout={alignment_info['brownout_applied']}, thermal={alignment_info['thermal_applied']})"
    )

    # 2) Parse runner cfgs
    teacher_agent_cfg = load_cfg_from_registry(args_cli.teacher_task, args_cli.teacher_agent)
    student_agent_cfg = load_cfg_from_registry(args_cli.student_task, args_cli.student_agent)
    teacher_agent_cfg.seed = int(args_cli.seed)
    student_agent_cfg.seed = int(args_cli.seed)
    if args_cli.device is not None:
        teacher_agent_cfg.device = args_cli.device
        student_agent_cfg.device = args_cli.device

    # 3) Create envs
    teacher_env = gym.make(args_cli.teacher_task, cfg=teacher_env_cfg)
    student_env = gym.make(args_cli.student_task, cfg=student_env_cfg)
    teacher_env = RslRlVecEnvWrapper(teacher_env, clip_actions=teacher_agent_cfg.clip_actions)
    student_env = RslRlVecEnvWrapper(student_env, clip_actions=student_agent_cfg.clip_actions)
    teacher_base = teacher_env.unwrapped
    student_base = student_env.unwrapped

    # 4) Build runners
    teacher_runner = OnPolicyRunner(teacher_env, teacher_agent_cfg.to_dict(), log_dir=None, device=teacher_agent_cfg.device)
    student_runner = OnPolicyRunner(student_env, student_agent_cfg.to_dict(), log_dir=str(run_dir), device=student_agent_cfg.device)

    teacher_ckpt = _resolve_checkpoint(args_cli.teacher_checkpoint)
    print(f"[INFO] Loading teacher checkpoint: {teacher_ckpt}")
    teacher_runner.load(teacher_ckpt)

    if args_cli.student_init_checkpoint:
        student_ckpt = _resolve_checkpoint(args_cli.student_init_checkpoint)
        print(f"[INFO] Loading student init checkpoint: {student_ckpt}")
        student_runner.load(student_ckpt)

    teacher_policy = teacher_runner.get_inference_policy(device=teacher_base.device)
    teacher_policy_nn = _get_policy_module(teacher_runner)
    student_policy_nn = _get_policy_module(student_runner)
    student_policy_nn.train()

    optimizer = torch.optim.Adam(
        student_policy_nn.parameters(),
        lr=float(args_cli.distill_lr),
        weight_decay=float(args_cli.weight_decay),
    )
    mse = torch.nn.MSELoss(reduction="mean")

    # Reset both envs and start synchronized rollouts.
    obs_teacher = _obs_from_reset_output(teacher_env.reset())
    obs_student = _obs_from_reset_output(student_env.reset())
    _sync_commands(teacher_base, student_base)
    _set_named_command_constant(teacher_base, "risk_factor", float(args_cli.fixed_risk_factor))
    _set_named_command_constant(student_base, "risk_factor", float(args_cli.fixed_risk_factor))

    metrics: list[dict[str, Any]] = []
    for update_idx in range(1, int(args_cli.num_updates) + 1):
        step_losses: list[float] = []
        mismatch_events = 0
        beta = _scheduled_beta(
            update_idx=update_idx,
            total_updates=int(args_cli.num_updates),
            start_beta=float(args_cli.dagger_start_beta),
            end_beta=float(args_cli.dagger_end_beta),
        )

        for _ in range(int(args_cli.steps_per_update)):
            _sync_commands(teacher_base, student_base)
            _set_named_command_constant(teacher_base, "risk_factor", float(args_cli.fixed_risk_factor))
            _set_named_command_constant(student_base, "risk_factor", float(args_cli.fixed_risk_factor))
            obs_teacher = teacher_env.get_observations()
            obs_student = student_env.get_observations()

            with torch.no_grad():
                teacher_action = teacher_policy(obs_teacher)

            student_action = _student_action_with_grad(student_policy_nn, obs_student)
            loss = mse(student_action, teacher_action.detach())

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if float(args_cli.grad_clip) > 0.0:
                torch.nn.utils.clip_grad_norm_(student_policy_nn.parameters(), float(args_cli.grad_clip))
            optimizer.step()
            step_losses.append(float(loss.item()))

            # DAgger-style rollout: expose student to its own action distribution progressively.
            exec_action = beta * teacher_action + (1.0 - beta) * student_action.detach()

            with torch.no_grad():
                obs_teacher, _, done_teacher, _ = teacher_env.step(exec_action)
                obs_student, _, done_student, _ = student_env.step(exec_action)
                _safe_reset_recurrent(teacher_policy_nn, done_teacher)
                _safe_reset_recurrent(student_policy_nn, done_student)

                done_t = done_teacher > 0.5
                done_s = done_student > 0.5
                if bool(torch.any(torch.logical_xor(done_t, done_s)).item()):
                    mismatch_events += 1
                    obs_teacher = _obs_from_reset_output(teacher_env.reset())
                    obs_student = _obs_from_reset_output(student_env.reset())
                    _sync_commands(teacher_base, student_base)
                    _set_named_command_constant(teacher_base, "risk_factor", float(args_cli.fixed_risk_factor))
                    _set_named_command_constant(student_base, "risk_factor", float(args_cli.fixed_risk_factor))

        mean_loss = float(np.mean(step_losses)) if len(step_losses) > 0 else float("nan")
        row = {
            "update": int(update_idx),
            "mse_action": mean_loss,
            "mismatch_events": int(mismatch_events),
            "dagger_beta": float(beta),
        }
        metrics.append(row)

        if update_idx % 10 == 0 or update_idx == 1:
            print(
                f"[Update {update_idx:05d}/{args_cli.num_updates}] "
                f"mse_action={mean_loss:.6f} beta={beta:.3f} mismatch={mismatch_events}"
            )

        if update_idx % int(args_cli.save_interval) == 0 or update_idx == int(args_cli.num_updates):
            ckpt_path = run_dir / f"student_distill_update_{update_idx:05d}.pt"
            _runner_save(student_runner, ckpt_path)
            print(f"[SAVE] {ckpt_path}")

    summary = {
        "timestamp": _now_stamp(),
        "teacher_task": args_cli.teacher_task,
        "student_task": args_cli.student_task,
        "teacher_checkpoint": teacher_ckpt,
        "student_init_checkpoint": args_cli.student_init_checkpoint,
        "num_updates": int(args_cli.num_updates),
        "steps_per_update": int(args_cli.steps_per_update),
        "num_envs": int(args_cli.num_envs),
        "seed": int(args_cli.seed),
        "distill_lr": float(args_cli.distill_lr),
        "weight_decay": float(args_cli.weight_decay),
        "grad_clip": float(args_cli.grad_clip),
        "dagger_start_beta": float(args_cli.dagger_start_beta),
        "dagger_end_beta": float(args_cli.dagger_end_beta),
        "fixed_risk_factor": float(args_cli.fixed_risk_factor),
        "align_student_dynamics": str(args_cli.align_student_dynamics),
        "alignment_applied": alignment_info,
        "metrics": metrics,
        "final_mse_action": float(metrics[-1]["mse_action"]) if len(metrics) > 0 else float("nan"),
    }
    with (run_dir / "distill_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    # Keep a stable filename for resume in train.py.
    final_ckpt = run_dir / "student_distill_final.pt"
    _runner_save(student_runner, final_ckpt)
    print(f"[DONE] Distillation complete. Final checkpoint: {final_ckpt}")

    teacher_env.close()
    student_env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
