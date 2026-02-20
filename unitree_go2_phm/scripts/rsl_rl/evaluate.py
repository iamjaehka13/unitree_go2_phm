# =============================================================================
# evaluate.py — PHM Paper Evaluation Script
# =============================================================================
# Runs a trained policy under controlled degradation scenarios and collects
# quantitative metrics for paper tables and figures.
#
# Usage:
#   python evaluate.py --task Unitree-Go2-Phm-v1 \
#       --checkpoint /path/to/model_3000.pt \
#       --num_envs 512 --num_episodes 100 \
#       --output_dir ./eval_results/phm_aware \
#       --headless
# =============================================================================

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate trained policy under degradation scenarios.")
parser.add_argument("--task", type=str, required=True, help="Gym task ID")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pt)")
parser.add_argument("--num_envs", type=int, default=512, help="Number of parallel envs")
parser.add_argument("--num_episodes", type=int, default=100, help="Min episodes per scenario to collect")
parser.add_argument("--output_dir", type=str, default="./eval_results", help="Output directory for CSV files")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Agent config entry point"
)

# local imports
import cli_args  # isort: skip
# Keep evaluate.py's required --checkpoint argument as the single source.
cli_args.add_rsl_rl_args(parser, include_checkpoint=False)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import carb
carb.settings.get_settings().set_int("/log/channels/omni.physx.tensors.plugin/level", 1)

"""Rest everything follows."""

import csv
import gymnasium as gym
import os
import json
import torch
import numpy as np
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnvCfg, DirectRLEnvCfg, DirectMARLEnvCfg
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

import unitree_go2_phm.tasks  # noqa: F401
from unitree_go2_phm.tasks.manager_based.unitree_go2_phm.phm.utils import compute_battery_voltage

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# =============================================================================
# Degradation Scenarios (Paper Table Rows)
# =============================================================================
SCENARIOS = {
    "fresh": {
        "description": "Brand new robot, no degradation",
        "fatigue_range": (0.0, 0.0),
        "health_margin": (1.0, 1.0),
        "temp_range": (25.0, 25.0),
        "soc_range": (1.0, 1.0),
    },
    "used": {
        "description": "Moderate wear, SOC 80-100%",
        "fatigue_range": (0.1, 0.3),
        "health_margin": (0.30, 0.55),
        "temp_range": (25.0, 35.0),
        "soc_range": (0.8, 1.0),
    },
    "aged": {
        "description": "Significant wear, warm motors, SOC 40-80%",
        "fatigue_range": (0.4, 0.7),
        "health_margin": (0.10, 0.25),
        "temp_range": (35.0, 55.0),
        "soc_range": (0.4, 0.8),
    },
    "critical": {
        "description": "Near end-of-life, hot motors, low battery",
        "fatigue_range": (0.7, 0.95),
        "health_margin": (0.02, 0.10),
        "temp_range": (65.0, 85.0),
        "soc_range": (0.1, 0.3),
    },
}


@dataclass
class _RewardTermSpec:
    name: str
    func: Any
    weight: float
    params: dict


def _extract_reward_term_specs(base_env) -> list[_RewardTermSpec]:
    """Extract active reward terms from env cfg in a robust, order-preserving way."""
    reward_cfg = getattr(getattr(base_env, "cfg", None), "rewards", None)
    if reward_cfg is None:
        return []

    if hasattr(reward_cfg, "__dataclass_fields__"):
        candidate_names = list(reward_cfg.__dataclass_fields__.keys())
    else:
        candidate_names = [k for k in vars(reward_cfg).keys() if not k.startswith("_")]

    terms: list[_RewardTermSpec] = []
    for name in candidate_names:
        term = getattr(reward_cfg, name, None)
        if term is None:
            continue
        if not hasattr(term, "func") or not hasattr(term, "weight"):
            continue

        try:
            weight = float(term.weight)
        except Exception:
            continue

        if abs(weight) <= 1e-12:
            # Skip inactive terms to keep breakdown focused.
            continue

        params = getattr(term, "params", None)
        if params is None:
            params = {}
        if not isinstance(params, dict):
            continue

        terms.append(_RewardTermSpec(name=name, func=term.func, weight=weight, params=params))

    return terms


class _RewardBreakdownCollector:
    """
    Recompute per-term reward contributions for analysis.

    The collector auto-detects whether manager-level reward scaling includes dt,
    by comparing reconstructed reward vs environment reward buffer at runtime.
    """

    def __init__(self, base_env):
        self.base_env = base_env
        self.step_dt = float(getattr(base_env, "step_dt", 1.0))
        self.term_specs = _extract_reward_term_specs(base_env)
        self.term_names = [t.name for t in self.term_specs]

        self._use_dt_scaling: bool | None = None
        self._decision_mae_no_dt: float | None = None
        self._decision_mae_with_dt: float | None = None
        self.last_reconstruction_mae: float | None = None

    @property
    def use_dt_scaling(self) -> bool | None:
        return self._use_dt_scaling

    def _as_reward_vector(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.as_tensor(x, device=self.base_env.device, dtype=torch.float32)
        if y.ndim == 0:
            y = y.repeat(self.base_env.num_envs)
        if y.ndim > 1:
            if y.shape[-1] == 1:
                y = y.squeeze(-1)
            else:
                y = torch.mean(y, dim=tuple(range(1, y.ndim)))
        return y

    def compute_step_contributions(self, reward_vec: torch.Tensor) -> dict[str, torch.Tensor]:
        if len(self.term_specs) == 0:
            return {}

        reward_vec = self._as_reward_vector(reward_vec)
        contrib_no_dt: dict[str, torch.Tensor] = {}
        sum_no_dt = torch.zeros_like(reward_vec)
        sum_with_dt = torch.zeros_like(reward_vec)

        for spec in self.term_specs:
            raw = spec.func(self.base_env, **spec.params)
            raw = self._as_reward_vector(raw)
            raw = torch.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)

            weighted = raw * float(spec.weight)
            contrib_no_dt[spec.name] = weighted
            sum_no_dt = sum_no_dt + weighted
            sum_with_dt = sum_with_dt + (weighted * self.step_dt)

        if self._use_dt_scaling is None:
            mae_no_dt = torch.mean(torch.abs(sum_no_dt - reward_vec)).item()
            mae_with_dt = torch.mean(torch.abs(sum_with_dt - reward_vec)).item()
            self._decision_mae_no_dt = float(mae_no_dt)
            self._decision_mae_with_dt = float(mae_with_dt)
            self._use_dt_scaling = mae_with_dt <= mae_no_dt

        if self._use_dt_scaling:
            contrib = {k: v * self.step_dt for k, v in contrib_no_dt.items()}
            recon = sum_with_dt
        else:
            contrib = contrib_no_dt
            recon = sum_no_dt

        self.last_reconstruction_mae = float(torch.mean(torch.abs(recon - reward_vec)).item())
        return contrib

    def meta(self) -> dict[str, Any]:
        return {
            "terms": list(self.term_names),
            "dt_scaled": bool(self._use_dt_scaling) if self._use_dt_scaling is not None else None,
            "decision_mae_no_dt": self._decision_mae_no_dt,
            "decision_mae_with_dt": self._decision_mae_with_dt,
            "last_reconstruction_mae": self.last_reconstruction_mae,
        }


def _thermal_failure_params(base_env) -> tuple[float, bool, float]:
    """Read thermal termination settings from env cfg (with safe defaults)."""
    threshold_temp = 90.0
    use_case_proxy = False
    coil_to_case_delta_c = 5.0
    term_cfg = getattr(getattr(base_env, "cfg", None), "terminations", None)
    thermal_failure = getattr(term_cfg, "thermal_failure", None) if term_cfg is not None else None
    params = getattr(thermal_failure, "params", None)
    if isinstance(params, dict):
        threshold_temp = float(params.get("threshold_temp", threshold_temp))
        use_case_proxy = bool(params.get("use_case_proxy", use_case_proxy))
        coil_to_case_delta_c = float(params.get("coil_to_case_delta_c", coil_to_case_delta_c))
    return threshold_temp, use_case_proxy, coil_to_case_delta_c


def _case_temperature_tensor(phm):
    for name in ("motor_case_temp", "case_temp", "motor_temp_case", "housing_temp", "motor_housing_temp"):
        if hasattr(phm, name):
            val = getattr(phm, name)
            if isinstance(val, torch.Tensor):
                return val
    return None


def _temperature_tensor_for_eval(base_env) -> torch.Tensor:
    phm = base_env.phm_state
    _, use_case_proxy, coil_to_case_delta_c = _thermal_failure_params(base_env)
    if use_case_proxy:
        case_temp = _case_temperature_tensor(phm)
        if case_temp is not None:
            return case_temp
        if hasattr(phm, "coil_temp"):
            return phm.coil_temp - float(coil_to_case_delta_c)
    return phm.coil_temp


def _sync_long_term_buffer_after_scenario(base_env, env_ids: torch.Tensor):
    """Reset long-term snapshots so slope/trend observations match injected PHM state."""
    if env_ids.numel() == 0:
        return

    ltb = getattr(base_env, "phm_long_term_buffer", None)
    if ltb is None:
        return

    fatigue_now = base_env.phm_state.fatigue_index[env_ids]

    if hasattr(ltb, "fatigue_snapshots"):
        ltb.fatigue_snapshots[env_ids] = 0.0
        ltb.fatigue_snapshots[env_ids, 0, :] = fatigue_now
    if hasattr(ltb, "snapshot_index"):
        ltb.snapshot_index[env_ids] = 1
    if hasattr(ltb, "fill_count"):
        ltb.fill_count[env_ids] = 1
    if hasattr(ltb, "is_buffer_filled"):
        ltb.is_buffer_filled[env_ids] = False
    if hasattr(ltb, "step_timer"):
        ltb.step_timer[env_ids] = 1
    if hasattr(ltb, "thermal_overload_duration"):
        ltb.thermal_overload_duration[env_ids] = 0.0


def _sync_runtime_state_after_scenario(base_env, env_ids: torch.Tensor):
    """Synchronize derived runtime state after direct PHM state injection."""
    if env_ids.numel() == 0:
        return

    phm = base_env.phm_state

    # 1) Electrical state sync
    soc = phm.soc[env_ids]
    zero_load = torch.zeros_like(soc)
    true_voltage = compute_battery_voltage(soc, zero_load)
    true_voltage = torch.nan_to_num(true_voltage, nan=25.0).clamp(0.0, 60.0)

    if hasattr(phm, "battery_voltage_true"):
        phm.battery_voltage_true[env_ids] = true_voltage

    sensor_bias = phm.voltage_sensor_bias[env_ids] if hasattr(phm, "voltage_sensor_bias") else torch.zeros_like(true_voltage)
    observed_voltage = torch.clamp(true_voltage + sensor_bias, 0.0, 60.0)
    phm.battery_voltage[env_ids] = observed_voltage

    if hasattr(phm, "min_voltage_log"):
        phm.min_voltage_log[env_ids] = true_voltage

    # 2) Always refresh BMS prediction cache.
    # Brownout may use this cache or measured channels depending on env config.
    if hasattr(base_env, "_predict_instant_voltage_ivp"):
        predicted_all = base_env._predict_instant_voltage_ivp(use_noisy_state=True, use_nominal_model=True)
        predicted_voltage = predicted_all[env_ids]
    else:
        predicted_voltage = observed_voltage

    if hasattr(phm, "bms_voltage_pred"):
        phm.bms_voltage_pred[env_ids] = predicted_voltage

    # 3) Brownout state sync (use same thresholds as env.step)
    if hasattr(phm, "brownout_latched") and hasattr(phm, "brownout_scale"):
        def _scalar_attr(name: str, default: float) -> float:
            value = getattr(base_env, name, default)
            if isinstance(value, torch.Tensor):
                return float(value.detach().cpu().item())
            return float(value)

        brownout_enter_v = _scalar_attr("_brownout_enter_v", 24.5)
        brownout_recover_v = _scalar_attr("_brownout_recover_v", 25.0)
        brownout_scale_low = _scalar_attr("_const_brownout_scale_low", 0.5)
        brownout_scale_high = _scalar_attr("_const_brownout_scale_high", 1.0)
        brownout_source = str(getattr(base_env, "_brownout_voltage_source", "bms_pred")).strip().lower()

        if brownout_source == "true_voltage" and hasattr(phm, "battery_voltage_true"):
            brownout_voltage = phm.battery_voltage_true[env_ids]
        elif brownout_source == "sensor_voltage" and hasattr(phm, "battery_voltage"):
            brownout_voltage = phm.battery_voltage[env_ids]
        else:
            brownout_voltage = predicted_voltage

        is_low_voltage = brownout_voltage < brownout_enter_v
        is_recovered = brownout_voltage > brownout_recover_v
        current_latch = phm.brownout_latched[env_ids]

        new_latch = torch.where(is_low_voltage, torch.ones_like(current_latch), current_latch)
        new_latch = torch.where(is_recovered, torch.zeros_like(new_latch), new_latch)
        phm.brownout_latched[env_ids] = new_latch

        phm.brownout_scale[env_ids] = torch.where(
            new_latch,
            torch.full_like(predicted_voltage, brownout_scale_low),
            torch.full_like(predicted_voltage, brownout_scale_high),
        )

    # 4) Align long-term PHM buffer with directly injected scenario state.
    _sync_long_term_buffer_after_scenario(base_env, env_ids)

    # 5) Apply degradation and effort limits immediately
    if hasattr(base_env, "_apply_physical_degradation"):
        base_env._apply_physical_degradation(env_ids=env_ids)

    if hasattr(base_env, "_compute_thermal_limits"):
        thermal_limits = base_env._compute_thermal_limits(env_ids=env_ids)
        if hasattr(phm, "brownout_scale"):
            final_limits = thermal_limits * phm.brownout_scale[env_ids].unsqueeze(-1)
        else:
            final_limits = thermal_limits

        base_env.robot.data.joint_effort_limits[env_ids] = final_limits
        if hasattr(base_env.robot, "write_joint_effort_limit_to_sim"):
            base_env.robot.write_joint_effort_limit_to_sim(final_limits, env_ids=env_ids)


def apply_scenario(env, scenario_name: str, env_ids: torch.Tensor):
    """Force-inject a specific degradation scenario for controlled evaluation."""
    base_env = env.unwrapped if hasattr(env, "unwrapped") else env
    scenario = dict(SCENARIOS[scenario_name])
    device = base_env.device
    env_ids = env_ids.to(device=device, dtype=torch.long)
    n = len(env_ids)
    num_motors = base_env.phm_state.fatigue_index.shape[1]

    fmin, fmax = scenario["fatigue_range"]
    fatigue = torch.rand((n, num_motors), device=device) * (fmax - fmin) + fmin
    base_env.phm_state.fatigue_index[env_ids] = fatigue

    mmin, mmax = scenario["health_margin"]
    margin = torch.rand((n, num_motors), device=device) * (mmax - mmin) + mmin
    base_env.phm_state.motor_health_capacity[env_ids] = torch.clamp(fatigue + margin, max=1.0)

    tmin, tmax = scenario["temp_range"]
    threshold_temp, use_case_proxy, coil_to_case_delta_c = _thermal_failure_params(base_env)
    # RealObs thermal termination may use case-proxy 72C; reduce injected critical/aged
    # case-equivalent temperature to avoid immediate-start failures in evaluation.
    if use_case_proxy:
        if scenario_name == "critical":
            case_tmax = max(30.0, threshold_temp - 2.0)
            case_tmin = max(25.0, case_tmax - 12.0)
            tmin = case_tmin + coil_to_case_delta_c
            tmax = case_tmax + coil_to_case_delta_c
        elif scenario_name == "aged":
            case_tmax = max(30.0, threshold_temp - 8.0)
            case_tmin = max(25.0, case_tmax - 14.0)
            tmin = case_tmin + coil_to_case_delta_c
            tmax = case_tmax + coil_to_case_delta_c

    temp = torch.rand((n, num_motors), device=device) * (tmax - tmin) + tmin
    base_env.phm_state.coil_temp[env_ids] = temp
    case_temp_tensor = _case_temperature_tensor(base_env.phm_state)
    if case_temp_tensor is not None:
        if use_case_proxy:
            delta = torch.full((n, num_motors), float(coil_to_case_delta_c), device=device)
        else:
            if scenario_name == "fresh":
                dmin, dmax = 0.0, 1.0
            elif scenario_name == "used":
                dmin, dmax = 1.0, 3.5
            elif scenario_name == "aged":
                dmin, dmax = 2.5, 6.5
            else:
                dmin, dmax = 3.5, 8.0
            delta = torch.rand((n, num_motors), device=device) * (dmax - dmin) + dmin
        case_temp = torch.clamp(base_env.phm_state.coil_temp[env_ids] - delta, min=25.0)
        case_temp_tensor[env_ids] = case_temp

    if hasattr(base_env.phm_state, "temp_derivative"):
        base_env.phm_state.temp_derivative[env_ids] = 0.0
    if hasattr(base_env.phm_state, "case_temp_derivative"):
        base_env.phm_state.case_temp_derivative[env_ids] = 0.0

    smin, smax = scenario["soc_range"]
    soc = torch.rand(n, device=device) * (smax - smin) + smin
    base_env.phm_state.soc[env_ids] = soc

    # Controlled evaluation: neutralize reset-time hidden random biases.
    if hasattr(base_env.phm_state, "friction_bias"):
        base_env.phm_state.friction_bias[env_ids] = 1.0
    if hasattr(base_env.phm_state, "voltage_sensor_bias"):
        base_env.phm_state.voltage_sensor_bias[env_ids] = 0.0
    if hasattr(base_env.phm_state, "encoder_offset"):
        base_env.phm_state.encoder_offset[env_ids] = 0.0

    _sync_runtime_state_after_scenario(base_env, env_ids)


def collect_step_metrics(env) -> dict:
    """Extract per-step metrics from environment state."""
    unwrapped = env.unwrapped
    robot = unwrapped.scene["robot"]
    phm = unwrapped.phm_state

    # Velocity tracking error
    cmd = unwrapped.command_manager.get_command("base_velocity")[:, :2]
    actual_vel = robot.data.root_lin_vel_b[:, :2]
    tracking_error = torch.norm(cmd - actual_vel, dim=1)

    # Angular velocity tracking
    cmd_ang = unwrapped.command_manager.get_command("base_velocity")[:, 2]
    actual_ang = robot.data.root_ang_vel_b[:, 2]
    ang_tracking_error = torch.abs(cmd_ang - actual_ang)

    temp_for_eval = _temperature_tensor_for_eval(unwrapped)

    metrics = {
        "tracking_error_xy": tracking_error.cpu(),
        "tracking_error_ang": ang_tracking_error.cpu(),
        "avg_temp": torch.mean(temp_for_eval, dim=1).cpu(),
        "max_temp": torch.max(temp_for_eval, dim=1)[0].cpu(),
        "avg_fatigue": torch.mean(phm.fatigue_index, dim=1).cpu(),
        "max_fatigue": torch.max(phm.fatigue_index, dim=1)[0].cpu(),
        "soc": phm.soc.cpu(),
        "battery_voltage": (
            phm.battery_voltage_true.cpu()
            if hasattr(phm, "battery_voltage_true")
            else phm.battery_voltage.cpu()
        ),
        "total_power": torch.sum(phm.avg_power_log, dim=1).cpu(),
        "max_saturation": torch.max(phm.torque_saturation, dim=1)[0].cpu(),
    }

    # SOH (State of Health)
    soh = phm.motor_health_capacity - phm.fatigue_index
    metrics["min_soh"] = torch.min(soh, dim=1)[0].cpu()

    return metrics


def _terminal_scalar(terminal_metrics: dict, terminal_lookup: dict[int, int], env_idx: int, key: str) -> float | None:
    """Read terminal snapshot scalar for one environment; return None if unavailable."""
    pos = terminal_lookup.get(env_idx)
    if pos is None:
        return None
    val = terminal_metrics.get(key, None)
    if val is None:
        return None
    return float(val[pos].detach().cpu().item())


def _reset_recurrent_state(policy_nn, num_envs: int, device: torch.device | str):
    """Reset recurrent hidden state across all envs, if the policy supports it."""
    if not hasattr(policy_nn, "reset"):
        return
    done_mask = torch.ones(num_envs, dtype=torch.bool, device=device)
    policy_nn.reset(done_mask)


def _safe_reset_recurrent(policy_nn, dones: torch.Tensor):
    """Reset recurrent state only when the policy exposes a reset method."""
    if hasattr(policy_nn, "reset"):
        policy_nn.reset(dones)


def _obs_from_reset_output(reset_output):
    """Normalize Gymnasium reset output across API versions."""
    if isinstance(reset_output, tuple):
        return reset_output[0]
    return reset_output


def run_evaluation(
    env,
    policy,
    policy_nn,
    scenario_name: str,
    num_target_episodes: int,
    temp_metric_semantics: str = "coil_hotspot",
) -> dict:
    """Run evaluation for one scenario, collecting episode-level statistics."""
    print(f"\n{'='*60}")
    print(f"  Evaluating scenario: {scenario_name}")
    print(f"  Description: {SCENARIOS[scenario_name]['description']}")
    print(f"  Target episodes: {num_target_episodes}")
    print(f"{'='*60}")

    episode_metrics = defaultdict(list)
    # Per-env episode step counters
    num_envs = env.num_envs
    ep_step_counter = torch.zeros(num_envs, dtype=torch.long)
    ep_tracking_errors = [[] for _ in range(num_envs)]
    ep_ang_errors = [[] for _ in range(num_envs)]
    ep_power_history = [[] for _ in range(num_envs)]
    ep_total_reward = torch.zeros(num_envs, dtype=torch.float32)

    reward_breakdown = _RewardBreakdownCollector(env.unwrapped)
    ep_term_signed = {name: torch.zeros(num_envs, dtype=torch.float32) for name in reward_breakdown.term_names}
    ep_term_abs = {name: torch.zeros(num_envs, dtype=torch.float32) for name in reward_breakdown.term_names}

    completed_episodes = 0
    total_steps = 0
    max_steps = num_target_episodes * 1500  # Safety: prevent infinite loop
    last_reported_episodes = 0

    # Reset and inject scenario.
    # This prevents hidden-state/episode carry-over across scenarios.
    obs = _obs_from_reset_output(env.reset())
    _reset_recurrent_state(policy_nn, num_envs=env.num_envs, device=env.unwrapped.device)
    all_env_ids = torch.arange(num_envs, device=env.unwrapped.device)
    apply_scenario(env, scenario_name, all_env_ids)
    # Ensure first action uses observations consistent with injected scenario.
    obs = env.get_observations()

    while completed_episodes < num_target_episodes and total_steps < max_steps:
        # NOTE:
        # Keep env.step() out of torch.inference_mode(). Otherwise internal simulator
        # buffers can become inference tensors and later fail on reset() with
        # "Inplace update to inference tensor outside InferenceMode".
        with torch.no_grad():
            actions = policy(obs)
        # Defensive clone: if policy wrapper returns inference tensors internally,
        # convert to a regular tensor before passing into the environment.
        actions = actions.clone()
        obs, rewards, dones, _ = env.step(actions)
        dones = torch.as_tensor(dones, device=env.unwrapped.device, dtype=torch.bool)
        if dones.ndim > 1:
            dones = torch.any(dones, dim=tuple(range(1, dones.ndim)))
        _safe_reset_recurrent(policy_nn, dones)

        # Collect step metrics
        step_metrics = collect_step_metrics(env)
        step_reward = reward_breakdown._as_reward_vector(rewards).detach().cpu()
        ep_total_reward += step_reward

        step_term_contrib = reward_breakdown.compute_step_contributions(reward_vec=rewards)
        done_mask_cpu = dones.detach().cpu()
        for term_name, term_val in step_term_contrib.items():
            term_cpu = term_val.detach().cpu().to(torch.float32)
            # done envs are already reset inside env.step(); final-step term values come from terminal snapshot.
            term_cpu[done_mask_cpu] = 0.0
            ep_term_signed[term_name] += term_cpu
            ep_term_abs[term_name] += torch.abs(term_cpu)

        terminal_metrics = getattr(env.unwrapped, "_last_terminal_metrics", {})
        terminal_ids = getattr(env.unwrapped, "_last_terminal_env_ids", None)
        terminal_lookup: dict[int, int] = {}
        if isinstance(terminal_ids, torch.Tensor) and terminal_ids.numel() > 0 and isinstance(terminal_metrics, dict):
            terminal_lookup = {int(env_id.item()): i for i, env_id in enumerate(terminal_ids.detach().cpu())}

        total_steps += 1
        ep_step_counter += 1

        # Track per-env step data
        for i in range(num_envs):
            trk_xy = step_metrics["tracking_error_xy"][i].item()
            trk_ang = step_metrics["tracking_error_ang"][i].item()
            total_power = step_metrics["total_power"][i].item()
            # Use terminal cache when available (env.step resets done envs before returning).
            trk_xy_terminal = _terminal_scalar(terminal_metrics, terminal_lookup, i, "tracking_error_xy")
            trk_ang_terminal = _terminal_scalar(terminal_metrics, terminal_lookup, i, "tracking_error_ang")
            power_terminal = _terminal_scalar(terminal_metrics, terminal_lookup, i, "total_power")
            if trk_xy_terminal is not None:
                trk_xy = trk_xy_terminal
            if trk_ang_terminal is not None:
                trk_ang = trk_ang_terminal
            if power_terminal is not None:
                total_power = power_terminal
            ep_tracking_errors[i].append(trk_xy)
            ep_ang_errors[i].append(trk_ang)
            ep_power_history[i].append(total_power)

        # Process completed episodes
        done_envs = dones.nonzero(as_tuple=False).squeeze(-1)
        if len(done_envs) > 0:
            for idx_t in done_envs:
                idx = idx_t.item()
                ep_len = ep_step_counter[idx].item()
                if ep_len < 2:
                    ep_step_counter[idx] = 0
                    ep_total_reward[idx] = 0.0
                    ep_tracking_errors[idx] = []
                    ep_ang_errors[idx] = []
                    ep_power_history[idx] = []
                    for term_name in reward_breakdown.term_names:
                        ep_term_signed[term_name][idx] = 0.0
                        ep_term_abs[term_name][idx] = 0.0
                    continue  # Skip trivially short episodes

                # Aggregate episode stats
                final_soc = _terminal_scalar(terminal_metrics, terminal_lookup, idx, "soc")
                final_avg_temp = _terminal_scalar(terminal_metrics, terminal_lookup, idx, "avg_temp")
                final_max_temp = _terminal_scalar(terminal_metrics, terminal_lookup, idx, "max_temp")
                final_max_fatigue = _terminal_scalar(terminal_metrics, terminal_lookup, idx, "max_fatigue")
                final_min_soh = _terminal_scalar(terminal_metrics, terminal_lookup, idx, "min_soh")
                final_max_saturation = _terminal_scalar(terminal_metrics, terminal_lookup, idx, "max_saturation")

                episode_metrics["episode_length"].append(ep_len)
                episode_metrics["mean_total_reward"].append((ep_total_reward[idx].item() / max(ep_len, 1)))
                episode_metrics["mean_tracking_error_xy"].append(np.mean(ep_tracking_errors[idx]))
                episode_metrics["mean_tracking_error_ang"].append(np.mean(ep_ang_errors[idx]))
                episode_metrics["mean_power"].append(np.mean(ep_power_history[idx]))
                episode_metrics["total_energy"].append(np.sum(ep_power_history[idx]) * env.unwrapped.step_dt)
                episode_metrics["final_soc"].append(
                    final_soc if final_soc is not None else step_metrics["soc"][idx].item()
                )
                final_avg_temp_val = final_avg_temp if final_avg_temp is not None else step_metrics["avg_temp"][idx].item()
                final_max_temp_val = final_max_temp if final_max_temp is not None else step_metrics["max_temp"][idx].item()
                # Keep legacy keys for backward compatibility, and add explicit semantic keys.
                avg_temp_key = f"final_avg_temp_{temp_metric_semantics}"
                max_temp_key = f"final_max_temp_{temp_metric_semantics}"
                episode_metrics["final_avg_temp"].append(final_avg_temp_val)
                episode_metrics[avg_temp_key].append(final_avg_temp_val)
                episode_metrics["final_max_temp"].append(final_max_temp_val)
                episode_metrics[max_temp_key].append(final_max_temp_val)
                episode_metrics["final_max_fatigue"].append(
                    final_max_fatigue if final_max_fatigue is not None else step_metrics["max_fatigue"][idx].item()
                )
                episode_metrics["final_min_soh"].append(
                    final_min_soh if final_min_soh is not None else step_metrics["min_soh"][idx].item()
                )
                episode_metrics["max_saturation"].append(
                    final_max_saturation if final_max_saturation is not None else step_metrics["max_saturation"][idx].item()
                )

                # Survival: did it survive the full episode?
                max_ep_steps = int(env.unwrapped.cfg.episode_length_s / env.unwrapped.step_dt)
                episode_metrics["survived"].append(1.0 if ep_len >= max_ep_steps - 1 else 0.0)

                if len(reward_breakdown.term_names) > 0:
                    # Add exact terminal-step reward terms cached before reset.
                    for term_name in reward_breakdown.term_names:
                        terminal_term = _terminal_scalar(
                            terminal_metrics,
                            terminal_lookup,
                            idx,
                            f"reward_term/{term_name}",
                        )
                        if terminal_term is not None:
                            ep_term_signed[term_name][idx] += float(terminal_term)
                            ep_term_abs[term_name][idx] += abs(float(terminal_term))

                    abs_total = 0.0
                    for term_name in reward_breakdown.term_names:
                        abs_total += float(ep_term_abs[term_name][idx].item())
                    denom = max(abs_total, 1e-8)

                    for term_name in reward_breakdown.term_names:
                        signed_sum = float(ep_term_signed[term_name][idx].item())
                        abs_sum = float(ep_term_abs[term_name][idx].item())
                        episode_metrics[f"reward_{term_name}_signed_mean"].append(signed_sum / max(ep_len, 1))
                        episode_metrics[f"reward_{term_name}_abs_mean"].append(abs_sum / max(ep_len, 1))
                        episode_metrics[f"reward_share_{term_name}"].append(abs_sum / denom)

                completed_episodes += 1

                # Reset per-env buffers
                ep_step_counter[idx] = 0
                ep_total_reward[idx] = 0.0
                ep_tracking_errors[idx] = []
                ep_ang_errors[idx] = []
                ep_power_history[idx] = []
                for term_name in reward_breakdown.term_names:
                    ep_term_signed[term_name][idx] = 0.0
                    ep_term_abs[term_name][idx] = 0.0

            # Re-inject scenario for reset envs
            done_ids_device = done_envs.to(env.unwrapped.device)
            apply_scenario(env, scenario_name, done_ids_device)
            # Avoid one-step stale observation for environments just reinjected.
            obs = env.get_observations()

        if completed_episodes > 0 and (completed_episodes // 20) > (last_reported_episodes // 20):
            surv = np.mean(episode_metrics["survived"][-20:]) if len(episode_metrics["survived"]) >= 20 else np.mean(episode_metrics["survived"])
            trk = np.mean(episode_metrics["mean_tracking_error_xy"][-20:]) if len(episode_metrics["mean_tracking_error_xy"]) >= 20 else np.mean(episode_metrics["mean_tracking_error_xy"])
            print(f"  [{completed_episodes}/{num_target_episodes}] "
                  f"Survival={surv:.2%} | TrackErr={trk:.4f}")
            last_reported_episodes = completed_episodes

    # Compute summary statistics
    summary = {}
    for key, values in episode_metrics.items():
        if len(values) == 0:
            summary[key] = {
                "mean": float("nan"),
                "std": float("nan"),
                "min": float("nan"),
                "max": float("nan"),
                "median": float("nan"),
                "count": 0,
            }
            continue
        arr = np.array(values)
        summary[key] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "median": float(np.median(arr)),
            "count": len(arr),
        }

    summary["reward_breakdown"] = reward_breakdown.meta()
    return summary


def _write_reward_breakdown_csv(all_results: dict, output_dir: str, timestamp: str) -> str | None:
    """Write scenario-wise reward contribution/share summary for quick tuning."""
    rows: list[dict[str, Any]] = []
    for scenario_name, summary in all_results.items():
        rb_meta = summary.get("reward_breakdown", {})
        dt_scaled = rb_meta.get("dt_scaled", None)
        recon_mae = rb_meta.get("last_reconstruction_mae", None)

        share_keys = [k for k in summary.keys() if k.startswith("reward_share_")]
        for share_key in sorted(share_keys):
            term_name = share_key[len("reward_share_") :]
            abs_key = f"reward_{term_name}_abs_mean"
            signed_key = f"reward_{term_name}_signed_mean"

            share_stats = summary.get(share_key, {})
            abs_stats = summary.get(abs_key, {})
            signed_stats = summary.get(signed_key, {})

            rows.append(
                {
                    "scenario": scenario_name,
                    "term": term_name,
                    "share_abs_mean": share_stats.get("mean", float("nan")),
                    "share_abs_std": share_stats.get("std", float("nan")),
                    "abs_contrib_mean": abs_stats.get("mean", float("nan")),
                    "abs_contrib_std": abs_stats.get("std", float("nan")),
                    "signed_contrib_mean": signed_stats.get("mean", float("nan")),
                    "signed_contrib_std": signed_stats.get("std", float("nan")),
                    "dt_scaled": dt_scaled,
                    "reconstruction_mae": recon_mae,
                }
            )

    if len(rows) == 0:
        return None

    csv_path = os.path.join(output_dir, f"reward_breakdown_{timestamp}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Main evaluation entry point."""
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    torch.manual_seed(args_cli.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args_cli.seed)
    np.random.seed(args_cli.seed)

    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # Create environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    if hasattr(env.unwrapped, "_enable_terminal_snapshot"):
        env.unwrapped._enable_terminal_snapshot = True
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    threshold_temp, use_case_proxy, coil_to_case_delta_c = _thermal_failure_params(env.unwrapped)
    temp_metric_semantics = "case_proxy" if use_case_proxy else "coil_hotspot"

    # Load trained policy
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = runner.alg.actor_critic

    # Create output directory
    output_dir = args_cli.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Run all scenarios
    all_results = {}
    for scenario_name in SCENARIOS:
        summary = run_evaluation(
            env,
            policy,
            policy_nn,
            scenario_name,
            args_cli.num_episodes,
            temp_metric_semantics=temp_metric_semantics,
        )
        all_results[scenario_name] = summary

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_path = os.path.join(output_dir, f"eval_{timestamp}.json")
    with open(result_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[DONE] Results saved to: {result_path}")

    meta_path = os.path.join(output_dir, f"eval_{timestamp}_meta.json")
    eval_meta = {
        "task": args_cli.task,
        "temperature_metric_semantics": temp_metric_semantics,
        "temperature_metric_field": f"final_max_temp_{temp_metric_semantics}",
        "thermal_termination_threshold_c": float(threshold_temp),
        "thermal_use_case_proxy": bool(use_case_proxy),
        "coil_to_case_delta_c": float(coil_to_case_delta_c),
        "scenario_injection_note": (
            "critical/aged injection ranges are adapted when case-proxy thermal termination is enabled "
            "to avoid immediate-start truncation."
        ),
    }
    with open(meta_path, "w") as f:
        json.dump(eval_meta, f, indent=2)
    print(f"[DONE] Evaluation metadata saved to: {meta_path}")

    reward_csv_path = _write_reward_breakdown_csv(all_results=all_results, output_dir=output_dir, timestamp=timestamp)
    if reward_csv_path is not None:
        print(f"[DONE] Reward breakdown CSV saved to: {reward_csv_path}")

    # Print summary table (paper-ready)
    print_paper_table(all_results, temp_metric_semantics=temp_metric_semantics)

    env.close()


def print_paper_table(results: dict, temp_metric_semantics: str = "coil_hotspot"):
    """Print a LaTeX-friendly summary table."""
    print("\n" + "=" * 90)
    print("  PAPER TABLE: Performance Under Degradation Scenarios")
    print("=" * 90)
    print(f"Note: Max Temp semantics = {temp_metric_semantics}")
    temp_col = "MaxTemp(case)" if temp_metric_semantics == "case_proxy" else "MaxTemp(coil)"
    header = f"{'Scenario':<12} | {'Survival%':>10} | {'Track Err':>10} | {'Power(W)':>10} | {'Energy(J)':>10} | {temp_col:>12} | {'Final SOC':>10}"
    print(header)
    print("-" * 90)

    for scenario_name, summary in results.items():
        surv = summary.get("survived", {}).get("mean", 0) * 100
        trk = summary.get("mean_tracking_error_xy", {}).get("mean", 0)
        pwr = summary.get("mean_power", {}).get("mean", 0)
        eng = summary.get("total_energy", {}).get("mean", 0)
        temp_key = f"final_max_temp_{temp_metric_semantics}"
        tmp = summary.get(temp_key, summary.get("final_max_temp", {})).get("mean", 0)
        soc = summary.get("final_soc", {}).get("mean", 0)

        row = f"{scenario_name:<12} | {surv:>9.1f}% | {trk:>10.4f} | {pwr:>10.1f} | {eng:>10.1f} | {tmp:>11.1f}°C | {soc:>10.3f}"
        print(row)

    print("=" * 90)

    # LaTeX output
    print("\n% --- LaTeX Table ---")
    print("\\begin{table}[h]")
    print("\\centering")
    print(f"\\caption{{Performance comparison under degradation scenarios (temperature semantics: {temp_metric_semantics})}}")
    print("\\label{tab:degradation_results}")
    print("\\begin{tabular}{lcccccc}")
    print("\\toprule")
    print(f"Scenario & Survival (\\%) & Track. Err & Power (W) & Energy (J) & Max Temp ({temp_metric_semantics}, °C) & Final SOC \\\\")
    print("\\midrule")
    for scenario_name, summary in results.items():
        surv = summary.get("survived", {}).get("mean", 0) * 100
        trk = summary.get("mean_tracking_error_xy", {}).get("mean", 0)
        trk_std = summary.get("mean_tracking_error_xy", {}).get("std", 0)
        pwr = summary.get("mean_power", {}).get("mean", 0)
        eng = summary.get("total_energy", {}).get("mean", 0)
        temp_key = f"final_max_temp_{temp_metric_semantics}"
        tmp = summary.get(temp_key, summary.get("final_max_temp", {})).get("mean", 0)
        soc = summary.get("final_soc", {}).get("mean", 0)
        print(f"{scenario_name.capitalize()} & {surv:.1f} & {trk:.4f}$\\pm${trk_std:.4f} & {pwr:.1f} & {eng:.1f} & {tmp:.1f} & {soc:.3f} \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")


if __name__ == "__main__":
    main()
    simulation_app.close()
