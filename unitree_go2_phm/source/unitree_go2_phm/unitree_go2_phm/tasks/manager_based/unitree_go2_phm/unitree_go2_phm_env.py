# =============================================================================
# unitree_go2_phm/unitree_go2_phm_env.py
# [Audit Status]: DIAMOND COPY (Debug Monitor Enabled)
#  1. Skating Solution: Implicit PD Control enabled
#  2. Physics Sync: Degraded gains -> PhysX
#  3. Debugging: Real-time Contact Force Monitor (Console Output)
# =============================================================================

from __future__ import annotations

import torch
import time
import logging
from typing import Any, Sequence

# [Isaac Lab Core]
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.assets import Articulation

# [PHM Components]
from .phm.interface import (
    init_phm_interface,
    update_phm_dynamics,
    reset_phm_interface,
    refresh_phm_sensors,
    clear_step_metrics,
)

# [PHM Constants]
from .phm.constants import (
    T_AMB,
    TEMP_WARN_THRESHOLD, 
    TEMP_CRITICAL_THRESHOLD,
    B_VISCOUS,
    WEAR_FRICTION_GAIN,
    ALPHA_CU,
    STICTION_NOMINAL,
    STICTION_WEAR_FACTOR,
    ALPHA_MAG,
)

# [PHM Utils]
from .phm.utils import compute_battery_voltage, compute_regenerative_efficiency, compute_component_losses


class UnitreeGo2PhmEnv(ManagerBasedRLEnv):
    """
    [PHM Enhanced Environment for Unitree Go2]
    
    Architecture:
    - Physics Loop (200Hz): Implicit PD Control (PhysX), Degradation Injection.
    - Control Loop (50Hz): Action Processing, Sensor Noise, Metric Aggregation.
    """

    def load_managers(self):
        if not hasattr(self, "robot"):
            self.robot: Articulation = self.scene["robot"]
        if not hasattr(self, "phm_state"):
            init_phm_interface(self)

        super().load_managers()

    def __init__(self, cfg: Any, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._dbg_first_reset = bool(getattr(cfg, "debug_first_reset", False))
        self._dbg_first_step = bool(getattr(cfg, "debug_first_step", False))
        # Disabled by default to avoid heavy console I/O in large-scale training.
        self._debug_contact_force_monitor = bool(getattr(cfg, "debug_contact_force_monitor", False))
        # Disabled by default to avoid extra tensor copies in training.
        self._enable_terminal_snapshot = bool(getattr(cfg, "enable_terminal_snapshot", False))
        self._last_terminal_env_ids = torch.empty(0, dtype=torch.long, device=self.device)
        self._last_terminal_metrics: dict[str, torch.Tensor] = {}

        # ---------------------------------------------------------------------
        # [Lifecycle Phase 2] PHM Interface Binding
        # ---------------------------------------------------------------------
        if not hasattr(self, "robot"):
            self.robot: Articulation = self.scene["robot"]
        if not hasattr(self, "phm_state"):
            init_phm_interface(self)
        
        if not hasattr(self, "phm_joint_indices"):
            # [Fix] slice(None) fallback은 PHMState(num_joints=NUM_MOTORS=12)와
            # shape 불일치를 유발할 수 있으므로, NUM_MOTORS 기반 인덱스로 대체.
            import logging
            from .phm.constants import NUM_MOTORS
            logging.warning(
                "[PHM] phm_joint_indices not set by init_phm_interface. "
                f"Using range({NUM_MOTORS}) as fallback."
            )
            self.phm_joint_indices = list(range(NUM_MOTORS))

        if isinstance(self.phm_joint_indices, slice):
            self._phm_joint_index_tensor = torch.arange(
                self.robot.data.joint_pos.shape[1], device=self.device, dtype=torch.long
            )
        else:
            self._phm_joint_index_tensor = torch.as_tensor(self.phm_joint_indices, device=self.device, dtype=torch.long)
        self._robot_to_phm_local = {
            int(robot_idx): i for i, robot_idx in enumerate(self._phm_joint_index_tensor.tolist())
        }

        # ---------------------------------------------------------------------
        # [Lifecycle Phase 3] Actuator Binding & Nominal Physics Backup
        # ---------------------------------------------------------------------
        self._nominal_effort_limits: torch.Tensor | None = None
        if hasattr(self.robot.data, "joint_effort_limits"):
             self._nominal_effort_limits = self.robot.data.joint_effort_limits.clone()
        
        self._nominal_stiffness = torch.zeros_like(self.robot.data.joint_pos)
        self._nominal_damping = torch.zeros_like(self.robot.data.joint_pos)
        found_phm_actuator = False

        for _, actuator in self.robot.actuators.items():
            if hasattr(actuator, "bind_phm_state"):
                actuator.bind_phm_state(self.phm_state)
                found_phm_actuator = True

            if hasattr(actuator, "bind_asset"):
                actuator.bind_asset(self.robot)

            act_idx = getattr(actuator, "joint_indices", None)
            if act_idx is None:
                act_idx = getattr(actuator, "_joint_ids", None)
            if act_idx is None:
                act_idx = slice(None)

            k_p = getattr(actuator, "nominal_kp", actuator.stiffness)
            k_d = getattr(actuator, "nominal_kd", actuator.damping)

            if isinstance(k_p, torch.Tensor):
                k_p_t = k_p.to(self.device)
                if k_p_t.ndim == 0:
                    self._nominal_stiffness[:, act_idx] = float(k_p_t.item())
                elif k_p_t.ndim == 1:
                    self._nominal_stiffness[:, act_idx] = k_p_t.unsqueeze(0)
                elif k_p_t.ndim == 2 and k_p_t.shape[1] == self.robot.data.joint_pos.shape[1]:
                    self._nominal_stiffness[:, act_idx] = k_p_t[:, act_idx]
                else:
                    self._nominal_stiffness[:, act_idx] = k_p_t
            else:
                self._nominal_stiffness[:, act_idx] = float(k_p)

            if isinstance(k_d, torch.Tensor):
                k_d_t = k_d.to(self.device)
                if k_d_t.ndim == 0:
                    self._nominal_damping[:, act_idx] = float(k_d_t.item())
                elif k_d_t.ndim == 1:
                    self._nominal_damping[:, act_idx] = k_d_t.unsqueeze(0)
                elif k_d_t.ndim == 2 and k_d_t.shape[1] == self.robot.data.joint_pos.shape[1]:
                    self._nominal_damping[:, act_idx] = k_d_t[:, act_idx]
                else:
                    self._nominal_damping[:, act_idx] = k_d_t
            else:
                self._nominal_damping[:, act_idx] = float(k_d)

        if not found_phm_actuator:
            raise RuntimeError("[Config Error] PHM Actuator not found in env configuration.")

        # Optional external fault profile for replay/fault-injection experiments.
        num_phm_joints = int(self._phm_joint_index_tensor.numel())
        self._external_kp_scale = torch.ones((self.num_envs, num_phm_joints), device=self.device)
        self._external_kd_scale = torch.ones((self.num_envs, num_phm_joints), device=self.device)

        # ---------------------------------------------------------------------
        # [Lifecycle Phase 4] Double-PD Prevention -> REMOVED for Implicit PD
        # ---------------------------------------------------------------------
        # [Fix] Implicit 모드에서는 PhysX가 토크를 계산해야 하므로 게인을 0으로 만들지 않습니다.
        # self.robot.write_joint_stiffness_to_sim(0.0, joint_ids=self.phm_joint_indices)
        # self.robot.write_joint_damping_to_sim(0.0, joint_ids=self.phm_joint_indices)

        # [Perf] Cache scalar constants for brownout logic (avoid per-step tensor creation)
        # Keep thresholds aligned with replay/real safety policy:
        # - 24.5V: pack hard-stop neighborhood
        # - 25.0V: recovery hysteresis to prevent latch chattering
        self._const_true = torch.tensor(True, device=self.device)
        self._const_false = torch.tensor(False, device=self.device)
        self._const_brownout_scale_low = torch.tensor(0.5, device=self.device)
        self._const_brownout_scale_high = torch.tensor(1.0, device=self.device)
        self._brownout_enter_v = torch.tensor(float(getattr(cfg, "brownout_enter_v", 24.5)), device=self.device)
        self._brownout_recover_v = torch.tensor(float(getattr(cfg, "brownout_recover_v", 25.0)), device=self.device)
        self._brownout_voltage_source = str(getattr(cfg, "brownout_voltage_source", "bms_pred")).strip().lower()
        self._init_command_transport_dr(cfg)

        clear_step_metrics(self)
        self._reward_term_specs_cache = self._iter_active_reward_terms()
        self._init_velocity_command_curriculum(cfg)
        self._init_push_curriculum(cfg)
        self._init_dr_curriculum(cfg)

    def clear_external_fault_profile(self):
        """Reset external actuator fault multipliers to nominal (1.0)."""
        self._external_kp_scale.fill_(1.0)
        self._external_kd_scale.fill_(1.0)

    def set_external_fault_profile(
        self, joint_names: Sequence[str], kp_scale: float = 1.0, kd_scale: float = 1.0
    ):
        """
        Apply constant external Kp/Kd scaling on selected joints.

        This is designed for reproducible fault-injection experiments.
        """
        self.clear_external_fault_profile()
        if len(joint_names) == 0:
            return

        robot_joint_ids, _ = self.robot.find_joints(list(joint_names))
        local_ids: list[int] = []
        for rid in robot_joint_ids:
            rid_int = int(rid.item()) if isinstance(rid, torch.Tensor) else int(rid)
            local = self._robot_to_phm_local.get(rid_int, None)
            if local is not None:
                local_ids.append(local)

        if len(local_ids) == 0:
            raise ValueError(
                f"[FaultProfile] None of joints {list(joint_names)} are part of PHM-controlled joints."
            )

        self._external_kp_scale[:, local_ids] = float(kp_scale)
        self._external_kd_scale[:, local_ids] = float(kd_scale)
        self._apply_physical_degradation()

    def reset(self, seed: int | None = None, options: dict | None = None):
        if getattr(self, "_dbg_first_reset", False):
            print(f"[DBG] Env.reset start (num_envs={self.num_envs})", flush=True)
            t0 = time.time()
        out = super().reset(seed=seed, options=options)
        if getattr(self, "_dbg_first_reset", False):
            print(f"[DBG] Env.reset done in {time.time() - t0:.2f}s", flush=True)
            self._dbg_first_reset = False
        return out

    def _init_velocity_command_curriculum(self, cfg: Any) -> None:
        """Initialize staged velocity-command range widening for locomotion training."""
        self._vel_cmd_curriculum_enable = bool(getattr(cfg, "velocity_cmd_curriculum_enable", True))
        self._vel_cmd_steps_per_iter = max(int(getattr(cfg, "velocity_cmd_curriculum_steps_per_iter", 24)), 1)
        self._vel_cmd_start_iter = max(int(getattr(cfg, "velocity_cmd_curriculum_start_iter", 160)), 0)
        self._vel_cmd_ramp_iters = max(int(getattr(cfg, "velocity_cmd_curriculum_ramp_iters", 340)), 0)
        self._vel_cmd_start_step = self._vel_cmd_start_iter * self._vel_cmd_steps_per_iter
        self._vel_cmd_ramp_steps = self._vel_cmd_ramp_iters * self._vel_cmd_steps_per_iter

        self._vel_cmd_initial_lin_x: tuple[float, float] = (-0.1, 0.1)
        self._vel_cmd_initial_lin_y: tuple[float, float] = (-0.1, 0.1)
        self._vel_cmd_initial_ang_z: tuple[float, float] = (-1.0, 1.0)

        self._vel_cmd_target_lin_x = tuple(getattr(cfg, "velocity_cmd_target_lin_vel_x", (-1.0, 1.0)))
        self._vel_cmd_target_lin_y = tuple(getattr(cfg, "velocity_cmd_target_lin_vel_y", (-0.4, 0.4)))
        self._vel_cmd_target_ang_z = tuple(getattr(cfg, "velocity_cmd_target_ang_vel_z", (-1.0, 1.0)))

        self._vel_cmd_last_alpha = -1.0
        self._vel_cmd_curriculum_started = False
        self._vel_cmd_curriculum_reached_target = False

        try:
            vel_term = self.command_manager.get_term("base_velocity")
            if hasattr(vel_term, "cfg") and hasattr(vel_term.cfg, "ranges"):
                rng = vel_term.cfg.ranges
                if hasattr(rng, "lin_vel_x"):
                    self._vel_cmd_initial_lin_x = tuple(rng.lin_vel_x)
                if hasattr(rng, "lin_vel_y"):
                    self._vel_cmd_initial_lin_y = tuple(rng.lin_vel_y)
                if hasattr(rng, "ang_vel_z"):
                    self._vel_cmd_initial_ang_z = tuple(rng.ang_vel_z)
        except Exception:
            # If command manager is not ready, keep safe defaults.
            pass

        if self._vel_cmd_curriculum_enable:
            logging.info(
                "[VelCmdCurriculum] enabled=True start_iter=%d ramp_iters=%d initial=(%s,%s,%s) target=(%s,%s,%s)",
                self._vel_cmd_start_iter,
                self._vel_cmd_ramp_iters,
                self._vel_cmd_initial_lin_x,
                self._vel_cmd_initial_lin_y,
                self._vel_cmd_initial_ang_z,
                self._vel_cmd_target_lin_x,
                self._vel_cmd_target_lin_y,
                self._vel_cmd_target_ang_z,
            )

    @staticmethod
    def _lerp_range(src: tuple[float, float], dst: tuple[float, float], alpha: float) -> tuple[float, float]:
        a = float(max(0.0, min(1.0, alpha)))
        return (
            float(src[0]) + (float(dst[0]) - float(src[0])) * a,
            float(src[1]) + (float(dst[1]) - float(src[1])) * a,
        )

    def _update_velocity_command_curriculum(self) -> None:
        """Expand base-velocity command ranges after a warmup period."""
        if not self._vel_cmd_curriculum_enable:
            return

        try:
            vel_term = self.command_manager.get_term("base_velocity")
        except Exception:
            return
        if not hasattr(vel_term, "cfg") or not hasattr(vel_term.cfg, "ranges"):
            return

        step = int(self.common_step_counter)
        if self._vel_cmd_ramp_steps <= 0:
            alpha = 1.0 if step >= self._vel_cmd_start_step else 0.0
        else:
            alpha = (step - self._vel_cmd_start_step) / float(self._vel_cmd_ramp_steps)
            alpha = float(max(0.0, min(1.0, alpha)))

        # Keep updates cheap when alpha does not change numerically.
        if abs(alpha - self._vel_cmd_last_alpha) <= 1e-9:
            return

        rng = vel_term.cfg.ranges
        rng.lin_vel_x = self._lerp_range(self._vel_cmd_initial_lin_x, self._vel_cmd_target_lin_x, alpha)
        rng.lin_vel_y = self._lerp_range(self._vel_cmd_initial_lin_y, self._vel_cmd_target_lin_y, alpha)
        rng.ang_vel_z = self._lerp_range(self._vel_cmd_initial_ang_z, self._vel_cmd_target_ang_z, alpha)

        # Force one-time immediate resample at curriculum transitions to avoid long stale commands.
        crossed_start = (self._vel_cmd_last_alpha <= 0.0) and (alpha > 0.0)
        reached_target = (self._vel_cmd_last_alpha < 1.0) and (alpha >= 1.0)
        if crossed_start or reached_target:
            vel_term.time_left[:] = 0.0

        if crossed_start and not self._vel_cmd_curriculum_started:
            self._vel_cmd_curriculum_started = True
            logging.info(
                "[VelCmdCurriculum] started at step=%d (iter≈%.1f)",
                step,
                float(step) / float(self._vel_cmd_steps_per_iter),
            )
        if reached_target and not self._vel_cmd_curriculum_reached_target:
            self._vel_cmd_curriculum_reached_target = True
            logging.info(
                "[VelCmdCurriculum] reached target at step=%d (iter≈%.1f)",
                step,
                float(step) / float(self._vel_cmd_steps_per_iter),
            )

        if hasattr(self, "extras"):
            self.extras["cmd/vel_curriculum_alpha"] = torch.tensor(alpha, device=self.device)
            self.extras["cmd/lin_vel_x_max"] = torch.tensor(float(rng.lin_vel_x[1]), device=self.device)
            self.extras["cmd/lin_vel_y_max"] = torch.tensor(float(rng.lin_vel_y[1]), device=self.device)
            self.extras["cmd/ang_vel_z_max"] = torch.tensor(float(rng.ang_vel_z[1]), device=self.device)

        self._vel_cmd_last_alpha = alpha

    def _init_push_curriculum(self, cfg: Any) -> None:
        """Initialize staged push-disturbance range widening for locomotion stability."""
        self._push_curriculum_enable = bool(getattr(cfg, "push_curriculum_enable", True))
        self._push_steps_per_iter = max(int(getattr(cfg, "push_curriculum_steps_per_iter", 24)), 1)
        self._push_start_iter = max(int(getattr(cfg, "push_curriculum_start_iter", 501)), 0)
        self._push_ramp_iters = max(int(getattr(cfg, "push_curriculum_ramp_iters", 499)), 0)
        self._push_start_step = self._push_start_iter * self._push_steps_per_iter
        self._push_ramp_steps = self._push_ramp_iters * self._push_steps_per_iter
        self._push_initial_xy = tuple(getattr(cfg, "push_curriculum_initial_xy", (0.0, 0.0)))
        self._push_target_xy = tuple(getattr(cfg, "push_curriculum_target_xy", (-0.5, 0.5)))
        self._push_last_alpha = -1.0
        self._push_curriculum_started = False
        self._push_curriculum_reached_target = False
        self._push_cfg_lookup_warned = False
        self._push_cfg_set_warned = False

        if self._push_curriculum_enable:
            logging.info(
                "[PushCurriculum] enabled=True start_iter=%d ramp_iters=%d initial_xy=%s target_xy=%s",
                self._push_start_iter,
                self._push_ramp_iters,
                self._push_initial_xy,
                self._push_target_xy,
            )

    def _init_dr_curriculum(self, cfg: Any) -> None:
        """Initialize staged DR widening (friction/mass/command-delay)."""
        self._dr_curriculum_enable = bool(getattr(cfg, "dr_curriculum_enable", True))
        self._dr_steps_per_iter = max(int(getattr(cfg, "dr_curriculum_steps_per_iter", 24)), 1)
        self._dr_start_iter = max(int(getattr(cfg, "dr_curriculum_start_iter", 501)), 0)
        self._dr_ramp_iters = max(int(getattr(cfg, "dr_curriculum_ramp_iters", 499)), 0)
        self._dr_start_step = self._dr_start_iter * self._dr_steps_per_iter
        self._dr_ramp_steps = self._dr_ramp_iters * self._dr_steps_per_iter
        self._dr_initial_friction_range = tuple(
            getattr(cfg, "dr_curriculum_initial_friction_range", (0.6, 1.25))
        )
        self._dr_target_friction_range = tuple(
            getattr(cfg, "dr_curriculum_target_friction_range", (0.5, 1.3))
        )
        self._dr_initial_mass_scale_range = tuple(
            getattr(cfg, "dr_curriculum_initial_mass_scale_range", (0.9, 1.1))
        )
        self._dr_target_mass_scale_range = tuple(
            getattr(cfg, "dr_curriculum_target_mass_scale_range", (0.8, 1.2))
        )
        self._dr_initial_cmd_delay_steps = max(
            int(getattr(cfg, "dr_curriculum_initial_cmd_delay_max_steps", self._cmd_delay_max_steps)),
            0,
        )
        self._dr_target_cmd_delay_steps = max(
            int(getattr(cfg, "dr_curriculum_target_cmd_delay_max_steps", self._cmd_delay_max_steps)),
            0,
        )
        # Ensure delay DR starts from the intended initial bound.
        self._cmd_delay_max_steps = self._dr_initial_cmd_delay_steps

        self._dr_last_alpha = -1.0
        self._dr_curriculum_started = False
        self._dr_curriculum_reached_target = False
        self._event_cfg_lookup_warned: dict[str, bool] = {}
        self._event_cfg_set_warned: dict[str, bool] = {}

        if self._dr_curriculum_enable:
            logging.info(
                "[DRCurriculum] enabled=True start_iter=%d ramp_iters=%d friction=%s->%s mass_scale=%s->%s "
                "delay_steps=%d->%d",
                self._dr_start_iter,
                self._dr_ramp_iters,
                self._dr_initial_friction_range,
                self._dr_target_friction_range,
                self._dr_initial_mass_scale_range,
                self._dr_target_mass_scale_range,
                self._dr_initial_cmd_delay_steps,
                self._dr_target_cmd_delay_steps,
            )

    def _init_command_transport_dr(self, cfg: Any) -> None:
        """Initialize command transport DR (delay/dropout as network/control jitter proxy)."""
        self._cmd_transport_dr_enable = bool(getattr(cfg, "cmd_transport_dr_enable", True))
        self._cmd_delay_max_steps = max(int(getattr(cfg, "cmd_delay_max_steps", 1)), 0)
        self._cmd_dropout_prob = float(getattr(cfg, "cmd_dropout_prob", 0.005))
        self._cmd_dropout_prob = float(max(0.0, min(1.0, self._cmd_dropout_prob)))
        self._cmd_delay_buffer: torch.Tensor | None = None
        self._cmd_last_applied: torch.Tensor | None = None

    def _apply_command_transport_dr(self, action: torch.Tensor) -> torch.Tensor:
        """Apply per-env random command delay and packet-drop hold."""
        if not self._cmd_transport_dr_enable:
            return action

        num_envs, num_actions = action.shape
        hist_len = max(self._cmd_delay_max_steps + 1, 1)

        if (
            self._cmd_delay_buffer is None
            or self._cmd_delay_buffer.shape[0] != hist_len
            or self._cmd_delay_buffer.shape[1] != num_envs
            or self._cmd_delay_buffer.shape[2] != num_actions
        ):
            self._cmd_delay_buffer = action.unsqueeze(0).repeat(hist_len, 1, 1).clone()
            self._cmd_last_applied = action.clone()

        self._cmd_delay_buffer = torch.roll(self._cmd_delay_buffer, shifts=1, dims=0)
        self._cmd_delay_buffer[0] = action

        if self._cmd_delay_max_steps > 0:
            delay_steps = torch.randint(
                low=0,
                high=self._cmd_delay_max_steps + 1,
                size=(num_envs,),
                device=self.device,
            )
        else:
            delay_steps = torch.zeros((num_envs,), dtype=torch.long, device=self.device)

        env_idx = torch.arange(num_envs, device=self.device)
        delayed_action = self._cmd_delay_buffer[delay_steps, env_idx, :]

        if self._cmd_dropout_prob > 0.0:
            drop_mask = torch.rand((num_envs,), device=self.device) < self._cmd_dropout_prob
            if torch.any(drop_mask):
                delayed_action = delayed_action.clone()
                if self._cmd_last_applied is None:
                    self._cmd_last_applied = delayed_action.clone()
                delayed_action[drop_mask] = self._cmd_last_applied[drop_mask]
        else:
            drop_mask = torch.zeros((num_envs,), dtype=torch.bool, device=self.device)

        self._cmd_last_applied = delayed_action.detach().clone()
        if hasattr(self, "extras"):
            self.extras["dr/cmd_delay_steps_mean"] = torch.mean(delay_steps.float())
            self.extras["dr/cmd_drop_rate"] = torch.mean(drop_mask.float())

        return delayed_action

    def _get_event_term_cfg(self, term_name: str) -> tuple[Any | None, str]:
        """Resolve event term cfg via EventManager API, with cfg fallback."""
        mgr = getattr(self, "event_manager", None)
        if mgr is not None and hasattr(mgr, "get_term_cfg"):
            try:
                return mgr.get_term_cfg(term_name), "manager"
            except Exception as err:
                if not self._event_cfg_lookup_warned.get(term_name, False):
                    logging.warning(
                        "[DRCurriculum] Failed to read event term '%s' from EventManager; falling back to cfg path. "
                        "error=%s",
                        term_name,
                        err,
                    )
                    self._event_cfg_lookup_warned[term_name] = True

        events_cfg = getattr(self.cfg, "events", None)
        if events_cfg is not None and hasattr(events_cfg, term_name):
            return getattr(events_cfg, term_name), "cfg"
        return None, "none"

    def _set_event_term_cfg(self, term_name: str, term_cfg: Any, source: str) -> None:
        """Write event term cfg back through public manager API when available."""
        if source != "manager":
            return
        mgr = getattr(self, "event_manager", None)
        if mgr is None or not hasattr(mgr, "set_term_cfg"):
            return
        try:
            mgr.set_term_cfg(term_name, term_cfg)
        except Exception as err:
            if not self._event_cfg_set_warned.get(term_name, False):
                logging.warning(
                    "[DRCurriculum] Failed to write event term '%s' through EventManager; keeping cfg-side update. "
                    "error=%s",
                    term_name,
                    err,
                )
                self._event_cfg_set_warned[term_name] = True

    def _update_dr_curriculum(self) -> None:
        """Widen DR range in the 501~1000-iter phase (friction/mass/latency)."""
        if not self._dr_curriculum_enable:
            return

        step = int(self.common_step_counter)
        if self._dr_ramp_steps <= 0:
            alpha = 1.0 if step >= self._dr_start_step else 0.0
        else:
            alpha = (step - self._dr_start_step) / float(self._dr_ramp_steps)
            alpha = float(max(0.0, min(1.0, alpha)))

        if abs(alpha - self._dr_last_alpha) <= 1e-9:
            return

        friction_range = self._lerp_range(self._dr_initial_friction_range, self._dr_target_friction_range, alpha)
        mass_scale_range = self._lerp_range(self._dr_initial_mass_scale_range, self._dr_target_mass_scale_range, alpha)
        delay_steps_f = (
            float(self._dr_initial_cmd_delay_steps)
            + (float(self._dr_target_cmd_delay_steps) - float(self._dr_initial_cmd_delay_steps)) * alpha
        )
        self._cmd_delay_max_steps = max(int(round(delay_steps_f)), 0)

        # Update physics-material randomization bounds.
        material_term, source = self._get_event_term_cfg("physics_material")
        if material_term is not None:
            params = getattr(material_term, "params", None)
            if isinstance(params, dict):
                params["static_friction_range"] = (float(friction_range[0]), float(friction_range[1]))
                params["dynamic_friction_range"] = (float(friction_range[0]), float(friction_range[1]))
            self._set_event_term_cfg("physics_material", material_term, source)

        # Update mass randomization scale bounds.
        mass_term, source = self._get_event_term_cfg("add_mass")
        if mass_term is not None:
            params = getattr(mass_term, "params", None)
            if isinstance(params, dict):
                params["mass_distribution_params"] = (float(mass_scale_range[0]), float(mass_scale_range[1]))
                params["operation"] = "scale"
            self._set_event_term_cfg("add_mass", mass_term, source)

        # Keep cfg-side mirror synchronized for readability.
        events_cfg = getattr(self.cfg, "events", None)
        if events_cfg is not None:
            if hasattr(events_cfg, "physics_material"):
                term = getattr(events_cfg, "physics_material")
                params = getattr(term, "params", None)
                if isinstance(params, dict):
                    params["static_friction_range"] = (float(friction_range[0]), float(friction_range[1]))
                    params["dynamic_friction_range"] = (float(friction_range[0]), float(friction_range[1]))
            if hasattr(events_cfg, "add_mass"):
                term = getattr(events_cfg, "add_mass")
                params = getattr(term, "params", None)
                if isinstance(params, dict):
                    params["mass_distribution_params"] = (float(mass_scale_range[0]), float(mass_scale_range[1]))
                    params["operation"] = "scale"

        crossed_start = (self._dr_last_alpha <= 0.0) and (alpha > 0.0)
        reached_target = (self._dr_last_alpha < 1.0) and (alpha >= 1.0)
        if crossed_start and not self._dr_curriculum_started:
            self._dr_curriculum_started = True
            logging.info(
                "[DRCurriculum] started at step=%d (iter≈%.1f)",
                step,
                float(step) / float(self._dr_steps_per_iter),
            )
        if reached_target and not self._dr_curriculum_reached_target:
            self._dr_curriculum_reached_target = True
            logging.info(
                "[DRCurriculum] reached target at step=%d (iter≈%.1f)",
                step,
                float(step) / float(self._dr_steps_per_iter),
            )

        if hasattr(self, "extras"):
            self.extras["dr/curriculum_alpha"] = torch.tensor(alpha, device=self.device)
            self.extras["dr/friction_min"] = torch.tensor(float(friction_range[0]), device=self.device)
            self.extras["dr/friction_max"] = torch.tensor(float(friction_range[1]), device=self.device)
            self.extras["dr/mass_scale_min"] = torch.tensor(float(mass_scale_range[0]), device=self.device)
            self.extras["dr/mass_scale_max"] = torch.tensor(float(mass_scale_range[1]), device=self.device)
            self.extras["dr/cmd_delay_max_steps"] = torch.tensor(float(self._cmd_delay_max_steps), device=self.device)

        self._dr_last_alpha = alpha

    def _get_push_event_term_cfg(self) -> tuple[Any | None, str]:
        """Resolve push event term cfg via stable public API, with cfg fallback."""
        mgr = getattr(self, "event_manager", None)
        if mgr is not None and hasattr(mgr, "get_term_cfg"):
            try:
                term_cfg = mgr.get_term_cfg("push_robot")
                return term_cfg, "manager"
            except Exception as err:
                if not self._push_cfg_lookup_warned:
                    logging.warning(
                        "[PushCurriculum] Failed to read push_robot term from EventManager; "
                        "falling back to cfg path. error=%s",
                        err,
                    )
                    self._push_cfg_lookup_warned = True

        events_cfg = getattr(self.cfg, "events", None)
        if events_cfg is not None and hasattr(events_cfg, "push_robot"):
            return getattr(events_cfg, "push_robot"), "cfg"

        return None, "none"

    def _update_push_curriculum(self) -> None:
        """Ramp external push disturbance from easy to target range."""
        if not self._push_curriculum_enable:
            return

        step = int(self.common_step_counter)
        if self._push_ramp_steps <= 0:
            alpha = 1.0 if step >= self._push_start_step else 0.0
        else:
            alpha = (step - self._push_start_step) / float(self._push_ramp_steps)
            alpha = float(max(0.0, min(1.0, alpha)))

        if abs(alpha - self._push_last_alpha) <= 1e-9:
            return

        push_xy = self._lerp_range(self._push_initial_xy, self._push_target_xy, alpha)
        push_term_cfg, source = self._get_push_event_term_cfg()
        if push_term_cfg is not None:
            params = getattr(push_term_cfg, "params", None)
            if isinstance(params, dict):
                vel_range = params.get("velocity_range", None)
                if isinstance(vel_range, dict):
                    vel_range["x"] = (float(push_xy[0]), float(push_xy[1]))
                    vel_range["y"] = (float(push_xy[0]), float(push_xy[1]))

            # Persist back to manager through public API.
            if source == "manager":
                mgr = getattr(self, "event_manager", None)
                if mgr is not None and hasattr(mgr, "set_term_cfg"):
                    try:
                        mgr.set_term_cfg("push_robot", push_term_cfg)
                    except Exception as err:
                        if not self._push_cfg_set_warned:
                            logging.warning(
                                "[PushCurriculum] Failed to write updated push_robot term via EventManager. "
                                "Using cfg-side update only. error=%s",
                                err,
                            )
                            self._push_cfg_set_warned = True

        # Keep config copy synchronized for logging/debug readability.
        events_cfg = getattr(self.cfg, "events", None)
        if events_cfg is not None and hasattr(events_cfg, "push_robot"):
            cfg_term = getattr(events_cfg, "push_robot")
            cfg_params = getattr(cfg_term, "params", None)
            if isinstance(cfg_params, dict):
                cfg_vel = cfg_params.get("velocity_range", None)
                if isinstance(cfg_vel, dict):
                    cfg_vel["x"] = (float(push_xy[0]), float(push_xy[1]))
                    cfg_vel["y"] = (float(push_xy[0]), float(push_xy[1]))

        crossed_start = (self._push_last_alpha <= 0.0) and (alpha > 0.0)
        reached_target = (self._push_last_alpha < 1.0) and (alpha >= 1.0)
        if crossed_start and not self._push_curriculum_started:
            self._push_curriculum_started = True
            logging.info(
                "[PushCurriculum] started at step=%d (iter≈%.1f)",
                step,
                float(step) / float(self._push_steps_per_iter),
            )
        if reached_target and not self._push_curriculum_reached_target:
            self._push_curriculum_reached_target = True
            logging.info(
                "[PushCurriculum] reached target at step=%d (iter≈%.1f)",
                step,
                float(step) / float(self._push_steps_per_iter),
            )

        if hasattr(self, "extras"):
            self.extras["cmd/push_curriculum_alpha"] = torch.tensor(alpha, device=self.device)
            self.extras["cmd/push_vel_xy_max"] = torch.tensor(float(push_xy[1]), device=self.device)

        self._push_last_alpha = alpha

    def _apply_physical_degradation(self, env_ids=None):
        """
        [PHM Dynamics Core - Physics Loop (200Hz)]
        Updates the GROUND TRUTH hardware state based on accumulated physics.
        This enables Implicit PD to handle degraded physics seamlessly.
        
        [Fix] 이중 적분 제거: update_phm_dynamics()가 이미 fatigue_index와 coil_temp를
        적분 완료했으므로, 여기서는 현재 상태를 그대로 사용하여 게인을 계산합니다.
        이전에는 한 번 더 Euler step을 적용하여 관측-물리 비대칭이 발생했습니다.
        """
        ids = slice(None) if env_ids is None else env_ids

        # 1. 현재 PHM 상태 직접 사용 (이미 update_phm_dynamics에서 적분 완료)
        fatigue = torch.clamp(self.phm_state.fatigue_index[ids], min=0.0)
        temp = torch.clamp(self.phm_state.coil_temp[ids], min=25.0)

        # 2. Actuator Gain Calculation (Physical Degradation)
        # [Fix #1] 연속적 열 감쇠 모델: voltage predictor의 저항 모델(ALPHA_CU)과 동일한 물리 사용.
        #   - T_AMB(25°C)~75°C: 저항 증가에 의한 점진적 게인 감소 (1/(1+α*ΔT))
        #   - 75°C~90°C: 추가 derating (절연 열화 / 감자 모사)
        # 이전: 75°C 이하에서 gain=1.0 (계단식) → 전압 예측과 물리적 비대칭 발생.
        gain_factor_fatigue = torch.clamp(1.0 - (fatigue * 0.6), min=0.2)
        # (a) 연속적 저항 기반 감쇠: gain ∝ 1/R ∝ 1/(1 + α*(T-T_AMB))
        resistance_derating = 1.0 / (1.0 + ALPHA_CU * (temp - T_AMB))
        # (a2) 자석 감자 효과: Kt ∝ (1 + α_mag*(T-T_AMB)), α_mag < 0이므로 고온 시 Kt 감소
        magnet_derating = torch.clamp(1.0 + ALPHA_MAG * (temp - T_AMB), min=0.5, max=1.0)
        # (b) 고온 구간 추가 derating (TEMP_WARN ~ TEMP_CRITICAL → 1.0 ~ 0.0)
        # [Fix] min=0.05: 90°C에서도 최소 5% 제어 권한을 유지하여
        # termination check 전에 Kp=0으로 인한 제어 불능 붕괴를 방지.
        severe_derating = torch.clamp(
            1.0 - (temp - TEMP_WARN_THRESHOLD) / max(TEMP_CRITICAL_THRESHOLD - TEMP_WARN_THRESHOLD, 1e-6),
            min=0.05, max=1.0,
        )
        gain_factor_thermal = resistance_derating * magnet_derating * severe_derating
        
        # [Fix #6] phm_joint_indices로 슬라이싱하여 (N, all_joints) vs (N, NUM_MOTORS) shape mismatch 방지
        joint_idx = self.phm_joint_indices
        nominal_kp = self._nominal_stiffness[ids][:, joint_idx] if not isinstance(joint_idx, slice) else self._nominal_stiffness[ids]
        current_kp = nominal_kp * gain_factor_fatigue * gain_factor_thermal
        # [Fix] 베어링 마모 시 Kd 하한을 Kp(min=0.2)보다 높게 유지 (min=0.4)
        # 물리: 마모 → 유격 증가로 감쇠가 줄어들지만, Kd가 지나치게 낮으면
        # 제어 불안정(진동/발산)이 발생하므로 안정성 확보를 위해 하한을 보존함.
        kd_factor_wear = torch.clamp(gain_factor_fatigue, min=0.4)
        nominal_kd = self._nominal_damping[ids][:, joint_idx] if not isinstance(joint_idx, slice) else self._nominal_damping[ids]
        current_kd = nominal_kd * kd_factor_wear * gain_factor_thermal

        # Apply optional external fault profile.
        current_kp = current_kp * self._external_kp_scale[ids]
        current_kd = current_kd * self._external_kd_scale[ids]

        # 3. Apply to Shared State (State for Observation/Reward)
        self.phm_state.degraded_stiffness[ids] = current_kp
        self.phm_state.degraded_damping[ids] = current_kd

        # 4. [CRITICAL] Apply to PhysX Simulator (Implicit Control Injection)
        # Actuator가 'joint_efforts=None'을 반환하므로, PhysX는 이 게인을 사용하여
        # 내부적으로 토크를 계산합니다. (Skating 제거 + 열화 반영)
        if hasattr(self.robot, "write_joint_stiffness_to_sim"):
            self.robot.write_joint_stiffness_to_sim(current_kp, joint_ids=self.phm_joint_indices, env_ids=env_ids)
            self.robot.write_joint_damping_to_sim(current_kd, joint_ids=self.phm_joint_indices, env_ids=env_ids)

    def step(self, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """
        [PHM-Aware Dual-Rate Step Function]
        """
        
        if getattr(self, "_dbg_first_step", False):
            print(f"[DBG] Env.step start action_shape={tuple(action.shape)} decimation={self.cfg.decimation}", flush=True)
            t_step0 = time.time()

        if self._enable_terminal_snapshot:
            self._last_terminal_env_ids = torch.empty(0, dtype=torch.long, device=self.device)
            self._last_terminal_metrics = {}

        # ---------------------------------------------------------------------
        # 1. Reset Step Accumulators (NOT Persistent State)
        # ---------------------------------------------------------------------
        if hasattr(self, "phm_state") and self.phm_state is not None:
            clear_step_metrics(self) 
            
            # Reset Log buffer for this step (Must be >= max possible voltage for min-tracking)
            if hasattr(self.phm_state, "min_voltage_log"):
                self.phm_state.min_voltage_log[:] = 33.6

        # ---------------------------------------------------------------------
        # 2. Action Processing & Sensor Sample-and-Hold
        # ---------------------------------------------------------------------
        action_in = self._apply_command_transport_dr(action.to(self.device))
        self.action_manager.process_action(action_in)
        self.recorder_manager.record_pre_step()
        
        # [Fix 1] Sensor Noise Synchronization
        if hasattr(self, "phm_state"):
            refresh_phm_sensors(self)

        # ---------------------------------------------------------------------
        # 3. Physics Decimation Loop (e.g., 4 steps x 5ms)
        # ---------------------------------------------------------------------
        for substep in range(self.cfg.decimation):
            # (A) Apply Action FIRST (substep 0 only)
            # [Fix #9] apply_action()을 voltage prediction보다 먼저 호출.
            # 이전: BMS가 joint_pos_target을 읽기 전에 apply_action이 호출되지 않아
            # 이전 step의 stale target으로 전압을 예측하여 brownout 판정이 1-step 지연됨.
            if substep == 0:
                self.action_manager.apply_action()

            # (B) Voltage Prediction & Brownout Logic (Control-step rate, substep 0 only)
            if substep == 0:
                # BMS Perception (Model Mismatch for Logic)
                # joint_pos_target이 이미 현재 action으로 설정된 상태
                v_bms_pred = self._predict_instant_voltage_ivp(use_noisy_state=True, use_nominal_model=True)

                # [Fix #8] Cache BMS predicted voltage for strategic observations.
                # Brownout source is configurable via cfg.brownout_voltage_source.
                if hasattr(self.phm_state, "bms_voltage_pred"):
                    self.phm_state.bms_voltage_pred[:] = v_bms_pred

                if hasattr(self.phm_state, "brownout_scale"):
                    current_scale = self.phm_state.brownout_scale
                    latched = self.phm_state.brownout_latched

                    if self._brownout_voltage_source == "true_voltage" and hasattr(self.phm_state, "battery_voltage_true"):
                        v_for_brownout = self.phm_state.battery_voltage_true
                    elif self._brownout_voltage_source == "sensor_voltage" and hasattr(self.phm_state, "battery_voltage"):
                        v_for_brownout = self.phm_state.battery_voltage
                    else:
                        v_for_brownout = v_bms_pred

                    # Latch Logic
                    is_low_voltage = v_for_brownout < self._brownout_enter_v
                    is_recovered = v_for_brownout > self._brownout_recover_v

                    new_latch = torch.where(is_low_voltage, self._const_true, latched)
                    new_latch = torch.where(is_recovered, self._const_false, new_latch)
                    self.phm_state.brownout_latched[:] = new_latch

                    # Scale Target
                    target_scale = torch.where(
                        new_latch, self._const_brownout_scale_low, self._const_brownout_scale_high
                    )

                    # LPF Smoothing
                    alpha = 0.1
                    new_scale = alpha * target_scale + (1.0 - alpha) * current_scale
                    self.phm_state.brownout_scale[:] = new_scale

            # [Fix #9] thermal_limits와 degradation을 substep 0에서만 계산.
            # 이전: 매 substep마다 재계산하여 4x 성능 낭비 (substep 간 fatigue/temp 변화 미미).
            # 수정: substep 0에서 1회 계산 후 나머지 substep에서 재사용.
            if substep == 0:
                if hasattr(self.phm_state, "brownout_scale"):
                    thermal_limits = self._compute_thermal_limits()
                    final_limits = thermal_limits * self.phm_state.brownout_scale.unsqueeze(-1)

                    if hasattr(self.robot, "write_joint_effort_limit_to_sim"):
                        self.robot.write_joint_effort_limit_to_sim(final_limits, env_ids=None)

                    self.robot.data.joint_effort_limits[:] = final_limits

                # (C) Apply Degradation BEFORE sim step (once per control step)
                self._apply_physical_degradation()

            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.scene.update(dt=self.physics_dt)
            
            # (D) Dynamics Integration (sim.step 이후 fresh torques/velocities로 상태 갱신)
            update_phm_dynamics(self, self.physics_dt)

            # (E) Min-Voltage Tracking (Fix #5)
            # 매 substep의 실제 battery_voltage로 최저값 추적.
            # 이전: substep 0의 PD 추정 기반 예측값만 캡처하여 실제 최저 전압과 괴리.
            if hasattr(self.phm_state, "min_voltage_log"):
                # [Fix #9] biased voltage 대신 true voltage 추적 (센서 바이어스 오염 방지)
                voltage_for_log = getattr(self.phm_state, "battery_voltage_true", self.phm_state.battery_voltage)
                self.phm_state.min_voltage_log[:] = torch.min(
                    self.phm_state.min_voltage_log, voltage_for_log
                )
            
            self._sim_step_counter += 1

        # ---------------------------------------------------------------------
        # 4. Post-Step Processing (RL Loop)
        # ---------------------------------------------------------------------
        
        # Rendering
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()
        if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
            self.sim.render()

        # Update Timers
        self.episode_length_buf += 1
        self.common_step_counter += 1

        # Terminations & Rewards
        self.reset_buf = self.termination_manager.compute()
        self.rew_buf = self.reward_manager.compute(dt=self.step_dt)

        # Resets
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self._cache_terminal_snapshot(reset_env_ids)
            self.recorder_manager.record_pre_reset(reset_env_ids)
            self._reset_idx(reset_env_ids)
            self.recorder_manager.record_post_reset(reset_env_ids)

        # Command & Event Updates
        self._update_velocity_command_curriculum()
        self._update_dr_curriculum()
        self._update_push_curriculum()
        self.command_manager.compute(dt=self.step_dt)
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)

        # Observations
        self.obs_buf = self.observation_manager.compute(update_history=True)

        # ---------------------------------------------------------------------
        # [DEBUG] Real-time Contact Force Monitor (Method 2)
        # ---------------------------------------------------------------------
        if self._debug_contact_force_monitor and self.common_step_counter % 60 == 0:
            # 1. Access Sensor Data
            if "contact_forces" in self.scene.sensors:
                sensor = self.scene["contact_forces"]
                
                # 2. Extract Z-force for Env 0 (Assuming [Env, Body, Axis])
                # Note: net_forces_w gives raw physics data, safer for debug
                if hasattr(sensor.data, "net_forces_w"):
                    net_forces = sensor.data.net_forces_w[0, :, 2] # Env 0, All Bodies, Z-axis
                    
                    # 3. Filter Active Contacts (> 1.0 N)
                    # We use >1.0N to filter out numerical noise/floating limbs
                    active_indices = torch.nonzero(net_forces > 1.0, as_tuple=False).squeeze(-1)
                    
                    if len(active_indices) > 0:
                        forces_str = ", ".join([f"{net_forces[i]:.1f}" for i in active_indices])
                        print(f"[Step {self.common_step_counter}] Active feet forces (N): [{forces_str}]")
                    else:
                        print(f"[Step {self.common_step_counter}] Warning: no ground contact detected")

        self.recorder_manager.record_post_step()

        # Metrics Logging
        self._log_phm_metrics()

        if getattr(self, "_dbg_first_step", False):
            print(f"[DBG] Env.step done in {time.time() - t_step0:.2f}s", flush=True)
            self._dbg_first_step = False

        return self.obs_buf, self.rew_buf, self.termination_manager.terminated, self.termination_manager.time_outs, self.extras

    def _reset_idx(self, env_ids: Sequence[int]):
        """
        [Audit Fix 7] Clear Separation of Reset Scopes
        """
        if isinstance(env_ids, slice):
            env_ids_t = torch.arange(self.num_envs, device=self.device, dtype=torch.long)[env_ids]
        else:
            env_ids_t = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if env_ids_t.numel() == 0:
            return

        # Curriculum gate input: count only non-timeout resets as instability signal.
        # Timeout resets are normal episode rollovers and should not freeze difficulty.
        non_timeout_count = float(env_ids_t.numel())
        try:
            if hasattr(self, "termination_manager") and hasattr(self.termination_manager, "time_outs"):
                time_out_mask = self.termination_manager.time_outs[env_ids_t]
                non_timeout_count = float(torch.sum(~time_out_mask).item())
        except Exception:
            pass
        self._curriculum_non_timeout_resets = non_timeout_count
        self._curriculum_reset_count = int(env_ids_t.numel())

        super()._reset_idx(env_ids_t)

        # Persistent State Reset (Fatigue, Latch, etc.)
        reset_phm_interface(self, env_ids_t)

        # [Fix #9] 리셋 환경의 센서 노이즈 재샘플링.
        # state.reset()이 encoder_noise/encoder_vel_noise를 0으로 초기화하므로,
        # 여기서 재샘플링하지 않으면 리셋 직후 첫 관측이 노이즈 없는 ground truth가 됨.
        # 이는 에이전트가 "리셋 직후에는 완벽한 센서 데이터" 패턴을 학습하는 원인.
        refresh_phm_sensors(self, env_ids=env_ids_t)

        # Command transport DR buffer cleanup for reset environments.
        if self._cmd_delay_buffer is not None:
            self._cmd_delay_buffer[:, env_ids_t, :] = 0.0
        if self._cmd_last_applied is not None:
            self._cmd_last_applied[env_ids_t] = 0.0

        # Note: step_energy_log, avg_power_log, friction_power, stall_timer,
        # brownout_scale, min_voltage_log are already reset inside reset_phm_interface().
        # Only reset fields NOT covered by reset_phm_interface:
        if hasattr(self, "phm_state"):
            if hasattr(self.phm_state, "instant_power"):
                self.phm_state.instant_power[env_ids_t] = 0.0

        # [Fix] 커리큘럼이 non-zero fatigue/temp를 설정했을 수 있으므로,
        # 리셋 직후 degraded gains를 즉시 재계산하여 observation-physics 일관성 확보
        if hasattr(self, "phm_state") and self.phm_state is not None:
            # [Fix #6] 리셋 환경만 대상으로 degradation 재계산
            self._apply_physical_degradation(env_ids=env_ids_t)

            # [Fix #4] 리셋 환경만 대상으로 effort limits 재계산.
            # 이전: 전체 환경의 limits를 덮어써서 nominal 복원이 무효화되고
            # 비리셋 환경에 불필요한 불연속이 발생했음.
            if hasattr(self.phm_state, "brownout_scale"):
                # [Fix #9] 리셋 대상 환경만 thermal limits 계산 (전체 4096 환경 연산 방지)
                thermal_limits = self._compute_thermal_limits(env_ids=env_ids_t)
                final_limits = thermal_limits * self.phm_state.brownout_scale[env_ids_t].unsqueeze(-1)
                self.robot.data.joint_effort_limits[env_ids_t] = final_limits
                if hasattr(self.robot, "write_joint_effort_limit_to_sim"):
                    self.robot.write_joint_effort_limit_to_sim(final_limits, env_ids=env_ids_t)

    # -------------------------------------------------------------------------
    # Helper Methods
    # -------------------------------------------------------------------------

    def _predict_instant_voltage_ivp(self, use_noisy_state: bool = False, use_nominal_model: bool = False) -> torch.Tensor:
        """
        [Fix 3] Instant Voltage Prediction with Model Mismatch Capability
        
        Electrical/mechanical losses are computed via shared compute_component_losses() (SSOT).
        Only friction computation is mode-specific (nominal vs actual) for model mismatch.
        """
        
        # 1. State Source Selection (Sensor Noise)
        joint_dim_robot = self.robot.data.joint_pos.shape[1]
        gt_pos = self.robot.data.joint_pos[:, self.phm_joint_indices]
        gt_vel = self.robot.data.joint_vel[:, self.phm_joint_indices]
        if use_noisy_state:
            # [Fix] BMS는 전압 센서에서 SOC를 역추정하므로, 전압 센서 바이어스가 SOC 오차 유발
            # OCV 기울기 ≈ 9.6V/100% → 전압 바이어스 1V ≈ SOC 오차 ~0.104
            soc_bias = getattr(self.phm_state, "voltage_sensor_bias", None)
            if soc_bias is not None and isinstance(soc_bias, torch.Tensor):
                soc_estimation_error = soc_bias / 9.6  # V → SOC fraction
                soc = torch.clamp(self.phm_state.soc + soc_estimation_error, 0.0, 1.0)
            else:
                soc = self.phm_state.soc

            if hasattr(self.phm_state, "encoder_meas_pos") and hasattr(self.phm_state, "encoder_meas_vel"):
                current_pos = self.phm_state.encoder_meas_pos
                joint_vel = self.phm_state.encoder_meas_vel
            else:
                pos_noise = getattr(self.phm_state, "encoder_noise", 0.0)
                pos_offset = getattr(self.phm_state, "encoder_offset", 0.0)
                if isinstance(pos_noise, torch.Tensor):
                    if pos_noise.ndim == 2 and pos_noise.shape[1] == joint_dim_robot:
                        pos_noise = pos_noise[:, self.phm_joint_indices]
                if isinstance(pos_offset, torch.Tensor):
                    if pos_offset.ndim == 2 and pos_offset.shape[1] == joint_dim_robot:
                        pos_offset = pos_offset[:, self.phm_joint_indices]
                current_pos = gt_pos + pos_offset + pos_noise

                vel_noise = getattr(self.phm_state, "encoder_vel_noise", 0.0)
                if isinstance(vel_noise, torch.Tensor):
                    if vel_noise.ndim == 2 and vel_noise.shape[1] == joint_dim_robot:
                        vel_noise = vel_noise[:, self.phm_joint_indices]
                joint_vel = gt_vel + vel_noise
        else:
            soc = self.phm_state.soc
            current_pos = gt_pos
            joint_vel = gt_vel

        # 2. Parameter Source Selection (Model Mismatch Logic)
        if use_nominal_model:
            # BMS View: Unknown degradation, assumes Nominal specs
            # _nominal_stiffness is (N, all_joints) → slice to PHM joints
            kp = self._nominal_stiffness 
            kd = self._nominal_damping
            if isinstance(kp, torch.Tensor) and kp.ndim == 2 and kp.shape[1] == joint_dim_robot:
                kp = kp[:, self.phm_joint_indices]
            if isinstance(kd, torch.Tensor) and kd.ndim == 2 and kd.shape[1] == joint_dim_robot:
                kd = kd[:, self.phm_joint_indices]
        else:
            # Physics View: Actual degraded hardware
            # degraded_stiffness is already (N, NUM_MOTORS) → no slicing needed
            kp = self.phm_state.degraded_stiffness
            kd = self.phm_state.degraded_damping

        # 3. Torque Source Selection
        if use_nominal_model:
            # BMS path: PD 공식으로 토크 추정 (PhysX 내부 데이터 접근 불가)
            target_pos = self.robot.data.joint_pos_target[:, self.phm_joint_indices]
            est_torque_demand = kp * (target_pos - current_pos) - kd * joint_vel
            torque_clamp = self._nominal_effort_limits[:, self.phm_joint_indices] if self._nominal_effort_limits is not None else 23.7
            est_torque_demand = torch.clamp(est_torque_demand, -torque_clamp, torque_clamp)
        else:
            # [Fix #3] Ground truth: PhysX가 실제 적용한 토크 사용.
            # 이전: PD 공식 추정 토크를 사용하여 interface.py의 전력 계산과 체계적 괴리 발생.
            # PhysX applied_torque는 접촉력, 관절 한계, 수치 솔버를 모두 반영한 실제 값.
            est_torque_demand = self.robot.data.applied_torque[:, self.phm_joint_indices]

        # 4. Power Calculation (SSOT: compute_component_losses from utils.py)
        vel_abs = torch.abs(joint_vel)
        
        # (a) Mode-specific friction (model mismatch by design)
        base_stiction = getattr(self.phm_state, "base_friction_torque", None)
        if base_stiction is not None and isinstance(base_stiction, torch.Tensor):
            if base_stiction.ndim == 2:
                base_stiction = base_stiction[:, self.phm_joint_indices]
        else:
            base_stiction = STICTION_NOMINAL

        if use_nominal_model:
            # BMS는 로봇별 실제 마찰을 모르므로 공칭 스펙 사용
            p_friction = B_VISCOUS * vel_abs * vel_abs + STICTION_NOMINAL * vel_abs
            temp_for_loss = T_AMB
        else:
            fatigue = torch.clamp(self.phm_state.fatigue_index[:, self.phm_joint_indices] if self.phm_state.fatigue_index.ndim == 2 else self.phm_state.fatigue_index, min=0.0)
            friction_bias = getattr(self.phm_state, "friction_bias", None)
            if friction_bias is not None:
                if friction_bias.ndim == 2:
                    friction_bias = friction_bias[:, self.phm_joint_indices]
            else:
                friction_bias = 1.0
            viscous_coeff = B_VISCOUS * (1.0 + fatigue * WEAR_FRICTION_GAIN) * friction_bias
            stiction_val = base_stiction * (1.0 + fatigue * STICTION_WEAR_FACTOR) * friction_bias
            p_friction = viscous_coeff * vel_abs * vel_abs + stiction_val * vel_abs
            coil_temp = self.phm_state.coil_temp[:, self.phm_joint_indices] if self.phm_state.coil_temp.ndim == 2 else self.phm_state.coil_temp
            temp_for_loss = coil_temp
        
        # (b) Electrical + mechanical losses via shared utility (SSOT with interface.py)
        p_copper, p_inverter, p_mech_total = compute_component_losses(
            torque=est_torque_demand,
            velocity=joint_vel,
            temp=temp_for_loss,
            external_friction_power=p_friction
        )
        
        # (c) Mechanical work with regen efficiency
        gross_mechanical_power = est_torque_demand * joint_vel
        net_mechanical_power = gross_mechanical_power - p_friction
        regen_eff = compute_regenerative_efficiency(joint_vel)
        mechanical_load = torch.where(
            net_mechanical_power < 0,
            net_mechanical_power * regen_eff,
            net_mechanical_power
        )
        
        # (d) Total Power = all losses + net mechanical work
        est_power_load = torch.sum(
            p_copper + p_inverter + p_mech_total + mechanical_load, dim=1
        )
        
        return compute_battery_voltage(soc, est_power_load)

    def _compute_thermal_limits(self, env_ids=None) -> torch.Tensor:
        """
        Computes physical torque limits based on Temperature, Fatigue, and Wear.
        [Fix #2] Fatigue에 의한 게인 감소도 effort limit에 반영.
        [Fix #6] 모든 PHM 텐서를 joint_idx로 슬라이싱하여 shape 안전성 확보.
        [Fix #9] env_ids 지원: _reset_idx에서 리셋 대상만 계산하여 불필요한 전체 연산 방지.
        
        Args:
            env_ids: 특정 환경만 계산할 경우 지정. None이면 전체 환경 계산.
        """
        if self._nominal_effort_limits is None: 
            if env_ids is None:
                return self.robot.data.joint_effort_limits.clone()
            return self.robot.data.joint_effort_limits[env_ids].clone()

        ids = slice(None) if env_ids is None else env_ids
        limits = self._nominal_effort_limits[ids].clone()
        joint_idx = self.phm_joint_indices

        # [Fix #6] 모든 PHM 텐서를 joint_idx로 슬라이싱 (shape mismatch 방지)
        coil_temp = self.phm_state.coil_temp[ids]
        coil_temp = coil_temp[:, joint_idx] if coil_temp.ndim == 2 else coil_temp
        fatigue = torch.nan_to_num(self.phm_state.fatigue_index[ids], nan=0.0).clamp(min=0.0)
        fatigue = fatigue[:, joint_idx] if fatigue.ndim == 2 else fatigue
        friction_bias = self.phm_state.friction_bias[ids]
        friction_bias = friction_bias[:, joint_idx] if friction_bias.ndim == 2 else friction_bias
        # [Fix #1 consistency] 연속적 열 감쇠 + 고온 derating (voltage predictor와 동일 물리)
        resistance_derating = 1.0 / (1.0 + ALPHA_CU * (coil_temp - T_AMB))
        # [Fix] min=0.05: _apply_physical_degradation과 동일 — 제어 불능 방지.
        severe_derating = torch.clamp(
            1.0 - (coil_temp - TEMP_WARN_THRESHOLD) / max(TEMP_CRITICAL_THRESHOLD - TEMP_WARN_THRESHOLD, 1e-6),
            min=0.05, max=1.0,
        )
        magnet_derating = torch.clamp(1.0 + ALPHA_MAG * (coil_temp - T_AMB), min=0.5, max=1.0)
        thermal_factor = resistance_derating * magnet_derating * severe_derating

        # [Fix #2] Fatigue에 의한 게인 감소 반영 (_apply_physical_degradation과 동일 공식)
        # 이전: effort limit이 fatigue를 무시하여, Kp가 줄었는데 effort limit은 그대로인 모순 발생.
        fatigue_factor = torch.clamp(1.0 - (fatigue * 0.6), min=0.2)

        # 복합 derating: thermal × fatigue
        combined_factor = thermal_factor * fatigue_factor
        if isinstance(joint_idx, slice):
            limits *= combined_factor
        else:
            limits[:, joint_idx] *= combined_factor
        
        # Friction Loss Compensation (Effective Torque Reduction)
        # [Fix #6] base_friction_torque도 joint_idx 슬라이싱
        base_stiction = getattr(self.phm_state, "base_friction_torque", None)
        if base_stiction is not None and isinstance(base_stiction, torch.Tensor):
            base_stiction_s = base_stiction[ids]
            base_stiction_s = base_stiction_s[:, joint_idx] if base_stiction_s.ndim == 2 else base_stiction_s
        else:
            base_stiction_s = STICTION_NOMINAL
        
        # [Fix #7] 정적 마찰(stiction)만 effort limit에서 차감.
        # 이전: 속도 의존 점성 마찰(viscous)도 차감하여 effort limit이 매 substep 변동,
        # 모터 peak torque가 본래 속도 독립적 특성인 물리 원칙에 위배됨.
        # [Fix] fatigue-dependent 마찰 증가 반영 (interface.py 마찰 계산과 동일 공식).
        # fatigue_factor는 모터의 전기적 토크 생성 능력 감소 (곱셈적),
        # stiction의 fatigue 스케일링은 베어링 마모에 의한 기계적 마찰 증가 (가산적).
        stiction_loss = base_stiction_s * (1.0 + fatigue * STICTION_WEAR_FACTOR) * friction_bias
        
        # [Fix] 마찰 보상 계수 0.5 적용: stiction은 이미 interface.py에서
        # (1) 마찰열 → 온도 상승 → thermal derating, (2) 마찰 전력 → SOC 감소 → brownout
        # 두 경로로 성능 저하에 기여하므로, effort limit 차감은 절반만 적용하여
        # 3중 경로에 의한 과도한 열화를 완화합니다.
        friction_total = stiction_loss * 0.5
        if isinstance(joint_idx, slice):
            limits -= friction_total
        else:
            limits[:, joint_idx] -= friction_total
        
        return torch.clamp(limits, min=0.0)

    def _iter_active_reward_terms(self) -> list[tuple[str, Any, float, dict[str, Any]]]:
        """Collect active reward terms from cfg for optional diagnostics."""
        reward_cfg = getattr(getattr(self, "cfg", None), "rewards", None)
        if reward_cfg is None:
            return []

        if hasattr(reward_cfg, "__dataclass_fields__"):
            candidate_names = list(reward_cfg.__dataclass_fields__.keys())
        else:
            candidate_names = [k for k in vars(reward_cfg).keys() if not k.startswith("_")]

        terms: list[tuple[str, Any, float, dict[str, Any]]] = []
        for name in candidate_names:
            term = getattr(reward_cfg, name, None)
            if term is None or not hasattr(term, "func") or not hasattr(term, "weight"):
                continue
            try:
                weight = float(term.weight)
            except Exception:
                continue
            if abs(weight) <= 1e-12:
                continue
            params = getattr(term, "params", None)
            if params is None:
                params = {}
            if not isinstance(params, dict):
                continue
            terms.append((name, term.func, weight, params))
        return terms

    def _as_reward_vector(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.as_tensor(x, device=self.device, dtype=torch.float32)
        if y.ndim == 0:
            y = y.repeat(self.num_envs)
        if y.ndim > 1:
            if y.shape[-1] == 1:
                y = y.squeeze(-1)
            else:
                y = torch.mean(y, dim=tuple(range(1, y.ndim)))
        return y

    def _compute_reward_term_contributions(
        self, env_ids: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], bool | None, float | None]:
        """
        Compute per-term weighted contributions for selected env_ids.

        Used only for terminal snapshot diagnostics when enabled.
        """
        if env_ids.numel() == 0:
            return {}, None, None
        if not hasattr(self, "rew_buf"):
            return {}, None, None

        term_specs = getattr(self, "_reward_term_specs_cache", None)
        if term_specs is None:
            term_specs = self._iter_active_reward_terms()
            self._reward_term_specs_cache = term_specs
        if len(term_specs) == 0:
            return {}, None, None

        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        target_reward = self._as_reward_vector(self.rew_buf)[env_ids]
        sum_no_dt = torch.zeros_like(target_reward)
        sum_with_dt = torch.zeros_like(target_reward)
        contrib_no_dt: dict[str, torch.Tensor] = {}

        for term_name, term_func, term_weight, term_params in term_specs:
            raw = term_func(self, **term_params)
            raw = torch.nan_to_num(self._as_reward_vector(raw), nan=0.0, posinf=0.0, neginf=0.0)
            weighted = raw * float(term_weight)
            weighted_sel = weighted[env_ids]
            contrib_no_dt[term_name] = weighted_sel
            sum_no_dt = sum_no_dt + weighted_sel
            sum_with_dt = sum_with_dt + (weighted_sel * float(self.step_dt))

        mae_no_dt = float(torch.mean(torch.abs(sum_no_dt - target_reward)).item())
        mae_with_dt = float(torch.mean(torch.abs(sum_with_dt - target_reward)).item())
        use_dt = bool(mae_with_dt <= mae_no_dt)

        if use_dt:
            contrib = {k: v * float(self.step_dt) for k, v in contrib_no_dt.items()}
            recon = sum_with_dt
        else:
            contrib = contrib_no_dt
            recon = sum_no_dt
        recon_mae = float(torch.mean(torch.abs(recon - target_reward)).item())
        return contrib, use_dt, recon_mae

    def _cache_terminal_snapshot(self, env_ids: torch.Tensor):
        """Cache terminal metrics before reset so evaluators can avoid post-reset contamination."""
        if not self._enable_terminal_snapshot or env_ids.numel() == 0:
            return
        if self.phm_state is None:
            return

        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        robot = self.scene["robot"]
        phm = self.phm_state

        cmd = self.command_manager.get_command("base_velocity")[env_ids]
        actual_vel = robot.data.root_lin_vel_b[env_ids][:, :2]
        actual_ang = robot.data.root_ang_vel_b[env_ids][:, 2]
        tracking_error_xy = torch.norm(cmd[:, :2] - actual_vel, dim=1)
        tracking_error_ang = torch.abs(cmd[:, 2] - actual_ang)

        total_power = torch.sum(phm.avg_power_log[env_ids], dim=1)

        # Keep terminal temperature metrics consistent with task thermal semantics:
        # RealObs may terminate on case/housing proxy, while privileged PHM uses coil.
        temps = phm.coil_temp[env_ids]
        term_cfg = getattr(getattr(self, "cfg", None), "terminations", None)
        thermal_failure = getattr(term_cfg, "thermal_failure", None) if term_cfg is not None else None
        params = getattr(thermal_failure, "params", None)
        use_case_proxy = False
        coil_to_case_delta_c = 5.0
        if isinstance(params, dict):
            use_case_proxy = bool(params.get("use_case_proxy", use_case_proxy))
            coil_to_case_delta_c = float(params.get("coil_to_case_delta_c", coil_to_case_delta_c))
        if use_case_proxy:
            case_like = None
            for name in (
                "motor_case_temp",
                "case_temp",
                "motor_temp_case",
                "housing_temp",
                "motor_housing_temp",
            ):
                if hasattr(phm, name):
                    val = getattr(phm, name)
                    if isinstance(val, torch.Tensor):
                        case_like = val[env_ids]
                        break
            if case_like is None:
                case_like = phm.coil_temp[env_ids] - float(coil_to_case_delta_c)
            temps = case_like

        avg_temp = torch.mean(temps, dim=1)
        max_temp = torch.max(temps, dim=1)[0]
        max_fatigue = torch.max(phm.fatigue_index[env_ids], dim=1)[0]
        soh = phm.motor_health_capacity[env_ids] - phm.fatigue_index[env_ids]
        min_soh = torch.min(soh, dim=1)[0]
        max_saturation = torch.max(phm.torque_saturation[env_ids], dim=1)[0]
        reward_terms, reward_dt_scaled, reward_recon_mae = self._compute_reward_term_contributions(env_ids)

        self._last_terminal_env_ids = env_ids.clone()
        self._last_terminal_metrics = {
            "tracking_error_xy": tracking_error_xy.detach().clone(),
            "tracking_error_ang": tracking_error_ang.detach().clone(),
            "total_power": total_power.detach().clone(),
            "soc": phm.soc[env_ids].detach().clone(),
            "avg_temp": avg_temp.detach().clone(),
            "max_temp": max_temp.detach().clone(),
            "max_fatigue": max_fatigue.detach().clone(),
            "min_soh": min_soh.detach().clone(),
            "max_saturation": max_saturation.detach().clone(),
        }
        for term_name, term_val in reward_terms.items():
            self._last_terminal_metrics[f"reward_term/{term_name}"] = term_val.detach().clone()
        if reward_dt_scaled is not None:
            dt_scaled_flag = 1.0 if reward_dt_scaled else 0.0
            self._last_terminal_metrics["reward_term/dt_scaled"] = torch.full(
                (env_ids.numel(),), dt_scaled_flag, device=self.device, dtype=torch.float32
            )
        if reward_recon_mae is not None:
            self._last_terminal_metrics["reward_term/recon_mae"] = torch.full(
                (env_ids.numel(),), float(reward_recon_mae), device=self.device, dtype=torch.float32
            )

    def _log_phm_metrics(self):
        if self.phm_state is None: return
        temps = self.phm_state.coil_temp
        term_cfg = getattr(getattr(self, "cfg", None), "terminations", None)
        thermal_failure = getattr(term_cfg, "thermal_failure", None) if term_cfg is not None else None
        params = getattr(thermal_failure, "params", None)
        use_case_proxy = False
        coil_to_case_delta_c = 5.0
        if isinstance(params, dict):
            use_case_proxy = bool(params.get("use_case_proxy", use_case_proxy))
            coil_to_case_delta_c = float(params.get("coil_to_case_delta_c", coil_to_case_delta_c))
        if use_case_proxy:
            case_like = None
            for name in (
                "motor_case_temp",
                "case_temp",
                "motor_temp_case",
                "housing_temp",
                "motor_housing_temp",
            ):
                if hasattr(self.phm_state, name):
                    val = getattr(self.phm_state, name)
                    if isinstance(val, torch.Tensor):
                        case_like = val
                        break
            if case_like is None:
                case_like = self.phm_state.coil_temp - float(coil_to_case_delta_c)
            temps = case_like

        self.extras["phm/avg_temp"] = torch.mean(temps)
        if hasattr(self.phm_state, "motor_case_temp"):
            self.extras["phm/avg_case_temp"] = torch.mean(self.phm_state.motor_case_temp)
        self.extras["phm/max_fatigue"] = torch.max(self.phm_state.fatigue_index)
        self.extras["phm/saturation_rate"] = torch.mean(self.phm_state.torque_saturation)
        if hasattr(self.phm_state, "fault_mask"):
            fault_mask = self.phm_state.fault_mask
            self.extras["phm/fault_joint_ratio"] = torch.mean(fault_mask)
            if fault_mask.ndim == 2 and fault_mask.shape[1] >= 12:
                fr = torch.max(fault_mask[:, 0:3], dim=1)[0]
                fl = torch.max(fault_mask[:, 3:6], dim=1)[0]
                rr = torch.max(fault_mask[:, 6:9], dim=1)[0]
                rl = torch.max(fault_mask[:, 9:12], dim=1)[0]
                self.extras["phm/fault_leg_fr_ratio"] = torch.mean(fr)
                self.extras["phm/fault_leg_fl_ratio"] = torch.mean(fl)
                self.extras["phm/fault_leg_rr_ratio"] = torch.mean(rr)
                self.extras["phm/fault_leg_rl_ratio"] = torch.mean(rl)
        if hasattr(self.phm_state, "fault_motor_id"):
            fault_motor_id = self.phm_state.fault_motor_id
            valid_fault = fault_motor_id >= 0
            if torch.any(valid_fault):
                self.extras["phm/fault_motor_id_mean"] = torch.mean(fault_motor_id[valid_fault].float())
            else:
                self.extras["phm/fault_motor_id_mean"] = torch.tensor(-1.0, device=self.device)
        if hasattr(self.phm_state, "min_voltage_log"):
             self.extras["phm/min_voltage"] = torch.min(self.phm_state.min_voltage_log)
        if hasattr(self.phm_state, "cell_voltage"):
            self.extras["phm/min_cell_voltage"] = torch.min(self.phm_state.cell_voltage)
        if hasattr(self.phm_state, "encoder_hold_flag"):
            self.extras["dr/encoder_hold_rate"] = torch.mean(self.phm_state.encoder_hold_flag)
        if hasattr(self, "_curriculum_effective_step"):
            self.extras["phm/curriculum_effective_step"] = torch.tensor(
                float(self._curriculum_effective_step), device=self.device
            )
        if hasattr(self, "_curriculum_term_ema"):
            self.extras["phm/curriculum_term_ema"] = torch.tensor(
                float(self._curriculum_term_ema), device=self.device
            )
        if hasattr(self, "_curriculum_track_ema"):
            self.extras["phm/curriculum_track_ema"] = torch.tensor(
                float(self._curriculum_track_ema), device=self.device
            )
        if hasattr(self, "_curriculum_p_fresh"):
            self.extras["phm/curriculum_p_fresh"] = torch.tensor(
                float(self._curriculum_p_fresh), device=self.device
            )
        if hasattr(self, "_curriculum_p_used"):
            self.extras["phm/curriculum_p_used"] = torch.tensor(
                float(self._curriculum_p_used), device=self.device
            )
        if hasattr(self, "_curriculum_p_aged"):
            self.extras["phm/curriculum_p_aged"] = torch.tensor(
                float(self._curriculum_p_aged), device=self.device
            )
        if hasattr(self, "_curriculum_p_crit"):
            self.extras["phm/curriculum_p_crit"] = torch.tensor(
                float(self._curriculum_p_crit), device=self.device
            )
