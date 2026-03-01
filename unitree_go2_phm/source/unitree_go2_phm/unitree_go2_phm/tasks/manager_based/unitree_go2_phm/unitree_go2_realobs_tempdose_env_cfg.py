from __future__ import annotations

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import TerminationTermCfg as TermTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import GaussianNoiseCfg

from . import mdp as phm_mdp
from .unitree_go2_realobs_env_cfg import RealObsObservationsCfg, RealObsTerminationsCfg, UnitreeGo2RealObsEnvCfg


@configclass
class RealObsTempDoseObservationsCfg(RealObsObservationsCfg):
    """RealObs + minimal PHM thermal dynamics (dT/dt + thermal dose)."""

    @configclass
    class PolicyCfg(RealObsObservationsCfg.PolicyCfg):
        thermal_rate = ObsTerm(
            func=phm_mdp.thermal_rate_realobs,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "rate_scale_c_per_s": 3.0,
                "use_case_proxy": True,
            },
            noise=GaussianNoiseCfg(std=0.02),
        )
        thermal_dose = ObsTerm(
            func=phm_mdp.thermal_overload_duration_obs,
            params={"asset_cfg": SceneEntityCfg("robot"), "scale": 0.01},
            noise=None,
        )

    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CriticCfg(RealObsObservationsCfg.CriticCfg):
        thermal_rate = ObsTerm(
            func=phm_mdp.thermal_rate_realobs,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "rate_scale_c_per_s": 3.0,
                "use_case_proxy": True,
            },
            noise=None,
        )
        thermal_dose = ObsTerm(
            func=phm_mdp.thermal_overload_duration_obs,
            params={"asset_cfg": SceneEntityCfg("robot"), "scale": 0.01},
            noise=None,
        )

    critic: CriticCfg = CriticCfg()


@configclass
class UnitreeGo2RealObsTempDoseEnvCfg(UnitreeGo2RealObsEnvCfg):
    """Non-breaking variant: keep RealObs-v1 intact, add thermal dynamics channels here."""

    observations: RealObsTempDoseObservationsCfg = RealObsTempDoseObservationsCfg()
    # TempDose side task reports coil-hotspot metrics to align with 90C hard-stop semantics.
    temperature_metric_semantics: str = "coil_hotspot"

    @configclass
    class TempDoseTerminationsCfg(RealObsTerminationsCfg):
        # Side ablation task: keep a spec-aligned hard thermal stop (coil 기준 90C).
        thermal_failure = TermTerm(
            func=phm_mdp.thermal_runaway,
            params={"threshold_temp": 90.0, "use_case_proxy": False},
        )

    terminations: TempDoseTerminationsCfg = TempDoseTerminationsCfg()
