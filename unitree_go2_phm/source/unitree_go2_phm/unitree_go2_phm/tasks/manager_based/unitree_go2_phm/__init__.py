# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##


gym.register(
    id="Unitree-Go2-Phm-v1",
    entry_point="unitree_go2_phm.tasks.manager_based.unitree_go2_phm.unitree_go2_phm_env:UnitreeGo2PhmEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.unitree_go2_phm_env_cfg:UnitreeGo2PhmEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.mdp.agents.rsl_rl_ppo_cfg:UnitreeGo2PhmPPORunnerCfg",
    },
)

gym.register(
    id="Unitree-Go2-Baseline-v1",
    entry_point="unitree_go2_phm.tasks.manager_based.unitree_go2_phm.unitree_go2_phm_env:UnitreeGo2PhmEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.unitree_go2_baseline_env_cfg:UnitreeGo2BaselineEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.mdp.agents.rsl_rl_baseline_cfg:UnitreeGo2BaselinePPORunnerCfg",
    },
)

gym.register(
    id="Unitree-Go2-BaselineTuned-v1",
    entry_point="unitree_go2_phm.tasks.manager_based.unitree_go2_phm.unitree_go2_phm_env:UnitreeGo2PhmEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.unitree_go2_baseline_tuned_env_cfg:UnitreeGo2BaselineTunedEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.mdp.agents.rsl_rl_baseline_tuned_cfg:UnitreeGo2BaselineTunedPPORunnerCfg",
    },
)

gym.register(
    id="Unitree-Go2-RealObs-v1",
    entry_point="unitree_go2_phm.tasks.manager_based.unitree_go2_phm.unitree_go2_phm_env:UnitreeGo2PhmEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.unitree_go2_realobs_env_cfg:UnitreeGo2RealObsEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.mdp.agents.rsl_rl_realobs_cfg:UnitreeGo2RealObsPPORunnerCfg",
    },
)

gym.register(
    id="Unitree-Go2-RealObsTempDose-v1",
    entry_point="unitree_go2_phm.tasks.manager_based.unitree_go2_phm.unitree_go2_phm_env:UnitreeGo2PhmEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.unitree_go2_realobs_tempdose_env_cfg:UnitreeGo2RealObsTempDoseEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.mdp.agents.rsl_rl_realobs_cfg:UnitreeGo2RealObsPPORunnerCfg",
    },
)
