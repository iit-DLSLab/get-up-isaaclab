# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Ant getup environment.
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##
from .getup_env import GetUpEnv


# Go2 environments
from .getup_env import Go2FlatEnvCfg, Go2RoughBlindEnvCfg

gym.register(
    id="GetUp-Go2-Flat",
    entry_point=GetUpEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": Go2FlatEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FlatPPORunnerCfg",
    },
)

gym.register(
    id="GetUp-Go2-Rough-Blind",
    entry_point=GetUpEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": Go2RoughBlindEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:RoughPPORunnerCfg",
    },
)



