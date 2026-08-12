# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0

import gymnasium as gym

from . import move_ces_product_g1_29dof_dex1_env_cfg


# NOTE: the task id has to contain "Wholebody" - both sim_main.py and
# dds/dds_create.py switch to the whole-body RL / DDS control path on that
# substring.
gym.register(
    id="Isaac-Move-CES-Product-G129-Dex1-Wholebody",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": move_ces_product_g1_29dof_dex1_env_cfg.MoveCESProductG129Dex1WholebodyEnvCfg,
    },
    disable_env_checker=True,
)
