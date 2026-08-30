# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""向 Gymnasium 注册 G1 29DoF + Dex1 Wholebody 的 CES 完整取放任务。"""

import gymnasium as gym

from . import move_ces_product_g1_29dof_dex1_env_cfg


# 任务 ID 必须包含 ``Wholebody``：原项目 ``sim_main.py`` 与
# ``dds/dds_create.py`` 依靠这个子串选择全身 RL/DDS 控制路径。
gym.register(
    id="Isaac-Move-CES-Product-G129-Dex1-Wholebody",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": move_ces_product_g1_29dof_dex1_env_cfg.MoveCESProductG129Dex1WholebodyEnvCfg,
    },
    disable_env_checker=True,
)
