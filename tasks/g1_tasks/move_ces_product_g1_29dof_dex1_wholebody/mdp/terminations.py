# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""CES Product 掉落终止条件，与奖励模块共用同一高度常量。"""
from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

from tasks.common_scene.base_scene_ces_pickplace_wholebody import PRODUCT_DROP_Z

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_dropped(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    drop_height: float = PRODUCT_DROP_Z,
) -> torch.Tensor:
    """当 Product 世界 Z 低于掉落高度时，逐环境返回 ``True``。"""
    obj: RigidObject = env.scene[object_cfg.name]
    return obj.data.root_pos_w[:, 2] < drop_height
