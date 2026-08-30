# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""CES LoadingLine 产品取放的离散奖励。

Product 位于包装桌/灰筐有效空间时奖励 +1，低于统一掉落高度时奖励 -1，
仍在上料托盘或搬运途中时为 0。掉落条件最后覆盖，优先级高于放置成功。
"""
from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

from tasks.common_scene.base_scene_ces_pickplace_wholebody import (
    PLACE_TRAY_HEIGHT,
    PRODUCT_DROP_Z,
    TABLE_SPAWN_POS,
    TABLE_TOP_Z,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

@cache
def _get_rewards_dds_instance():
    """延迟获取并缓存 DDS 奖励对象；不可用时缓存 ``None`` 避免重复异常。"""
    try:
        from dds.dds_master import dds_manager

        return dds_manager.get_object("rewards")
    except Exception:
        return None


def compute_reward(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    table_x: float = TABLE_SPAWN_POS[0],
    table_y: float = TABLE_SPAWN_POS[1],
    half_x: float = 1.15,
    half_y: float = 0.40,
    drop_height: float = PRODUCT_DROP_Z,
    place_z_min: float = TABLE_TOP_Z + 0.01,
    place_z_max: float = TABLE_TOP_Z + PLACE_TRAY_HEIGHT + 0.12,
) -> torch.Tensor:
    """按 Product 世界位置计算每个环境的 ``-1/0/+1`` 奖励。

    参数默认值直接来自 CES 场景常量。``_reward_interval`` 由原项目
    ``sim_main`` 设置，用于复用上一帧奖励，避免 DDS 和场景读取过于频繁。
    """
    interval = getattr(env, "_reward_interval", 1) or 1
    counter = getattr(env, "_reward_counter", 0)
    last = getattr(env, "_reward_last", None)
    if interval > 1 and last is not None and counter % interval != 0:
        env._reward_counter = counter + 1
        return last

    obj: RigidObject = env.scene[object_cfg.name]
    x = obj.data.root_pos_w[:, 0]
    y = obj.data.root_pos_w[:, 1]
    z = obj.data.root_pos_w[:, 2]

    on_table = (
        (x > table_x - half_x)
        & (x < table_x + half_x)
        & (y > table_y - half_y)
        & (y < table_y + half_y)
        & (z > place_z_min)
        & (z < place_z_max)
    )
    dropped = z < drop_height

    reward = on_table.to(device=env.device, dtype=torch.float)
    reward = torch.where(dropped, -torch.ones_like(reward), reward)

    rewards_dds = _get_rewards_dds_instance()
    if rewards_dds:
        rewards_dds.write_rewards_data(reward)
    env._reward_last = reward
    env._reward_counter = counter + 1
    return reward
