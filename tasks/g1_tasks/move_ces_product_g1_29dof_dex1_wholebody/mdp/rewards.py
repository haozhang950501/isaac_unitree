# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""CES LoadingLine product pick-and-place rewards.

+1 when Product is sitting on the packing table, -1 if it has fallen to the
floor, 0 otherwise (in transit / still on the tray).
"""
from __future__ import annotations

import torch
from typing import TYPE_CHECKING

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

_rewards_dds = None
_dds_initialized = False


def _get_rewards_dds_instance():
    global _rewards_dds, _dds_initialized
    if not _dds_initialized or _rewards_dds is None:
        try:
            from dds.dds_master import dds_manager

            _rewards_dds = dds_manager.get_object("rewards")
        except Exception:
            _rewards_dds = None
        _dds_initialized = True
    return _rewards_dds


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

    reward = torch.zeros(env.num_envs, device=env.device, dtype=torch.float)
    reward[on_table] = 1.0
    reward[dropped] = -1.0

    rewards_dds = _get_rewards_dds_instance()
    if rewards_dds:
        rewards_dds.write_rewards_data(reward)
    env._reward_last = reward
    env._reward_counter = counter + 1
    return reward
