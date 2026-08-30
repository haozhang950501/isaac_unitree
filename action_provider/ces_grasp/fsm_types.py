# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""CES Baseline 状态机共享的阶段、命令和坐标类型。"""
from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

    Tensor = torch.Tensor
else:
    # 保持该纯类型模块可在没有 PyTorch/Isaac Sim 的 CPU 检查环境中导入。
    Tensor = Any

RootPose = tuple[
    tuple[float, float, float],
    tuple[float, float, float, float],
]
WalkCommand = tuple[float, float, float, float]


class CesPickPlacePhase(enum.Enum):
    """CES 唯一 Baseline 主链路及安全失败阶段。"""
    SETTLE = "settle"
    GOTO_PICK = "goto_pick"
    UNFOLD = "unfold"
    DESCEND = "descend"
    GRASP = "grasp"
    LIFT = "lift"
    RETURN_HOME = "return_home"
    CARRY = "carry"
    GOTO_PLACE = "goto_place"
    PLACE_HOLD = "place_hold"
    PLACE_APPROACH = "place_approach"
    RELEASE = "release"
    RETRACT = "retract"
    DONE = "done"
    FAILED = "failed"


@dataclass
class CesCommand:
    """状态机单帧输出给动作提供器的互斥控制命令。

    ``arm_q`` 用于已批准的关节轨迹，``tcp_pos/tcp_quat`` 只用于抓取
    下降的全位姿 IK；``arm_q_ref`` 是可选零空间参考。``root_pin`` 为
    ``None`` 时允许 Wholebody 行走，否则每个物理子步钉住完整骨盆位姿。
    """

    gripper: float
    walk: WalkCommand | None = None
    root_pin: RootPose | None = None
    arm_q: Tensor | None = None
    tcp_pos: Tensor | None = None
    tcp_quat: Tensor | None = None
    arm_q_ref: Tensor | None = None
