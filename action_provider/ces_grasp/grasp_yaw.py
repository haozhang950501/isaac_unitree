# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""把 Dex1 夹持轴对正 Product 世界 X 短边的纯数学辅助函数。"""
from __future__ import annotations

import math

# pose 30 已接近世界 ±X 时跳过悬停偏航段，避免无意义的小角度插值。
YAW_ALIGN_SKIP_RAD = math.radians(8.0)


def wrap_pi(rad: float) -> float:
    """把弧度角归一化到 ``[-pi, pi]``。"""
    return (float(rad) + math.pi) % (2.0 * math.pi) - math.pi


def jaw_xy_yaw(jx: float, jy: float) -> float:
    """返回夹持轴 XY 投影偏航角：0 为世界 +X，π 为世界 -X。"""
    return math.atan2(float(jy), float(jx))


def closer_world_x_yaw(current_yaw: float) -> tuple[float, float]:
    """返回最近的世界 ±X 朝向以及从当前偏航到目标的最短角差。

    产品世界 X 短边宽约 36 mm，夹持轴朝 +X 或 -X 都能完成抓取；选择
    更近的一侧可以避免夹爪在托盘上方旋转接近 180°。
    """
    d_plus = wrap_pi(0.0 - float(current_yaw))
    d_minus = wrap_pi(math.pi - float(current_yaw))
    if abs(d_minus) < abs(d_plus):
        return math.pi, d_minus
    return 0.0, d_plus
