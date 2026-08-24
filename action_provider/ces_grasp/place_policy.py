# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""Pure helpers for the CES Place handoff policy."""
from __future__ import annotations


def z_only_descend_goal(
    live_xyz: tuple[float, float, float], target_z: float
) -> tuple[float, float, float]:
    """Preserve live X/Y and descend toward ``target_z`` without ever raising."""
    x, y, z = (float(value) for value in live_xyz)
    return x, y, min(z, float(target_z))
