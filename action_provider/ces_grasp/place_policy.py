# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""Pure helpers for the CES Place handoff policy."""
from __future__ import annotations

# Root-frame Z.  The G1 root is Z-up, same as world, so this is world height.
PLACE_DESCEND_POS_AXES = (2,)


def z_only_descend_goal(
    live_xyz: tuple[float, float, float], target_z: float
) -> tuple[float, float, float]:
    """Target height for Place descend; X/Y are hints, not a locked IK task.

    The controller only tracks Z.  Shoulder rotation may lower the arm and
    the live TCP is allowed to drift in X/Y.  Never raises above the live Z.
    """
    x, y, z = (float(value) for value in live_xyz)
    return x, y, min(z, float(target_z))
