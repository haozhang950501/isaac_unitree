# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""Pure helpers: square the Dex1 jaw onto the product's world-X faces."""
from __future__ import annotations

import math

# Skip the hover yaw-align leg when pose 30 is already on world ±X.
YAW_ALIGN_SKIP_RAD = math.radians(8.0)


def wrap_pi(rad: float) -> float:
    return (float(rad) + math.pi) % (2.0 * math.pi) - math.pi


def jaw_xy_yaw(jx: float, jy: float) -> float:
    """Yaw of the jaw's XY projection: 0 = world +X, π = world −X."""
    return math.atan2(float(jy), float(jx))


def closer_world_x_yaw(current_yaw: float) -> tuple[float, float]:
    """Nearest world ±X heading and wrapped delta from ``current_yaw``.

    Pinching the 36 mm world-X faces works with jaw = +X or −X.  The nearer
    one avoids a ~180° spin above the tray.
    """
    d_plus = wrap_pi(0.0 - float(current_yaw))
    d_minus = wrap_pi(math.pi - float(current_yaw))
    if abs(d_minus) < abs(d_plus):
        return math.pi, d_minus
    return 0.0, d_plus
