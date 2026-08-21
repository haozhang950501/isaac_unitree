# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""Task-agnostic manipulation primitives shared by the autonomous providers.

* :mod:`ik_solver`     - forward kinematics read-out plus damped-least-squares
                         differential inverse kinematics for a single G1 arm
                         (position-only or full 6-DoF).
* :mod:`interpolation` - Cartesian trajectory interpolation with ease-in/ease-out
                         position blending and quaternion slerp.
"""

from .ik_solver import ArmDiffIK
from .interpolation import (
    CartesianInterpolator,
    JointSpaceInterpolator,
    ease_in_out,
    lerp,
    scale_segment_times,
    slerp,
)

__all__ = [
    "ArmDiffIK",
    "CartesianInterpolator",
    "JointSpaceInterpolator",
    "ease_in_out",
    "lerp",
    "scale_segment_times",
    "slerp",
]
