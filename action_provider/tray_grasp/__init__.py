# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""Autonomous tray grasp-and-lift utilities.

This subpackage implements the three building blocks requested for the
"walk up, grab both tray handles, lift the tray" behaviour:

* :mod:`ik_solver`     - forward kinematics read-out plus damped-least-squares
                         differential inverse kinematics for a single G1 arm.
* :mod:`interpolation` - Cartesian / joint-space trajectory interpolation used
                         to generate smooth end-effector targets.
* :mod:`state_machine` - the high level finite state machine that sequences the
                         approach, reach, grasp and lift phases.
"""

from .ik_solver import ArmDiffIK
from .interpolation import CartesianInterpolator, ease_in_out, lerp, slerp
from .state_machine import GraspPhase, TrayGraspStateMachine

__all__ = [
    "ArmDiffIK",
    "CartesianInterpolator",
    "ease_in_out",
    "lerp",
    "slerp",
    "GraspPhase",
    "TrayGraspStateMachine",
]
