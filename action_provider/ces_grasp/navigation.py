# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""Pure-math navigation helpers for the CES carry walk.

The locomotion policy consumes commands in the robot body frame.  Scene goals
are expressed in the world frame, so a robot spawned at yaw=pi must first turn
toward a +world-X goal instead of blindly receiving +vx.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


def wrap_angle(angle: float) -> float:
    """Wrap an angle to [-pi, pi]."""
    return math.atan2(math.sin(angle), math.cos(angle))


def _clamp(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))


@dataclass(frozen=True)
class BodyWalkPlan:
    """One body-frame command and the geometry used to produce it."""

    command: tuple[float, float, float, float]
    distance: float
    desired_yaw: float
    yaw_error: float
    mode: str
    pose_ready: bool


def plan_turn_then_forward(
    *,
    x: float,
    y: float,
    yaw: float,
    target_xy: tuple[float, float],
    target_yaw: float,
    require_yaw: bool,
    pos_tolerance: float,
    yaw_tolerance: float,
    align_tolerance: float,
    min_vx: float,
    max_vx: float,
    distance_gain: float,
    yaw_gain: float,
    max_wz: float,
    height: float = 0.8,
) -> BodyWalkPlan:
    """Plan a safe body-frame gait command toward a world-frame target.

    Translation is deliberately forward-only.  When the target is behind the
    robot (the CES scene starts at yaw=pi while the place route is toward
    +world-X), the command rotates in place first.  This avoids asking the
    policy for the unstable negative-vx carry gait.
    """
    dx = target_xy[0] - x
    dy = target_xy[1] - y
    distance = math.hypot(dx, dy)

    if distance > pos_tolerance:
        desired_yaw = math.atan2(dy, dx)
        yaw_error = wrap_angle(desired_yaw - yaw)
        wz = _clamp(yaw_gain * yaw_error, max_wz)
        if abs(yaw_error) > align_tolerance:
            return BodyWalkPlan(
                command=(0.0, 0.0, wz, height),
                distance=distance,
                desired_yaw=desired_yaw,
                yaw_error=yaw_error,
                mode="turn_to_path",
                pose_ready=False,
            )

        vx = max(min_vx, min(max_vx, distance_gain * distance))
        return BodyWalkPlan(
            command=(vx, 0.0, wz, height),
            distance=distance,
            desired_yaw=desired_yaw,
            yaw_error=yaw_error,
            mode="forward",
            pose_ready=False,
        )

    final_yaw_error = wrap_angle(target_yaw - yaw)
    if require_yaw and abs(final_yaw_error) > yaw_tolerance:
        return BodyWalkPlan(
            command=(0.0, 0.0, _clamp(yaw_gain * final_yaw_error, max_wz), height),
            distance=distance,
            desired_yaw=target_yaw,
            yaw_error=final_yaw_error,
            mode="turn_at_goal",
            pose_ready=False,
        )

    return BodyWalkPlan(
        command=(0.0, 0.0, 0.0, height),
        distance=distance,
        desired_yaw=target_yaw,
        yaw_error=final_yaw_error,
        mode="arrived",
        pose_ready=True,
    )
