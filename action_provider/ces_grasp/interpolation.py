# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""CES trajectory interpolation with PyTorch and Isaac Lab primitives."""
from __future__ import annotations

from bisect import bisect_left

import torch


def ease_in_out(value: float) -> float:
    """Cubic smoothstep on a normalized scalar."""
    value = max(0.0, min(1.0, value))
    return value * value * (3.0 - 2.0 * value)


def scale_segment_times(
    durations, scale: float, min_time: float = 0.0
) -> list[float]:
    """Uniformly time-scale a path without changing its joint-space curve."""
    factor = max(1e-3, float(scale))
    return [max(float(min_time), float(duration) / factor) for duration in durations]


def _bounds(durations: list[float]) -> tuple[list[float], float]:
    bounds: list[float] = []
    total = 0.0
    for duration in durations:
        total += max(1e-3, float(duration))
        bounds.append(total)
    return bounds, total


def _segment(bounds: list[float], elapsed: float) -> tuple[int, float, float]:
    index = min(bisect_left(bounds, elapsed), len(bounds) - 1)
    start = 0.0 if index == 0 else bounds[index - 1]
    length = max(1e-6, bounds[index] - start)
    phase = max(0.0, min(1.0, (elapsed - start) / length))
    return index, length, phase


def _quat_slerp(q0: torch.Tensor, q1: torch.Tensor, phase: float) -> torch.Tensor:
    """Apply Isaac Lab's scalar quaternion slerp to each environment."""
    from isaaclab.utils.math import quat_slerp

    return torch.stack(
        [quat_slerp(a, b.clone(), phase) for a, b in zip(q0, q1)], dim=0
    )


def _monotone_cubic_slopes(
    points: list[torch.Tensor], durations: list[float]
) -> list[torch.Tensor]:
    """Shape-preserving C1 waypoint velocities with resting endpoints."""
    widths = [max(1e-3, float(duration)) for duration in durations]
    secants = [
        (points[index + 1] - points[index]) / widths[index]
        for index in range(len(widths))
    ]
    slopes = [torch.zeros_like(point) for point in points]
    for index in range(1, len(points) - 1):
        before, after = secants[index - 1], secants[index]
        same_direction = before * after > 0.0
        weight_before = 2.0 * widths[index] + widths[index - 1]
        weight_after = widths[index] + 2.0 * widths[index - 1]
        slope = torch.zeros_like(points[index])
        slope[same_direction] = (weight_before + weight_after) / (
            weight_before / before[same_direction]
            + weight_after / after[same_direction]
        )
        slopes[index] = slope
    return slopes


class CartesianInterpolator:
    """Time-parameterized Cartesian path with per-segment quaternion slerp."""

    def __init__(self, device: str):
        self.device = device
        self.points: list[torch.Tensor] = []
        self.quats: list[torch.Tensor] = []
        self.bounds: list[float] = []
        self.duration = 1.0
        self.elapsed = 0.0

    def reset(
        self,
        start_pos: torch.Tensor,
        goal_pos: torch.Tensor,
        duration: float,
        start_quat: torch.Tensor | None = None,
        goal_quat: torch.Tensor | None = None,
    ):
        self.reset_path([start_pos, goal_pos], [duration], start_quat, goal_quat)

    def reset_path(
        self,
        points: list[torch.Tensor],
        durations: list[float],
        start_quat: torch.Tensor | None = None,
        goal_quat: torch.Tensor | None = None,
        quats: list[torch.Tensor] | None = None,
    ):
        if len(points) != len(durations) + 1:
            raise ValueError(
                f"{len(points)} Cartesian points require {len(points) - 1} durations"
            )
        self.points = [point.clone() for point in points]
        self.bounds, self.duration = _bounds(durations)
        self.elapsed = 0.0
        if quats is not None:
            if len(quats) != len(points):
                raise ValueError(f"{len(points)} Cartesian points require {len(points)} quaternions")
            self.quats = [quat.clone() for quat in quats]
        elif start_quat is not None and goal_quat is not None:
            self.quats = [start_quat.clone()] + [
                goal_quat.clone() for _ in range(len(points) - 1)
            ]
        else:
            self.quats = []

    @property
    def has_path(self) -> bool:
        return bool(self.bounds) and len(self.points) >= 2

    @property
    def finished(self) -> bool:
        return not self.has_path or self.elapsed >= self.duration

    def step(self, dt: float):
        if not self.has_path:
            return None, None
        self.elapsed = min(self.duration, self.elapsed + dt)
        index, _length, phase = _segment(self.bounds, self.elapsed)
        phase = ease_in_out(phase)
        pos = torch.lerp(self.points[index], self.points[index + 1], phase)
        quat = (
            _quat_slerp(self.quats[index], self.quats[index + 1], phase)
            if self.quats
            else None
        )
        return pos, quat


class JointSpaceInterpolator:
    """Smoothstep or shape-preserving monotone Hermite joint path."""

    _METHODS = {"segment_smoothstep", "monotone_cubic_hermite"}

    def __init__(self, device: str):
        self.device = device
        self.clear()

    def reset(
        self,
        start_q: torch.Tensor,
        goal_q: torch.Tensor,
        duration: float,
        method: str = "segment_smoothstep",
    ):
        self.reset_path([start_q, goal_q], [duration], method=method)

    def reset_path(
        self,
        qs: list[torch.Tensor],
        durations: list[float],
        method: str = "segment_smoothstep",
    ):
        if len(qs) != len(durations) + 1:
            raise ValueError(f"{len(qs)} joint waypoints require {len(qs) - 1} durations")
        if method not in self._METHODS:
            raise ValueError(f"unsupported CES interpolation method {method!r}")
        self.points = [q.clone() for q in qs]
        self.bounds, self.duration = _bounds(durations)
        self.elapsed = 0.0
        self.method = method
        self.slopes = (
            _monotone_cubic_slopes(self.points, durations)
            if method == "monotone_cubic_hermite"
            else []
        )

    def clear(self):
        self.points: list[torch.Tensor] = []
        self.bounds: list[float] = []
        self.duration = 1.0
        self.elapsed = 0.0
        self.method = "segment_smoothstep"
        self.slopes: list[torch.Tensor] = []

    @property
    def has_path(self) -> bool:
        return bool(self.bounds) and len(self.points) >= 2

    @property
    def finished(self) -> bool:
        return not self.has_path or self.elapsed >= self.duration

    def step(self, dt: float) -> torch.Tensor | None:
        if not self.has_path:
            return None
        self.elapsed = min(self.duration, self.elapsed + dt)
        index, length, phase = _segment(self.bounds, self.elapsed)
        if self.method == "monotone_cubic_hermite":
            phase2, phase3 = phase * phase, phase * phase * phase
            return (
                (2.0 * phase3 - 3.0 * phase2 + 1.0) * self.points[index]
                + (phase3 - 2.0 * phase2 + phase) * length * self.slopes[index]
                + (-2.0 * phase3 + 3.0 * phase2) * self.points[index + 1]
                + (phase3 - phase2) * length * self.slopes[index + 1]
            )
        return torch.lerp(
            self.points[index], self.points[index + 1], ease_in_out(phase)
        )
