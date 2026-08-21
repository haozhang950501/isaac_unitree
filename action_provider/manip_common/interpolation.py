# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""Trajectory interpolation helpers for smooth end-effector motion.

The state machine plans discrete Cartesian way-points (approach above the
handle, descend onto it, lift up).  To avoid step changes in the IK target -
which would make the arm jerk and disturb the balancing whole-body policy - we
interpolate between way-points over a fixed duration with an ease-in/ease-out
profile.  Orientation targets are interpolated with spherical linear
interpolation (slerp) **per segment**, so a vertical descend can hold a grasp
quaternion instead of spinning the wrist on the way down.
"""
from __future__ import annotations

import torch


def ease_in_out(s: float) -> float:
    """Smoothstep easing on the normalized parameter ``s`` in [0, 1]."""
    s = max(0.0, min(1.0, s))
    return s * s * (3.0 - 2.0 * s)


_JOINT_INTERPOLATION_METHODS = {
    "segment_smoothstep",
    "monotone_cubic_hermite",
}


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
        before = secants[index - 1]
        after = secants[index]
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


def scale_segment_times(
    durations, scale: float, min_time: float = 0.0
) -> list[float]:
    """Replay the same path faster by dividing every segment time by ``scale``.

    Uniform time scaling leaves the joint-space curve untouched and multiplies
    every velocity by ``scale``, so a waypoint shape that was validated at the
    authored speed stays valid.  ``min_time`` floors each segment so a large
    scale cannot collapse one into a step change.
    """
    factor = max(1e-3, float(scale))
    return [max(float(min_time), float(d) / factor) for d in durations]


def lerp(a: torch.Tensor, b: torch.Tensor, s: float) -> torch.Tensor:
    """Linear interpolation between tensors ``a`` and ``b``."""
    return a + (b - a) * s


def slerp(q0: torch.Tensor, q1: torch.Tensor, s: float) -> torch.Tensor:
    """Spherical linear interpolation between two quaternions (w, x, y, z).

    Both inputs and the output have shape [N, 4].
    """
    q0 = q0 / torch.norm(q0, dim=-1, keepdim=True)
    q1 = q1 / torch.norm(q1, dim=-1, keepdim=True)
    dot = (q0 * q1).sum(dim=-1, keepdim=True)
    # take the shorter arc
    q1 = torch.where(dot < 0.0, -q1, q1)
    dot = torch.abs(dot)
    # nearly parallel -> fall back to normalized lerp
    out = torch.empty_like(q0)
    close = (dot > 0.9995).squeeze(-1)
    if close.any():
        out[close] = lerp(q0[close], q1[close], s)
        out[close] = out[close] / torch.norm(out[close], dim=-1, keepdim=True)
    far = ~close
    if far.any():
        theta = torch.acos(torch.clamp(dot[far], -1.0, 1.0))
        sin_theta = torch.sin(theta)
        w0 = torch.sin((1.0 - s) * theta) / sin_theta
        w1 = torch.sin(s * theta) / sin_theta
        out[far] = w0 * q0[far] + w1 * q1[far]
        out[far] = out[far] / torch.norm(out[far], dim=-1, keepdim=True)
    return out


class CartesianInterpolator:
    """Time-parameterized interpolation along a sequence of way-points.

    A single instance tracks one motion.  Call :meth:`reset` (two poses) or
    :meth:`reset_path` (a corner-turning path) when a new motion starts and
    :meth:`step` every control cycle to advance the target.

    Orientation is interpolated **per segment**.  Passing only ``start_quat``
    and ``goal_quat`` slerps on the first leg and then holds ``goal_quat``
    (the usual approach-then-descend pattern).
    """

    def __init__(self, device: str):
        self.device = device
        self.points: list[torch.Tensor] = []
        self.quats: list[torch.Tensor] = []
        self.bounds: list[float] = []      # cumulative segment end times
        self.duration = 1.0
        self.elapsed = 0.0
        self._use_quat = False

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
        """Interpolate through ``points``, spending ``durations[i]`` on each leg.

        Turning a corner rather than cutting straight to the goal is what keeps the
        gripper out of the furniture: a straight line from the arm's resting pose to
        a way-point above a bench passes through the side of that bench.

        Args:
            points: ``len(durations) + 1`` positions, each shaped [N, 3].
            durations: seconds to spend on each leg.
            start_quat: orientation at the start, or ``None`` for position only.
            goal_quat: orientation at the end, or ``None`` for position only.
            quats: optional per-waypoint orientations (length ``len(points)``).
                Overrides ``start_quat`` / ``goal_quat`` when given.
        """
        if len(points) != len(durations) + 1:
            raise ValueError(
                f"[CartesianInterpolator] {len(points)} points needs "
                f"{len(points) - 1} durations, got {len(durations)}"
            )
        self.points = [p.clone() for p in points]
        self.bounds = []
        total = 0.0
        for d in durations:
            total += max(1e-3, float(d))
            self.bounds.append(total)
        self.duration = total
        self.elapsed = 0.0

        if quats is not None:
            if len(quats) != len(points):
                raise ValueError(
                    f"[CartesianInterpolator] {len(points)} points needs "
                    f"{len(points)} quats, got {len(quats)}"
                )
            self.quats = [q.clone() for q in quats]
            self._use_quat = True
        elif start_quat is not None and goal_quat is not None:
            # first leg rotates to the goal, remaining legs hold it
            n = len(points)
            self.quats = [start_quat.clone()] + [goal_quat.clone() for _ in range(n - 1)]
            self._use_quat = True
        else:
            self.quats = []
            self._use_quat = False

    @property
    def has_path(self) -> bool:
        return len(self.bounds) > 0 and len(self.points) >= 2

    @property
    def finished(self) -> bool:
        if not self.has_path:
            return True
        return self.elapsed >= self.duration

    def step(self, dt: float):
        """Advance by ``dt`` seconds and return the interpolated (pos, quat).

        ``quat`` is ``None`` when the motion carries no orientation target.
        An empty path returns ``(None, None)`` instead of indexing ``bounds[0]``.
        """
        if not self.has_path:
            return None, None
        self.elapsed = min(self.duration, self.elapsed + dt)
        # locate the active leg, then ease within it so each corner is approached
        # and left smoothly
        i = 0
        while i < len(self.bounds) - 1 and self.elapsed > self.bounds[i]:
            i += 1
        leg_start = 0.0 if i == 0 else self.bounds[i - 1]
        leg_len = max(1e-6, self.bounds[i] - leg_start)
        s = ease_in_out((self.elapsed - leg_start) / leg_len)
        pos = lerp(self.points[i], self.points[i + 1], s)
        quat = None
        if self._use_quat and i + 1 < len(self.quats):
            quat = slerp(self.quats[i], self.quats[i + 1], s)
        elif self._use_quat and self.quats:
            quat = self.quats[-1]
        return pos, quat


class JointSpaceInterpolator:
    """Time-parameterized joint-space interpolation along a sequence of q.

    ``segment_smoothstep`` preserves the legacy behavior: every leg eases to a
    complete stop at its next waypoint. ``monotone_cubic_hermite`` preserves
    the waypoint q values but carries a continuous, shape-preserving velocity
    through interior points; only the path endpoints stop.
    """

    def __init__(self, device: str):
        self.device = device
        self.points: list[torch.Tensor] = []
        self.bounds: list[float] = []
        self.duration = 1.0
        self.elapsed = 0.0
        self.method = "segment_smoothstep"
        self.slopes: list[torch.Tensor] = []

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
            raise ValueError(
                f"[JointSpaceInterpolator] {len(qs)} waypoints needs "
                f"{len(qs) - 1} durations, got {len(durations)}"
            )
        if method not in _JOINT_INTERPOLATION_METHODS:
            raise ValueError(
                f"[JointSpaceInterpolator] unsupported interpolation method {method!r}"
            )
        self.points = [q.clone() for q in qs]
        self.bounds = []
        total = 0.0
        for d in durations:
            total += max(1e-3, float(d))
            self.bounds.append(total)
        self.duration = total
        self.elapsed = 0.0
        self.method = method
        self.slopes = (
            _monotone_cubic_slopes(self.points, durations)
            if method == "monotone_cubic_hermite"
            else []
        )

    def clear(self):
        self.points = []
        self.bounds = []
        self.duration = 1.0
        self.elapsed = 0.0
        self.method = "segment_smoothstep"
        self.slopes = []

    @property
    def has_path(self) -> bool:
        return len(self.bounds) > 0 and len(self.points) >= 2

    @property
    def finished(self) -> bool:
        if not self.has_path:
            return True
        return self.elapsed >= self.duration

    @property
    def progress(self) -> float:
        if not self.has_path or self.duration <= 1e-9:
            return 1.0
        return min(1.0, self.elapsed / self.duration)

    def step(self, dt: float) -> torch.Tensor | None:
        """Advance by ``dt`` seconds and return the interpolated joint vector."""
        if not self.has_path:
            return None
        self.elapsed = min(self.duration, self.elapsed + dt)
        i = 0
        while i < len(self.bounds) - 1 and self.elapsed > self.bounds[i]:
            i += 1
        leg_start = 0.0 if i == 0 else self.bounds[i - 1]
        leg_len = max(1e-6, self.bounds[i] - leg_start)
        phase = max(0.0, min(1.0, (self.elapsed - leg_start) / leg_len))
        if self.method == "monotone_cubic_hermite":
            s2 = phase * phase
            s3 = s2 * phase
            h00 = 2.0 * s3 - 3.0 * s2 + 1.0
            h10 = s3 - 2.0 * s2 + phase
            h01 = -2.0 * s3 + 3.0 * s2
            h11 = s3 - s2
            return (
                h00 * self.points[i]
                + h10 * leg_len * self.slopes[i]
                + h01 * self.points[i + 1]
                + h11 * leg_len * self.slopes[i + 1]
            )
        return lerp(self.points[i], self.points[i + 1], ease_in_out(phase))
