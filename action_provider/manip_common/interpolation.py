# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""Trajectory interpolation helpers for smooth end-effector motion.

The state machine plans discrete Cartesian way-points (approach above the
handle, descend onto it, lift up).  To avoid step changes in the IK target -
which would make the arm jerk and disturb the balancing whole-body policy - we
interpolate between way-points over a fixed duration with an ease-in/ease-out
profile.  Orientation targets are interpolated with spherical linear
interpolation (slerp).
"""
from __future__ import annotations

import torch


def ease_in_out(s: float) -> float:
    """Smoothstep easing on the normalized parameter ``s`` in [0, 1]."""
    s = max(0.0, min(1.0, s))
    return s * s * (3.0 - 2.0 * s)


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
    far = ~close
    if far.any():
        theta = torch.acos(torch.clamp(dot[far], -1.0, 1.0))
        sin_theta = torch.sin(theta)
        w0 = torch.sin((1.0 - s) * theta) / sin_theta
        w1 = torch.sin(s * theta) / sin_theta
        out[far] = w0 * q0[far] + w1 * q1[far]
    out = out / torch.norm(out, dim=-1, keepdim=True)
    return out


class CartesianInterpolator:
    """Time-parameterized interpolation along a sequence of way-points.

    A single instance tracks one motion.  Call :meth:`reset` (two poses) or
    :meth:`reset_path` (a corner-turning path) when a new motion starts and
    :meth:`step` every control cycle to advance the target.  Orientation always
    interpolates once across the whole motion, not per segment.
    """

    def __init__(self, device: str):
        self.device = device
        self.points: list[torch.Tensor] = []
        self.bounds: list[float] = []      # cumulative segment end times
        self.start_quat: torch.Tensor | None = None
        self.goal_quat: torch.Tensor | None = None
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
        self._use_quat = start_quat is not None and goal_quat is not None
        self.start_quat = start_quat.clone() if self._use_quat else None
        self.goal_quat = goal_quat.clone() if self._use_quat else None

    @property
    def finished(self) -> bool:
        return self.elapsed >= self.duration

    def step(self, dt: float):
        """Advance by ``dt`` seconds and return the interpolated (pos, quat).

        ``quat`` is ``None`` when the motion carries no orientation target.
        """
        self.elapsed = min(self.duration, self.elapsed + dt)
        # locate the active leg, then ease within it so each corner is approached
        # and left smoothly
        i = 0
        while i < len(self.bounds) - 1 and self.elapsed > self.bounds[i]:
            i += 1
        leg_start = 0.0 if i == 0 else self.bounds[i - 1]
        leg_len = self.bounds[i] - leg_start
        pos = lerp(
            self.points[i], self.points[i + 1], ease_in_out((self.elapsed - leg_start) / leg_len)
        )
        quat = None
        if self._use_quat:
            quat = slerp(self.start_quat, self.goal_quat, ease_in_out(self.elapsed / self.duration))
        return pos, quat
