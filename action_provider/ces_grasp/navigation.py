# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""Leg-based body-frame walk planning for the CES carry (HOLD -> place stand).

The whole-body policy ignores small velocity commands: tapping keyboard WASD
does not move the robot, only a held key ramps the command high enough to make
it step.  So this planner never scales the command with the remaining distance.
Each leg drives a single motion primitive at a fixed magnitude and releases it
inside a dead band -- the same shape as holding a key and then letting go.

The route is a list of ``WalkLeg`` primitives in the world frame:

* ``reverse`` -- walk straight backwards along the leg heading (negative vx).
* ``turn``    -- turn in place to the leg heading.
* ``forward`` -- walk straight forwards along the leg heading (positive vx).

Progress is measured by projecting the goal onto the leg travel axis, so
sideways drift can never make a leg spin forever.  Heading and sideways drift
are corrected with fixed-magnitude ``wz`` / ``vy`` commands behind a hysteresis
dead band, never with a vanishing proportional term.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


def wrap_angle(angle: float) -> float:
    """Wrap an angle to [-pi, pi]."""
    return math.atan2(math.sin(angle), math.cos(angle))


@dataclass(frozen=True)
class WalkGait:
    """Command magnitudes and dead bands for one carry route.

    ``vx`` / ``vy`` / ``wz`` are the only translation / rotation magnitudes ever
    sent; anything smaller is below the policy's stepping threshold.
    ``stop_margin`` compensates the distance still travelled while the command
    ramps back to zero.
    """

    vx: float
    vy: float
    wz: float
    stop_margin: float
    yaw_tol: float
    lateral_tol: float
    align_yaw: float
    realign_yaw: float
    leg_settle: float
    height: float = 0.8


@dataclass(frozen=True)
class WalkLeg:
    """One primitive: hold ``yaw`` and drive ``kind`` until ``target_xy``.

    ``stop_margin`` overrides the gait default; use a small one where stopping
    late is harmless and a large one where overshooting hits something.
    """

    name: str
    kind: str  # "reverse" | "turn" | "forward"
    yaw: float
    target_xy: tuple[float, float] | None = None
    stop_margin: float | None = None


@dataclass(frozen=True)
class WalkStep:
    """One body-frame command plus the geometry that produced it."""

    command: tuple[float, float, float, float]
    leg_index: int
    leg_name: str
    mode: str
    remaining: float
    lateral: float
    yaw_error: float
    route_done: bool


def _signed(magnitude: float, error: float) -> float:
    return magnitude if error > 0.0 else -magnitude


def leg_error(
    leg: WalkLeg, x: float, y: float, yaw: float
) -> tuple[float, float, float]:
    """Return (remaining along travel axis, sideways offset, heading error).

    ``remaining`` is signed: negative means the robot already overshot.
    ``lateral`` is expressed in the body-left direction so it maps straight onto
    the sign of a ``vy`` command.
    """
    yaw_error = wrap_angle(leg.yaw - yaw)
    if leg.kind == "turn" or leg.target_xy is None:
        return 0.0, 0.0, yaw_error

    sign = -1.0 if leg.kind == "reverse" else 1.0
    axis = (sign * math.cos(leg.yaw), sign * math.sin(leg.yaw))
    dx = leg.target_xy[0] - x
    dy = leg.target_xy[1] - y
    remaining = dx * axis[0] + dy * axis[1]
    lateral = dx * (-math.sin(yaw)) + dy * math.cos(yaw)
    return remaining, lateral, yaw_error


def _line_crossing(
    p: tuple[float, float],
    u: tuple[float, float],
    q: tuple[float, float],
    v: tuple[float, float],
) -> float | None:
    """Distance ``a`` along ``u`` from ``p`` where ``p + a*u`` meets ``q + b*v``."""
    det = v[0] * u[1] - u[0] * v[1]
    if abs(det) < 1e-6:
        return None
    dx, dy = q[0] - p[0], q[1] - p[1]
    return (v[0] * dy - dx * v[1]) / det


def build_carry_route(
    *,
    pick_xy: tuple[float, float],
    pick_yaw: float,
    place_xy: tuple[float, float],
    place_yaw: float,
    place_stop_margin: float | None = None,
    backoff_trim: float = 0.0,
) -> list[WalkLeg]:
    """Back out of the machine, turn toward the table, then walk in.

    The robot reverses along its pick heading until it reaches the point where
    the place-stand approach line crosses that backing line -- i.e. until it is
    aligned with the gray tote, so the in-place turn leaves the place stand
    straight ahead.  For the CES layout (pick yaw = pi, place yaw = pi/2) the
    turn is a right turn and the final leg walks toward world +Y.

    ``backoff_trim`` shortens the reverse leg.  Backing up carries the largest
    steps, so under-shooting on purpose is safer: the last leg strafes the
    residual sideways error away.
    """
    back_dir = (-math.cos(pick_yaw), -math.sin(pick_yaw))
    fwd_dir = (math.cos(place_yaw), math.sin(place_yaw))
    a = _line_crossing(pick_xy, back_dir, place_xy, fwd_dir)
    if a is None or a <= 0.0:
        raise ValueError("place stand is not reachable by backing out of the pick stand")
    a = max(0.0, a - max(0.0, backoff_trim))
    corner = (pick_xy[0] + a * back_dir[0], pick_xy[1] + a * back_dir[1])
    # Backing out too far is free -- the last leg strafes that error away -- but
    # stopping short of the table leaves the arm over-reaching, so the approach
    # leg can carry its own tighter margin.
    return [
        WalkLeg("backoff", "reverse", pick_yaw, corner),
        WalkLeg("turn_to_table", "turn", place_yaw),
        WalkLeg(
            "approach_place", "forward", place_yaw, place_xy,
            stop_margin=place_stop_margin,
        ),
    ]


class LegWalkPlanner:
    """Walk a fixed route of primitives, one at a time, with zero-command pauses."""

    def __init__(self, legs: list[WalkLeg], gait: WalkGait):
        if not legs:
            raise ValueError("walk route needs at least one leg")
        self.legs = list(legs)
        self.gait = gait
        self.reset()

    def reset(self):
        self._i = 0
        self._settling = 0.0
        self._route_done = False
        self._fix_yaw = False
        self._fix_lateral = False

    @property
    def leg_index(self) -> int:
        return self._i

    @property
    def leg(self) -> WalkLeg:
        return self.legs[self._i]

    @property
    def route_done(self) -> bool:
        return self._route_done

    def step(self, x: float, y: float, yaw: float, dt: float = 0.0) -> WalkStep:
        """Command for the current pose; advances legs as they complete."""
        if self._route_done:
            return self._zero("arrived", *leg_error(self.leg, x, y, yaw), done=True)

        self._settling = max(0.0, self._settling - max(dt, 0.0))
        for _ in range(len(self.legs) + 1):
            leg = self.legs[self._i]
            remaining, lateral, yaw_error = leg_error(leg, x, y, yaw)
            if not self._leg_done(leg, remaining, yaw_error):
                if self._settling > 0.0:
                    return self._zero("settle", remaining, lateral, yaw_error)
                return self._drive(leg, remaining, lateral, yaw_error)
            # Leg reached: coast to a stop before starting the next primitive.
            if self._settling > 0.0:
                return self._zero("settle", remaining, lateral, yaw_error)
            if self._i + 1 >= len(self.legs):
                self._route_done = True
                return self._zero("arrived", remaining, lateral, yaw_error, done=True)
            self._i += 1
            self._settling = self.gait.leg_settle
            self._fix_yaw = False
            self._fix_lateral = False
        self._route_done = True
        return self._zero("arrived", 0.0, 0.0, 0.0, done=True)

    # ------------------------------------------------------------------ impl --
    def _leg_done(self, leg: WalkLeg, remaining: float, yaw_error: float) -> bool:
        if leg.kind == "turn":
            return abs(yaw_error) <= self.gait.yaw_tol
        margin = self.gait.stop_margin if leg.stop_margin is None else leg.stop_margin
        return remaining <= margin

    def _zero(
        self,
        mode: str,
        remaining: float,
        lateral: float,
        yaw_error: float,
        done: bool = False,
    ) -> WalkStep:
        return WalkStep(
            command=(0.0, 0.0, 0.0, self.gait.height),
            leg_index=self._i,
            leg_name=self.legs[self._i].name,
            mode=mode,
            remaining=remaining,
            lateral=lateral,
            yaw_error=yaw_error,
            route_done=done,
        )

    def _hysteresis(self, active: bool, error: float, enter: float) -> bool:
        """Dead band with hysteresis so a fixed-magnitude fix cannot chatter."""
        return abs(error) > (0.4 * enter if active else enter)

    def _drive(
        self, leg: WalkLeg, remaining: float, lateral: float, yaw_error: float
    ) -> WalkStep:
        gait = self.gait
        if leg.kind == "turn":
            return self._make(
                leg, (0.0, 0.0, _signed(gait.wz, yaw_error)), "turn",
                remaining, lateral, yaw_error,
            )

        if abs(yaw_error) > gait.realign_yaw:
            # Too crooked to keep translating: turn back onto the leg heading.
            self._fix_yaw = True
            return self._make(
                leg, (0.0, 0.0, _signed(gait.wz, yaw_error)), f"{leg.kind}_realign",
                remaining, lateral, yaw_error,
            )

        vx = -gait.vx if leg.kind == "reverse" else gait.vx
        self._fix_yaw = self._hysteresis(self._fix_yaw, yaw_error, gait.align_yaw)
        self._fix_lateral = self._hysteresis(
            self._fix_lateral, lateral, gait.lateral_tol
        )
        wz = _signed(gait.wz, yaw_error) if self._fix_yaw else 0.0
        vy = _signed(gait.vy, lateral) if self._fix_lateral else 0.0
        return self._make(leg, (vx, vy, wz), leg.kind, remaining, lateral, yaw_error)

    def _make(
        self,
        leg: WalkLeg,
        cmd: tuple[float, float, float],
        mode: str,
        remaining: float,
        lateral: float,
        yaw_error: float,
    ) -> WalkStep:
        return WalkStep(
            command=(cmd[0], cmd[1], cmd[2], self.gait.height),
            leg_index=self._i,
            leg_name=leg.name,
            mode=mode,
            remaining=remaining,
            lateral=lateral,
            yaw_error=yaw_error,
            route_done=False,
        )
