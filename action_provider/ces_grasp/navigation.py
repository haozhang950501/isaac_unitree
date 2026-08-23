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
* ``turn``    -- keep reversing while turning onto the leg heading (an arc).
* ``forward`` -- walk straight forwards along the leg heading (positive vx).

A pure yaw command (``vx = vy = 0``) is not enough to make the policy step, so
``turn`` rides on the reverse command that is already known to walk and sweeps
the heading around an arc of radius ``turn_vx / wz``.  ``build_carry_route``
stops the reverse leg one radius short so the arc ends back on the approach
line.

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

    ``turn_vx`` is the reverse magnitude a ``turn`` leg keeps sending to hold
    the gait alive.  If the robot still travels ``turn_max_drift`` without
    finishing the turn, the leg flips that translation and retries at ``wz_max``
    to walk the drift back off.

    ``lateral_arrive`` / ``yaw_arrive`` are how square to the goal a
    ``lateral_first`` leg wants to end up.  They are **reported, not enforced**:
    the stop line always wins, so missing them shows up as ``on_target=False``
    for the caller to log rather than as extra driving next to the table.
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
    turn_vx: float = 0.0
    wz_max: float | None = None
    turn_max_drift: float = 0.0
    lateral_arrive: float = 0.0
    yaw_arrive: float = 0.0


@dataclass(frozen=True)
class WalkLeg:
    """One primitive: hold ``yaw`` and drive ``kind`` until ``target_xy``.

    ``stop_margin`` overrides the gait default; use a small one where stopping
    late is harmless and a large one where overshooting hits something.
    ``settle_before`` overrides the zero-command pause taken before this leg;
    set it to zero where breaking the gait would make the next primitive start
    from a standstill.

    ``lateral_first`` makes the leg aim at the whole ``target_xy`` rather than
    only at the travel axis: it strafes onto the axis while walking in, and its
    stop line is final -- once reached the leg stops for good and never drives
    forward again.  Set it where something stands just past the goal.
    """

    name: str
    kind: str  # "reverse" | "turn" | "forward"
    yaw: float
    target_xy: tuple[float, float] | None = None
    stop_margin: float | None = None
    settle_before: float | None = None
    lateral_first: bool = False


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
    # False once a lateral_first leg has stopped without getting square to its
    # goal.  Purely informational: the stop line is not negotiable.
    on_target: bool = True


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
    turn_lead: float = 0.0,
) -> list[WalkLeg]:
    """Back out of the machine, turn toward the table, then walk in.

    The robot reverses along its pick heading until it reaches the point where
    the place-stand approach line crosses that backing line -- i.e. until it is
    aligned with the gray tote, so the turn leaves the place stand straight
    ahead.  For the CES layout (pick yaw = pi, place yaw = pi/2) the turn is a
    right turn and the final leg walks toward world +Y.

    ``turn_lead`` stops the reverse leg one arc radius short of that crossing.
    The turn keeps reversing, so it sweeps forward along the approach line by
    exactly one radius and lands back on it.

    ``backoff_trim`` shortens the reverse leg further.  Backing up carries the
    largest steps, so under-shooting on purpose is safer: the last leg strafes
    the residual sideways error away.
    """
    back_dir = (-math.cos(pick_yaw), -math.sin(pick_yaw))
    fwd_dir = (math.cos(place_yaw), math.sin(place_yaw))
    a = _line_crossing(pick_xy, back_dir, place_xy, fwd_dir)
    if a is None or a <= 0.0:
        raise ValueError("place stand is not reachable by backing out of the pick stand")
    a = max(0.0, a - max(0.0, backoff_trim) - max(0.0, turn_lead))
    corner = (pick_xy[0] + a * back_dir[0], pick_xy[1] + a * back_dir[1])
    # Backing out too far is free -- the last leg strafes that error away -- but
    # stopping short of the table leaves the arm over-reaching, so the approach
    # leg can carry its own tighter margin.
    return [
        WalkLeg("backoff", "reverse", pick_yaw, corner),
        # No pause before the arc: coasting to a standstill first is exactly the
        # case the policy will not start turning from.
        WalkLeg("turn_to_table", "turn", place_yaw, settle_before=0.0),
        # No pause after the arc either: zeroing vx kills the gait, and this
        # policy will not start a forward walk from a standstill.  Hand
        # body-frame +vx (W) straight over; at place yaw = π/2 that is world +Y.
        WalkLeg(
            "approach_place", "forward", place_yaw, place_xy,
            stop_margin=place_stop_margin, lateral_first=True,
            settle_before=0.0,
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
        self._braked = False
        self._leg_start_xy: tuple[float, float] | None = None

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
        if self._leg_start_xy is None:
            self._leg_start_xy = (x, y)
        for _ in range(len(self.legs) + 1):
            leg = self.legs[self._i]
            remaining, lateral, yaw_error = leg_error(leg, x, y, yaw)
            if not self._leg_done(leg, remaining, lateral, yaw_error):
                if self._settling > 0.0:
                    return self._zero("settle", remaining, lateral, yaw_error)
                return self._drive(leg, x, y, remaining, lateral, yaw_error)
            # Leg reached: coast to a stop before starting the next primitive.
            if self._settling > 0.0:
                return self._zero("settle", remaining, lateral, yaw_error)
            if self._i + 1 >= len(self.legs):
                self._route_done = True
                return self._zero("arrived", remaining, lateral, yaw_error, done=True)
            self._i += 1
            nxt = self.legs[self._i]
            settle = (
                self.gait.leg_settle
                if nxt.settle_before is None
                else nxt.settle_before
            )
            self._settling = settle
            self._leg_start_xy = (x, y)
            self._fix_yaw = False
            self._fix_lateral = False
            self._braked = False
        self._route_done = True
        return self._zero("arrived", 0.0, 0.0, 0.0, done=True)

    # ------------------------------------------------------------------ impl --
    def _stop_margin(self, leg: WalkLeg) -> float:
        return self.gait.stop_margin if leg.stop_margin is None else leg.stop_margin

    def _on_target(self, lateral: float, yaw_error: float) -> bool:
        gait = self.gait
        square = gait.lateral_arrive <= 0.0 or abs(lateral) <= gait.lateral_arrive
        aimed = gait.yaw_arrive <= 0.0 or abs(yaw_error) <= gait.yaw_arrive
        return square and aimed

    def _leg_done(
        self, leg: WalkLeg, remaining: float, lateral: float, yaw_error: float
    ) -> bool:
        if leg.kind == "turn":
            return abs(yaw_error) <= self.gait.yaw_tol
        if leg.lateral_first:
            # The table sits ~0.12 m past this goal and the gait cannot be
            # slowed (anything under the dead band simply does not walk), so a
            # forward command at 0.45 m/s always coasts ~0.25 m.  That makes the
            # stop line the one thing this leg may not negotiate: reach it and
            # the leg is over, whatever is left sideways or in heading.  Holding
            # out for a better pose here is what drove the robot into the table.
            if self._braked or remaining <= self._stop_margin(leg):
                self._braked = True
                return True
            return False
        return remaining <= self._stop_margin(leg)

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
            on_target=self._on_target(lateral, yaw_error),
        )

    def _hysteresis(self, active: bool, error: float, enter: float) -> bool:
        """Dead band with hysteresis so a fixed-magnitude fix cannot chatter."""
        return abs(error) > (0.4 * enter if active else enter)

    def _leg_drift(self, x: float, y: float) -> float:
        if self._leg_start_xy is None:
            return 0.0
        return math.hypot(x - self._leg_start_xy[0], y - self._leg_start_xy[1])

    def _drive(
        self,
        leg: WalkLeg,
        x: float,
        y: float,
        remaining: float,
        lateral: float,
        yaw_error: float,
    ) -> WalkStep:
        gait = self.gait
        if leg.kind == "turn":
            if (
                gait.turn_max_drift > 0.0
                and self._leg_drift(x, y) > gait.turn_max_drift
            ):
                # Travelled a whole arc without turning.  Zeroing vx here would
                # only freeze the robot -- vx is the one axis that starts a gait
                # -- so drive the arc the other way instead: it walks the drift
                # back off while trying to yaw at the higher rate.
                wz = _signed(gait.wz_max or gait.wz, yaw_error)
                return self._make(
                    leg, (gait.turn_vx, 0.0, wz), "turn_pinned",
                    remaining, lateral, yaw_error,
                )
            return self._make(
                leg, (-gait.turn_vx, 0.0, _signed(gait.wz, yaw_error)), "turn",
                remaining, lateral, yaw_error,
            )

        if abs(yaw_error) > gait.realign_yaw:
            # Too crooked to keep making progress: give up the travel axis and
            # arc back onto the leg heading.  The translation stays because the
            # policy will not yaw at all without one.  On a lateral_first leg it
            # always arcs backwards: whatever the goal is parked against is
            # straight ahead, so straightening up must not push into it.
            self._fix_yaw = True
            sign = 1.0 if leg.kind == "forward" and not leg.lateral_first else -1.0
            return self._make(
                leg,
                (sign * gait.turn_vx, 0.0, _signed(gait.wz, yaw_error)),
                f"{leg.kind}_realign",
                remaining, lateral, yaw_error,
            )

        vx = -gait.vx if leg.kind == "reverse" else gait.vx
        self._fix_yaw = self._hysteresis(self._fix_yaw, yaw_error, gait.align_yaw)
        self._fix_lateral = self._hysteresis(
            self._fix_lateral, lateral, gait.lateral_tol
        )
        wz = _signed(gait.wz, yaw_error) if self._fix_yaw else 0.0
        vy = _signed(gait.vy, lateral) if self._fix_lateral else 0.0
        mode = leg.kind
        if leg.lateral_first:
            if self._braked:
                # Past the stop line: settle here and stay settled.  Reversing
                # to try again only keeps the robot moving next to the table,
                # and the coast on the way out is what hit it last time.
                return self._zero(
                    f"{leg.kind}_stopped", remaining, lateral, yaw_error
                )
            # ``vy`` alone does not move the robot, it only steers a walk that
            # ``vx`` is already driving, so the sideways fix is a diagonal walk
            # on the way in -- there is no standing still to strafe.  All of the
            # correction has to happen in the run-in, before the stop line.
            if self._fix_lateral:
                mode = f"{leg.kind}_align"
        return self._make(leg, (vx, vy, wz), mode, remaining, lateral, yaw_error)

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
