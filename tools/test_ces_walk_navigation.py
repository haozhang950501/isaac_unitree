"""CPU-only checks for the CES carry walk route (back out → turn right → walk in).

The whole-body policy does not react to small velocity commands (tapping
keyboard WASD never makes the robot step), so these tests model that dead band
explicitly: a command below the threshold produces no motion at all.

Isaac Sim then showed something stronger: ``vx`` is the only axis that starts a
gait.  A pure yaw command leaves the robot standing, and so does a pure strafe
-- ``cmd_b=(+0.00,+0.30,+0.00)`` held for 18 s moved it exactly nowhere.  So the
model gates the whole command on ``vx``: below the ``vx`` dead band nothing
moves at all, however large ``vy`` or ``wz`` are.  The route must converge
under that, which means every correction has to ride on a live ``vx``.
"""
from __future__ import annotations

import importlib.util
import itertools
import math
from pathlib import Path
import sys
import unittest


MODULE_PATH = Path(__file__).parents[1] / "action_provider" / "ces_grasp" / "navigation.py"
SPEC = importlib.util.spec_from_file_location("ces_walk_navigation", MODULE_PATH)
navigation = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = navigation
SPEC.loader.exec_module(navigation)

# Live CES layout (see action_provider/ces_grasp/constants.py).
PICK_XY = (-3.1870, -1.3302)
PICK_YAW = math.pi
PLACE_XY = (-2.2669, -0.7717)
PLACE_YAW = 0.5 * math.pi
TRAY_XY = (-2.0869, -0.3117)

TURN_VX = 0.45
WZ = 1.20
GAIT = navigation.WalkGait(
    vx=0.45,
    vy=0.40,
    wz=WZ,
    stop_margin=0.15,
    yaw_tol=0.40,
    lateral_tol=0.10,
    align_yaw=0.20,
    realign_yaw=0.60,
    leg_settle=0.5,
    turn_vx=TURN_VX,
    wz_max=1.55,
    turn_max_drift=0.70,
    lateral_arrive=0.10,
    yaw_arrive=0.20,
)
PLACE_STOP_MARGIN = 0.30
# The table edge sits this far in front of the place stand, so the pelvis may
# never end up more than this past it.  Measured from X_B_PLACE and the table
# half-depth; the robot went down on the floor when it did.
TABLE_AHEAD_OF_STAND = 0.12
# Below these magnitudes the policy only sways in place.
POLICY_DEADBAND = (0.30, 0.25, 0.40)
# The reverse leg stops one arc radius short so the turn lands back on the
# place-stand approach line; that already under-shoots, so no extra trim.
BACKOFF_TRIM = 0.0
TURN_LEAD = TURN_VX / WZ


def route(backoff_trim=BACKOFF_TRIM, turn_lead=TURN_LEAD):
    return navigation.build_carry_route(
        pick_xy=PICK_XY,
        pick_yaw=PICK_YAW,
        place_xy=PLACE_XY,
        place_yaw=PLACE_YAW,
        place_stop_margin=PLACE_STOP_MARGIN,
        backoff_trim=backoff_trim,
        turn_lead=turn_lead,
    )


class Walker:
    """Coarse stand-in for the locomotion policy plus the command ramp."""

    def __init__(self, x, y, yaw, dt=0.02, lag=0.25):
        self.x, self.y, self.yaw = x, y, yaw
        self.dt = dt
        self.lag = lag
        self.cmd = [0.0, 0.0, 0.0]
        self.vel = [0.0, 0.0, 0.0]
        self.sent = []

    def drive(self, command):
        """Ramp the command, drop sub-threshold values, integrate the pose."""
        accel = (1.0, 1.0, 2.0)
        for i in range(3):
            delta = command[i] - self.cmd[i]
            limit = accel[i] * self.dt
            self.cmd[i] += max(-limit, min(limit, delta))
        self.sent.append(tuple(command[:3]))
        target = [
            self.cmd[i] if abs(self.cmd[i]) >= POLICY_DEADBAND[i] else 0.0
            for i in range(3)
        ]
        if target[0] == 0.0:
            # No forward/backward command, no gait: vy and wz do nothing on
            # their own, they only steer a walk that vx has already started.
            target = [0.0, 0.0, 0.0]
        blend = min(1.0, self.dt / self.lag)
        for i in range(3):
            self.vel[i] += (target[i] - self.vel[i]) * blend
        self.yaw = navigation.wrap_angle(self.yaw + self.vel[2] * self.dt)
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        self.x += (self.vel[0] * c - self.vel[1] * s) * self.dt
        self.y += (self.vel[0] * s + self.vel[1] * c) * self.dt


class MistrackingWalker(Walker):
    """Walker whose realized yaw / strafe rates differ from what was commanded.

    The policy is not calibrated, so the route has to survive tracking that is
    off: ``lag`` also stands in for how far the robot coasts after the command
    goes back to zero, which is the term that put it into the table.
    """

    def __init__(self, *args, yaw_gain=1.0, lat_gain=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.yaw_gain, self.lat_gain = yaw_gain, lat_gain

    def drive(self, command):
        scaled = list(command)
        scaled[1] *= self.lat_gain
        scaled[2] *= self.yaw_gain
        super().drive(scaled)


def _run_until_settled(planner, walker, max_time=40.0, coast=1.5):
    """Return how far past the place stand the pelvis ever got."""
    fwd = (math.cos(PLACE_YAW), math.sin(PLACE_YAW))

    def past():
        return ((walker.x - PLACE_XY[0]) * fwd[0]
                + (walker.y - PLACE_XY[1]) * fwd[1])

    worst = past()
    for _ in range(int(max_time / walker.dt)):
        step = planner.step(walker.x, walker.y, walker.yaw, walker.dt)
        worst = max(worst, past())
        if step.route_done:
            break
        walker.drive(step.command)
    # Keep integrating after the planner lets go: the robot does not stop dead.
    for _ in range(int(coast / walker.dt)):
        walker.drive((0.0, 0.0, 0.0, GAIT.height))
        worst = max(worst, past())
    return worst


def run_route(start=None, max_time=60.0, gait=GAIT, legs=None, arrive_hold=0.35):
    """Walk the route, then coast for ``arrive_hold`` like the FSM stand check."""
    planner = navigation.LegWalkPlanner(legs or route(), gait)
    x, y, yaw = start or (PICK_XY[0], PICK_XY[1], PICK_YAW)
    walker = Walker(x, y, yaw)
    steps = []
    for _ in range(int(max_time / walker.dt)):
        step = planner.step(walker.x, walker.y, walker.yaw, walker.dt)
        steps.append(step)
        if step.route_done:
            for _ in range(int(arrive_hold / walker.dt)):
                walker.drive(step.command)
            return walker, steps
        walker.drive(step.command)
    raise AssertionError("carry route did not converge")


class TestCarryRoute(unittest.TestCase):
    def test_route_is_backoff_then_right_turn_then_forward(self):
        legs = route(backoff_trim=0.0, turn_lead=0.0)
        self.assertEqual([leg.kind for leg in legs], ["reverse", "turn", "forward"])
        # Back out along world +X (robot faces -X) until aligned with the tote.
        self.assertAlmostEqual(legs[0].target_xy[0], PLACE_XY[0], places=3)
        self.assertAlmostEqual(legs[0].target_xy[1], PICK_XY[1], places=3)
        self.assertAlmostEqual(legs[0].target_xy[0], TRAY_XY[0] - 0.18, places=3)
        # pi -> pi/2 is a right turn (negative wz).
        self.assertLess(navigation.wrap_angle(legs[1].yaw - legs[0].yaw), 0.0)
        self.assertEqual(legs[2].target_xy, PLACE_XY)

    def test_backoff_trim_stops_short_of_the_crossing(self):
        full = route(backoff_trim=0.0, turn_lead=0.0)[0].target_xy
        trimmed = route(backoff_trim=0.20, turn_lead=0.0)[0].target_xy
        # Backing runs along world +X, so trimming lands at a smaller x.
        self.assertAlmostEqual(trimmed[0], full[0] - 0.20, places=6)
        self.assertAlmostEqual(trimmed[1], full[1], places=6)
        self.assertLess(
            math.hypot(trimmed[0] - PICK_XY[0], trimmed[1] - PICK_XY[1]),
            math.hypot(full[0] - PICK_XY[0], full[1] - PICK_XY[1]),
        )

    def test_backoff_stops_short_by_the_turn_lead(self):
        """The reverse leg ends one arc radius before the crossing."""
        crossing = route(backoff_trim=0.0, turn_lead=0.0)[0].target_xy
        corner = route()[0].target_xy
        self.assertAlmostEqual(corner[0], crossing[0] - TURN_LEAD, places=6)
        self.assertAlmostEqual(corner[1], crossing[1], places=6)

    def test_turn_arc_lands_back_on_the_approach_line(self):
        """Reversing through the 90 deg turn sweeps exactly one radius each way."""
        corner = route()[0].target_xy
        radius = TURN_VX / WZ
        # Unicycle with v<0, w<0 from yaw=pi to yaw=pi/2.
        end = (corner[0] + radius, corner[1] - radius)
        self.assertAlmostEqual(end[0], PLACE_XY[0], places=6)
        # The arc leaves the robot further from the table, so the last leg has
        # room for its stop margin instead of arriving on top of the table.
        self.assertLess(end[1], corner[1])
        self.assertGreater(PLACE_XY[1] - end[1], 0.5)

    def test_first_command_backs_up_before_any_turn(self):
        planner = navigation.LegWalkPlanner(route(), GAIT)
        step = planner.step(PICK_XY[0], PICK_XY[1], PICK_YAW, 0.02)
        self.assertEqual(step.leg_name, "backoff")
        self.assertLess(step.command[0], 0.0)
        self.assertEqual(step.command[2], 0.0)

    def test_turn_leg_is_clockwise_and_keeps_reversing(self):
        planner = navigation.LegWalkPlanner(route(), GAIT)
        corner = route()[0].target_xy
        for _ in range(40):
            step = planner.step(corner[0], corner[1], PICK_YAW, 0.02)
        self.assertEqual(step.leg_name, "turn_to_table")
        self.assertEqual(step.mode, "turn")
        # A pure yaw command never steps, so the arc rides the reverse command.
        self.assertAlmostEqual(step.command[0], -GAIT.turn_vx, places=6)
        self.assertLess(step.command[2], 0.0)
        self.assertAlmostEqual(abs(step.command[2]), GAIT.wz, places=6)

    def test_no_settle_between_backoff_and_turn(self):
        """Coasting to a standstill is what stops the turn from ever starting."""
        _walker, steps = run_route()
        modes = [(step.leg_name, step.mode) for step in steps]
        # The reverse command is handed straight over to the arc.
        last_backoff = max(i for i, m in enumerate(modes) if m[0] == "backoff")
        first_turn = min(i for i, m in enumerate(modes) if m[0] == "turn_to_table")
        self.assertEqual(first_turn, last_backoff + 1)
        self.assertEqual(modes[last_backoff][1], "reverse")
        self.assertEqual(modes[first_turn][1], "turn")
        # Same for the last leg: a pause after the turn is what stops W
        # from ever starting, so the robot never walks world +Y.
        last_turn = max(i for i, m in enumerate(modes) if m[0] == "turn_to_table")
        first_approach = min(
            i for i, m in enumerate(modes) if m[0] == "approach_place"
        )
        self.assertEqual(first_approach, last_turn + 1)
        self.assertGreater(steps[first_approach].command[0], 0.0)
        yaw = PLACE_YAW - steps[first_approach].yaw_error
        vx, vy = steps[first_approach].command[0], steps[first_approach].command[1]
        world_y = vx * math.sin(yaw) + vy * math.cos(yaw)
        self.assertGreater(world_y, 0.0)

    def test_turn_reverses_the_arc_after_max_drift(self):
        """If the arc travels without turning, drive it back at the higher rate."""
        planner = navigation.LegWalkPlanner(route(), GAIT)
        corner = route()[0].target_xy
        planner.step(corner[0], corner[1], PICK_YAW, 0.02)
        step = planner.step(corner[0], corner[1], PICK_YAW, 0.02)
        self.assertEqual(step.mode, "turn")
        self.assertAlmostEqual(step.command[0], -GAIT.turn_vx, places=6)
        # Same heading, but well past the drift budget.
        drifted = corner[0] + GAIT.turn_max_drift + 0.05
        step = planner.step(drifted, corner[1], PICK_YAW, 0.02)
        self.assertEqual(step.leg_name, "turn_to_table")
        self.assertEqual(step.mode, "turn_pinned")
        # Flipped, not zeroed: zero vx would just freeze the robot in place.
        self.assertAlmostEqual(step.command[0], GAIT.turn_vx, places=6)
        self.assertEqual(step.command[1], 0.0)
        self.assertAlmostEqual(abs(step.command[2]), GAIT.wz_max, places=6)

    def test_no_command_steers_without_driving(self):
        """vy / wz alone do nothing, so they may never be sent without vx.

        This is the bug that stalled the approach in Isaac Sim: it held
        ``cmd_b=(+0.00,+0.30,+0.00)`` for 18 s and the robot never moved.
        """
        for start in (None,
                      (PICK_XY[0] + 0.10, PICK_XY[1] - 0.12, PICK_YAW - 0.25)):
            _walker, steps = run_route(start=start)
            for step in steps:
                vx, vy, wz = step.command[:3]
                if vy == 0.0 and wz == 0.0:
                    continue
                self.assertGreaterEqual(
                    abs(vx), POLICY_DEADBAND[0],
                    msg=f"{step.mode} sent vy={vy:+.2f} wz={wz:+.2f} "
                        f"with vx={vx:+.2f}: the robot would just stand there",
                )

    def test_every_command_clears_the_policy_dead_band(self):
        _walker, steps = run_route()
        for step in steps:
            for axis, value in enumerate(step.command[:3]):
                if value != 0.0:
                    self.assertGreaterEqual(
                        abs(value), POLICY_DEADBAND[axis],
                        msg=f"{step.mode} sent {value:+.3f} on axis {axis}",
                    )

    def test_route_converges_with_dead_band_policy(self):
        walker, steps = run_route()
        self.assertLess(math.hypot(walker.x - PLACE_XY[0], walker.y - PLACE_XY[1]), 0.25)
        # The stand coordinate is the goal on both axes now: x is the sideways
        # one, and it must land inside the arrival window, not just anywhere on
        # the approach line.
        self.assertLess(abs(walker.x - PLACE_XY[0]), GAIT.lateral_arrive)
        # Never past the stand: the table is 12 cm in front of it.
        self.assertLess(walker.y, PLACE_XY[1] + 0.02)
        self.assertLess(abs(navigation.wrap_angle(walker.yaw - PLACE_YAW)), GAIT.yaw_tol)
        modes = [step.mode for step in steps]
        self.assertEqual(modes.index("reverse"), 0)
        self.assertLess(modes.index("turn"), modes.index("forward"))

    def test_backoff_never_walks_forward_into_the_machine(self):
        _walker, steps = run_route()
        for step in steps:
            if step.leg_name == "backoff":
                self.assertLessEqual(step.command[0], 0.0)

    def test_sideways_drift_is_corrected_at_full_magnitude(self):
        legs = route()
        planner = navigation.LegWalkPlanner(legs, GAIT)
        # 0.25 m short of the stand along world +X on the final +Y leg: facing
        # +Y that is the robot's right, so vy must be negative and full size.
        planner._i = 2  # noqa: SLF001 - drive the last leg directly
        step = planner.step(PLACE_XY[0] - 0.25, PLACE_XY[1] - 0.5, PLACE_YAW, 0.02)
        self.assertEqual(step.leg_name, "approach_place")
        self.assertAlmostEqual(step.command[1], -GAIT.vy, places=6)
        # Correcting sideways is a diagonal walk, never a pure strafe: vy alone
        # does not start a gait, it only steers one that vx is already driving.
        self.assertEqual(step.mode, "forward_align")
        self.assertAlmostEqual(step.command[0], GAIT.vx, places=6)

    def test_approach_walks_in_once_it_is_on_the_axis(self):
        planner = navigation.LegWalkPlanner(route(), GAIT)
        planner._i = 2  # noqa: SLF001
        step = planner.step(PLACE_XY[0], PLACE_XY[1] - 0.5, PLACE_YAW, 0.02)
        self.assertEqual(step.mode, "forward")
        self.assertAlmostEqual(step.command[0], GAIT.vx, places=6)
        self.assertEqual(step.command[1], 0.0)

    def test_approach_stops_at_the_line_even_if_it_is_off_sideways(self):
        """The stop line wins over the pose, because the table is right there.

        Holding out for a square pose next to the table is what crashed twice:
        the fixes all need a live ``vx``, so "keep correcting" means "keep
        driving".  Arriving crooked is the arm's problem; arriving late is a
        crash.  It still has to say so via ``on_target``.
        """
        planner = navigation.LegWalkPlanner(route(), GAIT)
        planner._i = 2  # noqa: SLF001
        step = planner.step(PLACE_XY[0] - 0.30, PLACE_XY[1], PLACE_YAW, 0.02)
        self.assertTrue(step.route_done)
        self.assertEqual(step.command[:3], (0.0, 0.0, 0.0))
        self.assertFalse(step.on_target)

    def test_approach_stops_at_the_line_even_if_it_is_crooked(self):
        planner = navigation.LegWalkPlanner(route(), GAIT)
        planner._i = 2  # noqa: SLF001
        skew = GAIT.yaw_arrive + math.radians(4.0)
        step = planner.step(PLACE_XY[0], PLACE_XY[1], PLACE_YAW - skew, 0.02)
        self.assertTrue(step.route_done)
        self.assertEqual(step.command[:3], (0.0, 0.0, 0.0))
        self.assertFalse(step.on_target)

    def test_a_square_arrival_reports_on_target(self):
        planner = navigation.LegWalkPlanner(route(), GAIT)
        planner._i = 2  # noqa: SLF001
        step = planner.step(PLACE_XY[0], PLACE_XY[1], PLACE_YAW, 0.02)
        self.assertTrue(step.route_done)
        self.assertTrue(step.on_target)

    def test_every_axis_keeps_room_above_the_dead_band(self):
        """A fix magnitude close to the dead band is a fix that may never apply.

        ``vy`` used to sit 20% above the band while ``vx`` had 50% and ``wz``
        200%, so it was the first axis to vanish once the policy tracked a bit
        under the command: the sideways error froze and the leg just paced.
        """
        for axis, value in enumerate((GAIT.vx, GAIT.vy, GAIT.wz)):
            self.assertGreaterEqual(
                abs(value), 1.5 * POLICY_DEADBAND[axis],
                msg=f"axis {axis} commands {value} against a "
                    f"{POLICY_DEADBAND[axis]} dead band: too little room",
            )

    def test_pose_windows_stay_inside_their_fix_bands(self):
        """A pose window tighter than its fix band can never be reported met.

        The fixes only arm above ``lateral_tol`` / ``align_yaw``; below that
        nothing is commanded, so anything tighter would flag ``on_target=False``
        on a pose the planner has no way to improve.
        """
        self.assertGreaterEqual(GAIT.lateral_arrive, GAIT.lateral_tol)
        self.assertGreaterEqual(GAIT.yaw_arrive, GAIT.align_yaw)

    def test_approach_arrives_on_the_stand_coordinate(self):
        planner = navigation.LegWalkPlanner(route(), GAIT)
        planner._i = 2  # noqa: SLF001
        step = planner.step(PLACE_XY[0], PLACE_XY[1], PLACE_YAW, 0.02)
        self.assertTrue(step.route_done)
        self.assertEqual(step.command[:3], (0.0, 0.0, 0.0))

    def test_no_approach_ever_reaches_the_table(self):
        """Sweep the approach and assert the pelvis never reaches the table edge.

        The single-scenario checks missed two crashes, because what kills the
        robot is the *combination* of a crooked start, imperfect tracking and
        command lag.  So sweep them together and hold one line: the pelvis may
        never get within ``TABLE_AHEAD_OF_STAND`` of the stand's front.
        """
        worst, worst_case = -9.9, None
        for dy, dx, dyaw_deg, yaw_gain, lat_gain, lag in itertools.product(
            (-0.90, -0.60, -0.30),
            (-0.30, -0.10, 0.0, 0.10, 0.30),
            (-30, -14, 0, 14, 30),
            (0.7, 1.0, 1.5),
            (0.7, 1.0),
            (0.25, 0.45),
        ):
            walker = MistrackingWalker(
                PLACE_XY[0] + dx, PLACE_XY[1] + dy,
                PLACE_YAW + math.radians(dyaw_deg),
                yaw_gain=yaw_gain, lat_gain=lat_gain, lag=lag,
            )
            planner = navigation.LegWalkPlanner(route(), GAIT)
            planner._i = 2  # noqa: SLF001
            past = _run_until_settled(planner, walker)
            if past > worst:
                worst, worst_case = past, (dy, dx, dyaw_deg, yaw_gain, lat_gain, lag)
        self.assertLess(
            worst, TABLE_AHEAD_OF_STAND,
            msg=f"pelvis ended {worst*1000:+.0f}mm past the stand, table edge is "
                f"at {TABLE_AHEAD_OF_STAND*1000:.0f}mm, worst case "
                f"dy/dx/dyaw/yaw_gain/lat_gain/lag={worst_case}",
        )

    def test_the_stop_is_latched_and_never_lets_go(self):
        """Once stopped it stays stopped, even if the pose looks fixable.

        Reversing to have another go keeps the robot walking right next to the
        table, and the coast on the way out is what hit it.
        """
        planner = navigation.LegWalkPlanner(route(), GAIT)
        planner._i = 2  # noqa: SLF001
        planner.step(PLACE_XY[0], PLACE_XY[1] - PLACE_STOP_MARGIN, PLACE_YAW, 0.02)
        # Now feed it poses that the old logic would have driven out of.
        for dy, dx, dyaw in ((0.20, 0.0, 0.0), (0.0, -0.30, 0.0),
                             (-0.40, 0.0, 0.5), (0.0, 0.0, -0.6)):
            step = planner.step(
                PLACE_XY[0] + dx, PLACE_XY[1] + dy, PLACE_YAW + dyaw, 0.02
            )
            self.assertEqual(
                step.command[:3], (0.0, 0.0, 0.0),
                msg=f"restarted with {step.mode} at dy={dy} dx={dx} dyaw={dyaw}",
            )

    def test_approach_never_commands_forward_inside_the_stop_margin(self):
        _walker, steps = run_route()
        for step in steps:
            if step.leg_name != "approach_place":
                continue
            if step.remaining <= PLACE_STOP_MARGIN:
                self.assertLessEqual(
                    step.command[0], 0.0,
                    msg=f"{step.mode} still pushed vx={step.command[0]:+.3f} "
                        f"with remaining={step.remaining:+.3f}",
                )

    def test_large_heading_error_arcs_back_before_translating(self):
        planner = navigation.LegWalkPlanner(route(), GAIT)
        step = planner.step(PICK_XY[0], PICK_XY[1], PICK_YAW - 0.9, 0.02)
        self.assertEqual(step.mode, "reverse_realign")
        # Realigning gives up the travel axis but keeps reversing, otherwise the
        # yaw command would be ignored just like a stationary turn.
        self.assertAlmostEqual(step.command[0], -GAIT.turn_vx, places=6)
        self.assertGreater(step.command[2], 0.0)

    def test_route_survives_a_crooked_start(self):
        walker, _steps = run_route(
            start=(PICK_XY[0] + 0.10, PICK_XY[1] - 0.12, PICK_YAW - 0.25)
        )
        self.assertLess(math.hypot(walker.x - PLACE_XY[0], walker.y - PLACE_XY[1]), 0.25)

    def test_planner_reset_restarts_at_the_backoff_leg(self):
        planner = navigation.LegWalkPlanner(route(), GAIT)
        # Already standing at the goal: each leg still waits out its settle pause.
        for _ in range(200):
            planner.step(PLACE_XY[0], PLACE_XY[1], PLACE_YAW, 0.02)
        self.assertTrue(planner.route_done)
        planner.reset()
        self.assertFalse(planner.route_done)
        self.assertEqual(planner.leg.name, "backoff")


if __name__ == "__main__":
    unittest.main()
