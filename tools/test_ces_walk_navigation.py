"""CPU-only checks for the CES carry walk route (back out → turn right → walk in).

The whole-body policy does not react to small velocity commands (tapping
keyboard WASD never makes the robot step), so these tests model that dead band
explicitly: a command below the threshold produces no motion at all.  The route
must still converge under that model.
"""
from __future__ import annotations

import importlib.util
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

GAIT = navigation.WalkGait(
    vx=0.45,
    vy=0.30,
    wz=0.70,
    stop_margin=0.15,
    yaw_tol=0.15,
    lateral_tol=0.10,
    align_yaw=0.30,
    realign_yaw=0.60,
    leg_settle=0.5,
)
# Below these magnitudes the policy only sways in place.
POLICY_DEADBAND = (0.30, 0.25, 0.40)
# Backing up overshoots, so the reverse leg aims short on purpose.
BACKOFF_TRIM = 0.20


def route(backoff_trim=BACKOFF_TRIM):
    return navigation.build_carry_route(
        pick_xy=PICK_XY,
        pick_yaw=PICK_YAW,
        place_xy=PLACE_XY,
        place_yaw=PLACE_YAW,
        place_stop_margin=0.12,
        backoff_trim=backoff_trim,
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
        blend = min(1.0, self.dt / self.lag)
        for i in range(3):
            self.vel[i] += (target[i] - self.vel[i]) * blend
        self.yaw = navigation.wrap_angle(self.yaw + self.vel[2] * self.dt)
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        self.x += (self.vel[0] * c - self.vel[1] * s) * self.dt
        self.y += (self.vel[0] * s + self.vel[1] * c) * self.dt


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
        legs = route(backoff_trim=0.0)
        self.assertEqual([leg.kind for leg in legs], ["reverse", "turn", "forward"])
        # Back out along world +X (robot faces -X) until aligned with the tote.
        self.assertAlmostEqual(legs[0].target_xy[0], PLACE_XY[0], places=3)
        self.assertAlmostEqual(legs[0].target_xy[1], PICK_XY[1], places=3)
        self.assertAlmostEqual(legs[0].target_xy[0], TRAY_XY[0] - 0.18, places=3)
        # pi -> pi/2 is a right turn (negative wz).
        self.assertLess(navigation.wrap_angle(legs[1].yaw - legs[0].yaw), 0.0)
        self.assertEqual(legs[2].target_xy, PLACE_XY)

    def test_backoff_trim_stops_short_of_the_crossing(self):
        full = route(backoff_trim=0.0)[0].target_xy
        trimmed = route()[0].target_xy
        # Backing runs along world +X, so trimming lands at a smaller x.
        self.assertAlmostEqual(trimmed[0], full[0] - BACKOFF_TRIM, places=6)
        self.assertAlmostEqual(trimmed[1], full[1], places=6)
        self.assertLess(
            math.hypot(trimmed[0] - PICK_XY[0], trimmed[1] - PICK_XY[1]),
            math.hypot(full[0] - PICK_XY[0], full[1] - PICK_XY[1]),
        )

    def test_first_command_backs_up_before_any_turn(self):
        planner = navigation.LegWalkPlanner(route(), GAIT)
        step = planner.step(PICK_XY[0], PICK_XY[1], PICK_YAW, 0.02)
        self.assertEqual(step.leg_name, "backoff")
        self.assertLess(step.command[0], 0.0)
        self.assertEqual(step.command[2], 0.0)

    def test_turn_leg_is_clockwise_and_holds_position(self):
        planner = navigation.LegWalkPlanner(route(), GAIT)
        corner = route()[0].target_xy
        for _ in range(40):
            step = planner.step(corner[0], corner[1], PICK_YAW, 0.02)
        self.assertEqual(step.leg_name, "turn_to_table")
        self.assertEqual(step.command[0], 0.0)
        self.assertLess(step.command[2], 0.0)
        self.assertAlmostEqual(abs(step.command[2]), GAIT.wz, places=6)

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
        self.assertLess(math.hypot(walker.x - PLACE_XY[0], walker.y - PLACE_XY[1]), 0.15)
        self.assertLess(abs(navigation.wrap_angle(walker.yaw - PLACE_YAW)), GAIT.yaw_tol)
        modes = [step.mode for step in steps]
        self.assertEqual(modes.index("reverse"), 0)
        self.assertLess(modes.index("turn"), modes.index("forward"))
        self.assertIn("settle", modes)

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
        self.assertAlmostEqual(step.command[0], GAIT.vx, places=6)
        self.assertAlmostEqual(step.command[1], -GAIT.vy, places=6)

    def test_large_heading_error_turns_before_translating(self):
        planner = navigation.LegWalkPlanner(route(), GAIT)
        step = planner.step(PICK_XY[0], PICK_XY[1], PICK_YAW - 0.9, 0.02)
        self.assertEqual(step.mode, "reverse_realign")
        self.assertEqual(step.command[0], 0.0)
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
