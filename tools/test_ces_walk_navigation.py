"""CPU-only checks for the CES world-to-body walk planner."""
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


def plan(*, x=0.0, y=0.0, yaw=0.0, target=(1.0, 0.0), target_yaw=0.0, require_yaw=True):
    return navigation.plan_turn_then_forward(
        x=x,
        y=y,
        yaw=yaw,
        target_xy=target,
        target_yaw=target_yaw,
        require_yaw=require_yaw,
        pos_tolerance=0.06,
        yaw_tolerance=0.12,
        align_tolerance=0.22,
        min_vx=0.10,
        max_vx=0.22,
        distance_gain=0.65,
        yaw_gain=1.20,
        max_wz=0.55,
    )


class TestCesWalkNavigation(unittest.TestCase):
    def test_yaw_pi_turns_before_world_positive_x_motion(self):
        result = plan(yaw=math.pi, target=(1.0, 0.1), require_yaw=False)
        self.assertEqual(result.mode, "turn_to_path")
        self.assertEqual(result.command[0], 0.0)
        self.assertLess(result.command[2], 0.0)

    def test_aligned_robot_walks_forward(self):
        result = plan(yaw=0.05, target=(1.0, 0.1), require_yaw=False)
        self.assertEqual(result.mode, "forward")
        self.assertGreater(result.command[0], 0.0)

    def test_translation_command_is_never_negative_vx(self):
        for degrees in range(-180, 181, 15):
            result = plan(yaw=math.radians(degrees), target=(1.0, 0.25), require_yaw=False)
            self.assertGreaterEqual(result.command[0], 0.0)

    def test_goal_rotates_to_place_yaw_then_arrives(self):
        turning = plan(target=(0.01, 0.01), yaw=0.0, target_yaw=math.pi / 2)
        self.assertEqual(turning.mode, "turn_at_goal")
        self.assertGreater(turning.command[2], 0.0)

        arrived = plan(target=(0.01, 0.01), yaw=math.pi / 2, target_yaw=math.pi / 2)
        self.assertTrue(arrived.pose_ready)
        self.assertEqual(arrived.command[:3], (0.0, 0.0, 0.0))

    def test_rotated_ces_route_converges_with_forward_only_motion(self):
        x, y, yaw = -3.187, -1.330, math.pi
        goals = [
            ((-2.55, -1.25), math.pi, False),
            ((-2.27, -0.77), math.pi / 2, True),
        ]
        for target, target_yaw, require_yaw in goals:
            for _ in range(3000):
                result = plan(
                    x=x,
                    y=y,
                    yaw=yaw,
                    target=target,
                    target_yaw=target_yaw,
                    require_yaw=require_yaw,
                )
                vx, _vy, wz, _height = result.command
                self.assertGreaterEqual(vx, 0.0)
                dt = 0.02
                yaw = navigation.wrap_angle(yaw + wz * dt)
                x += vx * math.cos(yaw) * dt
                y += vx * math.sin(yaw) * dt
                if result.pose_ready:
                    break
            else:
                self.fail(f"route did not converge to {target}")

        self.assertLess(math.hypot(x + 2.27, y + 0.77), 0.06)
        self.assertLess(abs(navigation.wrap_angle(math.pi / 2 - yaw)), 0.12)


if __name__ == "__main__":
    unittest.main()
