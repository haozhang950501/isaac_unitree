"""CPU-only contract tests for the approved CES smooth Pick design."""
from __future__ import annotations

import importlib.util
import json
import math
import sys
import types
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
POSES_ROOT = ROOT / "action_provider" / "ces_grasp" / "poses"
SMOOTH_DIR = POSES_ROOT / "ces_pick_smooth_v1"
JOINT_NAMES = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]
JOINT_POSES = [
    "00_ready",
    "10_forward_lift_retract",
    "20_right_shift_wrist_down",
    "30_pre_grasp_vertical",
]
Q_REF_POSE = "40_grasp_posture_ref"
RETURN_POSE = "05_chest_carry"
PLACE_POSE = "15_place_forward_release"
RETURN_COMMANDED_POSES = [
    "30_pre_grasp_vertical",
    "20_right_shift_wrist_down",
    RETURN_POSE,
]


class FakeTensor(np.ndarray):
    """Small numpy-backed subset used to exercise the torch-free interpolator."""

    def clone(self):
        return self.copy().view(FakeTensor)


def fake_tensor(values) -> FakeTensor:
    return np.asarray(values, dtype=float).view(FakeTensor)


def load_interpolation_module():
    fake_torch = types.ModuleType("torch")
    fake_torch.Tensor = FakeTensor
    fake_torch.zeros_like = lambda value: np.zeros_like(value).view(FakeTensor)
    sys.modules["torch"] = fake_torch
    path = ROOT / "action_provider" / "manip_common" / "interpolation.py"
    spec = importlib.util.spec_from_file_location("ces_smooth_interpolation_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


class SmoothPickManifestTests(unittest.TestCase):
    def test_manifest_preserves_runtime_q_ref_contract(self):
        manifest = read_json(SMOOTH_DIR / "trajectory_manifest.json")
        self.assertTrue(manifest["runtime_ready"])
        self.assertEqual(manifest["name"], "ces_pick_smooth_v1")
        self.assertEqual(
            manifest["pose_files"],
            [
                f"{name}.json"
                for name in [*JOINT_POSES, Q_REF_POSE, RETURN_POSE, PLACE_POSE]
            ],
        )
        self.assertNotIn("05_forward_reach.json", manifest["pose_files"])
        self.assertNotIn("25_pre_grasp_xy_aligned.json", manifest["pose_files"])
        self.assertEqual(
            [segment["future_project_control"] for segment in manifest["segments"]],
            [
                "joint_space",
                "joint_space",
                "joint_space",
                "cartesian_vertical_ik_with_dynamic_q_ref",
            ],
        )
        self.assertEqual(
            manifest["interpolation"]["method"], "monotone_cubic_hermite"
        )
        handoff = manifest["future_project_handoff"]
        self.assertEqual(handoff["joint_space_through"], "30_pre_grasp_vertical")
        self.assertEqual(handoff["dynamic_q_ref_from"], "30_pre_grasp_vertical")
        self.assertEqual(handoff["dynamic_q_ref_to"], Q_REF_POSE)
        return_path = manifest["return_path"]
        self.assertEqual(
            return_path["logical_waypoints"],
            [Q_REF_POSE, *RETURN_COMMANDED_POSES],
        )
        self.assertEqual(
            return_path["commanded_waypoints"], RETURN_COMMANDED_POSES
        )
        self.assertIn("never hard-command", return_path["start_policy"])
        self.assertEqual(
            [float(segment["duration_s"]) for segment in return_path["segments"]],
            [0.8, 1.2, 3.0],
        )
        self.assertEqual(return_path["interpolation"], "monotone_cubic_hermite")
        return_validation = manifest["validation"]["return_path"]
        self.assertTrue(return_validation["validation_passed"])
        self.assertTrue(return_validation["pose_20_is_drawer_clearance_waypoint"])
        self.assertTrue(return_validation["via_20_to_05"])
        self.assertLess(return_validation["max_abs_velocity_rad_s"], 0.8)
        place_path = manifest["place_path"]
        self.assertEqual(
            place_path["logical_waypoints"], [RETURN_POSE, PLACE_POSE]
        )
        self.assertEqual(place_path["commanded_waypoints"], [PLACE_POSE])
        self.assertEqual(place_path["interpolation"], "segment_smoothstep")
        self.assertEqual(
            [float(segment["duration_s"]) for segment in place_path["segments"]],
            [3.2],
        )
        self.assertNotIn("vertical_compensation", place_path)
        self.assertEqual(place_path["final_role"], "release_posture_at_pose_15")
        place_validation = manifest["validation"]["place_path"]
        self.assertTrue(place_validation["validation_passed"])
        self.assertTrue(place_validation["pose_05_is_exact_carry_start"])
        self.assertTrue(place_validation["pose_25_removed"])

    def test_selected_waypoint_q_values_match_baseline(self):
        for name in [*JOINT_POSES, Q_REF_POSE]:
            smooth = read_json(SMOOTH_DIR / f"{name}.json")
            self.assertEqual(smooth["joint_order"], JOINT_NAMES)
        q30 = read_json(SMOOTH_DIR / "30_pre_grasp_vertical.json")["q"]
        q40 = read_json(SMOOTH_DIR / f"{Q_REF_POSE}.json")
        self.assertEqual(q40["control_role"], "q_ref_only")
        self.assertFalse(q40["unitree_arm_q_allowed"])
        self.assertEqual(q40["q"][4:], q30[4:])
        q05 = read_json(SMOOTH_DIR / f"{RETURN_POSE}.json")
        self.assertEqual(q05["control_role"], "return_joint_space_carry_endpoint")
        self.assertTrue(q05["unitree_arm_q_allowed"])
        self.assertEqual(
            q05["q"],
            [-0.04000009, -0.34000003, 0.52000004, -0.7499997, -0.34, -0.09, 0.90999967],
        )
        self.assertGreater(q05["validation"]["min_joint_limit_margin_rad"], 0.1)
        q15 = read_json(SMOOTH_DIR / f"{PLACE_POSE}.json")
        self.assertEqual(q15["control_role"], "place_joint_space_release_endpoint")
        self.assertTrue(q15["unitree_arm_q_allowed"])
        self.assertEqual(
            q15["q"],
            [
                -1.2266918,
                -0.2318979,
                1.4931674,
                1.2198853,
                -0.042756237,
                0.005500171,
                1.1593101,
            ],
        )
        self.assertGreater(q15["validation"]["min_joint_limit_margin_rad"], 0.5)

    def test_pose_library_loads_runtime_sequence(self):
        action_provider_package = types.ModuleType("action_provider")
        action_provider_package.__path__ = [str(ROOT / "action_provider")]
        package = types.ModuleType("action_provider.ces_grasp")
        package.__path__ = [str(ROOT / "action_provider" / "ces_grasp")]
        constants = types.ModuleType("action_provider.ces_grasp.constants")
        constants.RIGHT_ARM_JOINTS = JOINT_NAMES
        sys.modules["action_provider"] = action_provider_package
        sys.modules["action_provider.ces_grasp"] = package
        sys.modules[constants.__name__] = constants
        action_provider_package.ces_grasp = package
        package.constants = constants
        path = ROOT / "action_provider" / "ces_grasp" / "pose_library.py"
        spec = importlib.util.spec_from_file_location("ces_smooth_pose_library_test", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        waypoint_set = module.load_baseline_trajectory()
        self.assertEqual(waypoint_set.joint_waypoints, tuple(JOINT_POSES))
        self.assertEqual(waypoint_set.q_ref_from, "30_pre_grasp_vertical")
        self.assertEqual(waypoint_set.q_ref_to, Q_REF_POSE)
        self.assertEqual(waypoint_set.return_start, Q_REF_POSE)
        self.assertEqual(
            waypoint_set.return_waypoints, tuple(RETURN_COMMANDED_POSES)
        )
        self.assertEqual(
            waypoint_set.return_segment_durations, (0.8, 1.2, 3.0)
        )
        self.assertEqual(
            waypoint_set.return_interpolation_method, "monotone_cubic_hermite"
        )
        self.assertEqual(waypoint_set.place_start, RETURN_POSE)
        self.assertEqual(waypoint_set.place_waypoints, (PLACE_POSE,))
        self.assertEqual(waypoint_set.place_segment_durations, (3.2,))
        self.assertEqual(
            waypoint_set.place_interpolation_method, "segment_smoothstep"
        )
        self.assertEqual(
            waypoint_set.interpolation_method, "monotone_cubic_hermite"
        )


class PlacePathTests(unittest.TestCase):
    def test_05_to_15_smoothstep_hits_both_approved_q_values(self):
        module = load_interpolation_module()
        q05 = fake_tensor(read_json(SMOOTH_DIR / f"{RETURN_POSE}.json")["q"])
        q15 = fake_tensor(read_json(SMOOTH_DIR / f"{PLACE_POSE}.json")["q"])
        interpolator = module.JointSpaceInterpolator("cpu")
        interpolator.reset(q05, q15, 3.2, method="segment_smoothstep")
        np.testing.assert_allclose(interpolator.step(0.0), q05, atol=1e-9)
        interpolator.elapsed = interpolator.duration
        np.testing.assert_allclose(interpolator.step(0.0), q15, atol=1e-9)

class SmoothJointInterpolatorTests(unittest.TestCase):
    INTERPOLATION_METHOD = "monotone_cubic_hermite"

    @classmethod
    def setUpClass(cls):
        cls.module = load_interpolation_module()
        cls.manifest = read_json(SMOOTH_DIR / "trajectory_manifest.json")
        cls.qs = [
            fake_tensor(read_json(SMOOTH_DIR / f"{name}.json")["q"])
            for name in JOINT_POSES
        ]
        cls.durations = [
            float(segment["duration_s"])
            for segment in cls.manifest["segments"][: len(JOINT_POSES) - 1]
        ]

    def make_interpolator(self):
        interpolator = self.module.JointSpaceInterpolator("cpu")
        interpolator.reset_path(
            self.qs,
            self.durations,
            method=self.INTERPOLATION_METHOD,
        )
        return interpolator

    def sample(self, interpolator, elapsed: float) -> np.ndarray:
        interpolator.elapsed = max(0.0, min(interpolator.duration, elapsed))
        return np.asarray(interpolator.step(0.0), dtype=float)

    def test_hits_every_waypoint(self):
        interpolator = self.make_interpolator()
        np.testing.assert_allclose(self.sample(interpolator, 0.0), self.qs[0])
        for index, bound in enumerate(interpolator.bounds):
            np.testing.assert_allclose(
                self.sample(interpolator, bound), self.qs[index + 1], atol=1e-9
            )

    def test_interior_velocity_is_continuous_and_nonzero(self):
        interpolator = self.make_interpolator()
        epsilon = 1e-4
        for bound in interpolator.bounds[:-1]:
            at = self.sample(interpolator, bound)
            left = (at - self.sample(interpolator, bound - epsilon)) / epsilon
            right = (self.sample(interpolator, bound + epsilon) - at) / epsilon
            self.assertGreater(float(np.linalg.norm(left)), 0.05)
            np.testing.assert_allclose(left, right, atol=5e-4)

    def test_curve_does_not_overshoot_global_waypoint_range(self):
        interpolator = self.make_interpolator()
        low = np.min(np.asarray(self.qs), axis=0) - 1e-9
        high = np.max(np.asarray(self.qs), axis=0) + 1e-9
        for elapsed in np.linspace(0.0, interpolator.duration, 1001):
            q = self.sample(interpolator, float(elapsed))
            self.assertTrue(np.all(q >= low))
            self.assertTrue(np.all(q <= high))


class SmoothReturnInterpolatorTests(SmoothJointInterpolatorTests):
    """The authored 30->20->05 return must stay smooth and bounded."""

    INTERPOLATION_METHOD = "monotone_cubic_hermite"

    @classmethod
    def setUpClass(cls):
        cls.module = load_interpolation_module()
        manifest = read_json(SMOOTH_DIR / "trajectory_manifest.json")
        cls.qs = [
            fake_tensor(read_json(SMOOTH_DIR / f"{name}.json")["q"])
            for name in RETURN_COMMANDED_POSES
        ]
        # Segment 0 is live post-lift (logical 40 phase) -> authored 30.
        cls.durations = [
            float(segment["duration_s"])
            for segment in manifest["return_path"]["segments"][1:]
        ]

    def test_interior_velocity_is_continuous_and_nonzero(self):
        """Pose 20 is crossed continuously without skipping the clearance q."""
        interpolator = self.make_interpolator()
        epsilon = 1e-4
        bound = interpolator.bounds[0]
        at = self.sample(interpolator, bound)
        left = (at - self.sample(interpolator, bound - epsilon)) / epsilon
        right = (self.sample(interpolator, bound + epsilon) - at) / epsilon
        np.testing.assert_allclose(left, right, atol=5e-4)
        self.assertGreater(float(np.linalg.norm(left)), 0.05)

    def test_max_speed_return_stays_under_the_slew_clamp(self):
        interpolator = self.module.JointSpaceInterpolator("cpu")
        interpolator.reset_path(
            self.qs,
            self.module.scale_segment_times(self.durations, 3.0, min_time=0.4),
            method=self.INTERPOLATION_METHOD,
        )
        epsilon = 1e-4
        peak = 0.0
        for elapsed in np.linspace(epsilon, interpolator.duration - epsilon, 2001):
            rate = (
                self.sample(interpolator, float(elapsed) + epsilon)
                - self.sample(interpolator, float(elapsed) - epsilon)
            ) / (2.0 * epsilon)
            peak = max(peak, float(np.abs(rate).max()))
        self.assertLess(peak, 0.080 / (4 * 0.005))


class PickSpeedScaleTests(SmoothJointInterpolatorTests):
    """Speeding the pick up must only rescale time, never reshape the path."""

    SPEED = 1.5

    def make_interpolator(self):
        interpolator = self.module.JointSpaceInterpolator("cpu")
        interpolator.reset_path(
            self.qs,
            self.module.scale_segment_times(self.durations, self.SPEED),
            method="monotone_cubic_hermite",
        )
        return interpolator

    def test_scaling_floors_each_segment(self):
        scaled = self.module.scale_segment_times([1.2, 0.3], 3.0, min_time=0.4)
        self.assertAlmostEqual(scaled[0], 0.4)
        self.assertAlmostEqual(scaled[1], 0.4)

    def test_scaled_path_is_the_same_curve_in_normalized_time(self):
        authored = super().make_interpolator()
        fast = self.make_interpolator()
        self.assertAlmostEqual(fast.duration, authored.duration / self.SPEED, places=6)
        for fraction in np.linspace(0.0, 1.0, 401):
            np.testing.assert_allclose(
                self.sample(fast, float(fraction) * fast.duration),
                self.sample(authored, float(fraction) * authored.duration),
                atol=1e-9,
            )

    def test_peak_joint_rate_stays_under_the_slew_clamp(self):
        # action_provider 每步把关节增量截断在 ARM_SLEW_RAD=0.080 rad，
        # 控制周期 dt = 4 * 0.005 s，所以下发速度上限是 4.0 rad/s。
        interpolator = self.make_interpolator()
        epsilon = 1e-4
        peak = 0.0
        for elapsed in np.linspace(epsilon, interpolator.duration - epsilon, 2001):
            rate = (
                self.sample(interpolator, float(elapsed) + epsilon)
                - self.sample(interpolator, float(elapsed) - epsilon)
            ) / (2.0 * epsilon)
            peak = max(peak, float(np.abs(rate).max()))
        self.assertLess(peak, 0.080 / (4 * 0.005))


class GraspYawAlignTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = ROOT / "action_provider" / "ces_grasp" / "grasp_yaw.py"
        spec = importlib.util.spec_from_file_location("ces_grasp_yaw_test", path)
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    def test_closer_world_x_prefers_plus_x_on_a_tie(self):
        target, delta = self.module.closer_world_x_yaw(math.pi / 2)
        self.assertEqual(target, 0.0)
        self.assertAlmostEqual(delta, -math.pi / 2, places=6)

    def test_already_on_plus_x_is_a_no_op(self):
        target, delta = self.module.closer_world_x_yaw(0.0)
        self.assertEqual(target, 0.0)
        self.assertAlmostEqual(delta, 0.0, places=9)

    def test_pose_30_jaw_squares_onto_world_minus_x(self):
        """ces_pick_smooth_v1 pose 30 jaw is ~68° off pelvis +X = world −X at pick yaw=π."""
        pose = read_json(SMOOTH_DIR / "30_pre_grasp_vertical.json")
        col2 = pose["palm_orientation"]["rotation_matrix"]
        jx_p, jy_p = float(col2[0][2]), float(col2[1][2])
        # pick yaw=π: pelvis +X/+Y = world −X/−Y
        yaw = self.module.jaw_xy_yaw(-jx_p, -jy_p)
        target, delta = self.module.closer_world_x_yaw(yaw)
        self.assertAlmostEqual(target, math.pi, places=6)
        self.assertGreater(abs(delta), math.radians(60.0))
        self.assertLess(abs(delta), math.radians(80.0))
        self.assertAlmostEqual(
            abs(math.degrees(delta)),
            float(pose["dex1_two_finger_proxy"]["xr_jaw_axis_to_pelvis_x_angle_deg"]),
            delta=1.0,
        )


if __name__ == "__main__":
    unittest.main()
