# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""CES LoadingLine product pick-and-place constants.

Stand poses are derived from the scene Product / table poses and the measured
right-arm sweet spot in the pelvis frame:

    stand_xy = target_xy - (x_b * forward + y_b * left)
"""
from __future__ import annotations

import math

from tasks.common_scene.base_scene_ces_pickplace_wholebody import (
    PRODUCT_POS,
    ROBOT_INIT_POS,
    TABLE_SPAWN_POS,
    TABLE_TOP_Z,
)

# Dex1: q increases → jaws close.  gap ≈ 0.050 - 2q (m)
# Product AABB 36 x 138.5 x 25.5 mm; pinch the 36 mm world-X face, fingers down.
TCP_LOCAL = (0.0, 0.115, 0.0)
GRIPPER_OPEN = -0.010
GRIPPER_CLOSED = 0.019  # gap ≈ 12 mm; PD squeeze without punching through
STAND_PELVIS_Z = 0.755

EE_BODY = "right_hand_base_link"
RIGHT_ARM_JOINTS = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]
LEFT_ARM_JOINTS = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
]
RIGHT_GRIPPER_JOINTS = ["right_hand_Joint1_1", "right_hand_Joint2_1"]
LEFT_GRIPPER_JOINTS = ["left_hand_Joint1_1", "left_hand_Joint2_1"]

# Stand closer so the right arm can reach deeper into the tray.
X_B_PICK = 0.30
Y_B_PICK = -0.38
Z_B_PICK = 0.066
X_B_PLACE = 0.46
Y_B_PLACE = -0.18

# Slide in above the lip with fingers already vertical, then drop onto Product.
APPROACH_HEIGHT = 0.080
LIFT_HEIGHT = 0.08  # just above tray; chest height, not shoulder
APPROACH_STANDOFF = 0.18  # rotate to vertical fully outside the pocket
LIFT_RETRACT = 0.04  # tiny XY clear; do not IK-retract (that flips the arm up)
GRASP_INSET = 0.020  # world -X into the drawer; 10 mm left the +X jaw in the mouth slot
GRASP_SHIFT_Y = 0.0  # world Y along the long axis; + is toward world +Y
# AABB Z is mid-thickness (25.5 mm).  TCP a bit above the top face so the
# pads pinch the upper rim and do not drive into the pocket.
PRODUCT_HALF_Z = 0.01275
GRASP_Z_CLEARANCE = 0.012
GRASP_Z_OFFSET = PRODUCT_HALF_Z + GRASP_Z_CLEARANCE  # ≈ 0.025, above the mounting step
PLACE_APPROACH_HEIGHT = 0.10
PLACE_CLEARANCE = 0.018  # object-root z above tabletop (half of 25.5 mm + 5 mm)

SETTLE_TIME = 1.0
# Snap mode pins the pelvis for the whole station stay.  Walk mode never
# pins: same as DDSRLActionProvider / Move-Cylinder — policy balances.
STAND_MIN_TIME = 0.6
STAND_STABLE_TIME = 0.5    # consecutive seconds of "standing"
STAND_TILT_MAX = 0.18
STAND_YAW_RATE_MAX = 0.45
STAND_XY_SPEED_MAX = 0.12
APPROACH_TIME = 2.8
UNFOLD_TIME = 3.2
ORIENT_TIME = 2.2
SLIDE_TIME = 2.4
DESCEND_TIME = 2.0
GRASP_TIME = 3.0
GRASP_POS_TOL = 0.055  # visual alignment is tighter than TCP residual
GRASP_WAIT_MAX = 3.0
LIFT_TIME = 3.2
CARRY_TIME = 0.6  # freeze the lift pose; do not Cartesian-tuck (that dumps the part)
HOLD_TIME = 0.3
PLACE_APPROACH_TIME = 2.8
PLACE_DESCEND_TIME = 3.5
RELEASE_TIME = 0.6
RETRACT_TIME = 1.4

# Default hanging pose → this seed, then DiffIK to the vertical pre-grasp.
# Order matches RIGHT_ARM_JOINTS.  Elbow ~90°, wrist starts pointing down.
RIGHT_ARM_READY = (0.40, -0.42, 0.18, 1.20, 0.0, 0.95, 0.0)
# Shoulder-height carry (pitch ~66°).  Keep roll close to the WB default so
# the torso does not lean; elbow bent so the hand sits at the shoulder, not overhead.
RIGHT_ARM_CARRY = (1.15, -0.22, 0.0, 1.00, 0.0, 0.45, 0.0)
ARM_SLEW_RAD = 0.080  # max |Δq| per control step (~4 rad/s at 50 Hz)
ARM_SLEW_RAD_LIFT = 0.012  # keep pads on the part while friction-lifting
STOP_AFTER = "place"

# Walk mode slews the pelvis in the world XY (policy cannot be trusted at
# yaw=π).  The ONNX gait only animates the legs.  Manipulation pins the root.
WALK_PLACE_TIMEOUT = 50.0
WALK_GOTO_TIMEOUT = 20.0
WALK_POS_ARRIVE = 0.06
WALK_YAW_ARRIVE = 0.12
WALK_GUIDE_SPEED = 0.28  # m/s along the world-XY line to the stand
WALK_GUIDE_YAW_RATE = 0.90  # rad/s
WALK_ANIM_VX = 0.20  # body-vx shown to the policy so the feet cycle
WALK_ARRIVE_HOLD = 0.35

CARRY_OFFSET_B = (0.40, -0.22, 0.16)
# Pre-extend the right arm so DiffIK is not starting from a folded pose.
READY_OFFSET_B = (0.42, -0.24, 0.14)


def heading_yaw(root_quat) -> float:
    """Heading from body +X projected on the ground (valid while pitched)."""
    w = float(root_quat[0, 0])
    x = float(root_quat[0, 1])
    y = float(root_quat[0, 2])
    z = float(root_quat[0, 3])
    fx = 1.0 - 2.0 * (y * y + z * z)
    fy = 2.0 * (x * y + w * z)
    return math.atan2(fy, fx)


def yaw_quat(yaw: float) -> tuple[float, float, float, float]:
    return (math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0))


def wrap_angle(a: float) -> float:
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def forward_left(yaw: float) -> tuple[tuple[float, float], tuple[float, float]]:
    """Body x = forward, body y = left, for a Z-up yaw (0 = +X)."""
    c, s = math.cos(yaw), math.sin(yaw)
    return (c, s), (-s, c)


def stand_xy(target_xy: tuple[float, float], yaw: float, x_b: float, y_b: float) -> tuple[float, float]:
    fwd, left = forward_left(yaw)
    return (
        target_xy[0] - (x_b * fwd[0] + y_b * left[0]),
        target_xy[1] - (x_b * fwd[1] + y_b * left[1]),
    )


# Pick: stand left of the tray so the right arm unfolds into the mouth.
# Yaw π = face -X (cluster is yawed +180°).  Walk commands stay in the body
# frame, so this heading is fine — do not treat world +X as "forward".
SPAWN_STAND_XY = (ROBOT_INIT_POS[0], ROBOT_INIT_POS[1])
SPAWN_STAND_YAW = math.pi

PICK_STAND_YAW = math.pi
PICK_TARGET_XY = (PRODUCT_POS[0], PRODUCT_POS[1])
PICK_STAND_XY = stand_xy(PICK_TARGET_XY, PICK_STAND_YAW, X_B_PICK, Y_B_PICK)

# Place on the table top, inset from the south lip.  Product long axis is
# ~139 mm along world Y; the old y=-0.50 sat on the edge and left half hanging.
PLACE_STAND_YAW = 0.5 * math.pi
PLACE_TARGET_XY = (TABLE_SPAWN_POS[0], TABLE_SPAWN_POS[1] - 0.06)
PLACE_STAND_XY = stand_xy(PLACE_TARGET_XY, PLACE_STAND_YAW, X_B_PLACE, Y_B_PLACE)
PLACE_Z = TABLE_TOP_Z + PLACE_CLEARANCE

# Walk vias stay east of CES (x > -2.95). Place via is in the aisle so the
# robot can face the table, then walk body-forward (no reverse off the tray).
PICK_VIA_XY = (ROBOT_INIT_POS[0] - 0.15, -1.70)
PLACE_VIA_XY = (-2.55, -1.25)

# Reward / termination boxes (world).  Table AABB x[-3.32,-0.85] y[-0.63,0.13].
TABLE_REWARD_HALF_XY = (1.15, 0.40)
DROP_HEIGHT = 0.32
PLACE_HEIGHT_MIN = TABLE_TOP_Z + 0.01
PLACE_HEIGHT_MAX = TABLE_TOP_Z + 0.22
