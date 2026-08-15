# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""CES 取放常量。站位：stand_xy = target_xy - (x_b*前 + y_b*左)。"""
from __future__ import annotations

import math

from tasks.common_scene.base_scene_ces_pickplace_wholebody import (
    PLACE_TRAY_HEIGHT,
    PRODUCT_POS,
    ROBOT_INIT_POS,
    TABLE_SPAWN_POS,
    TABLE_TOP_Z,
)

# Dex1：q 增大则闭合。gap ≈ 0.050 - 2q (m)
# 产品 AABB 36×138.5×25.5 mm；夹世界 X 短边，手指朝下。
TCP_LOCAL = (0.0, 0.115, 0.0)
GRIPPER_OPEN = -0.010
GRIPPER_CLOSED = 0.019  # gap ≈ 12 mm
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

# 站位：target - (x_b*前 + y_b*左)。抓取站靠近托盘，右手能伸进上料口。
X_B_PICK = 0.30
Y_B_PICK = -0.38
X_B_PLACE = 0.46
Y_B_PLACE = -0.18

APPROACH_HEIGHT = 0.080
LIFT_HEIGHT = 0.08  # 刚过托盘沿，胸口高度
APPROACH_STANDOFF = 0.18
# 抬起时 TCP 往左（pick yaw=π 时世界 -Y），前臂躲开抽屉沿，朝向不变。
LIFT_SHIFT_Y = -0.06
GRASP_INSET = 0.020  # 世界 -X 收进抽屉，避免 +X 指卡槽
GRASP_SHIFT_Y = 0.0
PRODUCT_HALF_Z = 0.01275
GRASP_Z_CLEARANCE = 0.022  # 夹上沿，太深会咬凹槽
GRASP_Z_OFFSET = PRODUCT_HALF_Z + GRASP_Z_CLEARANCE  # ≈ 0.035
# 放置：TCP 停在灰筐沿上方，不要 IK 贴桌（腕会抖）。
PLACE_RELEASE_ABOVE_TABLE = 0.08

SETTLE_TIME = 1.0
STAND_MIN_TIME = 0.6
STAND_STABLE_TIME = 0.5
STAND_TILT_MAX = 0.18
STAND_YAW_RATE_MAX = 0.45
STAND_XY_SPEED_MAX = 0.12
APPROACH_TIME = 2.8
UNFOLD_TIME = 3.2
ORIENT_TIME = 2.2
SLIDE_TIME = 2.4
DESCEND_TIME = 1.1
GRASP_TIME = 1.0
GRASP_POS_TOL = 0.055
GRASP_WAIT_MAX = 0.6
LIFT_TIME = 2.2
CARRY_TIME = 0.6  # 冻抬起 q，不要再笛卡尔收臂（件会掉）
HOLD_TIME = 0.3
PLACE_APPROACH_TIME = 2.8
# snap 后若 TCP 已靠近放置点，只落 Z，避免长距离追 XY 拧臂。
PLACE_HOLD_XY_M = 0.40
PLACE_DESCEND_TIME = 0.0  # 跳过：悬停松爪，件自由落下
RELEASE_TIME = 0.8
RETRACT_TIME = 0.8

# 旧 FSM 垂臂种子，顺序同 RIGHT_ARM_JOINTS。
RIGHT_ARM_READY = (0.40, -0.42, 0.18, 1.20, 0.0, 0.95, 0.0)
ARM_SLEW_RAD = 0.080
ARM_SLEW_RAD_LIFT = 0.012  # 夹持后慢跟，垫面不瞬移
STOP_AFTER = "place"

# 路点 JSON 按关节名匹配，不改 DDS 下标。抓取：00→30 硬 q，30→40 只作 q_ref。
WAYPOINT_SET_DEFAULT = "ces_pick_natural_v2"
WAYPOINT_LEAD_IN_TIME = 0.25
WAYPOINT_LEAD_IN_TOL = 0.05

# walk 仍未接通；下列给以后双足换站用。
WALK_PLACE_TIMEOUT = 50.0
WALK_GOTO_TIMEOUT = 20.0
WALK_POS_ARRIVE = 0.06
WALK_YAW_ARRIVE = 0.12
WALK_GUIDE_SPEED = 0.28
WALK_GUIDE_YAW_RATE = 0.90
WALK_ANIM_VX = 0.20
WALK_ARRIVE_HOLD = 0.35


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


# 抓取站在托盘左侧，右手伸进上料口。yaw=π 朝 -X（整簇已转 +180°）。
SPAWN_STAND_XY = (ROBOT_INIT_POS[0], ROBOT_INIT_POS[1])
SPAWN_STAND_YAW = math.pi

PICK_STAND_YAW = math.pi
PICK_TARGET_XY = (PRODUCT_POS[0], PRODUCT_POS[1])
PICK_STAND_XY = stand_xy(PICK_TARGET_XY, PICK_STAND_YAW, X_B_PICK, Y_B_PICK)

# 放置站面向桌子（yaw=π/2）。目标往桌内收 6 cm，避免件露沿。
PLACE_STAND_YAW = 0.5 * math.pi
PLACE_TARGET_XY = (TABLE_SPAWN_POS[0], TABLE_SPAWN_POS[1] - 0.06)
PLACE_STAND_XY = stand_xy(PLACE_TARGET_XY, PLACE_STAND_YAW, X_B_PLACE, Y_B_PLACE)
PLACE_Z = TABLE_TOP_Z + PLACE_TRAY_HEIGHT + PLACE_RELEASE_ABOVE_TABLE

# walk 绕开 CES（x > -2.95），过道里再转向桌子。
PICK_VIA_XY = (ROBOT_INIT_POS[0] - 0.15, -1.70)
PLACE_VIA_XY = (-2.55, -1.25)

TABLE_REWARD_HALF_XY = (1.15, 0.40)
DROP_HEIGHT = 0.32
PLACE_HEIGHT_MIN = TABLE_TOP_Z + 0.01
PLACE_HEIGHT_MAX = TABLE_TOP_Z + PLACE_TRAY_HEIGHT + 0.12
