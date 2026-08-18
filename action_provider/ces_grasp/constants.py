# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""CES 取放常量。站位：stand_xy = target_xy - (x_b*前 + y_b*左)。"""
from __future__ import annotations

import math

from action_provider.ces_grasp.navigation import WalkGait, build_carry_route
from tasks.common_scene.base_scene_ces_pickplace_wholebody import (
    PICK_STAND_X_B,
    PICK_STAND_XY as SCENE_PICK_STAND_XY,
    PICK_STAND_Y_B,
    PLACE_TRAY_HEIGHT,
    PRODUCT_POS,
    ROBOT_INIT_POS,
    ROBOT_STAND_YAW,
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
# 抓取站位与 spawn 同源（机器人直接生成在抓取站，不再瞬移）。
X_B_PICK = PICK_STAND_X_B
Y_B_PICK = PICK_STAND_Y_B
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
# 抬起后按抓取路点的逆序回到初始臂姿（00），夹着件走路更稳、也不挡视线。
RETURN_LEAD_IN_TIME = 0.8  # 抬起 q → 30 的过渡段
RETURN_TIME = 2.5  # 无关节路点时：单段回默认臂姿
CARRY_TIME = 0.6  # 冻臂 q，不要再笛卡尔收臂（件会掉）
HOLD_TIME = 0.3
# 放置：从初始臂姿先抬到放置高度（贴身、只往前挪一点），再水平伸到灰筐上方。
PLACE_RAISE_FORWARD = 0.10  # 抬臂段顺机体前方挪一点，避免死折肘
PLACE_RAISE_TIME = 2.0
PLACE_REACH_TIME = 2.2
PLACE_APPROACH_TIME = PLACE_RAISE_TIME + PLACE_REACH_TIME
# 放置 IK 只跟位置（不给朝向目标），靠下面的窗口把姿态锁在初始臂姿附近：
# 肩内外旋/肩偏摆贴住初始值 → 肘不外翻；腕三轴几乎不转。
PLACE_ROLL_WINDOW = 0.45
PLACE_YAW_WINDOW = 0.50
PLACE_WRIST_WINDOW = 0.30
PLACE_ELBOW_MIN = 0.20  # 肘不许伸直锁死
PLACE_DESCEND_TIME = 0.0  # 跳过：悬停松爪，件自由落下
RELEASE_TIME = 0.8
# 收臂：按放置轨迹的逆序（灰筐上方 → 抬臂点 → 初始臂姿），别横扫桌沿。
RETRACT_TIME = 1.6
RETRACT_HOME_TIME = 1.6

# 旧 FSM 垂臂种子，顺序同 RIGHT_ARM_JOINTS。
RIGHT_ARM_READY = (0.40, -0.42, 0.18, 1.20, 0.0, 0.95, 0.0)
ARM_SLEW_RAD = 0.080
ARM_SLEW_RAD_LIFT = 0.012  # 夹持后慢跟，垫面不瞬移
STOP_AFTER = "place"

# 路点 JSON 按关节名匹配，不改 DDS 下标。抓取：00→30 硬 q，30→40 只作 q_ref。
WAYPOINT_SET_DEFAULT = "ces_pick_natural_v2"
WAYPOINT_LEAD_IN_TIME = 0.25
WAYPOINT_LEAD_IN_TOL = 0.05

# HOLD 后的双足换站。策略命令是机体系 [vx, vy, wz, height]。
# 关键约束：策略对小指令不迈步（键盘点动走不动，必须长按把指令拉起来），
# 所以平移/转向都用固定幅值 + 死区，不用比例控制。幅值参考键盘长按能走的量级。
WALK_VX = 0.45  # 前进/后退幅值；< 0.3 基本只前后晃不迈步
WALK_VY = 0.30  # 侧移纠偏幅值
WALK_WZ = 0.70  # 原地转向幅值（键盘 Z/X 长按约 0.7~1.0）
# 指令归零后策略还会多走一点：提前 stop_margin 松"油门"。
# 若实测停不到位（走过头撞桌 / 差太多够不到灰筐），只调这两个值。
WALK_STOP_MARGIN = 0.15
WALK_STOP_MARGIN_PLACE = 0.12  # 最后一段停短了手臂就要多伸，别留太大余量
# 后退段步子容易迈大：目标点再收 20 cm，宁可退不够（第③段侧移能补）。
WALK_BACKOFF_TRIM = 0.20
WALK_YAW_ARRIVE = 0.15  # 转向到位死区 ≈ 8.6°
WALK_LATERAL_TOL = 0.10  # 侧向偏差超过才侧移纠偏
WALK_ALIGN_YAW = 0.30  # 平移中朝向纠偏死区
WALK_REALIGN_YAW = 0.60  # 歪太多：停下平移，先转正
WALK_LEG_SETTLE = 0.5  # 每段之间零指令停稳，避免后退接右转时混合指令
WALK_ARRIVE_HOLD = 0.35
WALK_PLACE_TIMEOUT = 60.0
WALK_GOTO_TIMEOUT = 20.0
WALK_VX_ACCEL = 1.00
WALK_VY_ACCEL = 1.00
WALK_WZ_ACCEL = 2.00
WALK_ABORT_TILT = 0.55
WALK_ABORT_HOLD = 0.40


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
# 机器人 spawn 就在抓取站，所以 SPAWN_* 与 PICK_* 相同，启动不瞬移。
PICK_STAND_YAW = math.radians(ROBOT_STAND_YAW)
PICK_TARGET_XY = (PRODUCT_POS[0], PRODUCT_POS[1])
PICK_STAND_XY = SCENE_PICK_STAND_XY
SPAWN_STAND_XY = (ROBOT_INIT_POS[0], ROBOT_INIT_POS[1])
SPAWN_STAND_YAW = PICK_STAND_YAW

# 放置站面向桌子（yaw=π/2）。PLACE_TARGET_XY 就是灰色托盘中心
# （见 base_scene 的 place_gray_tray_on_table：桌心 y-0.06）。
PLACE_STAND_YAW = 0.5 * math.pi
PLACE_TARGET_XY = (TABLE_SPAWN_POS[0], TABLE_SPAWN_POS[1] - 0.06)
PLACE_STAND_XY = stand_xy(PLACE_TARGET_XY, PLACE_STAND_YAW, X_B_PLACE, Y_B_PLACE)
PLACE_Z = TABLE_TOP_Z + PLACE_TRAY_HEIGHT + PLACE_RELEASE_ABOVE_TABLE

# HOLD 后的换站路线（机器人在 pick 站朝 -X，后退即走世界 +X）：
# ① 后退到与灰色托盘 / 放置站对齐（backoff 角点）
# ② 原地右转 yaw π → π/2，正对桌子
# ③ 正向走进放置站
CARRY_WALK_GAIT = WalkGait(
    vx=WALK_VX,
    vy=WALK_VY,
    wz=WALK_WZ,
    stop_margin=WALK_STOP_MARGIN,
    yaw_tol=WALK_YAW_ARRIVE,
    lateral_tol=WALK_LATERAL_TOL,
    align_yaw=WALK_ALIGN_YAW,
    realign_yaw=WALK_REALIGN_YAW,
    leg_settle=WALK_LEG_SETTLE,
)
CARRY_WALK_LEGS = build_carry_route(
    pick_xy=PICK_STAND_XY,
    pick_yaw=PICK_STAND_YAW,
    place_xy=PLACE_STAND_XY,
    place_yaw=PLACE_STAND_YAW,
    place_stop_margin=WALK_STOP_MARGIN_PLACE,
    backoff_trim=WALK_BACKOFF_TRIM,
)
WALK_BACKOFF_XY = CARRY_WALK_LEGS[0].target_xy

TABLE_REWARD_HALF_XY = (1.15, 0.40)
DROP_HEIGHT = 0.32
PLACE_HEIGHT_MIN = TABLE_TOP_Z + 0.01
PLACE_HEIGHT_MAX = TABLE_TOP_Z + PLACE_TRAY_HEIGHT + 0.12
