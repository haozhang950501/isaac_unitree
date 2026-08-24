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
    PLACE_TRAY_CENTER_XY,
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
# 旧 Place 回退：TCP 停在灰筐沿上方，不要 IK 贴桌（腕会抖）。
PLACE_RELEASE_ABOVE_TABLE = 0.08
# 新 05→15 Place：15 展开后只跟世界 Z 往下压，肩膀可转，X/Y 允许偏移。
# 初始目标让 TCP 停在灰筐上沿 20 mm；阿里云实测前的保守可调值。
PLACE_FINAL_TCP_ABOVE_TRAY = 0.020

# 按 s 到右臂起动之间的等待就是这三个计时器。SETTLE / GOTO_PICK 全程由
# _apply_snap 每帧把骨盆写成 STAND_PELVIS_Z、速度清零，所以 is_standing()
# 从第一帧就为真 —— 原来的 1.0/0.6/0.5（合计 2.1 s）纯粹是空烧。
# 缩短的只是计时门槛，is_standing() 的判定本身没动：真站不稳仍会一直等，
# 最坏由 _navigate 的 6 s 超时兜底。若改成非 snap 起步（机器人要自己走到
# 抓取站、或 spawn 时会晃），把这三个值调回 1.0/0.6/0.5。
SETTLE_TIME = 0.3
STAND_MIN_TIME = 0.2
STAND_STABLE_TIME = 0.2
STAND_TILT_MAX = 0.18
STAND_YAW_RATE_MAX = 0.45
STAND_XY_SPEED_MAX = 0.12
APPROACH_TIME = 2.8
UNFOLD_TIME = 3.2
ORIENT_TIME = 2.2
SLIDE_TIME = 2.4
DESCEND_TIME = 1.1
# Pose 30 leaves the Dex1 jaw ~68° off world X.  Hover (lock XY/Z) and slerp
# yaw onto ±X before the Z drop, so DiffIK does not twist while entering the tray.
GRASP_YAW_ALIGN_TIME = 0.55
GRASP_TIME = 1.0
GRASP_POS_TOL = 0.055
GRASP_WAIT_MAX = 0.6
LIFT_TIME = 2.2
# 旧清单回退：抬起后先从实时 q 过渡到第一个逆向路点。smooth_v1 已把同一
# 0.8 s 写进 manifest 的 40(live)→30 return segment，随后走30→20→05避开抽屉边缘。
RETURN_LEAD_IN_TIME = 0.8
RETURN_TIME = 2.5  # 无关节路点时：单段回默认臂姿
CARRY_TIME = 0.6  # 冻臂 q，不要再笛卡尔收臂（件会掉）
HOLD_TIME = 0.3
# 旧清单回退：snap 换站前先在抓取站把件抬过桌面高度。Smooth V1 的新
# Place 清单保持胸前 05 直接换站，到站后再走人工 05→15，不使用这个常量。
# q：肩俯仰上抬、肘略收、roll/yaw/腕贴 00，避免外翻和大腕旋。
PLACE_PRE_RAISE_Q = (-0.60, -0.20, 0.00, 1.15, 0.00, 0.00, 0.00)
PLACE_PRE_RAISE_TIME = 1.6
# 旧清单回退：从抬臂姿态先升到放置高度，再水平伸到灰筐上方。
PLACE_RAISE_FORWARD = 0.06  # 抬臂段顺机体前方挪一点，避免死折肘
PLACE_RAISE_TIME = 1.6
PLACE_REACH_TIME = 2.2
PLACE_APPROACH_TIME = PLACE_RAISE_TIME + PLACE_REACH_TIME
# 放置 IK 只跟位置（不给朝向目标），靠下面的窗口把姿态锁在携带臂姿附近：
# 肩内外旋/肩偏摆贴住初始值 → 肘不外翻；腕三轴几乎不转。
PLACE_ROLL_WINDOW = 0.45
PLACE_YAW_WINDOW = 0.50
PLACE_WRIST_WINDOW = 0.30
PLACE_ELBOW_MIN = 0.20  # 肘不许伸直锁死
PLACE_DESCEND_TIME = 1.2
RELEASE_TIME = 0.8
# 收臂：按放置轨迹的逆序（灰筐上方 → 抬臂点 → 携带姿态），别横扫桌沿。
RETRACT_TIME = 1.6
RETRACT_HOME_TIME = 1.6

# 旧 FSM 垂臂种子，顺序同 RIGHT_ARM_JOINTS。
RIGHT_ARM_READY = (0.40, -0.42, 0.18, 1.20, 0.0, 0.95, 0.0)
ARM_SLEW_RAD = 0.080
ARM_SLEW_RAD_LIFT = 0.012  # 夹持后慢跟，垫面不瞬移
STOP_AFTER = "place"

# 路点 JSON 按关节名匹配，不改 DDS 下标。smooth：00→10→20→30 连续速度，
# 30→40 只作 q_ref；natural_v2 / natural_v1 仍可通过 CLI 回退。
WAYPOINT_SET_DEFAULT = "ces_pick_smooth_v1"
WAYPOINT_LEAD_IN_TIME = 0.25
WAYPOINT_LEAD_IN_TOL = 0.05

# Pick 提速：UNFOLD(00→30) / LIFT / RETURN_HOME 这三段关节轨迹的时长整体除以
# 这个倍率。均匀时间缩放不改关节空间曲线，只把每个速度乘以倍率，所以 URDF-viz
# 里确认过的姿态和 monotone_cubic_hermite 的连续性都原样保留。
# DESCEND / GRASP 不缩放：一个是落 Z 的对位精度，一个是夹爪闭合时间。
PICK_SPEED_SCALE = 1.5
PICK_SPEED_MIN = 0.25
# 关节下发被 _slew_arm 限在 ARM_SLEW_RAD/dt = 0.080/0.02 = 4.0 rad/s，而
# smooth_v1 原速峰值只有 0.89 rad/s，约 4.5 倍处轨迹才开始被截断。上限留一档
# 余量，也因为件只靠垫面摩擦夹着，抬臂/回臂再快会甩脱。
PICK_SPEED_MAX = 3.0
# 缩放后单段时长的下限，避免高倍率把某一段压成阶跃。
PICK_SEGMENT_MIN_TIME = 0.40


def clamp_pick_speed(scale: float | None) -> float:
    if scale is None:
        return PICK_SPEED_SCALE
    return min(PICK_SPEED_MAX, max(PICK_SPEED_MIN, float(scale)))

# HOLD 后的双足换站。策略命令是机体系 [vx, vy, wz, height]。
# 关键约束：策略对小指令不迈步（键盘点动走不动，必须长按把指令拉起来），
# 所以平移/转向都用固定幅值 + 死区，不用比例控制。幅值参考键盘长按能走的量级。
WALK_VX = 0.45  # 前进/后退幅值；< 0.3 基本只前后晃不迈步
WALK_REVERSE_VX = 0.45
# 侧移纠偏幅值。键盘 y_vel 上限 0.5，这里取 80%，与 vx/wz 各取上限 ~75% 一致。
# 之前的 0.30 离死区只剩 20% 余量（vx 有 50%、wz 有 200%），实机侧移响应稍弱
# 就整条被死区吃掉 —— 表现为 lat 一直冻在 0.12 不动、来回踱步。
WALK_VY = 0.40
# 键盘 yaw_vel 上限 1.57（send_commands_keyboard.py），后退用了上限的 75%，
# 转向也按同一比例给，之前的 0.70 只有 45%，策略吃不动。
WALK_WZ = 1.20
WALK_WZ_MAX = 1.55  # 漂移保护触发后的升级幅值，仍在键盘上限内
# 纯偏航（vx=vy=0）顶不起步态：转向段一直带着后退走，画一段弧把身子转过去。
# 用后退而不是前进：后退是唯一实测能迈步的模式，且第③段因此保留近 1 m 行程。
WALK_TURN_VX = 0.45
WALK_TURN_MAX_DRIFT = 0.70  # 转弧走了这么远还没转到位 = 策略没吃下 wz
# 指令归零后策略还会多走一点：提前 stop_margin 松"油门"。
# 若实测停不到位（走过头撞桌 / 差太多够不到灰筐），只调这两个值。
WALK_STOP_MARGIN = 0.15
# 最后一段：宁可停短，绝不许过。放置站骨盆离桌沿现在只有约 2 cm，走过头直接撞桌。
# 新 Place 不再追世界灰筐中心：05→15 后只落 Z，所以停位 XY 误差不会由手臂补偿；
# 用户明确接受不设 TCP X/Y 目标，安全上仍以不越过桌沿为第一优先级。
# 0.20 撞过桌：`vx=0.45` 松手后滑行约 0.25 m，比余量还大，到 0.20 才停必然冲过去。
# 又因为 `vx` 不能降到死区以下（低了根本不迈步），没法缓刹，只能提前松手。
# 取 0.30 ≈ 滑行量：滑行 0.10~0.30 时落点在放置站前 0.20~0.00，**不会越过站点**。
WALK_STOP_MARGIN_PLACE = 0.30
# 桌沿在放置站正前方约这么远（骨盆投影）。只用于硬性禁入判定，不参与规划。
# 2026-08-24 桌 −Y 12 cm 后近沿约在放置站前 2 cm（HeavyDuty 半宽 0.381）。
WALK_TABLE_AHEAD_OF_STAND = 0.02
# 骨盆离桌沿的最小安全距离：越过 (放置站 + AHEAD - SAFE) 就无条件停死并告警。
# 这是兜底闩锁，不依赖任何规划逻辑 —— 前两次撞桌都是规划分支漏了停止条件。
WALK_TABLE_SAFE = 0.06
# 后退段少退一个转弧半径，再提前一点给 wz 爬升和滑行留余量。
# 理论 R=vx/wz=0.375 m；实测转得太晚会把世界 X 走出放置站，后面还要用 −X 往回纠。
WALK_TURN_LEAD = WALK_TURN_VX / WALK_WZ + 0.20
# 距 backoff 目标这么远就开始带 wz 预转，转完时 X 才能落在放置站进入线上。
WALK_TURN_PREVIEW = 0.25
# 和 WALK_TURN_LEAD 同轴同向：lead 已经承担了"宁可退不够"，这里不用再收。
WALK_BACKOFF_TRIM = 0.0
# 转向的 stop_margin。实测（2026-08-19）在 0.25 处松手后还会多转约 14°，
# 也就是余转合计约 28°，所以提前到约 23° 松手。
WALK_YAW_ARRIVE = 0.40
WALK_LATERAL_TOL = 0.10  # 侧向偏差超过才侧移纠偏（回差：降到 0.04 才松手）
# 最后一段"希望"停成的姿态窗口：侧向 / 朝向。**只用于报告，不参与判停** ——
# 桌子就在放置站前 12 cm，为了摆正而继续在桌边挪动正是前两次撞桌的原因。
# 没进窗口只会打 off-target 告警，姿态靠进站途中的斜行去纠。
WALK_LATERAL_ARRIVE = 0.10
# 平移中朝向纠偏死区。实测转弧收尾会留 ~14° 残差，0.30（17°）够不到它，
# 于是 wz 一直是 0、歪着走完最后一段；收到 0.20（11.5°）让它能被纠回来。
WALK_ALIGN_YAW = 0.20
WALK_YAW_ARRIVE_FINAL = WALK_ALIGN_YAW
WALK_REALIGN_YAW = 0.60  # 歪太多：停下平移，先转正
WALK_LEG_SETTLE = 0.5  # 每段之间零指令停稳，避免后退接右转时混合指令
WALK_ARRIVE_HOLD = 0.35
# walk 到站后先钉盆、冻 05、夹爪不动，再开始 05→15。用来分流：
# 一钉就掉 = 动态根切运动学钉盆打断接触；钉住还能拿着、一伸 15 才掉 = 关节轨迹。
WALK_PLACE_HOLD_TIME = 1.5
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

# 放置站面向桌子（yaw=π/2）。站位按桌心 **Y 偏移之前** 的位置算，不跟
# 2026-08-24 的桌子 −Y 平移走，否则 15 的 +Y 伸手距离原样不变。
# PLACE_TARGET_XY 才是灰筐中心（随桌子一起 −Y）。
PLACE_STAND_YAW = 0.5 * math.pi
PLACE_TARGET_XY = PLACE_TRAY_CENTER_XY
_PLACE_STAND_FROM_XY = (-2.0869, -0.3117)
PLACE_STAND_XY = stand_xy(_PLACE_STAND_FROM_XY, PLACE_STAND_YAW, X_B_PLACE, Y_B_PLACE)
# 旧清单的世界 XY+Z 目标；Smooth V1 不再追这个 XY。
PLACE_Z = TABLE_TOP_Z + PLACE_TRAY_HEIGHT + PLACE_RELEASE_ABOVE_TABLE
# Smooth V1 在 15 到位后只向下逼近这个世界 Z；X/Y 不锁，肩膀可带动手臂下压。
# 若 15 已经低于该高度，状态机不会反向抬升，也不会继续向下压。
PLACE_FINAL_TCP_Z = TABLE_TOP_Z + PLACE_TRAY_HEIGHT + PLACE_FINAL_TCP_ABOVE_TRAY

# HOLD 后的换站路线（机器人在 pick 站朝 -X，后退即走世界 +X）：
# ① 后退到"转弧入弧点"（与放置站对齐的角点再少退一个转弧半径）
# ② 边后退边右转 yaw π → π/2，弧终点落回放置站进入线，正对桌子
# ③ 不停步，直接发 W（机体系 +vx）；转正后这就是世界 +Y，走进放置站
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
    turn_vx=WALK_TURN_VX,
    reverse_vx=WALK_REVERSE_VX,
    wz_max=WALK_WZ_MAX,
    turn_max_drift=WALK_TURN_MAX_DRIFT,
    turn_preview=WALK_TURN_PREVIEW,
    lateral_arrive=WALK_LATERAL_ARRIVE,
    yaw_arrive=WALK_YAW_ARRIVE_FINAL,
)
CARRY_WALK_LEGS = build_carry_route(
    pick_xy=PICK_STAND_XY,
    pick_yaw=PICK_STAND_YAW,
    place_xy=PLACE_STAND_XY,
    place_yaw=PLACE_STAND_YAW,
    place_stop_margin=WALK_STOP_MARGIN_PLACE,
    backoff_trim=WALK_BACKOFF_TRIM,
    turn_lead=WALK_TURN_LEAD,
)
WALK_BACKOFF_XY = CARRY_WALK_LEGS[0].target_xy

TABLE_REWARD_HALF_XY = (1.15, 0.40)
DROP_HEIGHT = 0.32
PLACE_HEIGHT_MIN = TABLE_TOP_Z + 0.01
PLACE_HEIGHT_MAX = TABLE_TOP_Z + PLACE_TRAY_HEIGHT + 0.12
