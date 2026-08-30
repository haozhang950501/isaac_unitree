# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""CES 持物换站的固定三段机体系导航器。

CES Baseline 的路线只有三段：从上料机前向后退出、保持后退并右转、
最后沿机体系 ``+vx`` 前进到放置站。这里不再提供任意路段编排能力，
避免用一套通用框架表达唯一一条已经通过仿真的路线。

双足策略存在两个必须保留的特点：小速度只会原地晃动，纯 ``vy`` 或
纯 ``wz`` 也无法启动步态。因此导航器只发送经过验证的固定幅值；所有
朝向和侧向修正都必须搭载有效 ``vx``，不能按剩余距离做比例降速。

最终进站以停止线为最高优先级。机器人即使尚未完全摆正，只要抵达
停止线就永久停止，防止为了修正姿态继续走入紧邻放置站的桌面。
"""
from __future__ import annotations

import enum
import math
from dataclasses import dataclass


def wrap_angle(angle: float) -> float:
    """把弧度角归一化到 ``[-pi, pi]``。"""
    return math.atan2(math.sin(angle), math.cos(angle))


class CarryWalkPhase(enum.Enum):
    """CES 固定换站路线的运行阶段。"""

    BACKOFF = "backoff"
    TURN = "turn_to_table"
    APPROACH = "approach_place"
    DONE = "done"

    @property
    def index(self) -> int:
        """返回用于日志显示的零基阶段序号。"""
        return tuple(type(self)).index(self)


@dataclass(frozen=True)
class CarryWalkConfig:
    """双足策略使用的固定指令幅值和安全死区。

    ``vx``、``vy``、``wz`` 是策略能稳定起步的指令幅值；
    ``reverse_vx`` 和 ``turn_vx`` 分别控制直线后退与转弧后退。
    ``stop_margin`` 用于补偿指令归零后的滑行距离。其余阈值只负责
    迟滞修正和到站报告，不允许覆盖最终停止线。
    """

    vx: float
    vy: float
    wz: float
    stop_margin: float
    yaw_tol: float
    lateral_tol: float
    align_yaw: float
    realign_yaw: float
    height: float
    turn_vx: float
    reverse_vx: float
    wz_max: float
    turn_max_drift: float
    turn_preview: float
    lateral_arrive: float
    yaw_arrive: float


@dataclass(frozen=True)
class CarryRoute:
    """由场景坐标确定的 CES 三段换站路线。"""

    pick_xy: tuple[float, float]
    pick_yaw: float
    backoff_xy: tuple[float, float]
    place_xy: tuple[float, float]
    place_yaw: float
    place_stop_margin: float


@dataclass(frozen=True)
class CarryWalkStep:
    """单帧步态命令及其对应的路线误差。

    ``remaining`` 是沿当前行进轴的剩余距离，负值表示已经越线；
    ``lateral`` 是机体系左向误差；``on_target`` 仅用于报告最终姿态，
    不会让机器人越过停止线继续纠偏。
    """

    command: tuple[float, float, float, float]
    phase: CarryWalkPhase
    mode: str
    remaining: float
    lateral: float
    yaw_error: float
    done: bool
    on_target: bool = True


def _signed(magnitude: float, error: float) -> float:
    """按误差方向返回固定幅值，零误差沿负方向但不会被实际调用。"""
    return magnitude if error > 0.0 else -magnitude


def _travel_error(
    target_xy: tuple[float, float],
    travel_yaw: float,
    direction: float,
    x: float,
    y: float,
    yaw: float,
) -> tuple[float, float, float]:
    """计算行进轴剩余距离、机体系侧向误差和目标朝向误差。

    ``direction`` 为 ``+1`` 表示沿目标朝向前进，为 ``-1`` 表示沿
    目标朝向后退。剩余距离投影到固定世界行进轴，因此侧向漂移不会
    让某个阶段永远无法结束。
    """
    axis = (
        direction * math.cos(travel_yaw),
        direction * math.sin(travel_yaw),
    )
    dx, dy = target_xy[0] - x, target_xy[1] - y
    remaining = dx * axis[0] + dy * axis[1]
    lateral = dx * (-math.sin(yaw)) + dy * math.cos(yaw)
    return remaining, lateral, wrap_angle(travel_yaw - yaw)


def _line_crossing(
    point_a: tuple[float, float],
    direction_a: tuple[float, float],
    point_b: tuple[float, float],
    direction_b: tuple[float, float],
) -> float | None:
    """返回第一条直线到两条世界坐标直线交点的有向距离。"""
    det = direction_b[0] * direction_a[1] - direction_a[0] * direction_b[1]
    if abs(det) < 1e-6:
        return None
    dx, dy = point_b[0] - point_a[0], point_b[1] - point_a[1]
    return (direction_b[0] * dy - dx * direction_b[1]) / det


def build_carry_route(
    *,
    pick_xy: tuple[float, float],
    pick_yaw: float,
    place_xy: tuple[float, float],
    place_yaw: float,
    place_stop_margin: float,
    backoff_trim: float = 0.0,
    turn_lead: float = 0.0,
) -> CarryRoute:
    """根据抓取站和放置站计算后退转弧的入弧点。

    后退线与放置站进站线的交点是理论转角。由于转向阶段仍在后退，
    机器人会画半径约为 ``turn_vx / wz`` 的圆弧，所以用
    ``turn_lead`` 提前结束直线后退，让圆弧末端回到进站线。
    ``backoff_trim`` 只允许进一步缩短后退距离，避免机器人靠近桌面。
    """
    back_direction = (-math.cos(pick_yaw), -math.sin(pick_yaw))
    place_direction = (math.cos(place_yaw), math.sin(place_yaw))
    distance = _line_crossing(
        pick_xy, back_direction, place_xy, place_direction
    )
    if distance is None or distance <= 0.0:
        raise ValueError("place stand is not reachable by backing out of the pick stand")
    distance = max(
        0.0,
        distance - max(0.0, backoff_trim) - max(0.0, turn_lead),
    )
    backoff_xy = (
        pick_xy[0] + distance * back_direction[0],
        pick_xy[1] + distance * back_direction[1],
    )
    return CarryRoute(
        pick_xy=pick_xy,
        pick_yaw=pick_yaw,
        backoff_xy=backoff_xy,
        place_xy=place_xy,
        place_yaw=place_yaw,
        place_stop_margin=place_stop_margin,
    )


class CarryWalkPlanner:
    """执行 CES 固定后退、转弧、进站路线的命令生成器。

    类内部只保存当前阶段、迟滞开关和转弧起点。阶段切换帧会立即返回
    下一阶段的有效速度，不插入零指令，避免步态策略停下后无法重新起步。
    """

    ACTIVE_PHASE_COUNT = 3

    def __init__(self, route: CarryRoute, config: CarryWalkConfig):
        """绑定固定场景路线和指令参数，并从 BACKOFF 阶段启动。"""
        self.route = route
        self.config = config
        self.reset()

    def reset(self) -> None:
        """从直线后退阶段重新开始，并清空所有迟滞状态。"""
        self.phase = CarryWalkPhase.BACKOFF
        self._fix_yaw = False
        self._fix_lateral = False
        self._phase_start_xy: tuple[float, float] | None = None

    def step(self, x: float, y: float, yaw: float, dt: float = 0.0) -> CarryWalkStep:
        """根据当前骨盆位姿生成一帧机体系步态命令。

        ``dt`` 为兼容调用方保留；固定路线没有阶段等待，因此不参与计算。
        同一帧最多跨越三个有效阶段，确保越过边界时直接交接下一条命令。
        """
        del dt
        if self._phase_start_xy is None:
            self._phase_start_xy = (x, y)

        for _ in range(self.ACTIVE_PHASE_COUNT):
            if self.phase is CarryWalkPhase.BACKOFF:
                remaining, lateral, yaw_error = _travel_error(
                    self.route.backoff_xy,
                    self.route.pick_yaw,
                    -1.0,
                    x,
                    y,
                    yaw,
                )
                if remaining > self.config.stop_margin:
                    return self._drive_backoff(
                        remaining, lateral, yaw_error, yaw
                    )
                self._advance(CarryWalkPhase.TURN, x, y)
                continue

            if self.phase is CarryWalkPhase.TURN:
                yaw_error = wrap_angle(self.route.place_yaw - yaw)
                if abs(yaw_error) > self.config.yaw_tol:
                    return self._drive_turn(x, y, yaw_error)
                self._advance(CarryWalkPhase.APPROACH, x, y)
                continue

            if self.phase is CarryWalkPhase.APPROACH:
                remaining, lateral, yaw_error = _travel_error(
                    self.route.place_xy,
                    self.route.place_yaw,
                    1.0,
                    x,
                    y,
                    yaw,
                )
                if remaining > self.route.place_stop_margin:
                    return self._drive_approach(
                        remaining, lateral, yaw_error
                    )
                self.phase = CarryWalkPhase.DONE
                return self._zero(
                    "arrived", remaining, lateral, yaw_error, done=True
                )

            break

        remaining, lateral, yaw_error = _travel_error(
            self.route.place_xy,
            self.route.place_yaw,
            1.0,
            x,
            y,
            yaw,
        )
        self.phase = CarryWalkPhase.DONE
        return self._zero("arrived", remaining, lateral, yaw_error, done=True)

    def _advance(self, phase: CarryWalkPhase, x: float, y: float) -> None:
        """切换阶段并重置该阶段独立的迟滞与漂移基准。"""
        self.phase = phase
        self._fix_yaw = False
        self._fix_lateral = False
        self._phase_start_xy = (x, y)

    def _on_target(self, lateral: float, yaw_error: float) -> bool:
        """判断最终姿态是否进入报告窗口，不参与停止决策。"""
        square = (
            self.config.lateral_arrive <= 0.0
            or abs(lateral) <= self.config.lateral_arrive
        )
        aimed = (
            self.config.yaw_arrive <= 0.0
            or abs(yaw_error) <= self.config.yaw_arrive
        )
        return square and aimed

    def _hysteresis(self, active: bool, error: float, enter: float) -> bool:
        """为固定幅值修正增加回差，避免在阈值附近来回切换。"""
        return abs(error) > (0.4 * enter if active else enter)

    def _phase_drift(self, x: float, y: float) -> float:
        """返回当前阶段起点到实时骨盆位置的平面距离。"""
        if self._phase_start_xy is None:
            return 0.0
        return math.hypot(
            x - self._phase_start_xy[0], y - self._phase_start_xy[1]
        )

    def _drive_backoff(
        self,
        remaining: float,
        lateral: float,
        yaw_error: float,
        yaw: float,
    ) -> CarryWalkStep:
        """生成直线后退命令，并在入弧点前提前叠加偏航。"""
        config = self.config
        if config.turn_preview > 0.0 and remaining <= config.turn_preview:
            turn_error = wrap_angle(self.route.place_yaw - yaw)
            return self._make(
                (-config.reverse_vx, 0.0, _signed(config.wz, turn_error)),
                "reverse_preturn",
                remaining,
                lateral,
                yaw_error,
            )

        if abs(yaw_error) > config.realign_yaw:
            self._fix_yaw = True
            return self._make(
                (-config.turn_vx, 0.0, _signed(config.wz, yaw_error)),
                "reverse_realign",
                remaining,
                lateral,
                yaw_error,
            )

        self._fix_yaw = self._hysteresis(
            self._fix_yaw, yaw_error, config.align_yaw
        )
        self._fix_lateral = self._hysteresis(
            self._fix_lateral, lateral, config.lateral_tol
        )
        wz = _signed(config.wz, yaw_error) if self._fix_yaw else 0.0
        vy = _signed(config.vy, lateral) if self._fix_lateral else 0.0
        return self._make(
            (-config.reverse_vx, vy, wz),
            "reverse",
            remaining,
            lateral,
            yaw_error,
        )

    def _drive_turn(self, x: float, y: float, yaw_error: float) -> CarryWalkStep:
        """保持有效后退速度完成转弧，漂移过大时反向平移重试。"""
        config = self.config
        if (
            config.turn_max_drift > 0.0
            and self._phase_drift(x, y) > config.turn_max_drift
        ):
            command = (
                config.turn_vx,
                0.0,
                _signed(config.wz_max, yaw_error),
            )
            mode = "turn_pinned"
        else:
            command = (
                -config.turn_vx,
                0.0,
                _signed(config.wz, yaw_error),
            )
            mode = "turn"
        return self._make(command, mode, 0.0, 0.0, yaw_error)

    def _drive_approach(
        self, remaining: float, lateral: float, yaw_error: float
    ) -> CarryWalkStep:
        """沿机体系前向轴进站；允许带偏航，但禁止侧移和反向。"""
        self._fix_yaw = self._hysteresis(
            self._fix_yaw, yaw_error, self.config.align_yaw
        )
        wz = _signed(self.config.wz, yaw_error) if self._fix_yaw else 0.0
        return self._make(
            (self.config.vx, 0.0, wz),
            "forward",
            remaining,
            lateral,
            yaw_error,
        )

    def _zero(
        self,
        mode: str,
        remaining: float,
        lateral: float,
        yaw_error: float,
        *,
        done: bool,
    ) -> CarryWalkStep:
        """构造停止命令，并附带最终姿态是否达标的诊断信息。"""
        return CarryWalkStep(
            command=(0.0, 0.0, 0.0, self.config.height),
            phase=self.phase,
            mode=mode,
            remaining=remaining,
            lateral=lateral,
            yaw_error=yaw_error,
            done=done,
            on_target=self._on_target(lateral, yaw_error),
        )

    def _make(
        self,
        command: tuple[float, float, float],
        mode: str,
        remaining: float,
        lateral: float,
        yaw_error: float,
    ) -> CarryWalkStep:
        """给三轴速度补上固定站立高度，形成策略的四维命令。"""
        return CarryWalkStep(
            command=(*command, self.config.height),
            phase=self.phase,
            mode=mode,
            remaining=remaining,
            lateral=lateral,
            yaw_error=yaw_error,
            done=False,
        )
