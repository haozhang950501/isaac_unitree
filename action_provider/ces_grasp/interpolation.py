# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""使用 PyTorch 与 Isaac Lab 基础算子的 CES 轨迹插值器。

笛卡尔位置采用分段 smoothstep，姿态采用 Isaac Lab 的四元数球面插值；
关节路径同时支持分段 smoothstep 和保持形状的单调三次 Hermite。
插值器只保存输入 Tensor，不自行搬运设备或修改原始路点。
"""
from __future__ import annotations

from bisect import bisect_left

import torch


def ease_in_out(value: float) -> float:
    """对归一化标量执行三次 smoothstep，并把输入限制在 ``[0, 1]``。"""
    value = max(0.0, min(1.0, value))
    return value * value * (3.0 - 2.0 * value)


def scale_segment_times(
    durations, scale: float, min_time: float = 0.0
) -> list[float]:
    """统一缩放各段时长，不改变关节空间曲线及路点数值。"""
    factor = max(1e-3, float(scale))
    return [max(float(min_time), float(duration) / factor) for duration in durations]


def _bounds(durations: list[float]) -> tuple[list[float], float]:
    """把各段时长转换为累计结束时间，并返回总时长。"""
    bounds: list[float] = []
    total = 0.0
    for duration in durations:
        total += max(1e-3, float(duration))
        bounds.append(total)
    return bounds, total


def _segment(bounds: list[float], elapsed: float) -> tuple[int, float, float]:
    """用二分查找定位当前段，并返回段索引、段长和归一化进度。"""
    index = min(bisect_left(bounds, elapsed), len(bounds) - 1)
    start = 0.0 if index == 0 else bounds[index - 1]
    length = max(1e-6, bounds[index] - start)
    phase = max(0.0, min(1.0, (elapsed - start) / length))
    return index, length, phase


def _quat_slerp(q0: torch.Tensor, q1: torch.Tensor, phase: float) -> torch.Tensor:
    """逐环境调用 Isaac Lab 的单四元数 SLERP。

    Isaac Lab 当前接口不支持批处理，因此这里保留显式逐环境调用；CES
    运行时只有一个环境，不会形成可观测的性能负担。
    """
    from isaaclab.utils.math import quat_slerp

    return torch.stack(
        [quat_slerp(a, b.clone(), phase) for a, b in zip(q0, q1)], dim=0
    )


def _monotone_cubic_slopes(
    points: list[torch.Tensor], durations: list[float]
) -> list[torch.Tensor]:
    """计算保持单调形状的 C1 路点速度，并令路径两端速度为零。"""
    widths = [max(1e-3, float(duration)) for duration in durations]
    secants = [
        (points[index + 1] - points[index]) / widths[index]
        for index in range(len(widths))
    ]
    slopes = [torch.zeros_like(point) for point in points]
    for index in range(1, len(points) - 1):
        before, after = secants[index - 1], secants[index]
        same_direction = before * after > 0.0
        weight_before = 2.0 * widths[index] + widths[index - 1]
        weight_after = widths[index] + 2.0 * widths[index - 1]
        slope = torch.zeros_like(points[index])
        slope[same_direction] = (weight_before + weight_after) / (
            weight_before / before[same_direction]
            + weight_after / after[same_direction]
        )
        slopes[index] = slope
    return slopes


class CartesianInterpolator:
    """按时间推进笛卡尔分段路径，并对每段姿态执行四元数插值。"""

    def __init__(self):
        """创建空笛卡尔路径；调用 ``reset_path`` 后才开始输出。"""
        self.points: list[torch.Tensor] = []
        self.quats: list[torch.Tensor] = []
        self.bounds: list[float] = []
        self.duration = 1.0
        self.elapsed = 0.0

    def reset_path(
        self,
        points: list[torch.Tensor],
        durations: list[float],
        start_quat: torch.Tensor | None = None,
        goal_quat: torch.Tensor | None = None,
        quats: list[torch.Tensor] | None = None,
    ) -> None:
        """装载一条分段路径，并校验位置、时长与姿态数量的一致性。"""
        if len(points) != len(durations) + 1:
            raise ValueError(
                f"{len(points)} Cartesian points require {len(points) - 1} durations"
            )
        self.points = [point.clone() for point in points]
        self.bounds, self.duration = _bounds(durations)
        self.elapsed = 0.0
        if quats is not None:
            if len(quats) != len(points):
                raise ValueError(f"{len(points)} Cartesian points require {len(points)} quaternions")
            self.quats = [quat.clone() for quat in quats]
        elif start_quat is not None and goal_quat is not None:
            self.quats = [start_quat.clone()] + [
                goal_quat.clone() for _ in range(len(points) - 1)
            ]
        else:
            self.quats = []

    @property
    def has_path(self) -> bool:
        """当前是否装载了至少一段有效路径。"""
        return bool(self.bounds) and len(self.points) >= 2

    @property
    def finished(self) -> bool:
        """当前路径是否为空或已推进到总时长。"""
        return not self.has_path or self.elapsed >= self.duration

    def step(self, dt: float) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """推进一个控制周期，返回当前位置与姿态；无路径时返回空值。"""
        if not self.has_path:
            return None, None
        self.elapsed = min(self.duration, self.elapsed + dt)
        index, _length, phase = _segment(self.bounds, self.elapsed)
        phase = ease_in_out(phase)
        pos = torch.lerp(self.points[index], self.points[index + 1], phase)
        quat = (
            _quat_slerp(self.quats[index], self.quats[index + 1], phase)
            if self.quats
            else None
        )
        return pos, quat


class JointSpaceInterpolator:
    """支持 smoothstep 和单调 Hermite 的关节空间分段插值器。"""

    _METHODS = {"segment_smoothstep", "monotone_cubic_hermite"}

    def __init__(self):
        """创建空关节路径，并初始化默认插值状态。"""
        self.clear()

    def reset(
        self,
        start_q: torch.Tensor,
        goal_q: torch.Tensor,
        duration: float,
        method: str = "segment_smoothstep",
    ) -> None:
        """装载仅含起点和终点的一段关节轨迹。"""
        self.reset_path([start_q, goal_q], [duration], method=method)

    def reset_path(
        self,
        qs: list[torch.Tensor],
        durations: list[float],
        method: str = "segment_smoothstep",
    ) -> None:
        """装载多路点关节轨迹，并按指定方法预计算插值数据。"""
        if len(qs) != len(durations) + 1:
            raise ValueError(f"{len(qs)} joint waypoints require {len(qs) - 1} durations")
        if method not in self._METHODS:
            raise ValueError(f"unsupported CES interpolation method {method!r}")
        self.points = [q.clone() for q in qs]
        self.bounds, self.duration = _bounds(durations)
        self.elapsed = 0.0
        self.method = method
        self.slopes = (
            _monotone_cubic_slopes(self.points, durations)
            if method == "monotone_cubic_hermite"
            else []
        )

    def clear(self) -> None:
        """清空当前路径，使下一次 ``step`` 安全返回空值。"""
        self.points: list[torch.Tensor] = []
        self.bounds: list[float] = []
        self.duration = 1.0
        self.elapsed = 0.0
        self.method = "segment_smoothstep"
        self.slopes: list[torch.Tensor] = []

    @property
    def has_path(self) -> bool:
        """当前是否装载了至少一段有效关节路径。"""
        return bool(self.bounds) and len(self.points) >= 2

    @property
    def finished(self) -> bool:
        """当前关节路径是否为空或已经完成。"""
        return not self.has_path or self.elapsed >= self.duration

    def step(self, dt: float) -> torch.Tensor | None:
        """推进一个控制周期并返回关节目标；无路径时返回空值。"""
        if not self.has_path:
            return None
        self.elapsed = min(self.duration, self.elapsed + dt)
        index, length, phase = _segment(self.bounds, self.elapsed)
        if self.method == "monotone_cubic_hermite":
            phase2, phase3 = phase * phase, phase * phase * phase
            return (
                (2.0 * phase3 - 3.0 * phase2 + 1.0) * self.points[index]
                + (phase3 - 2.0 * phase2 + phase) * length * self.slopes[index]
                + (-2.0 * phase3 + 3.0 * phase2) * self.points[index + 1]
                + (phase3 - phase2) * length * self.slopes[index + 1]
            )
        return torch.lerp(
            self.points[index], self.points[index + 1], ease_in_out(phase)
        )
