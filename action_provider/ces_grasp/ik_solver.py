# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""带动态零空间姿态参考的 CES 右臂全位姿差分 IK。

该求解器只实现 CES 实际使用的控制契约：同时跟踪 TCP 世界位置和姿态，
并可把 40 姿态作为动态 ``q_ref`` 投影到任务零空间。40 从不作为手臂
关节目标硬下发。官方通用控制器没有这个动态参考契约，因此保留 CES
专用的阻尼最小二乘实现。
"""
from __future__ import annotations

import torch
from isaaclab.utils.math import (
    compute_pose_error,
    matrix_from_quat,
    quat_apply,
    quat_inv,
    subtract_frame_transforms,
)


def _skew(vector: torch.Tensor) -> torch.Tensor:
    """把批量三维向量转换为叉乘使用的反对称矩阵。"""
    x, y, z = vector[:, 0], vector[:, 1], vector[:, 2]
    zero = torch.zeros_like(x)
    return torch.stack(
        (
            torch.stack((zero, -z, y), dim=-1),
            torch.stack((z, zero, -x), dim=-1),
            torch.stack((-y, x, zero), dim=-1),
        ),
        dim=-2,
    )


class ArmDiffIK:
    """面向 CES TCP 控制契约的阻尼最小二乘右臂 IK。

    求解过程全部在机器人当前设备上执行。阻尼矩阵、任务权重和零空间
    单位阵在构造时一次创建，避免 DESCEND 每帧重复分配固定 Tensor。
    """

    def __init__(
        self,
        robot,
        arm_joint_names: list[str],
        ee_body_name: str,
        device: str,
        damping: float = 0.08,
        gain: float = 1.0,
        max_delta_per_step: float = 0.06,
        tcp_offset: tuple[float, float, float] | None = None,
        w_pos: float = 1.0,
        w_rot: float = 0.3,
        max_iters: int = 6,
        pos_tol: float = 0.005,
    ):
        """解析关节/末端索引，并预创建求解所需的常量 Tensor。"""
        self.robot = robot
        self.device = device
        self.gain = float(gain)
        self.max_delta = float(max_delta_per_step)
        self.max_iters = max(1, int(max_iters))
        self.pos_tol = float(pos_tol)
        self.tcp_offset = torch.tensor(
            tcp_offset or (0.0, 0.0, 0.0),
            device=device,
            dtype=torch.float32,
        )
        self._has_tcp_offset = bool(
            float(torch.linalg.vector_norm(self.tcp_offset)) > 1e-8
        )

        joint_ids, _ = robot.find_joints(
            arm_joint_names,
            preserve_order=True,
        )
        self.joint_ids = list(joint_ids)
        body_ids, body_names = robot.find_bodies(
            ee_body_name,
            preserve_order=True,
        )
        if len(body_ids) != 1:
            raise ValueError(
                f"expected one end-effector body for {ee_body_name!r}, got {body_names}"
            )
        self.body_idx = int(body_ids[0])
        if robot.is_fixed_base:
            self.jacobi_body_idx = self.body_idx - 1
            self.jacobi_joint_ids = self.joint_ids
        else:
            self.jacobi_body_idx = self.body_idx
            self.jacobi_joint_ids = [joint_id + 6 for joint_id in self.joint_ids]

        limits = robot.data.joint_pos_limits[0, self.joint_ids].to(device)
        self.q_min, self.q_max = limits[:, 0], limits[:, 1]
        self._task_weights = torch.tensor(
            [float(w_pos)] * 3 + [float(w_rot)] * 3,
            device=device,
            dtype=torch.float32,
        )
        self._damping = (float(damping) ** 2) * torch.eye(
            6,
            device=device,
            dtype=torch.float32,
        )
        self._joint_eye = torch.eye(
            len(self.joint_ids),
            device=device,
            dtype=torch.float32,
        )

    def get_ee_pose_w(self) -> tuple[torch.Tensor, torch.Tensor]:
        """返回末端刚体原点的世界位置和 ``wxyz`` 四元数。"""
        return (
            self.robot.data.body_pos_w[:, self.body_idx],
            self.robot.data.body_quat_w[:, self.body_idx],
        )

    def get_tcp_pose_w(self) -> tuple[torch.Tensor, torch.Tensor]:
        """把局部 TCP 偏移旋转到世界系，返回实际夹持点位姿。"""
        position, quaternion = self.get_ee_pose_w()
        offset = self.tcp_offset.expand(position.shape[0], 3)
        return position + quat_apply(quaternion, offset), quaternion

    def _jacobian_b(self) -> torch.Tensor:
        """提取右臂雅可比并转换到浮动基座坐标系的 TCP 点。"""
        jacobian_w = self.robot.root_physx_view.get_jacobians()[
            :, self.jacobi_body_idx, :, self.jacobi_joint_ids
        ]
        base_rotation = matrix_from_quat(quat_inv(self.robot.data.root_quat_w))
        jacobian_b = jacobian_w.clone()
        jacobian_b[:, :3] = torch.bmm(base_rotation, jacobian_w[:, :3])
        jacobian_b[:, 3:] = torch.bmm(base_rotation, jacobian_w[:, 3:])
        if self._has_tcp_offset:
            _, quaternion = self.get_ee_pose_w()
            offset_w = quat_apply(
                quaternion,
                self.tcp_offset.expand(quaternion.shape[0], 3),
            )
            offset_b = torch.bmm(
                base_rotation,
                offset_w.unsqueeze(-1),
            ).squeeze(-1)
            # 刚体点平移后，线速度雅可比需要减去 [r]x Jw。
            jacobian_b[:, :3] -= torch.bmm(
                _skew(offset_b),
                jacobian_b[:, 3:],
            )
        return jacobian_b

    def solve(
        self,
        target_pos_w: torch.Tensor,
        target_quat_w: torch.Tensor,
        q_ref: torch.Tensor | None = None,
        q_ref_gain: float = 0.25,
    ) -> torch.Tensor:
        """求解全位姿关节目标，并可叠加动态零空间姿态参考。

        参数：
            target_pos_w: TCP 世界位置，形状为 ``(N, 3)``。
            target_quat_w: TCP 世界姿态，``wxyz``，形状为 ``(N, 4)``。
            q_ref: 可选动态关节参考；一维输入会扩展到所有环境。
            q_ref_gain: 零空间姿态误差增益，默认保持现有 0.25。

        返回：
            经过单步变化限幅和关节限位后的右臂关节目标。
        """
        root_pos = self.robot.data.root_pos_w
        root_quat = self.robot.data.root_quat_w
        tcp_pos_w, tcp_quat_w = self.get_tcp_pose_w()
        tcp_pos_b, tcp_quat_b = subtract_frame_transforms(
            root_pos,
            root_quat,
            tcp_pos_w,
            tcp_quat_w,
        )
        target_pos_b, target_quat_b = subtract_frame_transforms(
            root_pos,
            root_quat,
            target_pos_w,
            target_quat_w,
        )
        pos_error, rot_error = compute_pose_error(
            tcp_pos_b,
            tcp_quat_b,
            target_pos_b,
            target_quat_b,
            rot_error_type="axis_angle",
        )
        weights = self._task_weights.to(dtype=pos_error.dtype)
        error = torch.cat((pos_error, rot_error), dim=1) * weights
        jacobian = self._jacobian_b() * weights.view(1, 6, 1)

        q = self.robot.data.joint_pos[:, self.joint_ids].clone()
        q_ref_batch = None
        if q_ref is not None:
            q_ref_batch = q_ref.to(device=q.device, dtype=q.dtype)
            if q_ref_batch.dim() == 1:
                q_ref_batch = q_ref_batch.unsqueeze(0).expand_as(q)

        damping = self._damping.to(dtype=jacobian.dtype)
        joint_eye = self._joint_eye.to(dtype=jacobian.dtype).unsqueeze(0)
        for _ in range(self.max_iters):
            at_goal = float(
                torch.linalg.vector_norm(error[:, :3], dim=-1).max()
            ) < self.pos_tol
            if at_goal and q_ref_batch is None:
                break

            jacobian_t = jacobian.transpose(1, 2)
            system = jacobian @ jacobian_t + damping
            solved_error = torch.linalg.solve(system, error.unsqueeze(-1))
            delta_q = (jacobian_t @ solved_error).squeeze(-1)
            if q_ref_batch is not None:
                # 40 只投影到任务零空间，绝不能作为 ``arm_q`` 硬下发。
                jacobian_plus = torch.linalg.solve(system, jacobian).transpose(1, 2)
                null_projector = joint_eye - torch.bmm(
                    jacobian_plus,
                    jacobian,
                )
                posture_error = float(q_ref_gain) * (q_ref_batch - q)
                delta_q += (
                    null_projector @ posture_error.unsqueeze(-1)
                ).squeeze(-1)

            delta_q = torch.clamp(
                self.gain * delta_q,
                -self.max_delta,
                self.max_delta,
            )
            q = torch.clamp(q + delta_q, self.q_min, self.q_max)
            # 保持现有线性化契约：同一帧内不重新读取仿真位姿和雅可比。
            error -= (jacobian @ delta_q.unsqueeze(-1)).squeeze(-1)
            if at_goal:
                break
        return q

    def position_error_norm(self, target_pos_w: torch.Tensor) -> float:
        """返回第一个环境的 TCP 世界位置误差范数。"""
        tcp_pos_w, _ = self.get_tcp_pose_w()
        return float(
            torch.linalg.vector_norm(target_pos_w - tcp_pos_w, dim=-1)[0]
        )
