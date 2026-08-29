# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""CES right-arm differential IK with dynamic null-space posture control."""
from __future__ import annotations

import torch
from isaaclab.utils.math import (
    compute_pose_error,
    matrix_from_quat,
    quat_apply,
    quat_inv,
    subtract_frame_transforms,
)


def _skew(v: torch.Tensor) -> torch.Tensor:
    x, y, z = v[:, 0], v[:, 1], v[:, 2]
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
    """Damped least-squares IK for the CES arm/TCP control contract."""

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
        self.robot = robot
        self.device = device
        self.damping = damping
        self.gain = gain
        self.max_delta = max_delta_per_step
        self.w_pos = float(w_pos)
        self.w_rot = float(w_rot)
        self.max_iters = int(max_iters)
        self.pos_tol = float(pos_tol)
        self.tcp_offset = torch.tensor(
            tcp_offset or (0.0, 0.0, 0.0), device=device, dtype=torch.float32
        )

        joint_ids, joint_names = robot.find_joints(
            arm_joint_names, preserve_order=True
        )
        self.joint_ids = list(joint_ids)
        self.joint_names = list(joint_names)
        self.num_joints = len(self.joint_ids)
        body_ids, body_names = robot.find_bodies(ee_body_name, preserve_order=True)
        if len(body_ids) != 1:
            raise ValueError(
                f"expected one end-effector body for {ee_body_name!r}, got {body_names}"
            )
        self.body_idx = int(body_ids[0])
        self.body_name = body_names[0]
        if robot.is_fixed_base:
            self.jacobi_body_idx = self.body_idx - 1
            self.jacobi_joint_ids = self.joint_ids
        else:
            self.jacobi_body_idx = self.body_idx
            self.jacobi_joint_ids = [joint_id + 6 for joint_id in self.joint_ids]

        limits = robot.data.joint_pos_limits[0, self.joint_ids].to(device)
        self.q_min, self.q_max = limits[:, 0], limits[:, 1]

    def get_ee_pose_w(self):
        return (
            self.robot.data.body_pos_w[:, self.body_idx],
            self.robot.data.body_quat_w[:, self.body_idx],
        )

    def get_tcp_pose_w(self):
        pos, quat = self.get_ee_pose_w()
        offset = self.tcp_offset.expand(pos.shape[0], 3)
        return pos + quat_apply(quat, offset), quat

    def _jacobian_b(self) -> torch.Tensor:
        jac_w = self.robot.root_physx_view.get_jacobians()[
            :, self.jacobi_body_idx, :, self.jacobi_joint_ids
        ]
        base_rot = matrix_from_quat(quat_inv(self.robot.data.root_quat_w))
        jac_b = jac_w.clone()
        jac_b[:, :3] = torch.bmm(base_rot, jac_w[:, :3])
        jac_b[:, 3:] = torch.bmm(base_rot, jac_w[:, 3:])
        if float(torch.linalg.vector_norm(self.tcp_offset)) > 1e-8:
            _pos, quat = self.get_ee_pose_w()
            offset_w = quat_apply(quat, self.tcp_offset.expand(quat.shape[0], 3))
            offset_b = torch.bmm(base_rot, offset_w.unsqueeze(-1)).squeeze(-1)
            jac_b[:, :3] -= torch.bmm(_skew(offset_b), jac_b[:, 3:])
        return jac_b

    def solve(
        self,
        target_pos_w: torch.Tensor,
        target_quat_w: torch.Tensor | None = None,
        q_ref: torch.Tensor | None = None,
        q_ref_gain: float = 0.25,
        q_lo: torch.Tensor | None = None,
        q_hi: torch.Tensor | None = None,
        pos_axes: tuple[int, ...] | None = None,
    ) -> torch.Tensor:
        root_pos = self.robot.data.root_pos_w
        root_quat = self.robot.data.root_quat_w
        tcp_pos_w, tcp_quat_w = self.get_tcp_pose_w()
        target_frame_quat = tcp_quat_w if target_quat_w is None else target_quat_w
        tcp_pos_b, tcp_quat_b = subtract_frame_transforms(
            root_pos, root_quat, tcp_pos_w, tcp_quat_w
        )
        target_pos_b, target_quat_b = subtract_frame_transforms(
            root_pos, root_quat, target_pos_w, target_frame_quat
        )
        jac_b = self._jacobian_b()

        if target_quat_w is None:
            error = self.w_pos * (target_pos_b - tcp_pos_b)
            jac = self.w_pos * jac_b[:, :3]
            if pos_axes is not None:
                axes = [int(axis) for axis in pos_axes]
                error, jac = error[:, axes], jac[:, axes]
        else:
            pos_error, rot_error = compute_pose_error(
                tcp_pos_b,
                tcp_quat_b,
                target_pos_b,
                target_quat_b,
                rot_error_type="axis_angle",
            )
            error = torch.cat((self.w_pos * pos_error, self.w_rot * rot_error), dim=1)
            weights = torch.tensor(
                [self.w_pos] * 3 + [self.w_rot] * 3,
                device=self.device,
                dtype=jac_b.dtype,
            )
            jac = jac_b * weights.view(1, 6, 1)

        q = self.robot.data.joint_pos[:, self.joint_ids].clone()
        lo, hi = self.q_min, self.q_max
        if q_lo is not None:
            lo = torch.maximum(lo, q_lo.to(device=lo.device, dtype=lo.dtype))
        if q_hi is not None:
            hi = torch.minimum(hi, q_hi.to(device=hi.device, dtype=hi.dtype))
        q_ref_batch = None
        if q_ref is not None:
            q_ref_batch = q_ref.to(device=q.device, dtype=q.dtype)
            if q_ref_batch.dim() == 1:
                q_ref_batch = q_ref_batch.unsqueeze(0).expand_as(q)

        task_dim, joint_dim = jac.shape[1], jac.shape[2]
        damping = (self.damping**2) * torch.eye(
            task_dim, device=self.device, dtype=jac.dtype
        )
        joint_eye = torch.eye(joint_dim, device=self.device, dtype=jac.dtype)
        pos_dim = 3 if target_quat_w is not None or pos_axes is None else error.shape[1]
        for _ in range(max(1, self.max_iters)):
            at_goal = float(
                torch.linalg.vector_norm(error[:, :pos_dim], dim=-1).max()
            ) < self.pos_tol
            if at_goal and q_ref_batch is None:
                break
            jac_t = jac.transpose(1, 2)
            system = jac @ jac_t + damping
            solved_error = torch.linalg.solve(system, error.unsqueeze(-1))
            dq = (jac_t @ solved_error).squeeze(-1)
            if q_ref_batch is not None:
                # Keep 40 as a dynamic null-space reference, never a hard arm command.
                jac_plus = torch.linalg.solve(system, jac).transpose(1, 2)
                null_projector = joint_eye.unsqueeze(0) - torch.bmm(jac_plus, jac)
                posture_error = float(q_ref_gain) * (q_ref_batch - q)
                dq += (null_projector @ posture_error.unsqueeze(-1)).squeeze(-1)
            dq = torch.clamp(self.gain * dq, -self.max_delta, self.max_delta)
            q = torch.clamp(q + dq, lo, hi)
            error -= (jac @ dq.unsqueeze(-1)).squeeze(-1)
            if at_goal:
                break
        return q

    def position_error_norm(self, target_pos_w: torch.Tensor) -> float:
        tcp_pos_w, _ = self.get_tcp_pose_w()
        return float(torch.linalg.vector_norm(target_pos_w - tcp_pos_w, dim=-1)[0])
