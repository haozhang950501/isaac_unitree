# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""Differential inverse kinematics for a single Unitree G1 arm.

The solver reads forward kinematics (the end-effector pose) directly from the
physics articulation and solves the inverse problem with a damped-least-squares
(Levenberg-Marquardt) update on the geometric Jacobian:

.. math::

    \\Delta q = J^T (J J^T + \\lambda^2 I)^{-1} \\, \\Delta x
    q_{des}   = q_{cur} + k \\, \\Delta q

This mirrors Isaac Lab's own :class:`DifferentialIKController` /
``DifferentialInverseKinematicsAction`` conventions:

* the raw PhysX Jacobian returned by ``root_physx_view.get_jacobians()`` is
  expressed in the **world** frame,
* for a **floating base** articulation the body row index is used as-is and the
  joint columns are offset by 6 (the 6 floating-base DoFs come first),
* the world Jacobian is rotated into the robot **root** frame with
  ``R(base)^{-1}`` so it is consistent with a root-frame pose error.

Optional ``tcp_offset`` is expressed in the EE body frame.  The tracked point
is then ``p_tcp = p_ee + R(ee) * tcp_offset`` and the linear Jacobian is
shifted by the rigid-body term ``-skew(r) J_ω``.

Working entirely in the root frame keeps the tracking insensitive to the base
translation/rotation that happens while the whole-body policy keeps the robot
balanced and walking.
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


def _skew(v: torch.Tensor) -> torch.Tensor:
    """Batch skew-symmetric matrices for ``v`` of shape [N, 3]."""
    x, y, z = v[:, 0], v[:, 1], v[:, 2]
    o = torch.zeros_like(x)
    row0 = torch.stack((o, -z, y), dim=-1)
    row1 = torch.stack((z, o, -x), dim=-1)
    row2 = torch.stack((-y, x, o), dim=-1)
    return torch.stack((row0, row1, row2), dim=-2)


class ArmDiffIK:
    """Damped-least-squares differential IK for one arm (7 revolute joints)."""

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

        if tcp_offset is None:
            self.tcp_offset = torch.zeros(3, device=device)
        else:
            self.tcp_offset = torch.tensor(tcp_offset, device=device, dtype=torch.float32)

        # resolve joints (preserve the requested order, find_joints may reorder)
        joint_ids, joint_names = robot.find_joints(arm_joint_names, preserve_order=True)
        self.joint_ids = list(joint_ids)
        self.joint_names = list(joint_names)
        self.num_joints = len(self.joint_ids)

        # resolve the end-effector body
        body_ids, body_names = robot.find_bodies(ee_body_name, preserve_order=True)
        if len(body_ids) != 1:
            raise ValueError(
                f"[ArmDiffIK] expected exactly one ee body for '{ee_body_name}', got {body_names}"
            )
        self.body_idx = int(body_ids[0])
        self.body_name = body_names[0]

        # floating vs fixed base Jacobian indexing (see module docstring)
        if robot.is_fixed_base:
            self.jacobi_body_idx = self.body_idx - 1
            self.jacobi_joint_ids = list(self.joint_ids)
        else:
            self.jacobi_body_idx = self.body_idx
            self.jacobi_joint_ids = [j + 6 for j in self.joint_ids]

        # joint position limits for clamping (shape [num_joints, 2])
        limits = robot.data.joint_pos_limits[0, self.joint_ids].to(device)
        self.q_min = limits[:, 0]
        self.q_max = limits[:, 1]

    # ------------------------------------------------------------------ FK --
    def get_ee_pose_w(self):
        """Return the current end-effector body pose in the world frame ([N,3],[N,4])."""
        ee_pos_w = self.robot.data.body_pos_w[:, self.body_idx]
        ee_quat_w = self.robot.data.body_quat_w[:, self.body_idx]
        return ee_pos_w, ee_quat_w

    def get_tcp_pose_w(self):
        """Return the TCP pose in the world frame ([N,3],[N,4]).

        Orientation matches the EE body; position is offset by ``tcp_offset``.
        """
        ee_pos_w, ee_quat_w = self.get_ee_pose_w()
        n = ee_pos_w.shape[0]
        tcp_pos_w = ee_pos_w + quat_apply(ee_quat_w, self.tcp_offset.expand(n, 3))
        return tcp_pos_w, ee_quat_w

    def _jacobian_b(self) -> torch.Tensor:
        """Geometric Jacobian (6 x num_joints) of the TCP, in the root frame."""
        jac_w = self.robot.root_physx_view.get_jacobians()[
            :, self.jacobi_body_idx, :, self.jacobi_joint_ids
        ]
        base_rot = matrix_from_quat(quat_inv(self.robot.data.root_quat_w))
        jac_b = jac_w.clone()
        jac_b[:, :3, :] = torch.bmm(base_rot, jac_w[:, :3, :])
        jac_b[:, 3:, :] = torch.bmm(base_rot, jac_w[:, 3:, :])

        # J_v_tcp = J_v + ω × r  = J_v - skew(r) J_ω , r in the root frame
        if float(self.tcp_offset.norm()) > 1e-8:
            ee_pos_w, ee_quat_w = self.get_ee_pose_w()
            n = ee_pos_w.shape[0]
            r_w = quat_apply(ee_quat_w, self.tcp_offset.expand(n, 3))
            r_b = torch.bmm(base_rot, r_w.unsqueeze(-1)).squeeze(-1)
            jac_b[:, :3, :] = jac_b[:, :3, :] - torch.bmm(_skew(r_b), jac_b[:, 3:, :])
        return jac_b

    # ------------------------------------------------------------------ IK --
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
        """Compute absolute joint-position targets that drive the TCP to the goal.

        Args:
            target_pos_w: desired TCP position in world frame, shape [N, 3].
            target_quat_w: optional desired orientation (w,x,y,z), shape [N, 4].
                When ``None`` only position (3-DoF) is tracked and the arm keeps
                whatever orientation the redundancy resolution yields.
            q_ref: optional preferred posture, shape [num_joints] or [N, num_joints].
                Applied only in the Jacobian null space; it never replaces the
                TCP task update.
            q_ref_gain: null-space step gain ``k`` in
                ``dq = J^+ e + (I - J^+ J) k (q_ref - q)``.
            q_lo: optional per-joint lower bound, tightened against the hardware
                limit.  Use it to keep a posture (elbow / wrist) out of a range
                the task must not enter.
            q_hi: optional per-joint upper bound, same convention as ``q_lo``.
            pos_axes: if set (and ``target_quat_w`` is ``None``), only these
                root-frame position axes are tasks.  ``(2,)`` tracks height
                and lets X/Y drift as the shoulder lowers the arm.

        Returns:
            Desired joint positions for this arm, shape [N, num_joints].
        """
        robot = self.robot
        root_pos = robot.data.root_pos_w
        root_quat = robot.data.root_quat_w

        tcp_pos_w, tcp_quat_w = self.get_tcp_pose_w()
        dummy_quat = tcp_quat_w if target_quat_w is None else target_quat_w
        tcp_pos_b, tcp_quat_b = subtract_frame_transforms(root_pos, root_quat, tcp_pos_w, tcp_quat_w)
        tgt_pos_b, tgt_quat_b = subtract_frame_transforms(root_pos, root_quat, target_pos_w, dummy_quat)

        jac_b = self._jacobian_b()

        if target_quat_w is not None:
            pos_err, rot_err = compute_pose_error(
                tcp_pos_b, tcp_quat_b, tgt_pos_b, tgt_quat_b, rot_error_type="axis_angle"
            )
            error = torch.cat((self.w_pos * pos_err, self.w_rot * rot_err), dim=1)
            w = torch.tensor(
                [self.w_pos, self.w_pos, self.w_pos, self.w_rot, self.w_rot, self.w_rot],
                device=self.device,
                dtype=jac_b.dtype,
            )
            jac = jac_b * w.view(1, 6, 1)
        else:
            error = self.w_pos * (tgt_pos_b - tcp_pos_b)
            jac = jac_b[:, 0:3, :] * self.w_pos
            if pos_axes is not None:
                axes = [int(a) for a in pos_axes]
                error = error[:, axes]
                jac = jac[:, axes, :]

        q = robot.data.joint_pos[:, self.joint_ids].clone()
        lo, hi = self.q_min, self.q_max
        if q_lo is not None:
            lo = torch.maximum(lo, q_lo.to(device=lo.device, dtype=lo.dtype))
        if q_hi is not None:
            hi = torch.minimum(hi, q_hi.to(device=hi.device, dtype=hi.dtype))
        q_ref_b = None
        if q_ref is not None:
            q_ref_b = q_ref.to(device=q.device, dtype=q.dtype)
            if q_ref_b.dim() == 1:
                q_ref_b = q_ref_b.unsqueeze(0).expand_as(q)
        n_err = jac.shape[1]
        n_j = jac.shape[2]
        lam = (self.damping ** 2) * torch.eye(n_err, device=self.device, dtype=jac.dtype)
        eye_j = torch.eye(n_j, device=self.device, dtype=jac.dtype)
        pos_dim = 3 if target_quat_w is not None or pos_axes is None else error.shape[1]
        k_ref = float(q_ref_gain)
        for _ in range(max(1, self.max_iters)):
            pos_norm = torch.norm(error[:, :pos_dim], dim=-1).max()
            at_goal = float(pos_norm) < self.pos_tol
            if at_goal and q_ref_b is None:
                break
            jac_T = jac.transpose(1, 2)
            jjt_inv = torch.inverse(jac @ jac_T + lam)
            dq = (jac_T @ jjt_inv @ error.unsqueeze(-1)).squeeze(-1)
            if q_ref_b is not None:
                # dq = J^+ e + (I - J^+ J) k (q_ref - q)  — null space only
                j_plus = jac_T @ jjt_inv
                null_proj = eye_j.unsqueeze(0) - torch.bmm(j_plus, jac)
                dq = dq + (null_proj @ (k_ref * (q_ref_b - q)).unsqueeze(-1)).squeeze(-1)
            dq = torch.clamp(self.gain * dq, -self.max_delta, self.max_delta)
            q = torch.clamp(q + dq, lo, hi)
            # first-order residual (no extra physics FK between inner iters)
            error = error - (jac @ dq.unsqueeze(-1)).squeeze(-1)
            if at_goal:
                break
        return q

    def position_error_norm(self, target_pos_w: torch.Tensor) -> float:
        """Euclidean distance (m) between the TCP and a world-frame target."""
        tcp_pos_w, _ = self.get_tcp_pose_w()
        return float(torch.norm(target_pos_w - tcp_pos_w, dim=-1)[0].item())

    def joints_at_limit(self, margin: float = 0.02) -> int:
        """Count arm joints parked within ``margin`` rad of a travel limit."""
        q = self.robot.data.joint_pos[0, self.joint_ids]
        lo = (q - self.q_min) < margin
        hi = (self.q_max - q) < margin
        return int((lo | hi).sum().item())
