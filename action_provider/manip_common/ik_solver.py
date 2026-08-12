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

Working entirely in the root frame keeps the tracking insensitive to the base
translation/rotation that happens while the whole-body policy keeps the robot
balanced and walking.
"""
from __future__ import annotations

import torch

from isaaclab.utils.math import (
    compute_pose_error,
    matrix_from_quat,
    quat_inv,
    subtract_frame_transforms,
)


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
    ):
        self.robot = robot
        self.device = device
        self.damping = damping
        self.gain = gain
        self.max_delta = max_delta_per_step

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
        """Return the current end-effector pose in the world frame ([N,3],[N,4])."""
        ee_pos_w = self.robot.data.body_pos_w[:, self.body_idx]
        ee_quat_w = self.robot.data.body_quat_w[:, self.body_idx]
        return ee_pos_w, ee_quat_w

    def _jacobian_b(self) -> torch.Tensor:
        """Geometric Jacobian (6 x num_joints) of the ee body, in the root frame."""
        jac_w = self.robot.root_physx_view.get_jacobians()[
            :, self.jacobi_body_idx, :, self.jacobi_joint_ids
        ]
        base_rot = matrix_from_quat(quat_inv(self.robot.data.root_quat_w))
        jac_b = jac_w.clone()
        jac_b[:, :3, :] = torch.bmm(base_rot, jac_w[:, :3, :])
        jac_b[:, 3:, :] = torch.bmm(base_rot, jac_w[:, 3:, :])
        return jac_b

    # ------------------------------------------------------------------ IK --
    def solve(
        self,
        target_pos_w: torch.Tensor,
        target_quat_w: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute absolute joint-position targets that drive the ee to the goal.

        Args:
            target_pos_w: desired ee position in world frame, shape [N, 3].
            target_quat_w: optional desired ee orientation (w,x,y,z), shape [N, 4].
                When ``None`` only position (3-DoF) is tracked and the arm keeps
                whatever orientation the redundancy resolution yields.

        Returns:
            Desired joint positions for this arm, shape [N, num_joints].
        """
        robot = self.robot
        root_pos = robot.data.root_pos_w
        root_quat = robot.data.root_quat_w

        ee_pos_w, ee_quat_w = self.get_ee_pose_w()
        ee_pos_b, ee_quat_b = subtract_frame_transforms(root_pos, root_quat, ee_pos_w, ee_quat_w)
        tgt_pos_b, tgt_quat_b = subtract_frame_transforms(root_pos, root_quat, target_pos_w, target_quat_w)

        jac_b = self._jacobian_b()

        if target_quat_w is not None:
            pos_err, rot_err = compute_pose_error(
                ee_pos_b, ee_quat_b, tgt_pos_b, tgt_quat_b, rot_error_type="axis_angle"
            )
            error = torch.cat((pos_err, rot_err), dim=1)  # [N,6]
            jac = jac_b
        else:
            error = tgt_pos_b - ee_pos_b  # [N,3]
            jac = jac_b[:, 0:3, :]

        jac_T = jac.transpose(1, 2)
        n = jac.shape[1]
        lam = (self.damping ** 2) * torch.eye(n, device=self.device)
        dq = (jac_T @ torch.inverse(jac @ jac_T + lam) @ error.unsqueeze(-1)).squeeze(-1)
        dq = self.gain * dq
        dq = torch.clamp(dq, -self.max_delta, self.max_delta)

        q_cur = robot.data.joint_pos[:, self.joint_ids]
        q_des = q_cur + dq
        q_des = torch.clamp(q_des, self.q_min, self.q_max)
        return q_des

    def position_error_norm(self, target_pos_w: torch.Tensor) -> float:
        """Euclidean distance (m) between the ee and a world-frame target."""
        ee_pos_w, _ = self.get_ee_pose_w()
        return float(torch.norm(target_pos_w - ee_pos_w, dim=-1)[0].item())
