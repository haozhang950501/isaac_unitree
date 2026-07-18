# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""Autonomous "walk up, grab both tray handles and lift" action provider.

This provider reuses the whole-body RL policy from :class:`DDSRLActionProvider`
to keep the G1 balanced and to walk, but instead of streaming the arm / gripper
/ walking targets from DDS teleoperation it generates them autonomously:

* a finite state machine (:class:`TrayGraspStateMachine`) sequences the
  behaviour and produces the base velocity command, per-hand Cartesian goals and
  the gripper command,
* a damped-least-squares differential IK (:class:`ArmDiffIK`) turns the
  Cartesian goals into arm joint targets,
* Cartesian interpolation smooths the hand motion between way-points.

Like the base whole-body provider it drives the simulation itself and therefore
returns ``None`` (the controller runs in ``use_rl_action_mode``).
"""
from __future__ import annotations

from typing import Optional

import torch

import isaaclab.utils.math as _isaac_math
from isaaclab.utils.math import quat_apply

# Isaac Lab 2.x logs a deprecation warning on *every* call to the (still widely
# used internally) ``quat_rotate`` / ``quat_rotate_inverse`` helpers, which
# floods the console many times per control step once the whole-body policy and
# the articulation data buffers start running. Rebind the deprecated aliases to
# their fast, warning-free equivalents (exactly what the deprecation suggests).
if getattr(_isaac_math.quat_rotate, "__name__", "") != "quat_apply":
    _isaac_math.quat_rotate = _isaac_math.quat_apply
    _isaac_math.quat_rotate_inverse = _isaac_math.quat_apply_inverse

from action_provider.action_provider_wh_dds import DDSRLActionProvider
from action_provider.tray_grasp import ArmDiffIK, GraspPhase, TrayGraspStateMachine


class TrayGraspActionProvider(DDSRLActionProvider):
    """Autonomous tray grasp-and-lift built on top of the whole-body RL policy."""

    # end-effector frames of the two-finger (dex1) gripper
    LEFT_EE_BODY = "left_hand_base_link"
    RIGHT_EE_BODY = "right_hand_base_link"

    # tray handle centres expressed in the tray local frame (metres).
    # Derived from the baked tray geometry: the two end walls sit at
    # local x = +/-0.2725, y = 0, with the graspable band around z = 0.03.
    HANDLE_LOCAL_A = (0.2725, 0.0, 0.03)
    HANDLE_LOCAL_B = (-0.2725, 0.0, 0.03)

    def __init__(self, env, args_cli):
        super().__init__(env, args_cli)
        self.name = "TrayGraspActionProvider"
        self.device = env.device
        self.base_height_default = 0.8

        # control period: the provider advances 4 physics sub-steps per call
        self.control_dt = 4.0 * float(env.physics_dt)

        # command injected into the RL observation (overrides the DDS command)
        self.sm_command = torch.tensor(
            [[0.0, 0.0, 0.0, self.base_height_default]], device=self.device
        )

        # ---- IK solvers for both arms -----------------------------------
        left_arm = self.arm_joint_names[:7]
        right_arm = self.arm_joint_names[7:]
        self.left_ik = ArmDiffIK(self.env.scene["robot"], left_arm, self.LEFT_EE_BODY, self.device)
        self.right_ik = ArmDiffIK(self.env.scene["robot"], right_arm, self.RIGHT_EE_BODY, self.device)

        self._left_arm_idx_t = torch.tensor(self.left_ik.joint_ids, dtype=torch.long, device=self.device)
        self._right_arm_idx_t = torch.tensor(self.right_ik.joint_ids, dtype=torch.long, device=self.device)
        default = self.env.scene["robot"].data.default_joint_pos
        self._left_default = default[:, self.left_ik.joint_ids].clone()
        self._right_default = default[:, self.right_ik.joint_ids].clone()

        # ---- gripper joints ---------------------------------------------
        grip_names = ["left_hand_Joint1_1", "left_hand_Joint2_1",
                      "right_hand_Joint1_1", "right_hand_Joint2_1"]
        self._grip_idx_t = torch.tensor(
            [self.joint_to_index[n] for n in grip_names if n in self.joint_to_index],
            dtype=torch.long, device=self.device,
        )

        # tray handle local offsets
        self._handle_local = torch.tensor(
            [self.HANDLE_LOCAL_A, self.HANDLE_LOCAL_B], device=self.device
        )  # [2,3]

        # ---- state machine ----------------------------------------------
        self.sm = TrayGraspStateMachine(self)
        self._printed_bodies = False

    # --------------------------------------------------------- accessors --
    def get_base_pose_w(self):
        robot = self.env.scene["robot"]
        pos = robot.data.root_pos_w  # [1,3]
        q = robot.data.root_quat_w[0]  # (w,x,y,z)
        w, x, y, z = (float(q[0]), float(q[1]), float(q[2]), float(q[3]))
        yaw = torch.atan2(
            torch.tensor(2.0 * (w * z + x * y)),
            torch.tensor(1.0 - 2.0 * (y * y + z * z)),
        ).item()
        return pos, yaw

    def get_handle_positions_w(self):
        """Return (left_handle_w, right_handle_w) as [1,3] tensors.

        The robot faces +X, so its left hand takes the handle with the larger
        world Y and its right hand the one with the smaller world Y.
        """
        tray = self.env.scene["tray_fixture"]
        p = tray.data.root_pos_w  # [1,3]
        q = tray.data.root_quat_w  # [1,4]
        q2 = q.expand(2, 4)
        handles_w = p + quat_apply(q2, self._handle_local)  # [2,3]
        if float(handles_w[0, 1]) >= float(handles_w[1, 1]):
            left, right = handles_w[0:1], handles_w[1:2]
        else:
            left, right = handles_w[1:2], handles_w[0:1]
        return left.clone(), right.clone()

    # ----------------------------------------- override RL command source --
    def compute_current_observations(self):
        """Same observation layout as the base class but the walking command
        comes from the state machine rather than the ``run_command`` DDS topic."""
        command = self.sm_command
        self.ang_vel = self.env.scene["robot"].data.root_ang_vel_b
        self.projected_gravity = self.env.scene["robot"].data.projected_gravity_b
        self.joint_pos = self.env.scene["robot"].data.joint_pos
        self.joint_vel = self.env.scene["robot"].data.joint_vel
        action = self.action_buffer._circular_buffer.buffer[:, -1, :]
        current_actor_obs = torch.cat(
            [
                self.ang_vel * self.obs_scales["ang_vel"],
                self.projected_gravity * self.obs_scales["projected_gravity"],
                command * self.obs_scales["commands"],
                (self.joint_pos[:, self.all_obs_indices] - self.default_action_positions[:, self.all_obs_indices]) * self.obs_scales["joint_pos"],
                (self.joint_vel[:, self.all_obs_indices] - self.default_action_velocities[:, self.all_obs_indices]) * self.obs_scales["joint_vel"],
                action * self.obs_scales["actions"],
            ],
            dim=-1,
        )
        return current_actor_obs

    # ----------------------------------------------------------- main step --
    def get_action(self, env) -> Optional[torch.Tensor]:
        try:
            if not self._printed_bodies:
                self._printed_bodies = True
                robot = self.env.scene["robot"]
                print(f"[{self.name}] body_names = {robot.data.body_names}")
                print(f"[{self.name}] left ee body '{self.left_ik.body_name}' idx={self.left_ik.body_idx}")
                print(f"[{self.name}] right ee body '{self.right_ik.body_name}' idx={self.right_ik.body_idx}")
                print(f"[{self.name}] left arm joints  = {self.left_ik.joint_names}")
                print(f"[{self.name}] right arm joints = {self.right_ik.joint_names}")
                lee, lq = self.left_ik.get_ee_pose_w()
                ree, rq = self.right_ik.get_ee_pose_w()
                print(f"[{self.name}] init left ee pos={lee.cpu().numpy().tolist()} quat={lq.cpu().numpy().tolist()}")
                print(f"[{self.name}] init right ee pos={ree.cpu().numpy().tolist()} quat={rq.cpu().numpy().tolist()}")
                lh, rh = self.get_handle_positions_w()
                print(f"[{self.name}] handle L(world)={lh.cpu().numpy().tolist()} R(world)={rh.cpu().numpy().tolist()}")

            # 1) advance the finite state machine
            out = self.sm.update(self.control_dt)
            self.sm_command = out.command

            full_action = self._full_action_buf
            full_action.zero_()

            # 2) whole-body RL policy -> legs, default waist
            action_data = self.run_policy()
            full_action[self.action_to_indices] = action_data
            full_action[self.waist_to_all_indices] = self.default_waist_positions

            # 3) arms: IK when active, otherwise hold the default ready pose
            if out.arms_active and out.left_target_pos_w is not None:
                qL = self.left_ik.solve(out.left_target_pos_w)
                qR = self.right_ik.solve(out.right_target_pos_w)
            else:
                qL = self._left_default
                qR = self._right_default
            full_action.index_copy_(0, self._left_arm_idx_t, qL[0])
            full_action.index_copy_(0, self._right_arm_idx_t, qR[0])

            # 4) leg delay/clip/scale (identical to the base whole-body provider)
            delayed_actions = self.action_buffer.compute(full_action[self.old_action_indices].unsqueeze(0))
            cliped_actions = torch.clip(
                delayed_actions[:, self.action_to_indices], -self.clip_actions, self.clip_actions
            ).to(self.env.device)
            full_action[self.action_to_indices] = (
                cliped_actions * self.action_scale + self.default_action_positions[:, self.action_to_indices]
            )

            # 5) grippers
            if self._grip_idx_t.numel() > 0:
                full_action[self._grip_idx_t] = out.gripper_value

            # 6) advance the simulation (this provider owns the stepping)
            for _ in range(4):
                self.env.scene["robot"].set_joint_position_target(full_action)
                self.env.scene.write_data_to_sim()
                self.env.sim.step(render=False)
                self.env.scene.update(dt=self.env.physics_dt)
            self.env.sim.render()
            self.env.observation_manager.compute()

            # periodic tracking diagnostics
            if out.arms_active and out.left_target_pos_w is not None:
                self._diag_counter = getattr(self, "_diag_counter", 0) + 1
                if self._diag_counter % 25 == 0:
                    le = self.left_ik.position_error_norm(out.left_target_pos_w)
                    re = self.right_ik.position_error_norm(out.right_target_pos_w)
                    print(f"[{self.name}] {out.phase.name} L_err={le*1000:.0f}mm R_err={re*1000:.0f}mm grip={out.gripper_value:.3f}")

        except Exception as e:
            import traceback
            print(f"[{self.name}] get_action failed: {e}")
            traceback.print_exc()
            return None
        return None
