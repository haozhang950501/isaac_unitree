# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""Autonomous CES product pick-and-place action provider.

Pelvis is pinned at the pick / place stand (定位).  The Dex1 squeeze is a
real PD grasp: pad friction lifts the Product.  The right arm is
joint-locked so the default hang pose cannot pull it down.
"""
from __future__ import annotations

import math

import torch

from action_provider.action_provider_wh_dds import DDSRLActionProvider
from action_provider.ces_grasp import constants as C
from action_provider.ces_grasp.state_machine import CesPickPlaceStateMachine
from action_provider.manip_common import ArmDiffIK
from isaaclab.utils.math import quat_mul


class CESGraspActionProvider(DDSRLActionProvider):
    def __init__(self, env, args_cli):
        super().__init__(env, args_cli)
        self.name = "CESGrasp"
        self.dt = float(4 * env.physics_dt)
        self.station_mode = "snap"
        self.ik = ArmDiffIK(
            env.scene["robot"],
            C.RIGHT_ARM_JOINTS,
            C.EE_BODY,
            env.device,
            tcp_offset=C.TCP_LOCAL,
            max_delta_per_step=0.03,
            w_pos=1.0,
            w_rot=0.45,
            max_iters=8,
            pos_tol=0.005,
        )
        self.fsm = CesPickPlaceStateMachine(
            self,
            self.station_mode,
            stop_after=getattr(args_cli, "ces_stop_after", C.STOP_AFTER) or C.STOP_AFTER,
        )
        robot = env.scene["robot"]
        self._right_arm_idx = list(self.ik.joint_ids)
        left_ids, _ = robot.find_joints(C.LEFT_ARM_JOINTS, preserve_order=True)
        self._left_arm_idx = list(left_ids)
        rg_ids, _ = robot.find_joints(C.RIGHT_GRIPPER_JOINTS, preserve_order=True)
        lg_ids, _ = robot.find_joints(C.LEFT_GRIPPER_JOINTS, preserve_order=True)
        self._right_grip_idx = list(rg_ids)
        self._left_grip_idx = list(lg_ids)
        hands = robot.actuators.get("hands")
        if hands is not None:
            if torch.is_tensor(hands.stiffness):
                hands.stiffness[:] = 1800.0
                hands.damping[:] = 30.0
            else:
                hands.stiffness = 1800.0
                hands.damping = 30.0
            print("[CESGrasp] Dex1 hand PD kp=1800 kd=30")
        grip_t = torch.tensor(self._right_grip_idx, dtype=torch.long, device=env.device)
        n_g = len(self._right_grip_idx)
        robot.write_joint_stiffness_to_sim(
            torch.full((1, n_g), 1800.0, device=env.device), joint_ids=grip_t
        )
        robot.write_joint_damping_to_sim(
            torch.full((1, n_g), 30.0, device=env.device), joint_ids=grip_t
        )
        robot.write_joint_effort_limit_to_sim(
            torch.full((1, n_g), 80.0, device=env.device), joint_ids=grip_t
        )
        self._lock_arm_idx = torch.tensor(
            self._right_arm_idx + self._left_arm_idx,
            dtype=torch.long,
            device=env.device,
        )
        self._right_arm_default = robot.data.default_joint_pos[:, self._right_arm_idx].clone()
        self._left_arm_default = robot.data.default_joint_pos[:, self._left_arm_idx].clone()
        self._leg_default = robot.data.default_joint_pos[:, self.action_to_indices].clone()
        self._walk_cmd = [0.0, 0.0, 0.0, 0.8]
        self._hold_xy: tuple[float, float] | None = None
        self._hold_yaw: float | None = None
        self._q_right = self._right_arm_default[0].clone()
        self._squeeze = False
        self._err_logged = False
        self._err_t = -10.0
        print(
            f"[CESGrasp] v7 friction-grasp station_mode=snap "
            f"pick_stand={tuple(round(x, 3) for x in C.PICK_STAND_XY)} "
            f"place_stand={tuple(round(x, 3) for x in C.PLACE_STAND_XY)} "
            f"(no TCP weld; Dex1 PD squeeze + pad friction)"
        )

    def reset_task(self):
        """Return FSM / arm / gripper to settle so ``r`` can replay the pick."""
        self.fsm.reset()
        self._squeeze = False
        self._q_right = self._right_arm_default[0].clone()
        self._hold_xy = None
        self._hold_yaw = None
        self._err_logged = False
        self._err_t = -10.0
        print("[CESGrasp] task reset")

    # ------------------------------------------------------------------ ctx --
    @property
    def device(self):
        return self.env.device

    def get_base_pose_w(self):
        robot = self.env.scene["robot"]
        return robot.data.root_pos_w, robot.data.root_quat_w

    def get_object_pose_w(self):
        obj = self.env.scene["object"]
        return obj.data.root_pos_w, obj.data.root_quat_w

    def get_right_arm_q(self):
        robot = self.env.scene["robot"]
        return robot.data.joint_pos[:, self._right_arm_idx].clone()

    def _slew_arm(self, q_tgt: torch.Tensor) -> torch.Tensor:
        lim = C.ARM_SLEW_RAD_LIFT if self._squeeze else C.ARM_SLEW_RAD
        dq = q_tgt - self._q_right
        dq = torch.clamp(dq, -lim, lim)
        self._q_right = self._q_right + dq
        return self._q_right

    def get_product_aabb_center_w(self):
        import omni.usd
        from pxr import UsdGeom

        obj = self.env.scene["object"]
        _, quat = self.get_object_pose_w()
        path = obj.cfg.prim_path.replace("env_.*", "env_0")
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            return obj.data.root_pos_w.clone(), quat
        cache = UsdGeom.BBoxCache(
            0, [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy]
        )
        rng = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        if rng.IsEmpty():
            return obj.data.root_pos_w.clone(), quat
        mn, mx = rng.GetMin(), rng.GetMax()
        center = torch.tensor(
            [[0.5 * (mn[0] + mx[0]), 0.5 * (mn[1] + mx[1]), 0.5 * (mn[2] + mx[2])]],
            device=self.device,
            dtype=obj.data.root_pos_w.dtype,
        )
        return center, quat

    def get_heading(self) -> float:
        h = self.env.scene["robot"].data.heading_w
        return float(h[0].item()) if h.dim() else float(h.item())

    def stance_tilt(self) -> float:
        g = self.env.scene["robot"].data.projected_gravity_b[0]
        return float(torch.sqrt(g[0] * g[0] + g[1] * g[1]).item())

    def is_standing(self) -> bool:
        robot = self.env.scene["robot"]
        tilt = self.stance_tilt()
        ang = robot.data.root_ang_vel_w[0]
        lin = robot.data.root_lin_vel_w[0]
        z = float(robot.data.root_pos_w[0, 2])
        yaw_rate = float(ang[2].abs().item())
        xy_speed = float(torch.sqrt(lin[0] * lin[0] + lin[1] * lin[1]).item())
        return (
            tilt < C.STAND_TILT_MAX
            and yaw_rate < C.STAND_YAW_RATE_MAX
            and xy_speed < C.STAND_XY_SPEED_MAX
            and 0.68 < z < 0.88
        )

    def _apply_snap(self, xy: tuple[float, float], yaw: float):
        robot = self.env.scene["robot"]
        pose = robot.data.root_state_w[:, 0:7].clone()
        old_x = float(pose[0, 0])
        old_y = float(pose[0, 1])
        old_yaw = self.get_heading()
        dx = xy[0] - old_x
        dy = xy[1] - old_y
        dyaw = C.wrap_angle(yaw - old_yaw)
        pose[0, 0] = xy[0]
        pose[0, 1] = xy[1]
        pose[0, 2] = C.STAND_PELVIS_Z
        qw, qx, qy, qz = C.yaw_quat(yaw)
        pose[0, 3], pose[0, 4], pose[0, 5], pose[0, 6] = qw, qx, qy, qz
        vel = torch.zeros(1, 6, device=self.env.device, dtype=pose.dtype)
        robot.write_root_pose_to_sim(pose)
        robot.write_root_velocity_to_sim(vel)
        # Station teleport only: if the pelvis actually jumped, bring a
        # squeezed part along in world XY.  Same-stand pinning (dx≈0) leaves
        # the object in PhysX so lift is friction, not a TCP weld.
        jumped = (dx * dx + dy * dy) > 4e-4 or abs(dyaw) > 0.02
        if self._squeeze and jumped:
            self._translate_object_with_snap(old_x, old_y, dx, dy, dyaw)

    def _translate_object_with_snap(
        self, old_x: float, old_y: float, dx: float, dy: float, dyaw: float
    ):
        obj = self.env.scene["object"]
        op = obj.data.root_state_w[:, 0:7].clone()
        rel_x = float(op[0, 0]) - old_x
        rel_y = float(op[0, 1]) - old_y
        c, s = math.cos(dyaw), math.sin(dyaw)
        op[0, 0] = (old_x + dx) + c * rel_x - s * rel_y
        op[0, 1] = (old_y + dy) + s * rel_x + c * rel_y
        dq = torch.tensor(
            [C.yaw_quat(dyaw)], device=self.env.device, dtype=op.dtype
        )
        op[:, 3:7] = quat_mul(dq, op[:, 3:7])
        vel = torch.zeros(1, 6, device=self.env.device, dtype=op.dtype)
        obj.write_root_pose_to_sim(op)
        obj.write_root_velocity_to_sim(vel)

    def _write_locked_upper_body(self, full_action: torch.Tensor):
        """Kinematic-lock the arms only.  Dex1 stays on PD so pad contact
        can produce friction instead of teleporting through the mesh."""
        robot = self.env.scene["robot"]
        idx = self._lock_arm_idx
        pos = full_action.index_select(0, idx).unsqueeze(0)
        vel = torch.zeros_like(pos)
        robot.write_joint_state_to_sim(pos, vel, joint_ids=idx)

    def get_action(self, env):
        try:
            cmd = self.fsm.step()
            if self.fsm.phase.value == "settle":
                self._err_logged = False

            if cmd.snap_xy is not None and cmd.snap_yaw is not None:
                self._hold_xy, self._hold_yaw = cmd.snap_xy, cmd.snap_yaw
            pin_xy = cmd.snap_xy if cmd.snap_xy is not None else self._hold_xy
            pin_yaw = cmd.snap_yaw if cmd.snap_yaw is not None else self._hold_yaw
            if pin_xy is None:
                pin_xy, pin_yaw = C.SPAWN_STAND_XY, C.SPAWN_STAND_YAW

            self._walk_cmd = [0.0, 0.0, 0.0, 0.8]
            full_action = self._full_action_buf
            full_action.zero_()
            full_action[self.action_to_indices] = self._leg_default[0]
            full_action[self.waist_to_all_indices] = self.default_waist_positions[0]
            full_action[self._left_arm_idx] = self._left_arm_default[0]
            if cmd.arm_q is not None:
                full_action[self._right_arm_idx] = self._slew_arm(
                    cmd.arm_q.to(full_action.dtype)
                )
            elif cmd.tcp_pos is not None:
                try:
                    q_des = self.ik.solve(cmd.tcp_pos, cmd.tcp_quat)
                    full_action[self._right_arm_idx] = self._slew_arm(q_des[0])
                except Exception as e:
                    if not self._err_logged:
                        self._err_logged = True
                        print(f"[{self.name}] IK failed: {e}")
                        import traceback

                        traceback.print_exc()
                    full_action[self._right_arm_idx] = self._q_right
            elif self._squeeze:
                full_action[self._right_arm_idx] = self._q_right
            else:
                self._q_right = self._right_arm_default[0].clone()
                full_action[self._right_arm_idx] = self._q_right

            if cmd.gripper >= C.GRIPPER_CLOSED - 0.002:
                self._squeeze = True
            if self.fsm.phase.value in ("release", "retract", "settle"):
                self._squeeze = False
            grip = C.GRIPPER_CLOSED if self._squeeze else cmd.gripper
            full_action[self._right_grip_idx] = grip
            full_action[self._left_grip_idx] = C.GRIPPER_OPEN

            lock_upper = (
                cmd.arm_q is not None or cmd.tcp_pos is not None or self._squeeze
            )
            # While squeezing, drive the arm with PD only.  write_joint_state
            # teleports the pads off the part and kills friction.
            kinematic_arm = lock_upper and not self._squeeze
            robot = self.env.scene["robot"]
            for _ in range(4):
                self._apply_snap(pin_xy, pin_yaw)
                robot.set_joint_position_target(full_action)
                if kinematic_arm:
                    self._write_locked_upper_body(full_action)
                self.env.scene.write_data_to_sim()
                self.env.sim.step(render=False)
                if kinematic_arm:
                    self._write_locked_upper_body(full_action)
                self.env.scene.update(dt=self.env.physics_dt)

            self.env.sim.render()
            self.env.observation_manager.compute()
        except Exception as e:
            t = float(getattr(self.fsm, "t", 0.0))
            if (not self._err_logged) or (t - self._err_t > 2.0):
                self._err_logged = True
                self._err_t = t
                print(f"[{self.name}] CES grasp action failed: {e}")
                import traceback

                traceback.print_exc()
            return None
        return None
