# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""CES Baseline action provider: pinned pick, carry-walk, and pinned place."""
from __future__ import annotations

import math

import torch

from action_provider.action_provider_wh_dds import DDSRLActionProvider
from action_provider.ces_grasp import constants as C
from action_provider.ces_grasp.state_machine import (
    CesPickPlacePhase,
    CesPickPlaceStateMachine,
)
from action_provider.manip_common import ArmDiffIK
from isaaclab.utils.math import quat_mul


class CESGraspActionProvider(DDSRLActionProvider):
    def __init__(self, env, args_cli):
        super().__init__(env, args_cli)
        self.name = "CESGrasp"
        self.dt = float(4 * env.physics_dt)
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
            speed_scale=getattr(args_cli, "ces_pick_speed", None),
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
        self._walk_prime: list[float] | None = None
        self._hold_xy: tuple[float, float] | None = None
        self._hold_yaw: float | None = None
        self._hold_z: float | None = None
        self._grip_cmd: float | None = None
        self._last_policy_legs = None
        self._product_brake_pending = False
        self._q_right = self._right_arm_default[0].clone()
        self._squeeze = False
        self._err_logged = False
        self._err_t = -10.0
        print(
            f"[CESGrasp] Baseline Smooth V1 carry-walk "
            f"pick_stand={tuple(round(x, 3) for x in C.PICK_STAND_XY)} "
            f"place_stand={tuple(round(x, 3) for x in C.PLACE_STAND_XY)} "
            f"(no TCP weld; Dex1 PD squeeze + pad friction)"
        )

    def reset_task(self):
        """Return FSM / arm / gripper to settle so ``r`` can replay the pick."""
        self.fsm.reset()
        self._squeeze = False
        self._grip_cmd = None
        self._q_right = self._right_arm_default[0].clone()
        self._hold_xy = None
        self._hold_yaw = None
        self._hold_z = None
        self._walk_prime = None
        self._last_policy_legs = None
        self._product_brake_pending = False
        self.reset_walk_filt()
        self._err_logged = False
        self._err_t = -10.0
        print("[CESGrasp] task reset")

    def reset_walk_filt(self):
        """Reset the body-frame gait command ramp before releasing the pelvis."""
        self._walk_cmd = [0.0, 0.0, 0.0, 0.8]

    def prime_walk_filt(self, cmd):
        """预载步态滤波：释放骨盆的第一帧就直接给这个机体系指令，不从零 ramp。

        pick 结束后立刻满幅 S（-vx），避免等站稳再发时抱件前倾撞上 CES。
        """
        self._walk_prime = [float(cmd[0]), float(cmd[1]), float(cmd[2]), float(cmd[3])]

    def _begin_walk_policy(self, cmd: list[float]):
        """Clear standing-obs history and kick the root backward.

        Actor obs is a 10-frame stack. Pick fills it with vx=0, so the first
        gait inference still looks like standing and the policy takes one or
        two collecting steps toward CES before reverse shows up in the stack.
        Resetting makes every stacked frame a reverse command. The velocity
        kick stops the carry COM falling forward on the first unpinned step.
        """
        self.actor_obs_buffer.reset()
        self.action_buffer.reset()
        zeros = torch.zeros(
            self.num_envs,
            self.num_actions_all,
            dtype=torch.float,
            device=self.env.device,
        )
        self.action_buffer.compute(zeros)
        self._kick_walk_start_velocity(cmd[0])

    def _kick_walk_start_velocity(self, vx_body: float):
        """Give pelvis (and held product) the reverse world velocity immediately."""
        robot = self.env.scene["robot"]
        yaw = self.get_heading()
        fwd, _left = C.forward_left(yaw)
        vx_w = float(vx_body) * fwd[0]
        vy_w = float(vx_body) * fwd[1]
        dtype = robot.data.root_pos_w.dtype
        vel = torch.zeros(1, 6, device=self.env.device, dtype=dtype)
        vel[0, 0] = vx_w
        vel[0, 1] = vy_w
        robot.write_root_velocity_to_sim(vel)
        if self._squeeze:
            obj = self.env.scene["object"]
            ov = torch.zeros(1, 6, device=self.env.device, dtype=obj.data.root_pos_w.dtype)
            ov[0, 0] = vx_w
            ov[0, 1] = vy_w
            obj.write_root_velocity_to_sim(ov)
        print(
            f"[CESGrasp] walk start kick vx_b={vx_body:+.2f} "
            f"world=({vx_w:+.2f},{vy_w:+.2f}) (no forward collect)"
        )

    def _filter_walk_command(self, target) -> list[float]:
        """Rate-limit gait commands so HOLD does not hand the policy a step input."""
        target = [float(target[i]) for i in range(4)]
        limits = (C.WALK_VX_ACCEL, C.WALK_VY_ACCEL, C.WALK_WZ_ACCEL)
        for i, rate in enumerate(limits):
            delta = target[i] - self._walk_cmd[i]
            max_delta = rate * self.dt
            self._walk_cmd[i] += max(-max_delta, min(max_delta, delta))
        self._walk_cmd[3] = target[3]
        return list(self._walk_cmd)

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

    def get_right_gripper_q(self):
        robot = self.env.scene["robot"]
        return robot.data.joint_pos[0, self._right_grip_idx].detach().clone()

    def get_root_xy_speed(self) -> float:
        lin = self.env.scene["robot"].data.root_lin_vel_w[0]
        return float(torch.sqrt(lin[0] * lin[0] + lin[1] * lin[1]).item())

    def get_object_xy_speed(self) -> float:
        lin = self.env.scene["object"].data.root_lin_vel_w[0]
        return float(torch.sqrt(lin[0] * lin[0] + lin[1] * lin[1]).item())

    def _slew_arm(self, q_tgt: torch.Tensor) -> torch.Tensor:
        # 夹持冻臂时慢跟；抬起走规划关节轨迹，不能限太死。
        phase = self.fsm.phase.value
        slow = self._squeeze and phase in ("carry", "goto_place", "place_hold")
        lim = C.ARM_SLEW_RAD_LIFT if slow else C.ARM_SLEW_RAD
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

    def _apply_snap(
        self,
        xy: tuple[float, float],
        yaw: float,
        z: float | None = None,
        quat: tuple[float, float, float, float] | None = None,
    ):
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
        pose[0, 2] = float(z) if z is not None else C.STAND_PELVIS_Z
        if quat is not None:
            pose[0, 3] = quat[0]
            pose[0, 4] = quat[1]
            pose[0, 5] = quat[2]
            pose[0, 6] = quat[3]
        else:
            qw, qx, qy, qz = C.yaw_quat(yaw)
            pose[0, 3], pose[0, 4], pose[0, 5], pose[0, 6] = qw, qx, qy, qz
        vel = torch.zeros(1, 6, device=self.env.device, dtype=pose.dtype)
        robot.write_root_pose_to_sim(pose)
        robot.write_root_velocity_to_sim(vel)
        # 换站瞬移才带件；同站钉盆不焊 TCP，抬起靠摩擦。
        # 钉盆清零骨盆速度时，产品若还带着 walk 速度，垫面会把件夹飞。
        jumped = (dx * dx + dy * dy) > 4e-4 or abs(dyaw) > 0.02
        if quat is not None:
            # Live-quat pin: do not yaw-rotate the product around the pelvis.
            jumped = (dx * dx + dy * dy) > 4e-4
        if self._squeeze and jumped:
            self._translate_object_with_snap(old_x, old_y, dx, dy, dyaw)
            self._product_brake_pending = False
        elif self._squeeze and self._product_brake_pending:
            self._stop_held_product()
            self._product_brake_pending = False

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

    def _stop_held_product(self):
        """Match product velocity to the snapped pelvis (zeros). Pads stay put."""
        obj = self.env.scene["object"]
        vel = torch.zeros(1, 6, device=self.env.device, dtype=obj.data.root_pos_w.dtype)
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
            if self.fsm.phase is CesPickPlacePhase.SETTLE:
                self._err_logged = False

            walk_active = cmd.guide
            walk_failed_stop = (
                self.fsm.phase is CesPickPlacePhase.FAILED and self._squeeze
            )
            policy_active = walk_active or walk_failed_stop

            if not cmd.guide and cmd.snap_xy is not None and cmd.snap_yaw is not None:
                self._hold_xy, self._hold_yaw = cmd.snap_xy, cmd.snap_yaw
                self._hold_z = cmd.snap_z
            pin_quat = None
            if policy_active:
                pin_xy, pin_yaw, pin_z = None, None, None
                self._product_brake_pending = True
            elif cmd.snap_xy is not None and cmd.snap_yaw is not None:
                pin_xy = cmd.snap_xy
                pin_yaw = cmd.snap_yaw
                pin_z = cmd.snap_z
                pin_quat = cmd.snap_quat
            else:
                pin_xy = self._hold_xy
                pin_yaw = self._hold_yaw
                pin_z = self._hold_z
                if pin_xy is None:
                    pin_xy, pin_yaw, pin_z = C.SPAWN_STAND_XY, C.SPAWN_STAND_YAW, None

            target_walk = cmd.walk if walk_active else [0.0, 0.0, 0.0, 0.8]
            if policy_active:
                # pick 结束 / 右转接前进：直接拉满，不从零或变号 ramp（过 0 会停步）。
                if self._walk_prime is not None:
                    self._walk_cmd = list(self._walk_prime)
                    self._walk_prime = None
                    self._begin_walk_policy(self._walk_cmd)
                elif (
                    self._walk_cmd[0] * target_walk[0] < 0.0
                    and abs(target_walk[0]) >= 0.30
                ):
                    self._walk_cmd = [float(target_walk[i]) for i in range(4)]
                else:
                    self._filter_walk_command(target_walk)
                policy_action = self.run_policy(self._walk_cmd)
            else:
                if self._walk_prime is not None:
                    self._walk_cmd = list(self._walk_prime)
                else:
                    self.reset_walk_filt()
                self.compute_observations(self._walk_cmd)
                policy_action = None

            full_action = self._full_action_buf
            full_action.zero_()
            if policy_action is not None:
                full_action[self.action_to_indices] = policy_action.reshape(-1)
            elif self._squeeze and self._last_policy_legs is not None:
                # 仅到站钉盆：腿保持走路末姿态。走路时不要改腰，否则会越走越歪。
                full_action[self.action_to_indices] = self._last_policy_legs
            else:
                full_action[self.action_to_indices] = self._leg_default[0]
            full_action[self.waist_to_all_indices] = self.default_waist_positions[0]
            full_action[self._left_arm_idx] = self._left_arm_default[0]
            if cmd.arm_q is not None:
                if self.fsm.phase is CesPickPlacePhase.DESCEND:
                    print("[ces_verify] ERROR arm_q hard-set during DESCEND (40 must be q_ref only)")
                full_action[self._right_arm_idx] = self._slew_arm(
                    cmd.arm_q.to(full_action.dtype)
                )
            elif cmd.tcp_pos is not None:
                try:
                    q_des = self.ik.solve(
                        cmd.tcp_pos,
                        cmd.tcp_quat,
                        q_ref=cmd.arm_q_ref,
                    )
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
                if self._grip_cmd is None:
                    self._grip_cmd = C.GRIPPER_CLOSED
            if self.fsm.phase in (
                CesPickPlacePhase.RELEASE,
                CesPickPlacePhase.RETRACT,
                CesPickPlacePhase.SETTLE,
            ):
                self._squeeze = False
                self._grip_cmd = None

            history_action = full_action.clone()
            if policy_action is None:
                # The policy stores raw leg offsets in its action history, not
                # the default-offset joint targets used while the pelvis is pinned.
                history_action[self.action_to_indices] = 0.0
            delayed_actions = self.advance_action_history(history_action)
            if policy_action is not None:
                self.apply_delayed_policy_legs(full_action, delayed_actions)
                self._last_policy_legs = full_action[self.action_to_indices].clone()

            # 夹持力只在 RELEASE 才改。写在所有 action 之后，避免被腿/历史缓冲覆盖。
            if self._squeeze and self._grip_cmd is not None:
                full_action[self._right_grip_idx] = self._grip_cmd
            else:
                full_action[self._right_grip_idx] = cmd.gripper
            full_action[self._left_grip_idx] = C.GRIPPER_OPEN

            lock_upper = (
                cmd.arm_q is not None or cmd.tcp_pos is not None or self._squeeze
            )
            # 夹持中手臂只走 PD，write_joint_state 会把垫面瞬移开。
            kinematic_arm = lock_upper and not self._squeeze
            robot = self.env.scene["robot"]
            for _ in range(4):
                if pin_xy is not None and pin_yaw is not None:
                    self._apply_snap(pin_xy, pin_yaw, pin_z, quat=pin_quat)
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
