# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""Finite state machine that sequences the tray grasp-and-lift behaviour.

Phases::

    SETTLE  -> let physics settle, arms in the ready pose, grippers open
    WALK    -> closed-loop base navigation towards the stand pose in front of
               the tray (drives the whole-body policy's velocity command)
    APPROACH-> move both hands above their handle (Cartesian interpolation)
    DESCEND -> lower both hands onto the handles
    GRASP   -> close both grippers and dwell
    LIFT    -> raise both hands (and the grasped tray) by a fixed height
    HOLD    -> keep the tray lifted

The FSM only produces *targets* (a base velocity command, per-hand Cartesian
goals and a gripper command); turning the Cartesian goals into joint angles is
the IK solver's job and keeping the robot upright is the whole-body policy's
job.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import torch

from .interpolation import CartesianInterpolator


class GraspPhase(Enum):
    SETTLE = 0
    WALK = 1
    APPROACH = 2
    DESCEND = 3
    GRASP = 4
    LIFT = 5
    HOLD = 6


@dataclass
class SMOutput:
    command: torch.Tensor          # [1, 4] -> [vx, vy, wz, base_height] for the RL policy
    arms_active: bool              # whether IK targets should be tracked
    left_target_pos_w: torch.Tensor | None
    right_target_pos_w: torch.Tensor | None
    gripper_value: float           # commanded gripper joint position (open/closed)
    phase: GraspPhase


class TrayGraspStateMachine:
    def __init__(self, ctx, cfg: dict | None = None):
        """
        Args:
            ctx: the action provider, providing runtime accessors:
                 ``device``, ``base_height_default``,
                 ``get_base_pose_w() -> (pos[1,3], yaw)``,
                 ``get_handle_positions_w() -> (left[1,3], right[1,3])``,
                 ``left_ik`` / ``right_ik`` (with ``get_ee_pose_w()``).
            cfg: optional overrides for the tunable parameters below.
        """
        self.ctx = ctx
        self.device = ctx.device
        c = cfg or {}

        # -- navigation ----------------------------------------------------
        # Unicycle-style go-to-pose: while far away the robot turns to face the
        # stand point and walks forward (vx + wz, which locomotion policies
        # track much better than lateral vy); once close it fine-tunes the
        # position and aligns its heading to +X (yaw = 0) so it faces the tray.
        self.stand_distance = c.get("stand_distance", 0.52)   # m in front of the tray mid-point
        self.stand_lateral = c.get("stand_lateral", 0.0)      # m lateral offset of the stand point
        self.stand_distance = c.get("stand_distance", 0.48)   # (overrides above default)
        self.approach_run = c.get("approach_run", 0.40)       # straight final leg length (m)
        self.nav_kp = c.get("nav_kp", 1.5)                    # forward gain
        self.nav_kyaw = c.get("nav_kyaw", 1.8)
        self.nav_vmax = c.get("nav_vmax", 0.45)
        self.nav_vymax = c.get("nav_vymax", 0.20)
        self.nav_wmax = c.get("nav_wmax", 0.8)
        self.nav_vx_floor = c.get("nav_vx_floor", 0.18)       # min forward speed to beat gait deadband
        self.wp_tol = c.get("wp_tol", 0.16)                   # waypoint-1 arrival radius
        self.turn_tol = c.get("turn_tol", 0.12)               # heading alignment tolerance
        self.pos_tol = c.get("pos_tol", 0.10)
        self.yaw_tol = c.get("yaw_tol", 0.15)
        self.t_goto_wp1 = c.get("t_goto_wp1", 14.0)           # per-stage timeouts
        self.t_turn = c.get("t_turn", 5.0)
        self.t_goto_stand = c.get("t_goto_stand", 9.0)
        self.settle_time = c.get("settle_time", 1.0)
        self._walk_diag = 0

        # -- reach / grasp -------------------------------------------------
        # handle grasp point offset expressed in world frame (added to the raw
        # handle centre); tuned so the gripper base sits around the handle.
        self.grasp_offset_w = torch.tensor(
            c.get("grasp_offset_w", [0.0, 0.0, 0.02]), device=self.device
        ).unsqueeze(0)
        self.approach_offset_w = torch.tensor(
            c.get("approach_offset_w", [-0.10, 0.0, 0.12]), device=self.device
        ).unsqueeze(0)  # above and slightly back (towards the robot, -X)
        self.lift_offset_w = torch.tensor(
            c.get("lift_offset_w", [0.0, 0.0, 0.18]), device=self.device
        ).unsqueeze(0)

        self.approach_time = c.get("approach_time", 2.5)
        self.descend_time = c.get("descend_time", 2.0)
        self.grasp_time = c.get("grasp_time", 1.5)
        self.lift_time = c.get("lift_time", 3.0)

        self.gripper_open = c.get("gripper_open", 0.033)
        self.gripper_closed = c.get("gripper_closed", -0.02)

        # -- internal state ------------------------------------------------
        self.phase = GraspPhase.SETTLE
        self.phase_time = 0.0
        self._stand_pos_w: torch.Tensor | None = None
        self._left_grasp_w: torch.Tensor | None = None
        self._right_grasp_w: torch.Tensor | None = None
        self._left_interp = CartesianInterpolator(self.device)
        self._right_interp = CartesianInterpolator(self.device)
        self._left_target = None
        self._right_target = None
        self._walk_hold = 0.0
        self._wp1_w: torch.Tensor | None = None
        self._walk_sub = 0        # 0: goto wp1, 1: turn to +X, 2: goto stand
        self._sub_t0 = 0.0

    # --------------------------------------------------------------- utils --
    @staticmethod
    def _wrap(a: float) -> float:
        return (a + math.pi) % (2.0 * math.pi) - math.pi

    def _zero_cmd(self) -> torch.Tensor:
        return torch.tensor(
            [[0.0, 0.0, 0.0, self.ctx.base_height_default]], device=self.device
        )

    def _plan_grasp_points(self):
        left_h, right_h = self.ctx.get_handle_positions_w()
        self._left_grasp_w = left_h + self.grasp_offset_w
        self._right_grasp_w = right_h + self.grasp_offset_w
        mid = 0.5 * (left_h + right_h)
        # stand in front of the tray on the -X side, facing +X (yaw = 0)
        stand = mid.clone()
        stand[0, 0] = mid[0, 0] - self.stand_distance
        stand[0, 1] = mid[0, 1] + self.stand_lateral
        self._stand_pos_w = stand
        # intermediate way-point directly behind the stand point (-X) so the
        # final leg is a straight forward walk that ends facing +X
        wp1 = stand.clone()
        wp1[0, 0] = stand[0, 0] - self.approach_run
        self._wp1_w = wp1

    def _clip(self, v, lo, hi):
        return max(lo, min(hi, v))

    def _mk_cmd(self, vx, vy, wz) -> torch.Tensor:
        return torch.tensor([[vx, vy, wz, self.ctx.base_height_default]], device=self.device)

    def _unicycle_to(self, bx, by, base_yaw, target) -> tuple[torch.Tensor, float]:
        """Turn towards ``target`` (world xy) and walk forward. Returns (cmd, dist)."""
        tx, ty = float(target[0, 0]), float(target[0, 1])
        ex, ey = tx - bx, ty - by
        dist = math.hypot(ex, ey)
        heading = math.atan2(ey, ex)
        myaw = self._wrap(heading - base_yaw)
        gate = max(0.0, math.cos(myaw))
        vx = self._clip(self.nav_kp * dist * gate, 0.0, self.nav_vmax)
        if dist > self.wp_tol and gate > 0.5:
            vx = max(vx, self.nav_vx_floor)  # beat the gait deadband while far
        wz = self._clip(self.nav_kyaw * myaw, -self.nav_wmax, self.nav_wmax)
        return self._mk_cmd(vx, 0.0, wz), dist

    def _straight_to(self, bx, by, base_yaw, target) -> tuple[torch.Tensor, float, float]:
        """Walk to ``target`` while holding yaw = 0 (facing +X)."""
        tx, ty = float(target[0, 0]), float(target[0, 1])
        ex, ey = tx - bx, ty - by
        dist = math.hypot(ex, ey)
        cy, sy = math.cos(base_yaw), math.sin(base_yaw)
        vx = self._clip(self.nav_kp * (cy * ex + sy * ey), -self.nav_vmax, self.nav_vmax)
        vy = self._clip(self.nav_kp * (-sy * ex + cy * ey), -self.nav_vymax, self.nav_vymax)
        yaw_err = self._wrap(0.0 - base_yaw)
        wz = self._clip(self.nav_kyaw * yaw_err, -self.nav_wmax, self.nav_wmax)
        return self._mk_cmd(vx, vy, wz), dist, abs(yaw_err)

    # -------------------------------------------------------------- update --
    def update(self, dt: float) -> SMOutput:
        self.phase_time += dt
        method = getattr(self, f"_phase_{self.phase.name.lower()}")
        return method(dt)

    def _transition(self, new_phase: GraspPhase):
        print(f"[TrayFSM] {self.phase.name} -> {new_phase.name} (t={self.phase_time:.2f}s)")
        self.phase = new_phase
        self.phase_time = 0.0

    # -------------------------------------------------------------- phases --
    def _phase_settle(self, dt: float) -> SMOutput:
        if self.phase_time >= self.settle_time:
            self._plan_grasp_points()
            print(f"[TrayFSM] stand target = {self._stand_pos_w.cpu().numpy().tolist()}")
            print(f"[TrayFSM] left grasp   = {self._left_grasp_w.cpu().numpy().tolist()}")
            print(f"[TrayFSM] right grasp  = {self._right_grasp_w.cpu().numpy().tolist()}")
            self._transition(GraspPhase.WALK)
        return SMOutput(self._zero_cmd(), False, None, None, self.gripper_open, self.phase)

    def _begin_reach(self):
        left_ee, _ = self.ctx.left_ik.get_ee_pose_w()
        right_ee, _ = self.ctx.right_ik.get_ee_pose_w()
        self._left_interp.reset(left_ee, self._left_grasp_w + self.approach_offset_w, self.approach_time)
        self._right_interp.reset(right_ee, self._right_grasp_w + self.approach_offset_w, self.approach_time)
        self._transition(GraspPhase.APPROACH)

    def _phase_walk(self, dt: float) -> SMOutput:
        base_pos, base_yaw = self.ctx.get_base_pose_w()
        bx, by = float(base_pos[0, 0]), float(base_pos[0, 1])
        sub_t = self.phase_time - self._sub_t0
        cmd = self._mk_cmd(0.0, 0.0, 0.0)
        dist = 0.0
        yaw_err = abs(self._wrap(0.0 - base_yaw))

        if self._walk_sub == 0:
            # ---- stage 0: unicycle drive to the way-point behind the stand ----
            cmd, dist = self._unicycle_to(bx, by, base_yaw, self._wp1_w)
            if dist < self.wp_tol or sub_t > self.t_goto_wp1:
                print(f"[TrayFSM][WALK] wp1 reached (dist={dist:.3f}) -> turn")
                self._walk_sub = 1
                self._sub_t0 = self.phase_time
        elif self._walk_sub == 1:
            # ---- stage 1: turn in place to face +X ----------------------------
            wz = self._clip(self.nav_kyaw * self._wrap(0.0 - base_yaw), -self.nav_wmax, self.nav_wmax)
            cmd = self._mk_cmd(0.0, 0.0, wz)
            if yaw_err < self.turn_tol or sub_t > self.t_turn:
                print(f"[TrayFSM][WALK] aligned (yaw_err={yaw_err:.3f}) -> go straight")
                self._walk_sub = 2
                self._sub_t0 = self.phase_time
        else:
            # ---- stage 2: straight walk to the stand point (facing +X) --------
            cmd, dist, yaw_err = self._straight_to(bx, by, base_yaw, self._stand_pos_w)
            arrived = dist < self.pos_tol and yaw_err < self.yaw_tol
            if arrived:
                self._walk_hold += dt
            else:
                self._walk_hold = 0.0
            if self._walk_hold > 0.4 or sub_t > self.t_goto_stand:
                print(f"[TrayFSM] reached stand pose (dist={dist:.3f}, yaw_err={yaw_err:.3f})")
                self._begin_reach()
                left_ee, _ = self.ctx.left_ik.get_ee_pose_w()
                right_ee, _ = self.ctx.right_ik.get_ee_pose_w()
                return SMOutput(self._mk_cmd(0.0, 0.0, 0.0), True, left_ee, right_ee, self.gripper_open, self.phase)

        self._walk_diag += 1
        if self._walk_diag % 50 == 0:
            c = cmd[0].cpu().numpy()
            print(f"[TrayFSM][WALK{self._walk_sub}] t={sub_t:.1f}s base=({bx:+.2f},{by:+.2f}) "
                  f"yaw={base_yaw:+.2f} dist={dist:.3f} yaw_err={yaw_err:.3f} "
                  f"cmd=[vx{c[0]:+.2f} vy{c[1]:+.2f} wz{c[2]:+.2f}]")
        return SMOutput(cmd, False, None, None, self.gripper_open, self.phase)

    def _phase_approach(self, dt: float) -> SMOutput:
        lt, _ = self._left_interp.step(dt)
        rt, _ = self._right_interp.step(dt)
        self._left_target, self._right_target = lt, rt
        if self._left_interp.finished and self._right_interp.finished:
            self._left_interp.reset(lt, self._left_grasp_w, self.descend_time)
            self._right_interp.reset(rt, self._right_grasp_w, self.descend_time)
            self._transition(GraspPhase.DESCEND)
        return SMOutput(self._zero_cmd(), True, lt, rt, self.gripper_open, self.phase)

    def _phase_descend(self, dt: float) -> SMOutput:
        lt, _ = self._left_interp.step(dt)
        rt, _ = self._right_interp.step(dt)
        self._left_target, self._right_target = lt, rt
        if self._left_interp.finished and self._right_interp.finished:
            self._transition(GraspPhase.GRASP)
        return SMOutput(self._zero_cmd(), True, lt, rt, self.gripper_open, self.phase)

    def _phase_grasp(self, dt: float) -> SMOutput:
        # hold at the grasp points and close the grippers
        if self.phase_time >= self.grasp_time:
            self._left_interp.reset(self._left_grasp_w, self._left_grasp_w + self.lift_offset_w, self.lift_time)
            self._right_interp.reset(self._right_grasp_w, self._right_grasp_w + self.lift_offset_w, self.lift_time)
            self._transition(GraspPhase.LIFT)
        return SMOutput(
            self._zero_cmd(), True, self._left_grasp_w, self._right_grasp_w, self.gripper_closed, self.phase
        )

    def _phase_lift(self, dt: float) -> SMOutput:
        lt, _ = self._left_interp.step(dt)
        rt, _ = self._right_interp.step(dt)
        self._left_target, self._right_target = lt, rt
        if self._left_interp.finished and self._right_interp.finished:
            self._transition(GraspPhase.HOLD)
        return SMOutput(self._zero_cmd(), True, lt, rt, self.gripper_closed, self.phase)

    def _phase_hold(self, dt: float) -> SMOutput:
        lift_l = self._left_grasp_w + self.lift_offset_w
        lift_r = self._right_grasp_w + self.lift_offset_w
        return SMOutput(self._zero_cmd(), True, lift_l, lift_r, self.gripper_closed, self.phase)
