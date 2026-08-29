# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""Pick, grasp, lift, and return handlers for the CES Baseline FSM."""
from __future__ import annotations

import math
import logging

import torch
from isaaclab.utils.math import (
    quat_apply,
    quat_from_angle_axis,
    quat_from_matrix,
    quat_mul,
)

from action_provider.ces_grasp import constants as C
from action_provider.ces_grasp.fsm_types import CesPickPlacePhase
from action_provider.ces_grasp.grasp_yaw import (
    YAW_ALIGN_SKIP_RAD,
    closer_world_x_yaw,
    jaw_xy_yaw,
)
from action_provider.ces_grasp.interpolation import ease_in_out

logger = logging.getLogger("ces")


def top_down_grasp_quat(jaw_axis_w: torch.Tensor) -> torch.Tensor:
    """Return a top-down hand quaternion whose +X follows the horizontal jaw axis."""
    x = jaw_axis_w.clone()
    x[:, 2] = 0.0
    x = x / torch.clamp(torch.norm(x, dim=-1, keepdim=True), min=1e-6)
    down = torch.zeros_like(x)
    down[:, 2] = -1.0
    z = torch.linalg.cross(x, down)
    z = z / torch.clamp(torch.norm(z, dim=-1, keepdim=True), min=1e-6)
    y = torch.linalg.cross(z, x)
    return quat_from_matrix(torch.stack((x, y, z), dim=-1))


class CesPickMixin:
    """Handlers from SETTLE through RETURN_HOME."""

    def _offset_z(self, pos: torch.Tensor, dz: float) -> torch.Tensor:
        out = pos.clone()
        out[:, 2] += dz
        return out

    def _retract_from_ces(self, pos: torch.Tensor, dist: float) -> torch.Tensor:
        out = pos.clone()
        fwd, _ = C.forward_left(C.PICK_STAND_YAW)
        out[:, 0] -= dist * fwd[0]
        out[:, 1] -= dist * fwd[1]
        return out

    def _into_drawer(self, pos: torch.Tensor, dist: float) -> torch.Tensor:
        return self._retract_from_ces(pos, -dist)

    def _plan_grasp(self):
        pivot, _ = self.ctx.get_object_pose_w()
        obj_pos, _ = self.ctx.get_product_aabb_center_w()
        grasp = self._into_drawer(obj_pos, C.GRASP_INSET)
        grasp = self._offset_z(grasp, C.GRASP_Z_OFFSET)
        grasp[:, 1] += C.GRASP_SHIFT_Y
        self._grasp_pos_w = grasp
        p = [float(grasp[0, i]) for i in range(3)]
        c = [float(obj_pos[0, i]) for i in range(3)]
        v = [float(pivot[0, i]) for i in range(3)]
        logger.debug(
            f"[ces_fsm] Product pivot=({v[0]:.4f},{v[1]:.4f},{v[2]:.4f})  "
            f"AABB=({c[0]:.4f},{c[1]:.4f},{c[2]:.4f})  "
            f"grasp=({p[0]:.4f},{p[1]:.4f},{p[2]:.4f}) "
            f"inset={C.GRASP_INSET:.3f} y_shift={C.GRASP_SHIFT_Y:.3f} "
            f"z_off={C.GRASP_Z_OFFSET:.3f}"
        )
        jaw = torch.tensor([[1.0, 0.0, 0.0]], device=self.device, dtype=obj_pos.dtype)
        self._grasp_quat_w = top_down_grasp_quat(jaw)

    def _at_pick_stand(self) -> bool:
        if self.t < C.STAND_MIN_TIME:
            return False
        if self.t > 6.0:
            logger.warning("[ces_fsm] stand timeout - starting arm anyway")
            return True
        if self.ctx.is_standing():
            self.hold += self.ctx.dt
        else:
            self.hold = 0.0
        return self.hold >= C.STAND_STABLE_TIME

    def _start_grasp(self, err: float):
        now, _ = self.ctx.ik.get_tcp_pose_w()
        planned_z = float(self._grasp_pos_w[0, 2])
        self._grasp_pos_w = now.clone()
        if float(self._grasp_pos_w[0, 2]) < planned_z:
            self._grasp_pos_w[0, 2] = planned_z
        logger.info(
            f"[ces_fsm] close gripper, tcp_err={err*1000:.1f} mm "
            f"z={float(self._grasp_pos_w[0, 2]):.3f} (min {planned_z:.3f})"
        )
        self._grasp_arm_q = self.ctx.get_right_arm_q()[0].clone()
        logger.debug("[ces_fsm] grasp q=%s tcp_w=%s", self._grasp_arm_q, now[0])
        self._transition(CesPickPlacePhase.GRASP)

    def _begin_joint_lift(self):
        q_now = self.ctx.get_right_arm_q()[0].clone()
        if self._grasp_arm_q is None:
            self._grasp_arm_q = q_now
        up = self._offset_z(self._grasp_pos_w, C.LIFT_HEIGHT)
        up[:, 1] += C.LIFT_SHIFT_Y
        try:
            q_lift = self.ctx.ik.solve(up, self._grasp_quat_w, q_ref=q_now)[0]
        except Exception as exc:
            logger.warning("[ces_fsm] lift IK failed (%s) - hold grasp q", exc)
            q_lift = q_now
        self.joint_interp.reset(q_now, q_lift, self._lift_time)
        logger.info(
            f"[ces_fsm] joint lift z+{C.LIFT_HEIGHT:.3f} y{C.LIFT_SHIFT_Y:+.3f} "
            f"dur={self._lift_time:.2f}s (no per-frame IK)"
        )

    def _begin_return_home(self):
        """Play 40(live)->30 separately, then the approved 30->20->05 path."""
        q_now = self.ctx.get_right_arm_q()[0].clone()
        names = list(self._trajectory.return_waypoints)
        durations = list(self._return_segment_times)
        authored_qs = [self._pose_q(name) for name in names]
        self._return_followup_qs = []
        self._return_followup_durations = []
        self._return_followup_method = self._trajectory.return_interpolation_method
        self.joint_interp.reset(
            q_now,
            authored_qs[0],
            durations[0],
            method="segment_smoothstep",
        )
        if len(authored_qs) > 1:
            self._return_followup_qs = [authored_qs[0], *authored_qs[1:]]
            self._return_followup_durations = durations[1:]
        self._return_total_time = sum(durations)
        path = f"{self._trajectory.return_start}(live)→{'→'.join(names)}"
        logger.info(
            f"[ces_fsm] return arm to carry posture {path} "
            f"dur={self._return_total_time:.2f}s "
            f"lead_interp=segment_smoothstep "
            f"path_interp={self._trajectory.return_interpolation_method} "
            f"(gripper stays closed)"
        )

    def _hand_jaw_yaw(self, quat_w: torch.Tensor) -> float:
        axis = torch.tensor([[1.0, 0.0, 0.0]], device=quat_w.device, dtype=quat_w.dtype)
        jaw = quat_apply(quat_w, axis)
        return jaw_xy_yaw(float(jaw[0, 0]), float(jaw[0, 1]))

    def _yaw_align_grasp_quat(
        self, live_quat: torch.Tensor
    ) -> tuple[torch.Tensor, float, float]:
        yaw = self._hand_jaw_yaw(live_quat)
        target, delta = closer_world_x_yaw(yaw)
        axis = torch.tensor(
            [[0.0, 0.0, 1.0]], device=live_quat.device, dtype=live_quat.dtype
        )
        dq = quat_from_angle_axis(live_quat.new_tensor([delta]), axis)
        return quat_mul(dq, live_quat), target, delta

    def _begin_descend(self):
        now, live_quat = self.ctx.ik.get_tcp_pose_w()
        planned_z = float(self._grasp_pos_w[0, 2])
        goal = now.clone()
        goal[:, 2] = planned_z
        self._grasp_pos_w = goal.clone()
        aligned, target_yaw, delta = self._yaw_align_grasp_quat(live_quat)
        self._grasp_quat_w = aligned
        align_time = 1e-3 if abs(delta) < YAW_ALIGN_SKIP_RAD else C.GRASP_YAW_ALIGN_TIME
        self.interp.reset_path(
            [now, now.clone(), goal],
            [align_time, C.DESCEND_TIME],
            quats=[live_quat, aligned, aligned],
        )
        axis_name = "+X" if abs(target_yaw) < 1e-6 else "-X"
        logger.info(
            f"[ces_fsm] waypoint handoff yaw-align then descend "
            f"jaw_yaw={math.degrees(self._hand_jaw_yaw(live_quat)):+.1f}deg "
            f"-> world {axis_name} ({math.degrees(delta):+.1f}deg, {align_time:.2f}s) "
            f"z={float(now[0, 2]):.3f}->{planned_z:.3f} "
            f"(hold XY from 30; 40 is q_ref on the drop)"
        )
        self._transition(CesPickPlacePhase.DESCEND)

    def _handoff_cmd(self):
        tcp = self.interp.points[0] if self.interp.has_path else self._grasp_pos_w
        quat = self.interp.quats[0] if self.interp.quats else self._grasp_quat_w
        return self._cmd(tcp=tcp, quat=quat, arm_q_ref=self._q_ref_for_descend())

    def _q_ref_for_descend(self) -> torch.Tensor | None:
        if self._q_wp30 is None or self._q_wp40 is None:
            return None
        if not self.interp.has_path:
            return self._q_wp40
        align_end = self.interp.bounds[0] if len(self.interp.bounds) >= 2 else 0.0
        if self.interp.elapsed < align_end:
            return self._q_wp30
        drop_length = max(1e-6, self.interp.duration - align_end)
        s = min(1.0, (self.interp.elapsed - align_end) / drop_length)
        return torch.lerp(self._q_wp30, self._q_wp40, ease_in_out(s))

    def _pose_q(self, pose_name: str) -> torch.Tensor:
        return torch.tensor(
            self._trajectory.q_by_name[pose_name],
            device=self.device,
            dtype=torch.float32,
        )

    def _start_joint_waypoints(self):
        q_now = self.ctx.get_right_arm_q()[0].clone()
        names = self._trajectory.joint_waypoints
        qs = [self._pose_q(name) for name in names]
        self._q_wp30 = qs[-1].clone()
        self._q_wp40 = self._pose_q(self._trajectory.q_ref_to)
        durations = list(self._joint_segment_times)
        lead = torch.norm(q_now - qs[0]).item()
        if lead > C.WAYPOINT_LEAD_IN_TOL:
            qs = [q_now] + qs
            durations = [self._wp_lead_in_time] + durations
            logger.info(
                f"[ces_fsm] joint path lead-in {lead:.3f} rad "
                f"{'→'.join(names)} dur={sum(durations):.2f}s"
            )
        else:
            qs[0] = q_now
            logger.info(
                f"[ces_fsm] joint path {'→'.join(names)} "
                f"dur={sum(durations):.2f}s (start≈00)"
            )
        self.joint_interp.reset_path(qs, durations, method=self._trajectory.interpolation_method)

    def _step_settle(self):
        self.gripper = C.GRIPPER_OPEN
        if self.t >= C.SETTLE_TIME and (
            self.ctx.is_standing() or self.t >= C.SETTLE_TIME + C.STAND_MIN_TIME
        ):
            self._plan_grasp()
            root_pos, _ = self.ctx.get_base_pose_w()
            yaw = self.ctx.get_heading()
            x, y = float(root_pos[0, 0]), float(root_pos[0, 1])
            px, py = C.PICK_STAND_XY
            logger.info(
                f"[ces_fsm] settled on the pick stand "
                f"robot=({x:.2f},{y:.2f}) head={math.degrees(yaw):.0f}° "
                f"pick=({px:.2f},{py:.2f}) d={math.hypot(px-x, py-y):.2f} "
                f"(spawned here, no teleport)"
            )
            self._transition(CesPickPlacePhase.GOTO_PICK)
            if hasattr(self.ctx, "reset_walk_filt"):
                self.ctx.reset_walk_filt()
        return self._cmd()

    def _step_goto_pick(self):
        self.gripper = C.GRIPPER_OPEN
        if self._at_pick_stand():
            logger.info("[ces_fsm] at pick stand - joint waypoints 00->30")
            self._transition(CesPickPlacePhase.UNFOLD)
        return self._cmd()

    def _step_unfold(self):
        self.gripper = C.GRIPPER_OPEN
        if not self.joint_interp.has_path:
            self._start_joint_waypoints()
        q = self.joint_interp.step(self.ctx.dt)
        if q is None:
            q = self.ctx.get_right_arm_q()[0]
        if self.joint_interp.finished:
            self._begin_descend()
            return self._handoff_cmd()
        return self._cmd(arm_q=q)

    def _step_descend(self):
        pos, quat = self.interp.step(self.ctx.dt)
        if pos is None or quat is None:
            pos, quat = self._grasp_pos_w, self._grasp_quat_w
        q_ref = self._q_ref_for_descend()
        descend_limit = (
            self.interp.duration if self.interp.has_path else C.DESCEND_TIME
        ) + C.GRASP_WAIT_MAX
        if self.interp.finished or self.t > descend_limit:
            err = self.ctx.ik.position_error_norm(self._grasp_pos_w)
            self._start_grasp(err)
        return self._cmd(tcp=pos, quat=quat, arm_q_ref=q_ref)

    def _step_grasp(self):
        s = ease_in_out(min(1.0, self.t / max(C.GRASP_TIME, 1e-3)))
        self.gripper = C.GRIPPER_OPEN + s * (C.GRIPPER_CLOSED - C.GRIPPER_OPEN)
        q = self._grasp_arm_q
        if q is None:
            q = self.ctx.get_right_arm_q()[0].clone()
            self._grasp_arm_q = q
        if self.t >= C.GRASP_TIME:
            self.gripper = C.GRIPPER_CLOSED
            self._begin_joint_lift()
            self._transition(CesPickPlacePhase.LIFT)
        return self._cmd(arm_q=q)

    def _step_lift(self):
        self.gripper = C.GRIPPER_CLOSED
        q = self.joint_interp.step(self.ctx.dt)
        if q is None:
            q = self._grasp_arm_q
        if q is None:
            q = self.ctx.get_right_arm_q()[0]
        if self.joint_interp.finished or self.t > self._lift_time + 0.5:
            self._carry_arm_q = q.clone()
            logger.info("[ces_fsm] lift complete")
            logger.debug("[ces_fsm] lift q=%s", q)
            self._begin_return_home()
            self._transition(CesPickPlacePhase.RETURN_HOME)
        return self._cmd(arm_q=q)

    def _step_return_home(self):
        self.gripper = C.GRIPPER_CLOSED
        q = self.joint_interp.step(self.ctx.dt)
        if q is None:
            q = self._carry_arm_q
        if q is None:
            q = self.ctx.get_right_arm_q()[0]
        if self.joint_interp.finished and self._return_followup_qs:
            followup_qs = self._return_followup_qs
            followup_durations = self._return_followup_durations
            self._return_followup_qs = []
            self._return_followup_durations = []
            self.joint_interp.reset_path(
                followup_qs,
                followup_durations,
                method=self._return_followup_method,
            )
            logger.debug(
                f"[ces_fsm] return lead-in complete — play authored "
                f"follow-up path interp={self._return_followup_method}"
            )
            return self._cmd(arm_q=q)
        if self.joint_interp.finished or self.t > self._return_total_time + 1.0:
            self._carry_arm_q = q.clone()
            logger.info("[ces_fsm] arm at carry posture")
            logger.debug("[ces_fsm] carry q=%s", q)
            self._transition(CesPickPlacePhase.CARRY)
        return self._cmd(arm_q=q)
