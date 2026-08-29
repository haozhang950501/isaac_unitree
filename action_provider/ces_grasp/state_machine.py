# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""CES Baseline FSM facade.

The only supported runtime path is:
00→10→20→30 → 40(q_ref only) → grasp/lift → 40(live)→30→20→05
→ carry-walk → pin the live pelvis → 05→15 → release → 15→05.
"""
from __future__ import annotations

import math

import torch

from action_provider.ces_grasp import constants as C
from action_provider.ces_grasp.fsm_pick import CesPickMixin, top_down_grasp_quat
from action_provider.ces_grasp.fsm_place import CesPlaceMixin
from action_provider.ces_grasp.fsm_types import CesCommand, CesPickPlacePhase
from action_provider.ces_grasp.fsm_walk import CesWalkMixin
from action_provider.ces_grasp.pose_library import CesTrajectory, load_baseline_trajectory
from action_provider.manip_common import CartesianInterpolator, JointSpaceInterpolator
from action_provider.manip_common.interpolation import scale_segment_times

__all__ = [
    "CesCommand",
    "CesPickPlacePhase",
    "CesPickPlaceStateMachine",
    "top_down_grasp_quat",
]


class CesPickPlaceStateMachine(CesPickMixin, CesWalkMixin, CesPlaceMixin):
    """Shared-state facade with phase handlers grouped in three focused modules."""

    _PHASE_HANDLERS = {
        CesPickPlacePhase.SETTLE: "_step_settle",
        CesPickPlacePhase.GOTO_PICK: "_step_goto_pick",
        CesPickPlacePhase.UNFOLD: "_step_unfold",
        CesPickPlacePhase.DESCEND: "_step_descend",
        CesPickPlacePhase.GRASP: "_step_grasp",
        CesPickPlacePhase.LIFT: "_step_lift",
        CesPickPlacePhase.RETURN_HOME: "_step_return_home",
        CesPickPlacePhase.CARRY: "_step_carry",
        CesPickPlacePhase.GOTO_PLACE: "_step_goto_place",
        CesPickPlacePhase.PLACE_HOLD: "_step_place_hold",
        CesPickPlacePhase.PLACE_APPROACH: "_step_place_approach",
        CesPickPlacePhase.RELEASE: "_step_release",
        CesPickPlacePhase.RETRACT: "_step_retract",
        CesPickPlacePhase.DONE: "_step_done",
        CesPickPlacePhase.FAILED: "_step_failed",
    }

    def __init__(self, ctx, speed_scale: float | None = None):
        self.ctx = ctx
        self.device = ctx.device
        self.interp = CartesianInterpolator(self.device)
        self.joint_interp = JointSpaceInterpolator(self.device)
        self._trajectory: CesTrajectory = load_baseline_trajectory()
        self.speed_scale = C.clamp_pick_speed(speed_scale)

        self._joint_segment_times = tuple(
            self._scaled(
                self._trajectory.joint_segment_durations,
                C.PICK_SEGMENT_MIN_TIME,
            )
        )
        self._return_segment_times = tuple(
            self._scaled(
                self._trajectory.return_segment_durations,
                C.PICK_SEGMENT_MIN_TIME,
            )
        )
        self._place_segment_times = tuple(
            self._scaled(
                self._trajectory.place_segment_durations,
                C.PICK_SEGMENT_MIN_TIME,
            )
        )
        self._wp_lead_in_time = self._scaled([C.WAYPOINT_LEAD_IN_TIME])[0]
        self._lift_time = self._scaled([C.LIFT_TIME], C.PICK_SEGMENT_MIN_TIME)[0]

        self._planner = None
        self.phase = CesPickPlacePhase.SETTLE
        self.t = 0.0
        self.hold = 0.0
        self._log_t = 0.0
        self._walk = [0.0, 0.0, 0.0, 0.8]
        self.gripper = C.GRIPPER_OPEN
        self._grasp_pos_w = None
        self._grasp_quat_w = None
        self._grasp_arm_q = None
        self._carry_arm_q = None
        self._place_arm_q = None
        self._q_wp30 = None
        self._q_wp40 = None
        self._wp_arrive_names: list[str] = []
        self._wp_logged = 0
        self._verify_descend_xy = None
        self._logged_q_ref_once = False
        self._return_followup_qs: list[torch.Tensor] = []
        self._return_followup_durations: list[float] = []
        self._return_followup_method = self._trajectory.return_interpolation_method
        self._return_total_time = 0.0
        self._place_joint_total_time = 0.0
        self._dbg = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        self._walk_mode = "idle"
        self._walk_leg = 0
        self._turn_pinned_logged = False
        self._stopped_logged = False
        self._table_braked = False
        self._walk_unstable_t = 0.0
        self._place_lock_xy = None
        self._place_lock_yaw = None
        self._place_lock_z = None
        self._place_lock_quat = None
        self._place_hold_obj_z0 = None
        self._carry_drop_logged = False
        self._step_err_t = -10.0

        print(
            f"[ces_fsm] Baseline Smooth V1 walk-place "
            f"pick_stand=({C.PICK_STAND_XY[0]:.3f},{C.PICK_STAND_XY[1]:.3f}) "
            f"place_stand=({C.PLACE_STAND_XY[0]:.3f},{C.PLACE_STAND_XY[1]:.3f}) "
            f"arm_speed=x{self.speed_scale:.2f} "
            f"seg_s={'/'.join(f'{d:.2f}' for d in self._joint_segment_times)} "
            f"return={self._trajectory.return_start}(live)→"
            f"{'→'.join(self._trajectory.return_waypoints)} "
            f"return_s={'/'.join(f'{d:.2f}' for d in self._return_segment_times)} "
            f"place={self._trajectory.place_start}→"
            f"{'→'.join(self._trajectory.place_waypoints)} "
            f"place_s={'/'.join(f'{d:.2f}' for d in self._place_segment_times)}"
        )

    def _scaled(self, durations, min_time: float = 0.0) -> list[float]:
        return scale_segment_times(durations, self.speed_scale, min_time)

    def reset(self):
        self.phase = CesPickPlacePhase.SETTLE
        self.t = 0.0
        self.hold = 0.0
        self._log_t = 0.0
        self.gripper = C.GRIPPER_OPEN
        self._walk = [0.0, 0.0, 0.0, 0.8]
        self._grasp_pos_w = None
        self._grasp_quat_w = None
        self._grasp_arm_q = None
        self._carry_arm_q = None
        self._place_arm_q = None
        self._q_wp30 = None
        self._q_wp40 = None
        self._wp_arrive_names = []
        self._wp_logged = 0
        self._verify_descend_xy = None
        self._logged_q_ref_once = False
        self._return_followup_qs = []
        self._return_followup_durations = []
        self._return_total_time = 0.0
        self._place_joint_total_time = 0.0
        self.joint_interp.clear()
        self._dbg = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        self._walk_mode = "idle"
        self._walk_leg = 0
        if self._planner is not None:
            self._planner.reset()
        self._turn_pinned_logged = False
        self._stopped_logged = False
        self._table_braked = False
        self._walk_unstable_t = 0.0
        self._place_lock_xy = None
        self._place_lock_yaw = None
        self._place_lock_z = None
        self._place_lock_quat = None
        self._place_hold_obj_z0 = None
        self._carry_drop_logged = False
        self._step_err_t = -10.0

    def _transition(self, phase: CesPickPlacePhase):
        print(f"[ces_fsm] {self.phase.value} -> {phase.value}  t={self.t:.2f}s")
        self.phase = phase
        self.t = 0.0
        self.hold = 0.0
        if phase is CesPickPlacePhase.GOTO_PLACE and self._planner is not None:
            self._planner.reset()
            self._walk_leg = 0
            self._turn_pinned_logged = False
            self._stopped_logged = False
            self._table_braked = False

    def _cmd(
        self,
        tcp=None,
        quat=None,
        *,
        snap=False,
        snap_xy=None,
        snap_yaw=None,
        snap_z=None,
        snap_quat=None,
        guide=False,
        done=False,
        failed=False,
        arm_q=None,
        arm_q_ref=None,
    ) -> CesCommand:
        if not snap and not guide and self.phase in (
            CesPickPlacePhase.UNFOLD,
            CesPickPlacePhase.DESCEND,
            CesPickPlacePhase.GRASP,
            CesPickPlacePhase.LIFT,
            CesPickPlacePhase.RETURN_HOME,
        ):
            snap = True
            snap_xy = C.PICK_STAND_XY
            snap_yaw = C.PICK_STAND_YAW
        return CesCommand(
            tcp_pos=tcp,
            tcp_quat=quat,
            gripper=C.GRIPPER_CLOSED if self._squeezing() else self.gripper,
            walk=list(self._walk),
            snap_xy=snap_xy if snap else None,
            snap_yaw=snap_yaw if snap else None,
            guide=bool(guide),
            done=done,
            failed=failed,
            arm_q=arm_q,
            arm_q_ref=arm_q_ref,
            snap_z=snap_z if snap else None,
            snap_quat=snap_quat if snap else None,
        )

    def _squeezing(self) -> bool:
        if self.phase is CesPickPlacePhase.FAILED:
            return self._carry_arm_q is not None
        return self.phase in (
            CesPickPlacePhase.LIFT,
            CesPickPlacePhase.RETURN_HOME,
            CesPickPlacePhase.CARRY,
            CesPickPlacePhase.GOTO_PLACE,
            CesPickPlacePhase.PLACE_HOLD,
            CesPickPlacePhase.PLACE_APPROACH,
        )

    def _fmt_q(self, q: torch.Tensor) -> str:
        return "[" + ", ".join(f"{float(v):+.3f}" for v in q.reshape(-1)[:7]) + "]"

    def _maybe_log(self):
        self._log_t += self.ctx.dt
        interval = 0.25 if self.phase in (
            CesPickPlacePhase.GOTO_PICK,
            CesPickPlacePhase.GOTO_PLACE,
            CesPickPlacePhase.PLACE_HOLD,
            CesPickPlacePhase.PLACE_APPROACH,
        ) else 1.0
        if self._log_t < interval:
            return
        self._log_t = 0.0
        tcp_err = -1.0
        if self._grasp_pos_w is not None and self.phase in (
            CesPickPlacePhase.DESCEND,
            CesPickPlacePhase.GRASP,
            CesPickPlacePhase.LIFT,
        ):
            tcp_err = self.ctx.ik.position_error_norm(self._grasp_pos_w)
        extra = ""
        if self.phase is CesPickPlacePhase.GOTO_PLACE and self._walk_mode != "idle":
            x, y, remaining, yaw, lateral, _unused, yaw_err = self._dbg
            z = float(self.ctx.get_base_pose_w()[0][0, 2])
            extra = (
                f"leg={self._walk_leg} mode={self._walk_mode} "
                f"cmd_b=({self._walk[0]:+.2f},{self._walk[1]:+.2f},{self._walk[2]:+.2f}) "
                f"xy=({x:.2f},{y:.2f}) rem={remaining:+.2f} lat={lateral:+.2f} "
                f"head={math.degrees(yaw):.0f} err={math.degrees(yaw_err):+.0f}deg "
                f"z={z:.3f}"
            )
        if self.phase in (
            CesPickPlacePhase.PLACE_HOLD,
            CesPickPlacePhase.PLACE_APPROACH,
        ):
            extra = f"{self._carry_contact_line()} {extra}".strip()
        print(
            f"[ces_fsm] {self.phase.value} t={self.t:.1f}s "
            f"grip={self.gripper:.3f} tcp_err={tcp_err*1000:.1f}mm "
            f"tilt={self.ctx.stance_tilt():.2f} "
            f"standing={int(self.ctx.is_standing())} {extra}"
        )

    def step(self) -> CesCommand:
        self.t += self.ctx.dt
        self._walk = [0.0, 0.0, 0.0, 0.8]
        try:
            handler = getattr(self, self._PHASE_HANDLERS[self.phase])
            return handler()
        except Exception as exc:
            if self.t - self._step_err_t > 2.0:
                self._step_err_t = self.t
                print(f"[ces_fsm] {self.phase.value} step error t={self.t:.2f}s: {exc}")
                import traceback

                traceback.print_exc()
            return self._cmd(
                tcp=self._grasp_pos_w,
                quat=self._grasp_quat_w,
                arm_q=getattr(self.ctx, "_q_right", None),
            )
        finally:
            self._maybe_log()
