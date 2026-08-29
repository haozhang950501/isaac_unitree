# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""CES Baseline FSM facade.

The only supported runtime path is:
00→10→20→30 → 40(q_ref only) → grasp/lift → 40(live)→30→20→05
→ carry-walk → pin the live pelvis → 05→15 → release → 15→05.
"""
from __future__ import annotations

import logging

from action_provider.ces_grasp import constants as C
from action_provider.ces_grasp.fsm_pick import CesPickMixin, top_down_grasp_quat
from action_provider.ces_grasp.fsm_place import CesPlaceMixin
from action_provider.ces_grasp.fsm_types import CesCommand, CesPickPlacePhase
from action_provider.ces_grasp.fsm_walk import CesWalkMixin
from action_provider.ces_grasp.pose_library import CesTrajectory, load_baseline_trajectory
from action_provider.ces_grasp.interpolation import (
    CartesianInterpolator,
    JointSpaceInterpolator,
    scale_segment_times,
)

__all__ = [
    "CesCommand",
    "CesPickPlacePhase",
    "CesPickPlaceStateMachine",
    "top_down_grasp_quat",
]

logger = logging.getLogger("ces")


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
        self._reset_runtime()

        logger.info(
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
        self._reset_runtime()

    def _reset_runtime(self):
        self.phase = CesPickPlacePhase.SETTLE
        self.t = 0.0
        self.hold = 0.0
        self.gripper = C.GRIPPER_OPEN
        self._walk = [0.0, 0.0, 0.0, 0.8]
        self._grasp_pos_w = None
        self._grasp_quat_w = None
        self._grasp_arm_q = None
        self._carry_arm_q = None
        self._place_arm_q = None
        self._q_wp30 = None
        self._q_wp40 = None
        self._return_followup_qs = []
        self._return_followup_durations = []
        self._return_followup_method = self._trajectory.return_interpolation_method
        self._return_total_time = 0.0
        self._place_joint_total_time = 0.0
        self.joint_interp.clear()
        self._walk_mode = "idle"
        self._walk_leg = 0
        if self._planner is not None:
            self._planner.reset()
        self._turn_pinned_logged = False
        self._stopped_logged = False
        self._table_braked = False
        self._walk_unstable_t = 0.0
        self._place_lock_pose = None
        self._place_hold_obj_z0 = None
        self._carry_drop_logged = False
        self._step_err_t = -10.0

    def _transition(self, phase: CesPickPlacePhase):
        logger.info("[ces_fsm] %s -> %s t=%.2fs", self.phase.value, phase.value, self.t)
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
        walk=None,
        root_pin=None,
        arm_q=None,
        arm_q_ref=None,
    ) -> CesCommand:
        if walk is None and root_pin is None and self.phase in (
            CesPickPlacePhase.SETTLE,
            CesPickPlacePhase.GOTO_PICK,
            CesPickPlacePhase.UNFOLD,
            CesPickPlacePhase.DESCEND,
            CesPickPlacePhase.GRASP,
            CesPickPlacePhase.LIFT,
            CesPickPlacePhase.RETURN_HOME,
        ):
            root_pin = C.PICK_ROOT_PIN
        return CesCommand(
            gripper=C.GRIPPER_CLOSED if self._squeezing() else self.gripper,
            walk=None if walk is None else tuple(float(value) for value in walk),
            root_pin=root_pin,
            arm_q=arm_q,
            tcp_pos=tcp,
            tcp_quat=quat,
            arm_q_ref=arm_q_ref,
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

    def step(self) -> CesCommand:
        self.t += self.ctx.dt
        self._walk = [0.0, 0.0, 0.0, 0.8]
        try:
            handler = getattr(self, self._PHASE_HANDLERS[self.phase])
            return handler()
        except Exception as exc:
            if self.t - self._step_err_t > 2.0:
                self._step_err_t = self.t
                logger.exception(
                    "[ces_fsm] %s step error t=%.2fs: %s",
                    self.phase.value,
                    self.t,
                    exc,
                )
            safe_walk = (
                (0.0, 0.0, 0.0, 0.8)
                if self.phase in (CesPickPlacePhase.GOTO_PLACE, CesPickPlacePhase.FAILED)
                and self._carry_arm_q is not None
                else None
            )
            return self._cmd(
                tcp=self._grasp_pos_w,
                quat=self._grasp_quat_w,
                walk=safe_walk,
                arm_q=getattr(self.ctx, "_q_right", None),
            )
