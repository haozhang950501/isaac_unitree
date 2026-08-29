# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""Place, release, retract, and terminal handlers for the CES Baseline FSM."""
from __future__ import annotations

import math
import logging

from action_provider.ces_grasp import constants as C
from action_provider.ces_grasp.fsm_types import CesPickPlacePhase

logger = logging.getLogger("ces")


class CesPlaceMixin:
    def _frozen_place_arm(self):
        if self._place_arm_q is None:
            self._place_arm_q = self.ctx.get_right_arm_q()[0].clone()
        return self._place_arm_q

    def _begin_place_approach(self):
        q_now = self.ctx.get_right_arm_q()[0].clone()
        names = list(self._trajectory.place_waypoints)
        qs = [q_now, *[self._pose_q(name) for name in names]]
        durations = list(self._place_segment_times)
        self._place_joint_total_time = sum(durations)
        q05 = self._pose_q(self._trajectory.place_start)
        start_error = float((q_now - q05).abs().max())
        self.joint_interp.reset_path(
            qs,
            durations,
            method=self._trajectory.place_interpolation_method,
        )
        logger.info(
            f"[ces_fsm] Place joint path {self._trajectory.place_start}→"
            f"{'→'.join(names)} dur={self._place_joint_total_time:.2f}s "
            f"interp={self._trajectory.place_interpolation_method} "
            f"live_vs_05={start_error:.4f}rad "
            f"(pin live pelvis quat, arm PD, gripper locked, no gait)"
        )

    def _begin_retract(self):
        q_now = self.ctx.get_right_arm_q()[0].clone()
        home = self._carry_arm_q
        if home is None:
            home = self._pose_q(self._trajectory.place_start)
        duration = C.RETRACT_TIME + C.RETRACT_HOME_TIME
        self._place_arm_q = home.clone()
        self.joint_interp.reset(q_now, home, duration)
        logger.info("[ces_fsm] retract 15->05 dur=%.2fs", duration)

    def _log_place_result(self, tag: str):
        try:
            obj_pos, _ = self.ctx.get_object_pose_w()
        except Exception as exc:
            logger.debug("[ces_verify] %s product pose unavailable: %s", tag, exc)
            return
        x, y, z = (float(obj_pos[0, i]) for i in range(3))
        tx, ty = C.PLACE_TARGET_XY
        tray_top = C.TABLE_TOP_Z + C.PLACE_TRAY_HEIGHT
        logger.debug(
            f"[ces_verify] {tag} product=({x:.4f},{y:.4f},{z:.4f}) "
            f"tote=({tx:.4f},{ty:.4f},z={tray_top:.3f}) "
            f"dxy={math.hypot(x-tx,y-ty)*1000:.0f}mm "
            f"above_tray={(z-tray_top)*1000:+.0f}mm"
        )

    def _step_place_hold(self):
        self.gripper = C.GRIPPER_CLOSED
        q = self._carry_arm_q
        if q is None:
            q = self.ctx.get_right_arm_q()[0].clone()
            self._carry_arm_q = q
        self._watch_carry_drop("place_hold")
        if self.t >= C.WALK_PLACE_HOLD_TIME:
            self._begin_place_approach()
            self._transition(CesPickPlacePhase.PLACE_APPROACH)
        return self._place_body_cmd(arm_q=q)

    def _step_place_approach(self):
        self.gripper = C.GRIPPER_CLOSED
        self._watch_carry_drop("place_approach")
        q = self.joint_interp.step(self.ctx.dt)
        if q is None:
            q = self.ctx.get_right_arm_q()[0]
        if (
            self.joint_interp.finished
            or self.t > self._place_joint_total_time + 1.0
        ):
            self._place_arm_q = q.clone()
            logger.info("[ces_fsm] place pose 15 reached - opening gripper")
            logger.debug("[ces_fsm] place q15=%s", q)
            self._transition(CesPickPlacePhase.RELEASE)
        return self._place_body_cmd(arm_q=q)

    def _step_release(self):
        self.gripper = C.GRIPPER_OPEN
        q = self._frozen_place_arm()
        if self.t >= C.RELEASE_TIME:
            self._log_place_result("release_done")
            self._begin_retract()
            self._transition(CesPickPlacePhase.RETRACT)
        return self._place_body_cmd(arm_q=q)

    def _step_retract(self):
        self.gripper = C.GRIPPER_OPEN
        q = self.joint_interp.step(self.ctx.dt)
        if q is None:
            q = self._frozen_place_arm()
        if self.joint_interp.finished or self.t > self.joint_interp.duration + 1.0:
            self._place_arm_q = q.clone()
            logger.info("[ces_fsm] arm back at 05 carry - task done")
            self._log_place_result("task_done")
            self._transition(CesPickPlacePhase.DONE)
        return self._place_body_cmd(arm_q=q)

    def _step_done(self):
        q = self._place_arm_q if self._place_arm_q is not None else self._carry_arm_q
        return self._place_body_cmd(arm_q=q)

    def _step_failed(self):
        if self._carry_arm_q is not None:
            self.gripper = C.GRIPPER_CLOSED
            return self._cmd(
                arm_q=self._carry_arm_q, walk=(0.0, 0.0, 0.0, 0.8)
            )
        return self._cmd(arm_q=self._carry_arm_q)
