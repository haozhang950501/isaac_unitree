# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""Carry-walk and live-pelvis pinning handlers for the CES Baseline FSM."""
from __future__ import annotations

import math
import logging

from action_provider.ces_grasp import constants as C
from action_provider.ces_grasp.fsm_types import CesPickPlacePhase
from action_provider.ces_grasp.navigation import LegWalkPlanner

logger = logging.getLogger("ces")


class CesWalkMixin:
    def _walk_planner(self) -> LegWalkPlanner:
        if self._planner is None:
            self._planner = LegWalkPlanner(
                list(C.CARRY_WALK_LEGS), C.CARRY_WALK_GAIT
            )
            legs = " → ".join(
                f"{leg.name}({leg.kind})" for leg in C.CARRY_WALK_LEGS
            )
            corner = C.CARRY_WALK_LEGS[0].target_xy
            logger.info(
                f"[ces_fsm] carry route {legs} "
                f"backoff=({corner[0]:.3f},{corner[1]:.3f}) "
                f"tray_center=({C.PLACE_TARGET_XY[0]:.3f},{C.PLACE_TARGET_XY[1]:.3f}) "
                f"cmd |vx|={C.WALK_VX:.2f} |wz|={C.WALK_WZ:.2f} "
                f"turn_vx={C.WALK_TURN_VX:.2f} lead={C.WALK_TURN_LEAD:.3f}m "
                f"preturn={C.WALK_TURN_PREVIEW:.2f}m "
                f"(fixed magnitudes; policy ignores small or pure-yaw commands)"
            )
        return self._planner

    def _guide_route(self) -> bool:
        root_pos, _ = self.ctx.get_base_pose_w()
        x, y = float(root_pos[0, 0]), float(root_pos[0, 1])
        yaw = self.ctx.get_heading()
        planner = self._walk_planner()
        step = planner.step(x, y, yaw, self.ctx.dt)
        if step.leg_index != self._walk_leg:
            self._walk_leg = step.leg_index
            logger.info(
                f"[ces_fsm] walk leg {step.leg_index+1}/{len(planner.legs)} "
                f"{step.leg_name} at ({x:.3f},{y:.3f}) "
                f"head={math.degrees(yaw):.0f}deg"
            )
        if step.mode == "turn_pinned" and not self._turn_pinned_logged:
            self._turn_pinned_logged = True
            logger.warning(
                f"[ces_fsm] WALK WARNING: no turn after "
                f"{C.WALK_TURN_MAX_DRIFT:.2f} m of arc; retrying at "
                f"|wz|={C.WALK_WZ_MAX:.2f}"
            )
        if step.mode.endswith("_stopped") and not self._stopped_logged:
            self._stopped_logged = True
            square = "" if step.on_target else (
                f" OFF-TARGET: {step.lateral:+.3f} m sideways and "
                f"{math.degrees(step.yaw_error):+.0f}deg off square"
            )
            logger.warning(
                f"[ces_fsm] walk stopped at the place stand line "
                f"(rem={step.remaining:+.3f}m) and stays stopped.{square}"
            )
        if step.mode == "reverse_preturn" and self._walk_mode != "reverse_preturn":
            logger.info(
                f"[ces_fsm] walk pre-turn at ({x:.3f},{y:.3f}) "
                f"head={math.degrees(yaw):.0f}deg"
            )
        self._walk = list(step.command)
        self._walk_mode = step.mode
        if self._table_keepout_hit(x, y):
            return self._brake_for_table(x, y)
        if not step.route_done:
            self.hold = 0.0
            return False
        if self.ctx.is_standing():
            self.hold += self.ctx.dt
        else:
            self.hold = 0.0
        return self.hold >= C.WALK_ARRIVE_HOLD

    def _table_keepout_hit(self, x: float, y: float) -> bool:
        if self._table_braked:
            return True
        fwd, _ = C.forward_left(C.PLACE_STAND_YAW)
        past = (x - C.PLACE_STAND_XY[0]) * fwd[0] + (
            y - C.PLACE_STAND_XY[1]
        ) * fwd[1]
        return past > C.WALK_TABLE_AHEAD_OF_STAND - C.WALK_TABLE_SAFE

    def _brake_for_table(self, x: float, y: float) -> bool:
        if not self._table_braked:
            self._table_braked = True
            fwd, _ = C.forward_left(C.PLACE_STAND_YAW)
            past = (x - C.PLACE_STAND_XY[0]) * fwd[0] + (
                y - C.PLACE_STAND_XY[1]
            ) * fwd[1]
            logger.warning(
                f"[ces_fsm] WALK KEEP-OUT: pelvis is {past*1000:+.0f}mm past "
                f"the place stand. Stopping for good and placing from here — "
                f"raise WALK_STOP_MARGIN_PLACE (now {C.WALK_STOP_MARGIN_PLACE:.2f}) "
                f"if this guard fires."
            )
        self._walk = [0.0, 0.0, 0.0, C.CARRY_WALK_GAIT.height]
        self._walk_mode = "table_keepout"
        if self.ctx.is_standing():
            self.hold += self.ctx.dt
        else:
            self.hold = 0.0
        return self.hold >= C.WALK_ARRIVE_HOLD

    def _walk_to_place(self) -> bool:
        arrived = self._guide_route()
        if self.ctx.stance_tilt() > C.WALK_ABORT_TILT:
            self._walk_unstable_t += self.ctx.dt
        else:
            self._walk_unstable_t = 0.0
        if self._walk_unstable_t >= C.WALK_ABORT_HOLD:
            logger.error(
                f"[ces_fsm] WALK abort: tilt={self.ctx.stance_tilt():.2f} "
                f"for {self._walk_unstable_t:.2f}s"
            )
            self._walk = [0.0, 0.0, 0.0, 0.8]
            self._transition(CesPickPlacePhase.FAILED)
            return False
        if self.t > C.WALK_PLACE_TIMEOUT and not arrived:
            logger.error(
                f"[ces_fsm] WALK timeout (leg={self._walk_leg}) — marking FAILED"
            )
            self._transition(CesPickPlacePhase.FAILED)
            return False
        return arrived

    def _place_body_cmd(self, **kwargs):
        if self._place_lock_pose is None:
            raise RuntimeError("place root pose was not captured at walk arrival")
        return self._cmd(root_pin=self._place_lock_pose, **kwargs)

    def _watch_carry_drop(self, where: str):
        if self._carry_drop_logged or self._place_hold_obj_z0 is None:
            return
        try:
            obj_pos, _ = self.ctx.get_object_pose_w()
        except Exception:
            return
        delta = float(obj_pos[0, 2]) - self._place_hold_obj_z0
        if delta < -0.05:
            self._carry_drop_logged = True
            logger.warning(
                "[ces_fsm] DROP during %s dz=%.0fmm",
                where,
                delta * 1000.0,
            )

    def _step_carry(self):
        self.gripper = C.GRIPPER_CLOSED
        q = self._carry_arm_q
        if q is None:
            q = self.ctx.get_right_arm_q()[0].clone()
            self._carry_arm_q = q
        self._walk = [-C.WALK_VX, 0.0, 0.0, C.CARRY_WALK_GAIT.height]
        if hasattr(self.ctx, "prime_walk_filt"):
            self.ctx.prime_walk_filt(self._walk)
        logger.info("[ces_fsm] pick done - reverse walk starts immediately")
        self._transition(CesPickPlacePhase.GOTO_PLACE)
        return self._cmd(arm_q=q, walk=self._walk)

    def _step_goto_place(self):
        self.gripper = C.GRIPPER_CLOSED
        arrived = self._walk_to_place()
        q = self._carry_arm_q
        if q is None:
            q = self.ctx.get_right_arm_q()[0]
            self._carry_arm_q = q.clone()
        if self.phase is CesPickPlacePhase.FAILED:
            return self._cmd(arm_q=q, walk=(0.0, 0.0, 0.0, 0.8))
        if not arrived:
            return self._cmd(arm_q=q, walk=self._walk)

        root_pos, root_quat = self.ctx.get_base_pose_w()
        lock_pos = tuple(float(root_pos[0, i]) for i in range(3))
        lock_quat = tuple(float(root_quat[0, i]) for i in range(4))
        self._place_lock_pose = (lock_pos, lock_quat)
        q = self.ctx.get_right_arm_q()[0].clone()
        self._carry_arm_q = q
        if hasattr(self.ctx, "_q_right"):
            self.ctx._q_right = q.clone()

        xy, yaw = lock_pos[:2], self.ctx.get_heading()
        fwd, left = C.forward_left(C.PLACE_STAND_YAW)
        dx = C.PLACE_STAND_XY[0] - xy[0]
        dy = C.PLACE_STAND_XY[1] - xy[1]
        along = dx * fwd[0] + dy * fwd[1]
        side = dx * left[0] + dy * left[1]
        coast = C.WALK_STOP_MARGIN_PLACE - along
        reach = C.X_B_PLACE + along
        logger.info(
            f"[ces_fsm] walk arrived — lock actual pelvis "
            f"xy=({xy[0]:.3f},{xy[1]:.3f}) yaw={math.degrees(yaw):.1f}deg "
            f"(stand err dxy={math.hypot(dx,dy)*1000:.0f}mm "
            f"along={along*1000:+.0f}mm side={side*1000:+.0f}mm) "
            f"coast={coast*1000:+.0f}mm nominal_reach={reach*1000:.0f}mm "
            f"(pose15 release; no Cartesian XY correction)"
        )
        try:
            obj_pos, _ = self.ctx.get_object_pose_w()
            self._place_hold_obj_z0 = float(obj_pos[0, 2])
        except Exception:
            self._place_hold_obj_z0 = None
        self._carry_drop_logged = False
        logger.info(
            f"[ces_fsm] freeze 05 at place stand for "
            f"{C.WALK_PLACE_HOLD_TIME:.1f}s "
            f"(pin xy/z + live quat; arm PD, gripper locked, no gait)"
        )
        self._transition(CesPickPlacePhase.PLACE_HOLD)
        return self._place_body_cmd(arm_q=q)
