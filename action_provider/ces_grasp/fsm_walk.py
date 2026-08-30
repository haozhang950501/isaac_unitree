# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""CES 状态机的持物行走、桌面保护和到站钉盆阶段。"""
from __future__ import annotations

import logging
import math

from action_provider.ces_grasp import constants as C
from action_provider.ces_grasp.fsm_types import CesPickPlacePhase, WalkCommand
from action_provider.ces_grasp.navigation import (
    CarryWalkPhase,
    CarryWalkPlanner,
)

logger = logging.getLogger("ces")


class CesWalkMixin:
    """实现 ``CARRY``、``GOTO_PLACE`` 以及到站后的实际骨盆钉住。"""

    def _walk_planner(self) -> CarryWalkPlanner:
        """延迟创建固定路线导航器，避免模块加载阶段持有运行状态。"""
        if self._planner is None:
            self._planner = CarryWalkPlanner(
                C.CARRY_WALK_ROUTE,
                C.CARRY_WALK_CONFIG,
            )
            corner = C.CARRY_WALK_ROUTE.backoff_xy
            logger.info(
                f"[ces_fsm] carry route backoff(reverse) → "
                f"turn_to_table(turn) → approach_place(forward) "
                f"backoff=({corner[0]:.3f},{corner[1]:.3f}) "
                f"tray_center=({C.PLACE_TARGET_XY[0]:.3f},{C.PLACE_TARGET_XY[1]:.3f}) "
                f"cmd |vx|={C.WALK_VX:.2f} |wz|={C.WALK_WZ:.2f} "
                f"turn_vx={C.WALK_TURN_VX:.2f} lead={C.WALK_TURN_LEAD:.3f}m "
                f"preturn={C.WALK_TURN_PREVIEW:.2f}m "
                f"(fixed magnitudes; policy ignores small or pure-yaw commands)"
            )
        return self._planner

    def _guide_route(self) -> tuple[bool, WalkCommand]:
        """执行一帧路线规划，并返回是否稳定到站及对应步态命令。"""
        root_pos, _ = self.ctx.get_base_pose_w()
        x, y = float(root_pos[0, 0]), float(root_pos[0, 1])
        yaw = self.ctx.get_heading()
        planner = self._walk_planner()
        step = planner.step(x, y, yaw, self.ctx.dt)

        if (
            step.phase is not CarryWalkPhase.DONE
            and step.phase is not self._walk_phase
        ):
            self._walk_phase = step.phase
            logger.info(
                f"[ces_fsm] walk phase {step.phase.index + 1}/"
                f"{planner.ACTIVE_PHASE_COUNT} {step.phase.value} "
                f"at ({x:.3f},{y:.3f}) head={math.degrees(yaw):.0f}deg"
            )
        if step.mode == "turn_pinned" and not self._turn_pinned_logged:
            self._turn_pinned_logged = True
            logger.warning(
                f"[ces_fsm] WALK WARNING: no turn after "
                f"{C.WALK_TURN_MAX_DRIFT:.2f} m of arc; retrying at "
                f"|wz|={C.WALK_WZ_MAX:.2f}"
            )
        if step.done and not self._stopped_logged:
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
        self._walk_mode = step.mode

        # keep-out 独立于路线规划：任何规划遗漏都不能让骨盆越过桌前安全线。
        if self._table_keepout_hit(x, y):
            return self._brake_for_table(x, y)
        if not step.done:
            self.hold = 0.0
            return False, step.command

        # 导航器停止后还要确认机器人真正站稳，随后才捕获实际骨盆位姿。
        if self.ctx.is_standing():
            self.hold += self.ctx.dt
        else:
            self.hold = 0.0
        return self.hold >= C.WALK_ARRIVE_HOLD, step.command

    def _table_keepout_hit(self, x: float, y: float) -> bool:
        """判断骨盆是否越过桌前硬安全线；命中后保持闩锁。"""
        if self._table_braked:
            return True
        forward, _ = C.forward_left(C.PLACE_STAND_YAW)
        past = (x - C.PLACE_STAND_XY[0]) * forward[0] + (
            y - C.PLACE_STAND_XY[1]
        ) * forward[1]
        return past > C.WALK_TABLE_AHEAD_OF_STAND - C.WALK_TABLE_SAFE

    def _brake_for_table(self, x: float, y: float) -> tuple[bool, WalkCommand]:
        """触发桌面保护后永久归零步态，并等待机器人稳定。"""
        if not self._table_braked:
            self._table_braked = True
            forward, _ = C.forward_left(C.PLACE_STAND_YAW)
            past = (x - C.PLACE_STAND_XY[0]) * forward[0] + (
                y - C.PLACE_STAND_XY[1]
            ) * forward[1]
            logger.warning(
                f"[ces_fsm] WALK KEEP-OUT: pelvis is {past*1000:+.0f}mm past "
                f"the place stand. Stopping for good and placing from here — "
                f"raise WALK_STOP_MARGIN_PLACE (now {C.WALK_STOP_MARGIN_PLACE:.2f}) "
                f"if this guard fires."
            )
        self._walk_mode = "table_keepout"
        if self.ctx.is_standing():
            self.hold += self.ctx.dt
        else:
            self.hold = 0.0
        stop = (0.0, 0.0, 0.0, C.WALK_HEIGHT)
        return self.hold >= C.WALK_ARRIVE_HOLD, stop

    def _walk_to_place(self) -> tuple[bool, WalkCommand]:
        """组合路线、倾倒保护和超时保护，返回本帧安全步态命令。"""
        arrived, walk = self._guide_route()
        tilt = self.ctx.stance_tilt()
        if tilt > C.WALK_ABORT_TILT:
            self._walk_unstable_t += self.ctx.dt
        else:
            self._walk_unstable_t = 0.0
        if self._walk_unstable_t >= C.WALK_ABORT_HOLD:
            logger.error(
                f"[ces_fsm] WALK abort: tilt={tilt:.2f} "
                f"for {self._walk_unstable_t:.2f}s"
            )
            self._transition(CesPickPlacePhase.FAILED)
            return False, (0.0, 0.0, 0.0, C.WALK_HEIGHT)
        if self.t > C.WALK_PLACE_TIMEOUT and not arrived:
            phase = self._walk_phase.value if self._walk_phase is not None else "unknown"
            logger.error(
                f"[ces_fsm] WALK timeout (phase={phase}) — marking FAILED"
            )
            self._transition(CesPickPlacePhase.FAILED)
            return False, (0.0, 0.0, 0.0, C.WALK_HEIGHT)
        return arrived, walk

    def _place_body_cmd(self, **kwargs):
        """生成钉住实际到站骨盆位姿的放置阶段命令。"""
        if self._place_lock_pose is None:
            raise RuntimeError("place root pose was not captured at walk arrival")
        return self._cmd(root_pin=self._place_lock_pose, **kwargs)

    def _watch_carry_drop(self, where: str) -> None:
        """监视钉盆后产品高度突降，并且每次任务最多告警一次。"""
        if self._carry_drop_logged or self._place_hold_obj_z0 is None:
            return
        try:
            object_pos, _ = self.ctx.get_object_pose_w()
        except Exception:
            # 该诊断不能打断放置主链路；物体读取失败由动作提供器统一保护。
            return
        delta = float(object_pos[0, 2]) - self._place_hold_obj_z0
        if delta < -0.05:
            self._carry_drop_logged = True
            logger.warning(
                "[ces_fsm] DROP during %s dz=%.0fmm",
                where,
                delta * 1000.0,
            )

    def _step_carry(self):
        """Pick 完成后第一帧立即预载并下发反向步态。"""
        self.gripper = C.GRIPPER_CLOSED
        arm_q = self._carry_arm_q
        if arm_q is None:
            arm_q = self.ctx.get_right_arm_q()[0].clone()
            self._carry_arm_q = arm_q
        walk = (-C.WALK_VX, 0.0, 0.0, C.WALK_HEIGHT)
        self.ctx.prime_walk_filter(walk)
        logger.info("[ces_fsm] pick done - reverse walk starts immediately")
        self._transition(CesPickPlacePhase.GOTO_PLACE)
        return self._cmd(arm_q=arm_q, walk=walk)

    def _step_goto_place(self):
        """持物换站；到站后捕获实时骨盆位姿并切换到钉盆放置。"""
        self.gripper = C.GRIPPER_CLOSED
        arrived, walk = self._walk_to_place()
        arm_q = self._carry_arm_q
        if arm_q is None:
            arm_q = self.ctx.get_right_arm_q()[0]
            self._carry_arm_q = arm_q.clone()
        if self.phase is CesPickPlacePhase.FAILED:
            return self._cmd(
                arm_q=arm_q,
                walk=(0.0, 0.0, 0.0, C.WALK_HEIGHT),
            )
        if not arrived:
            return self._cmd(arm_q=arm_q, walk=walk)

        root_pos, root_quat = self.ctx.get_base_pose_w()
        lock_pos = tuple(float(root_pos[0, index]) for index in range(3))
        lock_quat = tuple(float(root_quat[0, index]) for index in range(4))
        self._place_lock_pose = (lock_pos, lock_quat)
        arm_q = self.ctx.get_right_arm_q()[0].clone()
        self._carry_arm_q = arm_q
        self.ctx.sync_right_arm_target(arm_q)

        xy, yaw = lock_pos[:2], self.ctx.get_heading()
        forward, left = C.forward_left(C.PLACE_STAND_YAW)
        dx = C.PLACE_STAND_XY[0] - xy[0]
        dy = C.PLACE_STAND_XY[1] - xy[1]
        along = dx * forward[0] + dy * forward[1]
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
            object_pos, _ = self.ctx.get_object_pose_w()
            self._place_hold_obj_z0 = float(object_pos[0, 2])
        except Exception:
            # 高度记录只服务掉落告警；失败时仍继续执行已验证的放置轨迹。
            self._place_hold_obj_z0 = None
        self._carry_drop_logged = False
        logger.info(
            f"[ces_fsm] freeze 05 at place stand for "
            f"{C.WALK_PLACE_HOLD_TIME:.1f}s "
            f"(pin xy/z + live quat; arm PD, gripper locked, no gait)"
        )
        self._transition(CesPickPlacePhase.PLACE_HOLD)
        return self._place_body_cmd(arm_q=arm_q)
