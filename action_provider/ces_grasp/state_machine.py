# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""Finite state machine for CES LoadingLine Product pick-and-place.

Phases::

    SETTLE          -> pin spawn, freeze default stance, arms idle
    GOTO_PICK       -> snap to pick stand
    UNFOLD          -> joint-space blend from hang to a bent-elbow ready pose
    APPROACH        -> rotate fingers down, then slide in over Product
    DESCEND         -> vertical drop onto the pinch pose
    GRASP           -> close Dex1 (ramped) and dwell
    LIFT            -> raise ~8 cm, keep the pinch orientation
    CARRY           -> freeze that arm pose (do not retarget the wrist)
    HOLD            -> keep squeezing, arm joints locked
    GOTO_PLACE      -> snap-locate to the table stand (friction carry)
    PLACE_APPROACH  -> after standing, reach over the table
    PLACE_DESCEND   -> gentle lower onto the table
    RELEASE         -> open gripper
    RETRACT         -> lift away from the part
    DONE / FAILED
"""
from __future__ import annotations

import enum
import math
from dataclasses import dataclass

import torch

from isaaclab.utils.math import quat_apply, quat_from_matrix

from action_provider.ces_grasp import constants as C
from action_provider.manip_common import CartesianInterpolator
from action_provider.manip_common.interpolation import ease_in_out


class CesPickPlacePhase(enum.Enum):
    SETTLE = "settle"
    GOTO_PICK = "goto_pick"
    UNFOLD = "unfold"
    APPROACH = "approach"
    DESCEND = "descend"
    GRASP = "grasp"
    LIFT = "lift"
    HOLD = "hold"
    CARRY = "carry"
    GOTO_PLACE = "goto_place"
    PLACE_APPROACH = "place_approach"
    PLACE_DESCEND = "place_descend"
    RELEASE = "release"
    RETRACT = "retract"
    DONE = "done"
    FAILED = "failed"


@dataclass
class CesCommand:
    tcp_pos: torch.Tensor | None
    tcp_quat: torch.Tensor | None
    gripper: float
    walk: list[float]
    snap_xy: tuple[float, float] | None
    snap_yaw: float | None
    guide: bool
    done: bool
    failed: bool
    arm_q: torch.Tensor | None


def top_down_grasp_quat(jaw_axis_w: torch.Tensor) -> torch.Tensor:
    """Hand +X = jaw (pinch), +Y = world down, +Z = remaining right-handed axis.

    ``jaw_axis_w`` is [N, 3]; the Z component is discarded so the jaws stay
    horizontal.
    """
    x = jaw_axis_w.clone()
    x[:, 2] = 0.0
    x = x / torch.clamp(torch.norm(x, dim=-1, keepdim=True), min=1e-6)
    down = torch.zeros_like(x)
    down[:, 2] = -1.0
    z = torch.linalg.cross(x, down)
    z = z / torch.clamp(torch.norm(z, dim=-1, keepdim=True), min=1e-6)
    y = torch.linalg.cross(z, x)
    rot = torch.stack((x, y, z), dim=-1)
    return quat_from_matrix(rot)


class CesPickPlaceStateMachine:
    def __init__(self, ctx, station_mode: str = "snap", stop_after: str = "place"):
        self.ctx = ctx
        requested = station_mode if station_mode in ("snap", "walk") else "snap"
        if requested == "walk":
            print("[ces_fsm] walk is disabled — using snap teleport")
        self.station_mode = "snap"
        self.stop_after = stop_after if stop_after in ("lift", "place") else C.STOP_AFTER
        self.device = ctx.device
        self.interp = CartesianInterpolator(self.device)
        self.phase = CesPickPlacePhase.SETTLE
        self.t = 0.0
        self.hold = 0.0
        self._nav_i = 0
        self._log_t = 0.0
        self._walk = [0.0, 0.0, 0.0, 0.8]
        self.gripper = C.GRIPPER_OPEN
        self._grasp_pos_w: torch.Tensor | None = None
        self._grasp_quat_w: torch.Tensor | None = None
        self._place_pos_w: torch.Tensor | None = None
        self._place_quat_w: torch.Tensor | None = None
        self._carry_quat_w: torch.Tensor | None = None
        self._hold_lift_pos: torch.Tensor | None = None
        self._q_unfold0: torch.Tensor | None = None
        self._q_unfold1: torch.Tensor | None = None
        self._q_carry0: torch.Tensor | None = None
        self._carry_arm_q: torch.Tensor | None = None
        self._dbg = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        self._spawn_yaw_applied = False
        self._step_err_t = -10.0
        print(
            f"[ces_fsm] v4 station_mode={self.station_mode} stop_after={self.stop_after} "
            f"pick_stand=({C.PICK_STAND_XY[0]:.3f},{C.PICK_STAND_XY[1]:.3f}) "
            f"place_stand=({C.PLACE_STAND_XY[0]:.3f},{C.PLACE_STAND_XY[1]:.3f}) "
            f"x_b_place={C.X_B_PLACE:.2f} "
            f"(snap locate; Dex1 friction grasp; arm locked while reaching)"
        )

    def reset(self):
        self.phase = CesPickPlacePhase.SETTLE
        self.t = 0.0
        self.hold = 0.0
        self._nav_i = 0
        self.gripper = C.GRIPPER_OPEN
        self._walk = [0.0, 0.0, 0.0, 0.8]
        self._grasp_pos_w = None
        self._grasp_quat_w = None
        self._place_pos_w = None
        self._place_quat_w = None
        self._carry_quat_w = None
        self._hold_lift_pos = None
        self._q_unfold0 = None
        self._q_unfold1 = None
        self._q_carry0 = None
        self._carry_arm_q = None
        self._dbg = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        self._spawn_yaw_applied = False
        self._step_err_t = -10.0

    def _transition(self, phase: CesPickPlacePhase):
        print(f"[ces_fsm] {self.phase.value} -> {phase.value}  t={self.t:.2f}s")
        self.phase = phase
        self.t = 0.0
        self.hold = 0.0
        self._nav_i = 0

    def _cmd(
        self,
        tcp=None,
        quat=None,
        snap=False,
        snap_xy=None,
        snap_yaw=None,
        guide=False,
        done=False,
        failed=False,
        arm_q=None,
    ) -> CesCommand:
        if (
            not snap
            and not guide
            and self.phase in (
                CesPickPlacePhase.UNFOLD,
                CesPickPlacePhase.APPROACH,
                CesPickPlacePhase.DESCEND,
                CesPickPlacePhase.GRASP,
                CesPickPlacePhase.LIFT,
            )
        ):
            snap = True
            snap_xy = C.PICK_STAND_XY
            snap_yaw = C.PICK_STAND_YAW
        return CesCommand(
            tcp_pos=tcp,
            tcp_quat=quat,
            gripper=C.GRIPPER_CLOSED if self._squeezing() else self.gripper,
            walk=list(self._walk),
            snap_xy=snap_xy if (snap or guide) else None,
            snap_yaw=snap_yaw if (snap or guide) else None,
            guide=bool(guide),
            done=done,
            failed=failed,
            arm_q=arm_q,
        )

    def _squeezing(self) -> bool:
        """True from lift through place-descend: jaws stay at GRIPPER_CLOSED."""
        if self.phase is CesPickPlacePhase.FAILED:
            return self._carry_arm_q is not None
        if self.phase is CesPickPlacePhase.DONE:
            return self.stop_after == "lift"
        return self.phase in (
            CesPickPlacePhase.LIFT,
            CesPickPlacePhase.CARRY,
            CesPickPlacePhase.HOLD,
            CesPickPlacePhase.GOTO_PLACE,
            CesPickPlacePhase.PLACE_APPROACH,
            CesPickPlacePhase.PLACE_DESCEND,
        )

    def _offset_z(self, pos: torch.Tensor, dz: float) -> torch.Tensor:
        out = pos.clone()
        out[:, 2] = out[:, 2] + dz
        return out

    def _retract_from_ces(self, pos: torch.Tensor, dist: float) -> torch.Tensor:
        """Offset toward the robot / away from CES along -pick-forward."""
        out = pos.clone()
        fwd, _ = C.forward_left(C.PICK_STAND_YAW)
        out[:, 0] = out[:, 0] - dist * fwd[0]
        out[:, 1] = out[:, 1] - dist * fwd[1]
        return out

    def _into_drawer(self, pos: torch.Tensor, dist: float) -> torch.Tensor:
        """Offset deeper into the tray (pick-stand forward)."""
        return self._retract_from_ces(pos, -dist)

    def _body_offset_w(self, offset_b: tuple[float, float, float]) -> torch.Tensor:
        root_pos, root_quat = self.ctx.get_base_pose_w()
        off = torch.tensor([offset_b], device=self.device, dtype=root_pos.dtype)
        return root_pos + quat_apply(root_quat, off)

    def _plan_grasp(self):
        pivot, _ = self.ctx.get_object_pose_w()
        obj_pos, _obj_quat = self.ctx.get_product_aabb_center_w()
        self._grasp_pos_w = self._into_drawer(obj_pos, C.GRASP_INSET)
        self._grasp_pos_w = self._offset_z(self._grasp_pos_w, C.GRASP_Z_OFFSET)
        self._grasp_pos_w = self._grasp_pos_w.clone()
        self._grasp_pos_w[:, 1] = self._grasp_pos_w[:, 1] + C.GRASP_SHIFT_Y
        p = [float(self._grasp_pos_w[0, i]) for i in range(3)]
        c = [float(obj_pos[0, i]) for i in range(3)]
        v = [float(pivot[0, i]) for i in range(3)]
        print(
            f"[ces_fsm] Product pivot=({v[0]:.4f},{v[1]:.4f},{v[2]:.4f})  "
            f"AABB=({c[0]:.4f},{c[1]:.4f},{c[2]:.4f})  "
            f"grasp=({p[0]:.4f},{p[1]:.4f},{p[2]:.4f}) "
            f"inset={C.GRASP_INSET:.3f} y_shift={C.GRASP_SHIFT_Y:.3f} "
            f"z_off={C.GRASP_Z_OFFSET:.3f}"
        )
        # Pinch the 36 mm world-X faces; fingers (hand +Y) point world-down.
        jaw = torch.tensor([[1.0, 0.0, 0.0]], device=self.device, dtype=obj_pos.dtype)
        self._grasp_quat_w = top_down_grasp_quat(jaw)
        self._carry_quat_w = self._grasp_quat_w.clone()

        place = torch.tensor(
            [[C.PLACE_TARGET_XY[0], C.PLACE_TARGET_XY[1], C.PLACE_Z]],
            device=self.device,
            dtype=obj_pos.dtype,
        )
        self._place_pos_w = place
        self._place_quat_w = self._grasp_quat_w.clone()

    def _nav_goals(self, kind: str) -> list[tuple[tuple[float, float], float, bool]]:
        """Walk targets: (xy, yaw, require_yaw)."""
        if kind == "pick":
            return [(C.PICK_STAND_XY, C.PICK_STAND_YAW, True)]
        return [
            (C.PLACE_VIA_XY, C.PICK_STAND_YAW, False),
            (C.PLACE_STAND_XY, C.PLACE_STAND_YAW, True),
        ]

    def _navigate(self, kind: str) -> tuple[bool, tuple[float, float], float, bool]:
        """Snap the pelvis to the stand and hold it (定位)."""
        xy, yaw, _ = self._nav_goals(kind)[-1]
        if self.t < C.STAND_MIN_TIME:
            return False, xy, yaw, True
        if self.t > 6.0:
            print("[ces_fsm] stand timeout — starting arm anyway")
            return True, xy, yaw, True
        if self.ctx.is_standing():
            self.hold += self.ctx.dt
        else:
            self.hold = 0.0
        arrived = self.hold >= C.STAND_STABLE_TIME
        return arrived, xy, yaw, True

    def _loco(self, kind: str) -> tuple[bool, tuple[float, float], float, bool]:
        """Slew the root toward ``_nav_goals``. Pelvis is not pinned until arrival."""
        goals = self._nav_goals(kind)
        i = min(self._nav_i, len(goals) - 1)
        xy, yaw, require_yaw = goals[i]
        last = i >= len(goals) - 1
        at_goal = self._guide_toward(xy, yaw, require_yaw)
        timeout = C.WALK_PLACE_TIMEOUT if kind == "place" else C.WALK_GOTO_TIMEOUT
        if at_goal and not last:
            self._nav_i += 1
            self.hold = 0.0
            print(f"[ces_fsm] {kind} waypoint {i + 1}/{len(goals)}")
            return self._loco(kind)
        if self.t > timeout and not at_goal:
            print(
                f"[ces_fsm] WALK timeout in {self.phase.value} "
                f"(nav_i={self._nav_i}) — marking FAILED, not arriving"
            )
            self._transition(CesPickPlacePhase.FAILED)
            return False, xy, yaw, False
        arrived = bool(at_goal and last)
        return arrived, xy, yaw, arrived

    def _guide_toward(
        self,
        target_xy: tuple[float, float],
        target_yaw: float,
        require_yaw: bool,
    ) -> bool:
        """FSM side of kinematic walk: destination is ``target_xy`` / ``target_yaw``.

        The action provider slews the root; this only decides arrival and
        feeds a small +vx so the ONNX policy keeps stepping.
        """
        root_pos, _root_quat = self.ctx.get_base_pose_w()
        x, y = float(root_pos[0, 0]), float(root_pos[0, 1])
        yaw = self.ctx.get_heading()
        dx, dy = target_xy[0] - x, target_xy[1] - y
        dist = math.hypot(dx, dy)
        yaw_err = C.wrap_angle(target_yaw - yaw)
        self._dbg = (x, y, dist, yaw, dx, dy, yaw_err)

        if dist > C.WALK_POS_ARRIVE:
            self.hold = 0.0
            self._walk = [C.WALK_ANIM_VX, 0.0, 0.0, 0.8]
            return False
        if require_yaw and abs(yaw_err) > C.WALK_YAW_ARRIVE:
            self.hold = 0.0
            self._walk = [0.0, 0.0, 0.0, 0.8]
            return False
        self._walk = [0.0, 0.0, 0.0, 0.8]
        self.hold += self.ctx.dt
        return self.hold >= C.WALK_ARRIVE_HOLD

    def _start_grasp(self, err: float):
        """Freeze XY at the actual TCP, but never close below the planned pinch Z."""
        now, _q_now = self.ctx.ik.get_tcp_pose_w()
        planned_z = float(self._grasp_pos_w[0, 2])
        self._grasp_pos_w = now.clone()
        if float(self._grasp_pos_w[0, 2]) < planned_z:
            self._grasp_pos_w[0, 2] = planned_z
        print(
            f"[ces_fsm] close gripper, tcp_err={err*1000:.1f} mm "
            f"z={float(self._grasp_pos_w[0, 2]):.3f} (min {planned_z:.3f})"
        )
        self._transition(CesPickPlacePhase.GRASP)

    def _maybe_log(self, extra: str = ""):
        self._log_t += self.ctx.dt
        interval = 0.25 if self.phase in (
            CesPickPlacePhase.APPROACH,
            CesPickPlacePhase.GOTO_PICK,
            CesPickPlacePhase.GOTO_PLACE,
        ) else 1.0
        if self._log_t < interval:
            return
        self._log_t = 0.0
        tcp_err = -1.0
        if self._grasp_pos_w is not None and self.phase in (
            CesPickPlacePhase.APPROACH,
            CesPickPlacePhase.DESCEND,
            CesPickPlacePhase.GRASP,
            CesPickPlacePhase.LIFT,
        ):
            tcp_err = self.ctx.ik.position_error_norm(self._grasp_pos_w)
        extra = extra or ""
        if self.phase in (CesPickPlacePhase.GOTO_PICK, CesPickPlacePhase.GOTO_PLACE):
            x, y, d, yaw, _dx, _dy, _yaw_err = self._dbg
            if d < 1e-6:
                return
            z = float(self.ctx.get_base_pose_w()[0][0, 2])
            extra = (
                f"nav={self._nav_i} guide=({self._walk[0]:+.2f}) "
                f"xy=({x:.2f},{y:.2f}) d={d:.2f} "
                f"head={math.degrees(yaw):.0f} z={z:.3f} {extra}"
            ).strip()
        print(
            f"[ces_fsm] {self.phase.value} t={self.t:.1f}s grip={self.gripper:.3f} "
            f"tcp_err={tcp_err*1000:.1f}mm tilt={self.ctx.stance_tilt():.2f} "
            f"standing={int(self.ctx.is_standing())} {extra}"
        )

    def _reset_approach_path(self):
        now, q_now = self.ctx.ik.get_tcp_pose_w()
        hover = self._offset_z(self._grasp_pos_w, C.APPROACH_HEIGHT)
        pre = self._retract_from_ces(hover, C.APPROACH_STANDOFF)
        self.interp.reset_path(
            [now, pre, hover],
            [C.ORIENT_TIME, C.SLIDE_TIME],
            q_now,
            self._grasp_quat_w,
        )
        n = [float(now[0, i]) for i in range(3)]
        h = [float(hover[0, i]) for i in range(3)]
        print(
            f"[ces_fsm] approach path tcp=({n[0]:.3f},{n[1]:.3f},{n[2]:.3f}) "
            f"hover=({h[0]:.3f},{h[1]:.3f},{h[2]:.3f}) dur={self.interp.duration:.2f}s"
        )

    def _begin_descend(self, reason: str):
        hover = self._offset_z(self._grasp_pos_w, C.APPROACH_HEIGHT)
        err = self.ctx.ik.position_error_norm(hover)
        now, _q_now = self.ctx.ik.get_tcp_pose_w()
        self.interp.reset(
            now,
            self._grasp_pos_w,
            C.DESCEND_TIME,
            self._grasp_quat_w,
            self._grasp_quat_w,
        )
        print(
            f"[ces_fsm] approach done ({reason}) tcp_err={err*1000:.1f} mm — descend"
        )
        self._transition(CesPickPlacePhase.DESCEND)

    def step(self) -> CesCommand:
        dt = self.ctx.dt
        self.t += dt
        self._walk = [0.0, 0.0, 0.0, 0.8]
        try:
            return self._step_body()
        except Exception as e:
            if self.t - self._step_err_t > 2.0:
                self._step_err_t = self.t
                print(f"[ces_fsm] {self.phase.value} step error t={self.t:.2f}s: {e}")
                import traceback

                traceback.print_exc()
            timeout = C.ORIENT_TIME + C.SLIDE_TIME + C.GRASP_WAIT_MAX
            if (
                self.phase is CesPickPlacePhase.APPROACH
                and self._grasp_pos_w is not None
                and self.t > timeout
            ):
                try:
                    self._begin_descend("exception-timeout")
                except Exception:
                    self._transition(CesPickPlacePhase.DESCEND)
            return self._cmd(
                tcp=self._grasp_pos_w,
                quat=self._grasp_quat_w,
                arm_q=getattr(self.ctx, "_q_right", None),
            )
        finally:
            self._maybe_log()

    def _step_body(self) -> CesCommand:
        if self.phase is CesPickPlacePhase.SETTLE:
            self.gripper = C.GRIPPER_OPEN
            if self.t >= C.SETTLE_TIME and (
                self.ctx.is_standing() or self.t >= C.SETTLE_TIME + C.STAND_MIN_TIME
            ):
                self._plan_grasp()
                root_pos, _root_quat = self.ctx.get_base_pose_w()
                yaw = self.ctx.get_heading()
                x, y = float(root_pos[0, 0]), float(root_pos[0, 1])
                px, py = C.PICK_STAND_XY
                print(
                    f"[ces_fsm] snap to pick stand "
                    f"robot=({x:.2f},{y:.2f}) head={math.degrees(yaw):.0f}° "
                    f"pick=({px:.2f},{py:.2f}) d={math.hypot(px - x, py - y):.2f}"
                )
                self._transition(CesPickPlacePhase.GOTO_PICK)
                if hasattr(self.ctx, "reset_walk_filt"):
                    self.ctx.reset_walk_filt()
            # Hold spawn pose until walking starts so the zero-cmd gait cannot
            # drift +X (was -2.04 → -0.43 during settle).
            return self._cmd(
                tcp=None,
                snap=True,
                snap_xy=C.SPAWN_STAND_XY,
                snap_yaw=C.SPAWN_STAND_YAW,
            )

        if self.phase is CesPickPlacePhase.GOTO_PICK:
            self.gripper = C.GRIPPER_OPEN
            arrived, xy, yaw, pin = self._navigate("pick")
            if arrived:
                self._q_unfold0 = self.ctx.get_right_arm_q()[0].clone()
                self._q_unfold1 = torch.tensor(
                    C.RIGHT_ARM_READY, device=self.device, dtype=self._q_unfold0.dtype
                )
                print("[ces_fsm] at pick stand — unfolding arm")
                self._transition(CesPickPlacePhase.UNFOLD)
            return self._cmd(tcp=None, snap=pin, snap_xy=xy, snap_yaw=yaw)

        if self.phase is CesPickPlacePhase.UNFOLD:
            self.gripper = C.GRIPPER_OPEN
            s = ease_in_out(self.t / max(C.UNFOLD_TIME, 1e-3))
            q = (1.0 - s) * self._q_unfold0 + s * self._q_unfold1
            if self.t >= C.UNFOLD_TIME:
                self._reset_approach_path()
                self._transition(CesPickPlacePhase.APPROACH)
            return self._cmd(tcp=None, arm_q=q)

        if self.phase is CesPickPlacePhase.APPROACH:
            timeout = C.ORIENT_TIME + C.SLIDE_TIME + C.GRASP_WAIT_MAX
            if self.t > timeout:
                self._begin_descend("timeout")
                pos, quat = self._grasp_pos_w, self._grasp_quat_w
                return self._cmd(tcp=pos, quat=quat)
            if not self.interp.has_path:
                self._reset_approach_path()
            pos, quat = self.interp.step(self.ctx.dt)
            if pos is None or quat is None:
                hover = self._offset_z(self._grasp_pos_w, C.APPROACH_HEIGHT)
                pos, quat = hover, self._grasp_quat_w
            if self.interp.finished:
                hover = self._offset_z(self._grasp_pos_w, C.APPROACH_HEIGHT)
                err = self.ctx.ik.position_error_norm(hover)
                if err < 0.06:
                    self._begin_descend("arrived")
                else:
                    pos, quat = hover, self._grasp_quat_w
            return self._cmd(tcp=pos, quat=quat)

        if self.phase is CesPickPlacePhase.DESCEND:
            pos, quat = self.interp.step(self.ctx.dt)
            if pos is None or quat is None:
                pos, quat = self._grasp_pos_w, self._grasp_quat_w
            if self.interp.finished or self.t > C.DESCEND_TIME + C.GRASP_WAIT_MAX:
                err = self.ctx.ik.position_error_norm(self._grasp_pos_w)
                pos, quat = self._grasp_pos_w, self._grasp_quat_w
                if err < C.GRASP_POS_TOL or self.t > C.DESCEND_TIME + C.GRASP_WAIT_MAX:
                    self._start_grasp(err)
            return self._cmd(tcp=pos, quat=quat)

        if self.phase is CesPickPlacePhase.GRASP:
            s = ease_in_out(min(1.0, self.t / max(C.GRASP_TIME, 1e-3)))
            self.gripper = C.GRIPPER_OPEN + s * (C.GRIPPER_CLOSED - C.GRIPPER_OPEN)
            if self.t >= C.GRASP_TIME:
                self.gripper = C.GRIPPER_CLOSED
                up = self._offset_z(self._grasp_pos_w, C.LIFT_HEIGHT)
                self._hold_lift_pos = up
                # Cartesian +Z only.  An XY retract with top-down orientation
                # makes DiffIK flip the arm overhead.
                self.interp.reset(
                    self._grasp_pos_w,
                    up,
                    C.LIFT_TIME,
                    self._grasp_quat_w,
                    self._grasp_quat_w,
                )
                self._transition(CesPickPlacePhase.LIFT)
            return self._cmd(tcp=self._grasp_pos_w, quat=self._grasp_quat_w)

        if self.phase is CesPickPlacePhase.LIFT:
            self.gripper = C.GRIPPER_CLOSED
            pos, quat = self.interp.step(self.ctx.dt)
            if pos is None or quat is None:
                pos = self._hold_lift_pos
                quat = self._grasp_quat_w
            if self.interp.finished or self.t > C.LIFT_TIME + 0.5:
                self._carry_arm_q = self.ctx.get_right_arm_q()[0].clone()
                print("[ces_fsm] lifted — freeze arm, keep gripper closed")
                self._transition(CesPickPlacePhase.CARRY)
            return self._cmd(tcp=pos, quat=quat)

        if self.phase is CesPickPlacePhase.CARRY:
            self.gripper = C.GRIPPER_CLOSED
            q = self._carry_arm_q
            if q is None:
                q = self.ctx.get_right_arm_q()[0].clone()
                self._carry_arm_q = q
            if self.t >= C.CARRY_TIME:
                if self.stop_after == "lift":
                    print("[ces_fsm] stop_after=lift — still squeezing")
                    self._transition(CesPickPlacePhase.DONE)
                else:
                    print("[ces_fsm] carry frozen — snap to table (arm locked, gripper closed)")
                    self._transition(CesPickPlacePhase.HOLD)
            return self._cmd(
                tcp=None,
                arm_q=q,
                snap=True,
                snap_xy=C.PICK_STAND_XY,
                snap_yaw=C.PICK_STAND_YAW,
            )

        if self.phase is CesPickPlacePhase.HOLD:
            self.gripper = C.GRIPPER_CLOSED
            q = self._carry_arm_q
            if q is None:
                q = self.ctx.get_right_arm_q()[0].clone()
                self._carry_arm_q = q
            if self.t >= C.HOLD_TIME:
                print("[ces_fsm] snap to table (arm frozen, product in gripper)")
                self._transition(CesPickPlacePhase.GOTO_PLACE)
            return self._cmd(
                tcp=None,
                arm_q=q,
                snap=True,
                snap_xy=C.PICK_STAND_XY,
                snap_yaw=C.PICK_STAND_YAW,
            )

        if self.phase is CesPickPlacePhase.GOTO_PLACE:
            self.gripper = C.GRIPPER_CLOSED
            arrived, xy, yaw, pin = self._navigate("place")
            q = self._carry_arm_q
            if q is None:
                q = self.ctx.get_right_arm_q()[0]
                self._carry_arm_q = q.clone()
            if arrived:
                print("[ces_fsm] at table stand — placing")
                now, q_now = self.ctx.ik.get_tcp_pose_w()
                hover = self._offset_z(self._place_pos_w, C.PLACE_APPROACH_HEIGHT)
                self.interp.reset(
                    now,
                    hover,
                    C.PLACE_APPROACH_TIME,
                    q_now,
                    self._place_quat_w,
                )
                self._transition(CesPickPlacePhase.PLACE_APPROACH)
            return self._cmd(tcp=None, arm_q=q, snap=pin, snap_xy=xy, snap_yaw=yaw)

        if self.phase is CesPickPlacePhase.PLACE_APPROACH:
            self.gripper = C.GRIPPER_CLOSED
            pos, quat = self.interp.step(self.ctx.dt)
            if pos is None or quat is None:
                hover = self._offset_z(self._place_pos_w, C.PLACE_APPROACH_HEIGHT)
                pos, quat = hover, self._place_quat_w
            if self.interp.finished or self.t > C.PLACE_APPROACH_TIME + 1.0:
                hover = self._offset_z(self._place_pos_w, C.PLACE_APPROACH_HEIGHT)
                self.interp.reset(
                    hover,
                    self._place_pos_w,
                    C.PLACE_DESCEND_TIME,
                    self._place_quat_w,
                    self._place_quat_w,
                )
                self._transition(CesPickPlacePhase.PLACE_DESCEND)
            return self._cmd(
                tcp=pos,
                quat=quat,
                snap=True,
                snap_xy=C.PLACE_STAND_XY,
                snap_yaw=C.PLACE_STAND_YAW,
            )

        if self.phase is CesPickPlacePhase.PLACE_DESCEND:
            self.gripper = C.GRIPPER_CLOSED
            pos, quat = self.interp.step(self.ctx.dt)
            if pos is None or quat is None:
                pos, quat = self._place_pos_w, self._place_quat_w
            if self.interp.finished or self.t > C.PLACE_DESCEND_TIME + 1.0:
                self._transition(CesPickPlacePhase.RELEASE)
            return self._cmd(
                tcp=pos,
                quat=quat,
                snap=True,
                snap_xy=C.PLACE_STAND_XY,
                snap_yaw=C.PLACE_STAND_YAW,
            )

        if self.phase is CesPickPlacePhase.RELEASE:
            self.gripper = C.GRIPPER_OPEN
            if self.t >= C.RELEASE_TIME:
                now = self._place_pos_w
                up = self._offset_z(now, C.PLACE_APPROACH_HEIGHT)
                self.interp.reset(now, up, C.RETRACT_TIME, self._place_quat_w, self._place_quat_w)
                self._transition(CesPickPlacePhase.RETRACT)
            return self._cmd(
                tcp=self._place_pos_w,
                quat=self._place_quat_w,
                snap=True,
                snap_xy=C.PLACE_STAND_XY,
                snap_yaw=C.PLACE_STAND_YAW,
            )

        if self.phase is CesPickPlacePhase.RETRACT:
            self.gripper = C.GRIPPER_OPEN
            pos, quat = self.interp.step(self.ctx.dt)
            if pos is None or quat is None:
                pos = self._offset_z(self._place_pos_w, C.PLACE_APPROACH_HEIGHT)
                quat = self._place_quat_w
            if self.interp.finished or self.t > C.RETRACT_TIME + 0.5:
                self._transition(CesPickPlacePhase.DONE)
            return self._cmd(
                tcp=pos,
                quat=quat,
                snap=True,
                snap_xy=C.PLACE_STAND_XY,
                snap_yaw=C.PLACE_STAND_YAW,
            )

        if self.phase is CesPickPlacePhase.DONE:
            if self.stop_after == "lift":
                self.gripper = C.GRIPPER_CLOSED
            if self.stop_after == "lift" and self._carry_arm_q is not None:
                return self._cmd(
                    tcp=None,
                    arm_q=self._carry_arm_q,
                    snap=True,
                    snap_xy=C.PICK_STAND_XY,
                    snap_yaw=C.PICK_STAND_YAW,
                    done=True,
                )
            if self.stop_after == "lift" and self._hold_lift_pos is not None:
                return self._cmd(
                    tcp=self._hold_lift_pos, quat=self._grasp_quat_w, done=True
                )
            return self._cmd(
                tcp=None,
                arm_q=self._carry_arm_q,
                snap=True,
                snap_xy=C.PLACE_STAND_XY,
                snap_yaw=C.PLACE_STAND_YAW,
                done=True,
            )

        # FAILED: keep the squeeze if we already picked
        if self._carry_arm_q is not None:
            self.gripper = C.GRIPPER_CLOSED
        return self._cmd(failed=True, arm_q=self._carry_arm_q)
