# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""CES LoadingLine 取放状态机。

路点开：SETTLE → GOTO_PICK → UNFOLD(00→30) → DESCEND(锁XY、只落Z、40 仅 q_ref)
→ GRASP → LIFT → CARRY/HOLD → GOTO_PLACE(snap) → PLACE_APPROACH(抬起姿态 IK)
→ RELEASE → RETRACT。
"""
from __future__ import annotations

import enum
import math
from dataclasses import dataclass

import torch

from isaaclab.utils.math import quat_apply, quat_from_matrix, subtract_frame_transforms

from action_provider.ces_grasp import constants as C
from action_provider.ces_grasp.pose_library import CesWaypointSet, load_waypoint_set
from action_provider.manip_common import CartesianInterpolator, JointSpaceInterpolator
from action_provider.manip_common.interpolation import ease_in_out, lerp


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
    arm_q_ref: torch.Tensor | None = None


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
    def __init__(
        self,
        ctx,
        station_mode: str = "snap",
        stop_after: str = "place",
        use_joint_waypoints: bool = False,
        waypoint_set: str | None = None,
    ):
        self.ctx = ctx
        requested = station_mode if station_mode in ("snap", "walk") else "snap"
        if requested == "walk":
            print("[ces_fsm] walk is disabled — using snap teleport")
        self.station_mode = "snap"
        self.stop_after = stop_after if stop_after in ("lift", "place") else C.STOP_AFTER
        self.use_joint_waypoints = bool(use_joint_waypoints)
        self.device = ctx.device
        self.interp = CartesianInterpolator(self.device)
        self.joint_interp = JointSpaceInterpolator(self.device)
        self._waypoints: CesWaypointSet | None = None
        self._q_wp30: torch.Tensor | None = None
        self._q_wp40: torch.Tensor | None = None
        self._wp_arrive_names: list[str] = []
        self._wp_logged = 0
        self._verify_descend_xy: tuple[float, float] | None = None
        self._logged_q_ref_once = False
        if self.use_joint_waypoints:
            self._waypoints = load_waypoint_set(waypoint_set or C.WAYPOINT_SET_DEFAULT)
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
        self._carry_arm_q: torch.Tensor | None = None
        self._grasp_arm_q: torch.Tensor | None = None
        self._place_arm_q: torch.Tensor | None = None
        self._dbg = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        self._spawn_yaw_applied = False
        self._step_err_t = -10.0
        extra = ""
        if self._waypoints is not None:
            extra = (
                f" waypoints={self._waypoints.name} "
                f"handoff={self._waypoints.joint_waypoints[-1]} "
                f"q_ref={self._waypoints.q_ref_from}->{self._waypoints.q_ref_to}"
            )
        print(
            f"[ces_fsm] drop-place station_mode={self.station_mode} stop_after={self.stop_after} "
            f"pick_stand=({C.PICK_STAND_XY[0]:.3f},{C.PICK_STAND_XY[1]:.3f}) "
            f"place_stand=({C.PLACE_STAND_XY[0]:.3f},{C.PLACE_STAND_XY[1]:.3f}) "
            f"x_b_place={C.X_B_PLACE:.2f} "
            f"(snap locate; Dex1 friction grasp; arm locked while reaching)"
            f"{extra}"
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
        self._carry_arm_q = None
        self._grasp_arm_q = None
        self._place_arm_q = None
        self._q_wp30 = None
        self._q_wp40 = None
        self._wp_arrive_names = []
        self._wp_logged = 0
        self._verify_descend_xy = None
        self._logged_q_ref_once = False
        self.joint_interp.clear()
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
        arm_q_ref=None,
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
            arm_q_ref=arm_q_ref,
        )

    def _squeezing(self) -> bool:
        """True from lift through place-approach: jaws stay at GRIPPER_CLOSED."""
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
        )

    def _place_drop_pos(self) -> torch.Tensor:
        """TCP target at drop height (table + PLACE_RELEASE_ABOVE_TABLE)."""
        if self._place_pos_w is None:
            return torch.tensor(
                [[C.PLACE_TARGET_XY[0], C.PLACE_TARGET_XY[1], C.PLACE_Z]],
                device=self.device,
            )
        return self._place_pos_w

    def _frozen_place_arm(self) -> torch.Tensor:
        if self._place_arm_q is None:
            self._place_arm_q = self.ctx.get_right_arm_q()[0].clone()
        return self._place_arm_q

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
        self._grasp_arm_q = self.ctx.get_right_arm_q()[0].clone()
        if self.use_joint_waypoints:
            pb = self._log_tcp("descend_end")
            if self._verify_descend_xy is not None:
                now_w, _ = self.ctx.ik.get_tcp_pose_w()
                dxy = math.hypot(
                    float(now_w[0, 0]) - self._verify_descend_xy[0],
                    float(now_w[0, 1]) - self._verify_descend_xy[1],
                )
                print(
                    f"[ces_verify] descend hold_xy dxy={dxy*1000:.1f}mm "
                    f"z_b={pb[2]:.4f} (target 0.101)"
                )
        self._transition(CesPickPlacePhase.GRASP)

    def _begin_joint_lift(self):
        """Raise with a one-shot IK goal, then joint-space playback.

        Per-frame DiffIK while the pads are squeezing makes the part and
        wrist chatter.  Solve the lift TCP once, then interpolate q.
        """
        q_now = self.ctx.get_right_arm_q()[0].clone()
        if self._grasp_arm_q is None:
            self._grasp_arm_q = q_now
        up = self._offset_z(self._grasp_pos_w, C.LIFT_HEIGHT)
        up = up.clone()
        up[:, 1] = up[:, 1] + C.LIFT_SHIFT_Y
        self._hold_lift_pos = up
        try:
            q_lift = self.ctx.ik.solve(up, self._grasp_quat_w, q_ref=q_now)[0]
        except Exception as exc:
            print(f"[ces_fsm] lift IK failed ({exc}) — hold grasp q")
            q_lift = q_now
        self.joint_interp.reset(q_now, q_lift, C.LIFT_TIME)
        print(
            f"[ces_fsm] joint lift z+{C.LIFT_HEIGHT:.3f} y{C.LIFT_SHIFT_Y:+.3f} "
            f"dur={C.LIFT_TIME:.2f}s (no per-frame IK)"
        )

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
        now, q_now = self.ctx.ik.get_tcp_pose_w()
        if self.use_joint_waypoints:
            # Z-only drop.  Do not retarget orientation to top_down_grasp_quat
            # (that made DiffIK twist shoulder/elbow).  Pose 40 is q_ref only.
            planned_z = float(self._grasp_pos_w[0, 2])
            goal = now.clone()
            goal[:, 2] = planned_z
            self._grasp_pos_w = goal.clone()
            self._grasp_quat_w = q_now.clone()
            self._carry_quat_w = q_now.clone()
            self._verify_descend_xy = (float(now[0, 0]), float(now[0, 1]))
            self.interp.reset(now, goal, C.DESCEND_TIME, q_now, q_now)
            print(
                f"[ces_fsm] waypoint handoff ({reason}) vertical descend "
                f"z={float(now[0, 2]):.3f}->{planned_z:.3f} "
                f"(hold XY+orientation from 30; 40 is q_ref only)"
            )
            self._log_tcp("handoff30")
            live_q = self.ctx.get_right_arm_q()[0]
            self._log_wp("live_30", live_q, self._q_wp30)
        else:
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

    def _handoff_cmd(self) -> CesCommand:
        """First descend frame: current TCP/orientation, not the goal quat."""
        tcp = self.interp.points[0] if self.interp.has_path else self._grasp_pos_w
        quat = self.interp.quats[0] if self.interp.quats else self._grasp_quat_w
        return self._cmd(tcp=tcp, quat=quat, arm_q_ref=self._q_ref_for_descend())

    def _fmt_q(self, q: torch.Tensor) -> str:
        return "[" + ", ".join(f"{float(v):+.3f}" for v in q.reshape(-1)[:7]) + "]"

    def _q_limit_margin(self, q: torch.Tensor) -> float:
        ik = getattr(self.ctx, "ik", None)
        if ik is None or not hasattr(ik, "q_min"):
            return float("nan")
        qv = q.reshape(-1)[:7]
        lo = qv - ik.q_min
        hi = ik.q_max - qv
        return float(torch.minimum(lo, hi).min().item())

    def _tcp_in_pelvis(self):
        root_pos, root_quat = self.ctx.get_base_pose_w()
        tcp_pos, tcp_quat = self.ctx.ik.get_tcp_pose_w()
        pos_b, quat_b = subtract_frame_transforms(root_pos, root_quat, tcp_pos, tcp_quat)
        return pos_b[0], quat_b[0]

    def _log_wp(self, name: str, q: torch.Tensor, prev_q: torch.Tensor | None = None):
        dq = ""
        if prev_q is not None:
            dmax = float((q.reshape(-1)[:7] - prev_q.reshape(-1)[:7]).abs().max().item())
            dq = f" dmax={dmax:.3f}rad"
        print(
            f"[ces_verify] wp={name} q={self._fmt_q(q)} "
            f"limit_margin={self._q_limit_margin(q):.3f}rad{dq}"
        )

    def _log_tcp(self, tag: str):
        pos_b, _ = self._tcp_in_pelvis()
        tcp_w, _ = self.ctx.ik.get_tcp_pose_w()
        pb = [float(pos_b[i]) for i in range(3)]
        pw = [float(tcp_w[0, i]) for i in range(3)]
        if "place" in tag:
            print(
                f"[ces_verify] {tag} tcp_w=({pw[0]:.4f},{pw[1]:.4f},{pw[2]:.4f}) "
                f"tcp_b=({pb[0]:.4f},{pb[1]:.4f},{pb[2]:.4f})"
            )
            return pb
        tgt = (0.320, -0.380, 0.181) if "30" in tag or "handoff" in tag else (0.320, -0.380, 0.101)
        err = (pb[0] - tgt[0], pb[1] - tgt[1], pb[2] - tgt[2])
        print(
            f"[ces_verify] {tag} tcp_w=({pw[0]:.4f},{pw[1]:.4f},{pw[2]:.4f}) "
            f"tcp_b=({pb[0]:.4f},{pb[1]:.4f},{pb[2]:.4f}) "
            f"vs({tgt[0]:.3f},{tgt[1]:.3f},{tgt[2]:.3f}) "
            f"err_mm=({err[0]*1000:+.1f},{err[1]*1000:+.1f},{err[2]*1000:+.1f})"
        )
        return pb

    def _flush_wp_arrivals(self, q: torch.Tensor):
        while (
            self._wp_logged < len(self.joint_interp.bounds)
            and self.joint_interp.elapsed + 1e-9 >= self.joint_interp.bounds[self._wp_logged]
        ):
            name = (
                self._wp_arrive_names[self._wp_logged]
                if self._wp_logged < len(self._wp_arrive_names)
                else f"seg{self._wp_logged}"
            )
            prev = None
            if self._wp_logged > 0:
                prev_name = self._wp_arrive_names[self._wp_logged - 1]
                if self._waypoints is not None and prev_name in self._waypoints.q_by_name:
                    prev = self._pose_q(prev_name)
            self._log_wp(name, q, prev)
            self._wp_logged += 1

    def _pose_q(self, pose_name: str) -> torch.Tensor:
        if self._waypoints is None:
            raise RuntimeError("joint waypoints are not loaded")
        q = self._waypoints.q_by_name[pose_name]
        return torch.tensor(q, device=self.device, dtype=torch.float32)

    def _start_joint_waypoints(self):
        """Play authored 00→10→20→25→30 in joint space from the live arm q."""
        wp = self._waypoints
        if wp is None:
            raise RuntimeError("joint waypoints are not loaded")
        q_now = self.ctx.get_right_arm_q()[0].clone()
        qs = [self._pose_q(name) for name in wp.joint_waypoints]
        self._q_wp30 = qs[-1].clone()
        self._q_wp40 = self._pose_q(wp.q_ref_to)
        durations = list(wp.joint_segment_durations)
        lead = torch.norm(q_now - qs[0]).item()
        if lead > C.WAYPOINT_LEAD_IN_TOL:
            qs = [q_now] + qs
            durations = [C.WAYPOINT_LEAD_IN_TIME] + durations
            self._wp_arrive_names = list(wp.joint_waypoints)
            print(
                f"[ces_fsm] joint path lead-in {lead:.3f} rad "
                f"{'→'.join(wp.joint_waypoints)} dur={sum(durations):.2f}s"
            )
        else:
            qs[0] = q_now
            self._wp_arrive_names = list(wp.joint_waypoints[1:])
            self._log_wp(wp.joint_waypoints[0], q_now)
            print(
                f"[ces_fsm] joint path {'→'.join(wp.joint_waypoints)} "
                f"dur={sum(durations):.2f}s (start≈00)"
            )
        self._wp_logged = 0
        self.joint_interp.reset_path(qs, durations)

    def _begin_place_approach(self):
        """从抬起姿态 IK 到桌上方。锁朝向，q_ref=lift；已靠近则只落 Z。"""
        now, q_now = self.ctx.ik.get_tcp_pose_w()
        drop = self._place_drop_pos()
        dxy = math.hypot(
            float(drop[0, 0] - now[0, 0]),
            float(drop[0, 1] - now[0, 1]),
        )
        goal = now.clone()
        goal[:, 2] = float(drop[0, 2])
        hold_xy = dxy <= C.PLACE_HOLD_XY_M
        if not hold_xy:
            goal[:, 0] = drop[0, 0]
            goal[:, 1] = drop[0, 1]
        self._place_pos_w = goal.clone()
        self._place_quat_w = q_now.clone()
        if self._carry_arm_q is None:
            self._carry_arm_q = self.ctx.get_right_arm_q()[0].clone()
        self.interp.reset(now, goal, C.PLACE_APPROACH_TIME, q_now, q_now)
        mode = "hold_xy+drop_z" if hold_xy else "slide_xy+drop_z"
        print(
            f"[ces_fsm] at table stand — IK place from lift ({mode}) "
            f"dxy={dxy*1000:.0f}mm z={float(now[0, 2]):.3f}->{float(goal[0, 2]):.3f} "
            f"(hold lift orientation; q_ref=lift)"
        )
        self._log_tcp("place_start")

    def _q_ref_for_place(self) -> torch.Tensor | None:
        """放置 IK 用抬起 q，避免重新解出怪姿态。"""
        return self._carry_arm_q

    def _q_ref_for_descend(self) -> torch.Tensor | None:
        """30→40 只作零空间提示，不作 arm_q。"""
        if self._q_wp30 is None or self._q_wp40 is None:
            return None
        if not self.interp.has_path:
            return self._q_wp40
        s = min(1.0, self.interp.elapsed / max(self.interp.duration, 1e-6))
        return lerp(self._q_wp30, self._q_wp40, ease_in_out(s))

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
                if self.use_joint_waypoints:
                    print("[ces_fsm] at pick stand — joint waypoints 00→30")
                else:
                    self._q_unfold0 = self.ctx.get_right_arm_q()[0].clone()
                    self._q_unfold1 = torch.tensor(
                        C.RIGHT_ARM_READY, device=self.device, dtype=self._q_unfold0.dtype
                    )
                    print("[ces_fsm] at pick stand — unfolding arm")
                self._transition(CesPickPlacePhase.UNFOLD)
            return self._cmd(tcp=None, snap=pin, snap_xy=xy, snap_yaw=yaw)

        if self.phase is CesPickPlacePhase.UNFOLD:
            self.gripper = C.GRIPPER_OPEN
            if self.use_joint_waypoints:
                if not self.joint_interp.has_path:
                    self._start_joint_waypoints()
                q = self.joint_interp.step(self.ctx.dt)
                if q is None:
                    q = self.ctx.get_right_arm_q()[0]
                self._flush_wp_arrivals(q)
                if self.joint_interp.finished:
                    self._begin_descend("joint-waypoint-handoff")
                    return self._handoff_cmd()
                return self._cmd(tcp=None, arm_q=q)
            s = ease_in_out(self.t / max(C.UNFOLD_TIME, 1e-3))
            q = (1.0 - s) * self._q_unfold0 + s * self._q_unfold1
            if self.t >= C.UNFOLD_TIME:
                self._reset_approach_path()
                self._transition(CesPickPlacePhase.APPROACH)
            return self._cmd(tcp=None, arm_q=q)

        if self.phase is CesPickPlacePhase.APPROACH:
            if self.use_joint_waypoints:
                self._begin_descend("joint-waypoint-handoff")
                return self._handoff_cmd()
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
            q_ref = self._q_ref_for_descend() if self.use_joint_waypoints else None
            if self.use_joint_waypoints and q_ref is not None and not self._logged_q_ref_once:
                self._logged_q_ref_once = True
                live = self.ctx.get_right_arm_q()[0]
                dw = ""
                if self._q_wp30 is not None:
                    dw = (
                        f" wrist_vs_30="
                        f"{self._fmt_q((live - self._q_wp30)[4:7])}"
                    )
                print(
                    f"[ces_verify] descend q_ref_only (arm_q is None) "
                    f"q_ref={self._fmt_q(q_ref)}{dw}"
                )
            if self.interp.finished or self.t > C.DESCEND_TIME + C.GRASP_WAIT_MAX:
                err = self.ctx.ik.position_error_norm(self._grasp_pos_w)
                pos, quat = self._grasp_pos_w, self._grasp_quat_w
                # Waypoint descend already holds XY from 30; do not wait on TCP tol.
                if (
                    self.use_joint_waypoints
                    or err < C.GRASP_POS_TOL
                    or self.t > C.DESCEND_TIME + C.GRASP_WAIT_MAX
                ):
                    self._start_grasp(err)
            return self._cmd(tcp=pos, quat=quat, arm_q_ref=q_ref)

        if self.phase is CesPickPlacePhase.GRASP:
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
            return self._cmd(tcp=None, arm_q=q)

        if self.phase is CesPickPlacePhase.LIFT:
            self.gripper = C.GRIPPER_CLOSED
            q = self.joint_interp.step(self.ctx.dt)
            if q is None:
                q = self._grasp_arm_q
            if q is None:
                q = self.ctx.get_right_arm_q()[0]
            if self.joint_interp.finished or self.t > C.LIFT_TIME + 0.5:
                self._carry_arm_q = q.clone()
                print(
                    f"[ces_fsm] lifted — freeze arm q={self._fmt_q(q)} "
                    f"(HOLD 用此 q 走路/换站)"
                )
                self._log_tcp("lift_done")
                self._transition(CesPickPlacePhase.CARRY)
            return self._cmd(tcp=None, arm_q=q)

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
                pb = self._tcp_in_pelvis()[0]
                tcp_w, _ = self.ctx.ik.get_tcp_pose_w()
                print(
                    f"[ces_fsm] HOLD tcp_w="
                    f"({float(tcp_w[0, 0]):.4f},{float(tcp_w[0, 1]):.4f},{float(tcp_w[0, 2]):.4f}) "
                    f"tcp_b=({float(pb[0]):.4f},{float(pb[1]):.4f},{float(pb[2]):.4f}) "
                    f"q={self._fmt_q(q)}"
                )
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
                self._begin_place_approach()
                self._transition(CesPickPlacePhase.PLACE_APPROACH)
            return self._cmd(tcp=None, arm_q=q, snap=pin, snap_xy=xy, snap_yaw=yaw)

        if self.phase is CesPickPlacePhase.PLACE_APPROACH:
            self.gripper = C.GRIPPER_CLOSED
            drop = self._place_pos_w if self._place_pos_w is not None else self._place_drop_pos()
            quat_hold = self._place_quat_w
            pos, quat = self.interp.step(self.ctx.dt)
            if pos is None or quat is None:
                pos, quat = drop, quat_hold
            if self.interp.finished or self.t > C.PLACE_APPROACH_TIME + 1.0:
                self._place_arm_q = self.ctx.get_right_arm_q()[0].clone()
                print("[ces_fsm] drop height reached — open gripper, let product fall")
                self._log_tcp("place_release")
                self._transition(CesPickPlacePhase.RELEASE)
            return self._cmd(
                tcp=pos,
                quat=quat,
                arm_q_ref=self._q_ref_for_place(),
                snap=True,
                snap_xy=C.PLACE_STAND_XY,
                snap_yaw=C.PLACE_STAND_YAW,
            )

        if self.phase is CesPickPlacePhase.PLACE_DESCEND:
            # Kept only as a fallback: never IK onto the tabletop.
            self._place_arm_q = self.ctx.get_right_arm_q()[0].clone()
            print("[ces_fsm] skip place-descend — release at current height")
            self._transition(CesPickPlacePhase.RELEASE)
            return self._cmd(
                arm_q=self._place_arm_q,
                snap=True,
                snap_xy=C.PLACE_STAND_XY,
                snap_yaw=C.PLACE_STAND_YAW,
            )

        if self.phase is CesPickPlacePhase.RELEASE:
            self.gripper = C.GRIPPER_OPEN
            q = self._frozen_place_arm()
            if self.t >= C.RELEASE_TIME:
                self._transition(CesPickPlacePhase.RETRACT)
            return self._cmd(
                tcp=None,
                arm_q=q,
                snap=True,
                snap_xy=C.PLACE_STAND_XY,
                snap_yaw=C.PLACE_STAND_YAW,
            )

        if self.phase is CesPickPlacePhase.RETRACT:
            self.gripper = C.GRIPPER_OPEN
            q = self._frozen_place_arm()
            if self.t >= C.RETRACT_TIME:
                self._transition(CesPickPlacePhase.DONE)
            return self._cmd(
                tcp=None,
                arm_q=q,
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
            q = self._place_arm_q if self._place_arm_q is not None else self._carry_arm_q
            return self._cmd(
                tcp=None,
                arm_q=q,
                snap=True,
                snap_xy=C.PLACE_STAND_XY,
                snap_yaw=C.PLACE_STAND_YAW,
                done=True,
            )

        # FAILED: keep the squeeze if we already picked
        if self._carry_arm_q is not None:
            self.gripper = C.GRIPPER_CLOSED
        return self._cmd(failed=True, arm_q=self._carry_arm_q)
