# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""Load authored CES right-arm waypoint JSON by joint name.

Viewer-only fields (``viewer_urdf``, ``ik_end``) are ignored.  ``q`` is
reordered onto :data:`RIGHT_ARM_JOINTS` so DDS motor indices are never used.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from action_provider.ces_grasp import constants as C

_POSES_ROOT = Path(__file__).resolve().parent / "poses"
_JOINT_SPACE = "joint_space"
_Q_REF_CONTROL = "cartesian_vertical_ik_with_dynamic_q_ref"


@dataclass(frozen=True)
class CesWaypointSet:
    name: str
    joint_names: tuple[str, ...]
    q_by_name: dict[str, tuple[float, ...]]
    joint_waypoints: tuple[str, ...]
    joint_segment_durations: tuple[float, ...]
    q_ref_from: str
    q_ref_to: str
    q_ref_duration: float


def _remap_q(joint_order: list[str], q: list[float], target_names: list[str]) -> tuple[float, ...]:
    if len(joint_order) != len(q):
        raise ValueError(
            f"pose joint_order has {len(joint_order)} names but q has {len(q)} values"
        )
    by_name = dict(zip(joint_order, q))
    missing = [n for n in target_names if n not in by_name]
    if missing:
        raise ValueError(f"pose is missing joints {missing}; match by name, not index")
    return tuple(float(by_name[n]) for n in target_names)


def _load_pose_q(path: Path, target_names: list[str]) -> tuple[str, tuple[float, ...]]:
    data = json.loads(path.read_text())
    name = str(data["name"])
    q = _remap_q(list(data["joint_order"]), list(data["q"]), target_names)
    return name, q


def load_waypoint_set(name: str | None = None) -> CesWaypointSet:
    """Load a pose directory under ``ces_grasp/poses/<name>``."""
    set_name = name or C.WAYPOINT_SET_DEFAULT
    root = _POSES_ROOT / set_name
    manifest_path = root / "trajectory_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"CES waypoint manifest not found: {manifest_path}")

    manifest = json.loads(manifest_path.read_text())
    target_names = list(C.RIGHT_ARM_JOINTS)
    q_by_name: dict[str, tuple[float, ...]] = {}
    for pose_file in manifest.get("pose_files", []):
        pose_name, q = _load_pose_q(root / pose_file, target_names)
        q_by_name[pose_name] = q

    joint_waypoints: list[str] = []
    joint_durations: list[float] = []
    q_ref_from = str(manifest.get("future_project_handoff", {}).get("dynamic_q_ref_from", ""))
    q_ref_to = str(manifest.get("future_project_handoff", {}).get("dynamic_q_ref_to", ""))
    q_ref_duration = 0.0
    handoff = str(manifest.get("future_project_handoff", {}).get("joint_space_through", ""))

    for seg in manifest.get("segments", []):
        control = seg.get("future_project_control")
        src, dst = str(seg["from"]), str(seg["to"])
        dur = float(seg["duration_s"])
        if control == _JOINT_SPACE:
            if not joint_waypoints:
                joint_waypoints.append(src)
            if joint_waypoints[-1] != src:
                raise ValueError(
                    f"joint-space segment {src}->{dst} does not continue from {joint_waypoints[-1]}"
                )
            joint_waypoints.append(dst)
            joint_durations.append(dur)
        elif control == _Q_REF_CONTROL:
            q_ref_from = src
            q_ref_to = dst
            q_ref_duration = dur

    if handoff and joint_waypoints and joint_waypoints[-1] != handoff:
        raise ValueError(
            f"joint-space path ends at {joint_waypoints[-1]}, expected handoff {handoff}"
        )
    missing_wp = [n for n in joint_waypoints + [q_ref_from, q_ref_to] if n and n not in q_by_name]
    if missing_wp:
        raise ValueError(f"waypoint set {set_name} missing poses {missing_wp}")
    if len(joint_waypoints) < 2 or not joint_durations:
        raise ValueError(f"waypoint set {set_name} has no joint-space segments")
    if not q_ref_from or not q_ref_to:
        raise ValueError(f"waypoint set {set_name} is missing 30->40 q_ref poses")

    return CesWaypointSet(
        name=str(manifest.get("name", set_name)),
        joint_names=tuple(target_names),
        q_by_name=q_by_name,
        joint_waypoints=tuple(joint_waypoints),
        joint_segment_durations=tuple(joint_durations),
        q_ref_from=q_ref_from,
        q_ref_to=q_ref_to,
        q_ref_duration=q_ref_duration,
    )
