# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""按关节名加载 CES 右手路点 JSON。忽略 viewer 字段，不使用 DDS 下标。"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from action_provider.ces_grasp import constants as C

_POSES_ROOT = Path(__file__).resolve().parent / "poses"
_JOINT_SPACE = "joint_space"
_Q_REF_CONTROL = "cartesian_vertical_ik_with_dynamic_q_ref"
_INTERPOLATION_METHODS = {
    "segment_smoothstep",
    "monotone_cubic_hermite",
}


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
    return_start: str
    return_waypoints: tuple[str, ...]
    return_segment_durations: tuple[float, ...]
    return_interpolation_method: str
    place_start: str
    place_waypoints: tuple[str, ...]
    place_segment_durations: tuple[float, ...]
    place_interpolation_method: str
    place_descend_duration: float
    interpolation_method: str


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
    data = json.loads(path.read_text(encoding="utf-8"))
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

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    interpolation = manifest.get("interpolation", {})
    interpolation_method = (
        str(interpolation.get("method", "segment_smoothstep"))
        if isinstance(interpolation, dict)
        else str(interpolation)
    )
    if interpolation_method not in _INTERPOLATION_METHODS:
        raise ValueError(
            f"waypoint set {set_name} uses unsupported interpolation "
            f"{interpolation_method!r}"
        )
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

    return_spec = manifest.get("return_path")
    return_interpolation_method = interpolation_method
    if return_spec is not None:
        if not isinstance(return_spec, dict):
            raise ValueError(f"waypoint set {set_name} return_path must be an object")
        return_interpolation_method = str(
            return_spec.get("interpolation", interpolation_method)
        )
        if return_interpolation_method not in _INTERPOLATION_METHODS:
            raise ValueError(
                f"waypoint set {set_name} return_path uses unsupported interpolation "
                f"{return_interpolation_method!r}"
            )
        logical_return = tuple(
            str(value) for value in return_spec.get("logical_waypoints", [])
        )
        return_waypoints = tuple(
            str(value) for value in return_spec.get("commanded_waypoints", [])
        )
        if len(logical_return) < 2 or logical_return[1:] != return_waypoints:
            raise ValueError(
                f"waypoint set {set_name} return_path must be "
                "logical_start + commanded_waypoints"
            )
        return_start = logical_return[0]
        return_durations: list[float] = []
        expected_src = return_start
        return_segments = list(return_spec.get("segments", []))
        if len(return_segments) != len(return_waypoints):
            raise ValueError(
                f"waypoint set {set_name} return_path has {len(return_segments)} "
                f"segments for {len(return_waypoints)} commanded waypoints"
            )
        for index, segment in enumerate(return_segments):
            src = str(segment["from"])
            dst = str(segment["to"])
            duration = float(segment["duration_s"])
            if src != expected_src or dst != return_waypoints[index]:
                raise ValueError(
                    f"return segment {src}->{dst} does not match expected "
                    f"{expected_src}->{return_waypoints[index]}"
                )
            if duration <= 0.0:
                raise ValueError(f"return segment {src}->{dst} duration must be positive")
            return_durations.append(duration)
            expected_src = dst
    else:
        # Compatibility for natural_v1/v2 manifests: keep the historical
        # live-post-lift lead-in followed by the reversed forward path to 00.
        return_start = q_ref_to
        return_waypoints = tuple(reversed(joint_waypoints))
        return_durations = [
            float(C.RETURN_LEAD_IN_TIME),
            *reversed(joint_durations),
        ]

    place_spec = manifest.get("place_path")
    place_start = ""
    place_waypoints: tuple[str, ...] = ()
    place_durations: list[float] = []
    place_interpolation_method = interpolation_method
    place_descend_duration = 0.0
    if place_spec is not None:
        if not isinstance(place_spec, dict):
            raise ValueError(f"waypoint set {set_name} place_path must be an object")
        place_interpolation_method = str(
            place_spec.get("interpolation", "segment_smoothstep")
        )
        if place_interpolation_method not in _INTERPOLATION_METHODS:
            raise ValueError(
                f"waypoint set {set_name} place_path uses unsupported interpolation "
                f"{place_interpolation_method!r}"
            )
        logical_place = tuple(
            str(value) for value in place_spec.get("logical_waypoints", [])
        )
        place_waypoints = tuple(
            str(value) for value in place_spec.get("commanded_waypoints", [])
        )
        if len(logical_place) < 2 or logical_place[1:] != place_waypoints:
            raise ValueError(
                f"waypoint set {set_name} place_path must be "
                "logical_start + commanded_waypoints"
            )
        place_start = logical_place[0]
        expected_src = place_start
        place_segments = list(place_spec.get("segments", []))
        if len(place_segments) != len(place_waypoints):
            raise ValueError(
                f"waypoint set {set_name} place_path has {len(place_segments)} "
                f"segments for {len(place_waypoints)} commanded waypoints"
            )
        for index, segment in enumerate(place_segments):
            src = str(segment["from"])
            dst = str(segment["to"])
            duration = float(segment["duration_s"])
            if src != expected_src or dst != place_waypoints[index]:
                raise ValueError(
                    f"place segment {src}->{dst} does not match expected "
                    f"{expected_src}->{place_waypoints[index]}"
                )
            if str(segment.get("runtime_control")) != _JOINT_SPACE:
                raise ValueError(
                    f"place segment {src}->{dst} must use {_JOINT_SPACE} control"
                )
            if duration <= 0.0:
                raise ValueError(f"place segment {src}->{dst} duration must be positive")
            place_durations.append(duration)
            expected_src = dst

        vertical = place_spec.get("vertical_compensation")
        if not isinstance(vertical, dict):
            raise ValueError(
                f"waypoint set {set_name} place_path needs vertical_compensation"
            )
        expected_vertical = {
            "runtime_control": "cartesian_position_only_ik",
            "xy_policy": "hold_live_tcp_xy_after_15",
            "z_policy": "descend_only_to_scene_tray_clearance",
        }
        for key, expected in expected_vertical.items():
            actual = str(vertical.get(key, ""))
            if actual != expected:
                raise ValueError(
                    f"waypoint set {set_name} vertical_compensation {key} "
                    f"must be {expected!r}, got {actual!r}"
                )
        if str(vertical.get("target_constant", "")) != "PLACE_FINAL_TCP_Z":
            raise ValueError(
                f"waypoint set {set_name} vertical compensation must target "
                "PLACE_FINAL_TCP_Z"
            )
        if vertical.get("orientation_target") is not None:
            raise ValueError(
                f"waypoint set {set_name} vertical compensation must be position-only"
            )
        expected_q_ref = place_waypoints[-1] if place_waypoints else ""
        if str(vertical.get("q_ref", "")) != expected_q_ref:
            raise ValueError(
                f"waypoint set {set_name} vertical compensation q_ref must be "
                f"{expected_q_ref!r}"
            )
        place_descend_duration = float(vertical.get("duration_s", 0.0))
        if place_descend_duration <= 0.0:
            raise ValueError(
                f"waypoint set {set_name} vertical compensation duration must be positive"
            )

    if handoff and joint_waypoints and joint_waypoints[-1] != handoff:
        raise ValueError(
            f"joint-space path ends at {joint_waypoints[-1]}, expected handoff {handoff}"
        )
    if return_start and q_ref_to and return_start != q_ref_to:
        raise ValueError(
            f"waypoint set {set_name} return starts at {return_start}, "
            f"expected q_ref phase {q_ref_to}"
        )
    if place_start and return_waypoints and place_start != return_waypoints[-1]:
        raise ValueError(
            f"waypoint set {set_name} place starts at {place_start}, "
            f"expected carry endpoint {return_waypoints[-1]}"
        )
    missing_wp = [
        n
        for n in [
            *joint_waypoints,
            q_ref_from,
            q_ref_to,
            return_start,
            *return_waypoints,
            place_start,
            *place_waypoints,
        ]
        if n and n not in q_by_name
    ]
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
        return_start=return_start,
        return_waypoints=tuple(return_waypoints),
        return_segment_durations=tuple(return_durations),
        return_interpolation_method=return_interpolation_method,
        place_start=place_start,
        place_waypoints=place_waypoints,
        place_segment_durations=tuple(place_durations),
        place_interpolation_method=place_interpolation_method,
        place_descend_duration=place_descend_duration,
        interpolation_method=interpolation_method,
    )
