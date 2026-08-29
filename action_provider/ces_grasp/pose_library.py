# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""Load the single CES Baseline runtime trajectory manifest."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from action_provider.ces_grasp import constants as C

_NAME = "ces_pick_smooth_v1"
_MANIFEST = (
    Path(__file__).resolve().parent / "poses" / _NAME / "trajectory_manifest.json"
)
_METHODS = {"segment_smoothstep", "monotone_cubic_hermite"}


@dataclass(frozen=True)
class CesTrajectory:
    name: str
    q_by_name: dict[str, tuple[float, ...]]
    joint_waypoints: tuple[str, ...]
    joint_segment_durations: tuple[float, ...]
    q_ref_from: str
    q_ref_to: str
    return_start: str
    return_waypoints: tuple[str, ...]
    return_segment_durations: tuple[float, ...]
    return_interpolation_method: str
    place_start: str
    place_waypoints: tuple[str, ...]
    place_segment_durations: tuple[float, ...]
    place_interpolation_method: str
    interpolation_method: str


def _path(data: dict, name: str) -> tuple[tuple[str, ...], tuple[float, ...], str]:
    spec = data.get(name)
    if not isinstance(spec, dict):
        raise ValueError(f"CES manifest is missing {name}")
    waypoints = tuple(str(value) for value in spec.get("waypoints", ()))
    durations = tuple(float(value) for value in spec.get("durations_s", ()))
    method = str(spec.get("interpolation", ""))
    if len(waypoints) < 2 or len(durations) != len(waypoints) - 1:
        raise ValueError(f"{name} requires one duration between each waypoint")
    if any(duration <= 0.0 for duration in durations):
        raise ValueError(f"{name} durations must be positive")
    if method not in _METHODS:
        raise ValueError(f"unsupported CES interpolation method {method!r}")
    return waypoints, durations, method


def load_baseline_trajectory() -> CesTrajectory:
    """Load and validate the fixed Smooth V1 pick/return/place contract."""
    data = json.loads(_MANIFEST.read_text(encoding="utf-8"))
    if data.get("schema_version") != 2 or data.get("name") != _NAME:
        raise ValueError("CES Baseline manifest schema/name mismatch")
    joint_order = tuple(str(name) for name in data.get("joint_order", ()))
    if joint_order != tuple(C.RIGHT_ARM_JOINTS):
        raise ValueError("CES manifest joint_order must match the runtime arm order")

    poses = data.get("poses")
    if not isinstance(poses, dict):
        raise ValueError("CES manifest poses must be an object")
    q_by_name = {
        str(name): tuple(float(value) for value in q)
        for name, q in poses.items()
    }
    invalid = [name for name, q in q_by_name.items() if len(q) != len(joint_order)]
    if invalid:
        raise ValueError(f"CES poses have invalid joint counts: {invalid}")

    forward, forward_times, forward_method = _path(data, "forward_path")
    return_path, return_times, return_method = _path(data, "return_path")
    place_path, place_times, place_method = _path(data, "place_path")
    q_ref = data.get("q_ref", {})
    q_ref_from, q_ref_to = str(q_ref.get("from", "")), str(q_ref.get("to", ""))
    if q_ref_from != forward[-1] or q_ref_to != return_path[0]:
        raise ValueError("CES q_ref must bridge forward pose 30 to live return pose 40")
    if return_path[-1] != place_path[0]:
        raise ValueError("CES place path must start at return pose 05")
    required = {*forward, q_ref_to, *return_path, *place_path}
    missing = sorted(required.difference(q_by_name))
    if missing:
        raise ValueError(f"CES Baseline is missing poses {missing}")

    return CesTrajectory(
        name=_NAME,
        q_by_name=q_by_name,
        joint_waypoints=forward,
        joint_segment_durations=forward_times,
        q_ref_from=q_ref_from,
        q_ref_to=q_ref_to,
        return_start=return_path[0],
        return_waypoints=return_path[1:],
        return_segment_durations=return_times,
        return_interpolation_method=return_method,
        place_start=place_path[0],
        place_waypoints=place_path[1:],
        place_segment_durations=place_times,
        place_interpolation_method=place_method,
        interpolation_method=forward_method,
    )
