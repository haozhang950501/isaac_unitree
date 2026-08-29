# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""Load the single supported CES Smooth V1 trajectory by joint name."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from action_provider.ces_grasp import constants as C

_BASELINE_NAME = "ces_pick_smooth_v1"
_BASELINE_ROOT = Path(__file__).resolve().parent / "poses" / _BASELINE_NAME
_JOINT_SPACE = "joint_space"
_Q_REF_CONTROL = "cartesian_vertical_ik_with_dynamic_q_ref"
_INTERPOLATION_METHODS = {"segment_smoothstep", "monotone_cubic_hermite"}


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


def _remap_q(
    joint_order: list[str], q: list[float], target_names: list[str]
) -> tuple[float, ...]:
    if len(joint_order) != len(q):
        raise ValueError(
            f"pose joint_order has {len(joint_order)} names but q has {len(q)} values"
        )
    by_name = dict(zip(joint_order, q))
    missing = [name for name in target_names if name not in by_name]
    if missing:
        raise ValueError(f"pose is missing joints {missing}; match by name, not index")
    return tuple(float(by_name[name]) for name in target_names)


def _load_pose_q(path: Path, target_names: list[str]) -> tuple[str, tuple[float, ...]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return str(data["name"]), _remap_q(
        list(data["joint_order"]), list(data["q"]), target_names
    )


def _interpolation(value, *, default: str) -> str:
    method = (
        str(value.get("method", default)) if isinstance(value, dict) else str(value or default)
    )
    if method not in _INTERPOLATION_METHODS:
        raise ValueError(f"unsupported CES interpolation method {method!r}")
    return method


def _commanded_path(
    spec: dict,
    *,
    path_name: str,
    default_interpolation: str,
    required_control: str | None = None,
) -> tuple[str, tuple[str, ...], tuple[float, ...], str]:
    logical = tuple(str(value) for value in spec.get("logical_waypoints", []))
    commanded = tuple(str(value) for value in spec.get("commanded_waypoints", []))
    if len(logical) < 2 or logical[1:] != commanded:
        raise ValueError(f"{path_name} must be logical_start + commanded_waypoints")

    segments = list(spec.get("segments", []))
    if len(segments) != len(commanded):
        raise ValueError(
            f"{path_name} has {len(segments)} segments for {len(commanded)} waypoints"
        )
    durations: list[float] = []
    expected_src = logical[0]
    for index, segment in enumerate(segments):
        src, dst = str(segment["from"]), str(segment["to"])
        duration = float(segment["duration_s"])
        if src != expected_src or dst != commanded[index]:
            raise ValueError(
                f"{path_name} segment {src}->{dst} does not match "
                f"{expected_src}->{commanded[index]}"
            )
        if required_control and str(segment.get("runtime_control")) != required_control:
            raise ValueError(f"{path_name} segment {src}->{dst} must use {required_control}")
        if duration <= 0.0:
            raise ValueError(f"{path_name} segment {src}->{dst} duration must be positive")
        durations.append(duration)
        expected_src = dst
    method = _interpolation(
        spec.get("interpolation", default_interpolation),
        default=default_interpolation,
    )
    return logical[0], commanded, tuple(durations), method


def load_baseline_trajectory() -> CesTrajectory:
    """Load and validate the fixed Smooth V1 pick/return/place contract."""
    manifest_path = _BASELINE_ROOT / "trajectory_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"CES Baseline manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if str(manifest.get("name")) != _BASELINE_NAME:
        raise ValueError(f"CES manifest must be {_BASELINE_NAME!r}")

    interpolation = _interpolation(
        manifest.get("interpolation"), default="monotone_cubic_hermite"
    )
    target_names = list(C.RIGHT_ARM_JOINTS)
    q_by_name: dict[str, tuple[float, ...]] = {}
    for pose_file in manifest.get("pose_files", []):
        pose_name, q = _load_pose_q(_BASELINE_ROOT / pose_file, target_names)
        q_by_name[pose_name] = q

    joint_waypoints: list[str] = []
    joint_durations: list[float] = []
    q_ref_from = ""
    q_ref_to = ""
    for segment in manifest.get("segments", []):
        src, dst = str(segment["from"]), str(segment["to"])
        control = str(segment.get("future_project_control"))
        if control == _JOINT_SPACE:
            if not joint_waypoints:
                joint_waypoints.append(src)
            if joint_waypoints[-1] != src:
                raise ValueError(f"forward segment {src}->{dst} is not continuous")
            joint_waypoints.append(dst)
            joint_durations.append(float(segment["duration_s"]))
        elif control == _Q_REF_CONTROL:
            q_ref_from, q_ref_to = src, dst

    handoff = str(manifest.get("future_project_handoff", {}).get("joint_space_through", ""))
    if not joint_waypoints or joint_waypoints[-1] != handoff:
        raise ValueError(f"forward path must end at handoff {handoff!r}")
    if not q_ref_from or not q_ref_to or q_ref_from != handoff:
        raise ValueError("forward path must define the 30->40 dynamic q_ref segment")

    return_spec = manifest.get("return_path")
    place_spec = manifest.get("place_path")
    if not isinstance(return_spec, dict) or not isinstance(place_spec, dict):
        raise ValueError("CES Baseline requires explicit return_path and place_path")
    return_start, return_waypoints, return_durations, return_method = _commanded_path(
        return_spec,
        path_name="return_path",
        default_interpolation=interpolation,
    )
    place_start, place_waypoints, place_durations, place_method = _commanded_path(
        place_spec,
        path_name="place_path",
        default_interpolation="segment_smoothstep",
        required_control=_JOINT_SPACE,
    )
    if return_start != q_ref_to:
        raise ValueError("return_path must start from the live 40 phase")
    if not return_waypoints or place_start != return_waypoints[-1]:
        raise ValueError("place_path must start from the return carry endpoint")

    required = {
        *joint_waypoints,
        q_ref_from,
        q_ref_to,
        return_start,
        *return_waypoints,
        place_start,
        *place_waypoints,
    }
    missing = sorted(required.difference(q_by_name))
    if missing:
        raise ValueError(f"CES Baseline is missing poses {missing}")

    return CesTrajectory(
        name=_BASELINE_NAME,
        q_by_name=q_by_name,
        joint_waypoints=tuple(joint_waypoints),
        joint_segment_durations=tuple(joint_durations),
        q_ref_from=q_ref_from,
        q_ref_to=q_ref_to,
        return_start=return_start,
        return_waypoints=return_waypoints,
        return_segment_durations=return_durations,
        return_interpolation_method=return_method,
        place_start=place_start,
        place_waypoints=place_waypoints,
        place_segment_durations=place_durations,
        place_interpolation_method=place_method,
        interpolation_method=interpolation,
    )
