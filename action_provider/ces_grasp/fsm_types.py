# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""Shared command and phase types for the CES Baseline state machine."""
from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Any

RootPose = tuple[
    tuple[float, float, float],
    tuple[float, float, float, float],
]
WalkCommand = tuple[float, float, float, float]


class CesPickPlacePhase(enum.Enum):
    SETTLE = "settle"
    GOTO_PICK = "goto_pick"
    UNFOLD = "unfold"
    DESCEND = "descend"
    GRASP = "grasp"
    LIFT = "lift"
    RETURN_HOME = "return_home"
    CARRY = "carry"
    GOTO_PLACE = "goto_place"
    PLACE_HOLD = "place_hold"
    PLACE_APPROACH = "place_approach"
    RELEASE = "release"
    RETRACT = "retract"
    DONE = "done"
    FAILED = "failed"


# Keep the nominal success path as data as well as documentation.
BASELINE_PHASE_ORDER = (
    CesPickPlacePhase.SETTLE,
    CesPickPlacePhase.GOTO_PICK,
    CesPickPlacePhase.UNFOLD,
    CesPickPlacePhase.DESCEND,
    CesPickPlacePhase.GRASP,
    CesPickPlacePhase.LIFT,
    CesPickPlacePhase.RETURN_HOME,
    CesPickPlacePhase.CARRY,
    CesPickPlacePhase.GOTO_PLACE,
    CesPickPlacePhase.PLACE_HOLD,
    CesPickPlacePhase.PLACE_APPROACH,
    CesPickPlacePhase.RELEASE,
    CesPickPlacePhase.RETRACT,
    CesPickPlacePhase.DONE,
)


@dataclass
class CesCommand:
    gripper: float
    walk: WalkCommand | None = None
    root_pin: RootPose | None = None
    arm_q: Any | None = None
    tcp_pos: Any | None = None
    tcp_quat: Any | None = None
    arm_q_ref: Any | None = None
