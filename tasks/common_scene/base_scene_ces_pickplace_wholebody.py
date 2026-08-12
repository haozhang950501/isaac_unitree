# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""Scene for the CES LoadingLine product pick-and-place task.

Layout:

* One bare packing table (``SM_HeavyDutyPackingTable`` only — crates / container
  hidden at startup).  Table-top Z is scaled to match the CES LoadingLine
  underside.
* ``ces_machine`` loads ``assets/bozhon/CESmachine_pickabble.usd`` with yaw
  **+180°** (LoadingLine faces south), no extra +X shift.
* Place table shifted **-3 m** along world -X.
* Robot stands further along **+Y**, facing **+X**.
* ``object`` wraps ``Root/LoadingLine/Tray_Assembly_01/Product`` (``spawn=None``).
* Warehouse ``Structure/walls`` are hidden at startup.
"""
import math
import os

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass

from tasks.common_config import CameraBaseCfg  # isort: skip

project_root = os.environ.get("PROJECT_ROOT")

# ---------------------------------------------------------------------------
# Heights (metres)
# ---------------------------------------------------------------------------
CES_SPAWN_Z = 0.9610
LOADING_LINE_BOTTOM_Z = 0.6173

_TABLE_LOCAL_HEIGHT = 0.9941
TABLE_SCALE_Z = LOADING_LINE_BOTTOM_Z / _TABLE_LOCAL_HEIGHT  # ≈ 0.621
TABLE_TOP_Z = LOADING_LINE_BOTTOM_Z

# ---------------------------------------------------------------------------
# CES pose — yaw = +180° (LoadingLine → -Y).
# ---------------------------------------------------------------------------
_CES_YAW_RAD = math.radians(180.0)
CES_SPAWN_ROT = (
    math.cos(_CES_YAW_RAD / 2.0),
    0.0,
    0.0,
    math.sin(_CES_YAW_RAD / 2.0),
)  # (0, 0, 0, 1)

_PRODUCT_LOCAL_AFTER_YAW = (-0.469837, -0.871558, -0.140226)
_CES_X_SHIFT = 0.0  # no +X bias (was 1.0 / 3.0)
_TABLE_X_SHIFT = -3.0  # metres along -X

PRODUCT_POS = (-2.2 + _CES_X_SHIFT, -2.2, 0.820774)  # (-2.2, -2.2, 0.821)
CES_SPAWN_POS = (
    PRODUCT_POS[0] - _PRODUCT_LOCAL_AFTER_YAW[0],
    PRODUCT_POS[1] - _PRODUCT_LOCAL_AFTER_YAW[1],
    CES_SPAWN_Z,
)  # ≈ (-1.7302, -1.3284, 0.961)

_PROD_YAW = math.radians(270.0)
PRODUCT_ROT = (
    math.cos(_PROD_YAW / 2.0),
    0.0,
    0.0,
    math.sin(_PROD_YAW / 2.0),
)

# Robot: further +Y, facing +X; stepped back along -X for more standoff.
ROBOT_INIT_POS = (-4.00, -0.80, 0.8)
ROBOT_INIT_ROT = (1.0, 0.0, 0.0, 0.0)  # face +X

# Table further -X.
TABLE_SPAWN_POS = (-0.70 + _TABLE_X_SHIFT, -3.60, 0.0)  # (-3.70, -3.60, 0)


def _hide_prim_tree(prim) -> None:
    """Make a prim (and collision APIs under it) invisible / non-colliding."""
    from pxr import Usd, UsdGeom, UsdPhysics

    if not prim.IsValid():
        return
    UsdGeom.Imageable(prim).MakeInvisible()
    for p in Usd.PrimRange(prim):
        if p.HasAPI(UsdPhysics.CollisionAPI):
            attr = UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr()
            if attr:
                attr.Set(False)


def hide_warehouse_walls(env, env_ids=None):
    """Hide warehouse Structure/walls (broken POLYFACE panels in the work cell)."""
    del env_ids
    try:
        import omni.usd
        from pxr import UsdGeom, UsdPhysics  # noqa: F401
    except ImportError:
        print("[ces_scene] omni.usd unavailable — cannot hide walls")
        return

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return

    n_hide = 0
    for i in range(env.num_envs):
        walls = stage.GetPrimAtPath(f"/World/envs/env_{i}/Room/Structure/walls")
        if walls.IsValid():
            _hide_prim_tree(walls)
            n_hide += 1
    if n_hide:
        print(f"[ces_scene] hid Structure/walls in {n_hide} env(s)")


def cleanup_packing_table(env, env_ids=None):
    """Keep only ``SM_HeavyDutyPackingTable``; hide crates / container on the USD."""
    del env_ids
    try:
        import omni.usd
    except ImportError:
        print("[ces_scene] omni.usd unavailable — cannot clean packing table")
        return

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return

    # Relative to each env's PackingTable root (UsdFileCfg defaultPrim = /Root).
    keep_name = "SM_HeavyDutyPackingTable_C02_01"
    n_clean = 0
    for i in range(env.num_envs):
        root = stage.GetPrimAtPath(f"/World/envs/env_{i}/PackingTable")
        if not root.IsValid():
            continue
        # Hide every direct clutter sibling under SM_CratePacking_Table_A1 and
        # the on-table container_h20; leave the HeavyDuty table prim alone.
        table_grp = stage.GetPrimAtPath(
            f"/World/envs/env_{i}/PackingTable/PackingTable_2/SM_CratePacking_Table_A1"
        )
        if table_grp.IsValid():
            for child in table_grp.GetChildren():
                if child.GetName() == keep_name:
                    continue
                _hide_prim_tree(child)
        container = stage.GetPrimAtPath(
            f"/World/envs/env_{i}/PackingTable/PackingTable_2/container_h20"
        )
        if container.IsValid():
            _hide_prim_tree(container)
        n_clean += 1
    if n_clean:
        print(f"[ces_scene] packing table cleaned (HeavyDuty only) in {n_clean} env(s)")


def ces_scene_startup(env, env_ids=None):
    """Startup hook: hide walls + strip packing-table clutter."""
    hide_warehouse_walls(env, env_ids)
    cleanup_packing_table(env, env_ids)


@configclass
class TableCESSceneCfgWH(InteractiveSceneCfg):
    """Warehouse room, one bare packing table, CES machine and pickable Product."""

    room_walls = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Room",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.0, 0.0], rot=[1.0, 0.0, 0.0, 0.0]),
        spawn=UsdFileCfg(
            usd_path=f"{project_root}/assets/objects/small_warehouse/small_warehouse_digital_twin.usd",
        ),
    )

    # Bare HeavyDuty table; crates/container removed at startup.  Z-scale brings
    # the tabletop down to LoadingLine underside height.
    packing_table = AssetBaseCfg(
        prim_path="/World/envs/env_.*/PackingTable",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=list(TABLE_SPAWN_POS),
            rot=[1.0, 0.0, 0.0, 0.0],
        ),
        spawn=UsdFileCfg(
            usd_path=f"{project_root}/assets/objects/PackingTable/PackingTable.usd",
            scale=(1.0, 1.0, TABLE_SCALE_Z),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
    )

    # Full CES + LoadingLine. Do NOT apply rigid_props on the root (nested Product RB).
    ces_machine = AssetBaseCfg(
        prim_path="/World/envs/env_.*/CESMachine",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=list(CES_SPAWN_POS),
            rot=list(CES_SPAWN_ROT),
        ),
        spawn=UsdFileCfg(
            usd_path=f"{project_root}/assets/bozhon/CESmachine_pickabble.usd",
        ),
    )

    object = RigidObjectCfg(
        prim_path="/World/envs/env_.*/CESMachine/Root/LoadingLine/Tray_Assembly_01/Product",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=PRODUCT_POS,
            rot=PRODUCT_ROT,
            lin_vel=(0.0, 0.0, 0.0),
            ang_vel=(0.0, 0.0, 0.0),
        ),
        spawn=None,
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    # Overview of shifted CES (+X) / table (-X) / robot facing +X.
    world_camera = CameraBaseCfg.get_camera_config(
        prim_path="/World/PerspectiveCamera",
        pos_offset=(-2.5, -4.5, 2.4),
        rot_offset=(-0.2706, 0.6533, 0.6533, -0.2706),
    )
