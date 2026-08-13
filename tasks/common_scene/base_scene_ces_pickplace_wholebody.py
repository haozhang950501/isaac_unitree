# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""Scene for the CES LoadingLine product pick-and-place task.

Layout:

* One bare packing table (``SM_HeavyDutyPackingTable`` only — crates / container
  hidden at startup).  Table-top Z is scaled to match the CES LoadingLine
  underside.
* Compact robot / CES / table cluster, then yaw **+180°** about the cluster
  XY centroid so the default viewport sees the robot from behind, looking at
  the machine front (same composition as the G1 trailing camera).
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
# Compact cluster in the pre-rotation frame, then +180° about the XY centroid.
#
# Before rotation (robot faces +X toward CES):
#   CES origin ≈ (-1.73, -1.33); robot ~1.9 m west; table on robot's right.
# After +180° the robot faces -X and the default +X/+Y viewport looks over
# its shoulder at the machine front.
# ---------------------------------------------------------------------------
_PRODUCT_LOCAL_IDENTITY = (0.469837, 0.871558, -0.140226)
_PRODUCT_YAW_REL_CES = 90.0  # world yaw was 270° when CES yaw was 180°

_CES_YAW_BEFORE = 180.0
_ROBOT_YAW_BEFORE = 0.0
_TABLE_YAW_BEFORE = 0.0
_CLUSTER_YAW = 180.0

_CES_BEFORE_XY = (-1.7302, -1.3284)
# 0.35 m closer than the old (-4.00, -0.80) standoff; still ~1.2 m to the face.
_ROBOT_BEFORE_XY = (_CES_BEFORE_XY[0] - 1.92, _CES_BEFORE_XY[1] + 0.48)
# Table on the robot's right-forward (was ~2.8 m to the right; now ~1.7 m).
_TABLE_BEFORE_XY = (_ROBOT_BEFORE_XY[0] + 0.50, _ROBOT_BEFORE_XY[1] - 1.70)

_CLUSTER_CX = (_ROBOT_BEFORE_XY[0] + _CES_BEFORE_XY[0] + _TABLE_BEFORE_XY[0]) / 3.0
_CLUSTER_CY = (_ROBOT_BEFORE_XY[1] + _CES_BEFORE_XY[1] + _TABLE_BEFORE_XY[1]) / 3.0


def _yaw_quat(deg: float) -> tuple[float, float, float, float]:
    """Z-up yaw quaternion (w, x, y, z)."""
    rad = math.radians(deg % 360.0)
    return (math.cos(rad / 2.0), 0.0, 0.0, math.sin(rad / 2.0))


def _rotate_xy(x: float, y: float, deg: float, cx: float, cy: float) -> tuple[float, float]:
    """Rotate (x, y) about (cx, cy) by yaw ``deg`` (CCW)."""
    rad = math.radians(deg)
    c, s = math.cos(rad), math.sin(rad)
    dx, dy = x - cx, y - cy
    return (cx + c * dx - s * dy, cy + s * dx + c * dy)


def _rz_xy(x: float, y: float, deg: float) -> tuple[float, float]:
    rad = math.radians(deg)
    c, s = math.cos(rad), math.sin(rad)
    return (c * x - s * y, s * x + c * y)


_CES_YAW = _CES_YAW_BEFORE + _CLUSTER_YAW
_ROBOT_XY = _rotate_xy(*_ROBOT_BEFORE_XY, _CLUSTER_YAW, _CLUSTER_CX, _CLUSTER_CY)
_CES_XY = _rotate_xy(*_CES_BEFORE_XY, _CLUSTER_YAW, _CLUSTER_CX, _CLUSTER_CY)
_TABLE_XY = _rotate_xy(*_TABLE_BEFORE_XY, _CLUSTER_YAW, _CLUSTER_CX, _CLUSTER_CY)
_PROD_LOCAL_XY = _rz_xy(_PRODUCT_LOCAL_IDENTITY[0], _PRODUCT_LOCAL_IDENTITY[1], _CES_YAW)

CES_SPAWN_POS = (_CES_XY[0], _CES_XY[1], CES_SPAWN_Z)
CES_SPAWN_ROT = _yaw_quat(_CES_YAW)

PRODUCT_POS = (
    CES_SPAWN_POS[0] + _PROD_LOCAL_XY[0],
    CES_SPAWN_POS[1] + _PROD_LOCAL_XY[1],
    CES_SPAWN_POS[2] + _PRODUCT_LOCAL_IDENTITY[2],
)
PRODUCT_ROT = _yaw_quat(_CES_YAW + _PRODUCT_YAW_REL_CES)

ROBOT_INIT_POS = (_ROBOT_XY[0], _ROBOT_XY[1], 0.8)
ROBOT_INIT_ROT = _yaw_quat(_ROBOT_YAW_BEFORE + _CLUSTER_YAW)  # face -X

# +X/+Y nudges the table away from the LoadingLine tray (上料口).
_TABLE_XY_NUDGE = (0.45, 0.35)
TABLE_SPAWN_POS = (_TABLE_XY[0] + _TABLE_XY_NUDGE[0], _TABLE_XY[1] + _TABLE_XY_NUDGE[1], 0.0)
TABLE_SPAWN_ROT = _yaw_quat(_TABLE_YAW_BEFORE + _CLUSTER_YAW)

# Trailing world camera: ~1.15 m behind the robot, looking along its -X heading.
_WORLD_CAM_POS = (ROBOT_INIT_POS[0] + 1.15, ROBOT_INIT_POS[1] - 0.08, 2.30)
_WORLD_CAM_ROT = (0.5, -0.5, -0.5, 0.5)  # ROS camera looking world -X


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


def _bind_physics_material(prim, shade_mat) -> int:
    from pxr import Usd, UsdPhysics, UsdShade

    n = 0
    for p in Usd.PrimRange(prim):
        if p.HasAPI(UsdPhysics.CollisionAPI):
            UsdShade.MaterialBindingAPI.Apply(p).Bind(
                shade_mat, UsdShade.Tokens.strongerThanDescendants
            )
            n += 1
    return n


def tune_product_grasp_physics(env, env_ids=None):
    """Split friction: sticky Dex1 pads, slippery tray, moderate Product.

    Combine-mode is max.  If Product itself is mu=10 it glues into the tray
    and Dex1 cannot lift it.  Pads carry the high mu; Product/tray stay low.
    """
    del env_ids
    mass_kg = 0.25
    pad_mu_s, pad_mu_d = 12.0, 10.0
    part_mu_s, part_mu_d = 0.80, 0.60
    tray_mu_s, tray_mu_d = 0.15, 0.10
    try:
        obj = env.scene["object"]
        view = obj.root_physx_view
        masses = view.get_masses()
        masses[:] = mass_kg
        n = masses.shape[0]
        import torch

        view.set_masses(masses, torch.arange(n, device=masses.device))
        print(f"[ces_scene] Product mass set to {mass_kg:.3f} kg")
    except Exception as exc:
        print(f"[ces_scene] Product mass write skipped: {exc}")

    try:
        import omni.usd
        from pxr import UsdPhysics, UsdShade
    except ImportError:
        return

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return
    for i in range(env.num_envs):
        mat_path = f"/World/envs/env_{i}/ProductGraspMaterial"
        mat_prim = stage.DefinePrim(mat_path, "Material")
        mat = UsdPhysics.MaterialAPI.Apply(mat_prim)
        mat.CreateStaticFrictionAttr(part_mu_s)
        mat.CreateDynamicFrictionAttr(part_mu_d)
        mat.CreateRestitutionAttr(0.0)
        part_mat = UsdShade.Material(mat_prim)

        pad_path = f"/World/envs/env_{i}/Dex1PadGraspMaterial"
        pad_prim = stage.DefinePrim(pad_path, "Material")
        pad_api = UsdPhysics.MaterialAPI.Apply(pad_prim)
        pad_api.CreateStaticFrictionAttr(pad_mu_s)
        pad_api.CreateDynamicFrictionAttr(pad_mu_d)
        pad_api.CreateRestitutionAttr(0.0)
        pad_mat = UsdShade.Material(pad_prim)

        tray_path = f"/World/envs/env_{i}/TraySlipMaterial"
        tray_prim = stage.DefinePrim(tray_path, "Material")
        tray_api = UsdPhysics.MaterialAPI.Apply(tray_prim)
        tray_api.CreateStaticFrictionAttr(tray_mu_s)
        tray_api.CreateDynamicFrictionAttr(tray_mu_d)
        tray_api.CreateRestitutionAttr(0.0)
        tray_mat = UsdShade.Material(tray_prim)

        product = stage.GetPrimAtPath(
            f"/World/envs/env_{i}/CESMachine/Root/LoadingLine/Tray_Assembly_01/Product"
        )
        n_prod = _bind_physics_material(product, part_mat) if product.IsValid() else 0
        if product.IsValid():
            UsdPhysics.MassAPI.Apply(product).CreateMassAttr(mass_kg)

        n_tray = 0
        tray = stage.GetPrimAtPath(
            f"/World/envs/env_{i}/CESMachine/Root/LoadingLine/Tray_Assembly_01"
        )
        if tray.IsValid():
            from pxr import Usd, UsdPhysics as _Phys

            for p in Usd.PrimRange(tray):
                if "/Product" in str(p.GetPath()):
                    continue
                if p.HasAPI(_Phys.CollisionAPI):
                    UsdShade.MaterialBindingAPI.Apply(p).Bind(
                        tray_mat, UsdShade.Tokens.strongerThanDescendants
                    )
                    n_tray += 1

        n_pad = 0
        robot = stage.GetPrimAtPath(f"/World/envs/env_{i}/Robot")
        if robot.IsValid():
            from pxr import Usd, UsdPhysics as _Phys

            for p in Usd.PrimRange(robot):
                if "right_hand" not in str(p.GetPath()).lower():
                    continue
                if p.HasAPI(_Phys.CollisionAPI):
                    UsdShade.MaterialBindingAPI.Apply(p).Bind(
                        pad_mat, UsdShade.Tokens.strongerThanDescendants
                    )
                    n_pad += 1
        print(
            f"[ces_scene] friction pads={pad_mu_s}/{pad_mu_d} "
            f"part={part_mu_s}/{part_mu_d} tray={tray_mu_s}/{tray_mu_d} "
            f"product_cols={n_prod} tray_cols={n_tray} right_hand_cols={n_pad}"
        )


def ces_scene_startup(env, env_ids=None):
    """Startup hook: hide walls, strip table clutter, make Product graspable."""
    hide_warehouse_walls(env, env_ids)
    cleanup_packing_table(env, env_ids)
    tune_product_grasp_physics(env, env_ids)


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

    packing_table = AssetBaseCfg(
        prim_path="/World/envs/env_.*/PackingTable",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=list(TABLE_SPAWN_POS),
            rot=list(TABLE_SPAWN_ROT),
        ),
        spawn=UsdFileCfg(
            usd_path=f"{project_root}/assets/objects/PackingTable/PackingTable.usd",
            scale=(1.0, 1.0, TABLE_SCALE_Z),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
    )

  
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

    # Over-the-shoulder: behind the robot, looking -X at the CES front.
    world_camera = CameraBaseCfg.get_camera_config(
        prim_path="/World/PerspectiveCamera",
        pos_offset=_WORLD_CAM_POS,
        rot_offset=_WORLD_CAM_ROT,
    )
