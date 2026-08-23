# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""Scene for the CES LoadingLine product pick-and-place task.

Layout:

* Packing table Z-scaled to the CES LoadingLine underside, with the gray
  ``container_h20`` tote kept on the top (crates still hidden).
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
# Authored gray tote on the packing table (container_h20).  World height
# follows the same Z scale as the table.
_PLACE_TRAY_LOCAL_HEIGHT = 0.16
PLACE_TRAY_HEIGHT = _PLACE_TRAY_LOCAL_HEIGHT * TABLE_SCALE_Z

# ---------------------------------------------------------------------------
# Compact cluster in the pre-rotation frame, then +180° about the XY centroid.
#
# Before rotation (robot faces +X toward CES):
#   CES origin ≈ (-1.73, -1.33); robot ~1.9 m west; table on robot's right.
# After +180° the robot faces -X and the default +X/+Y viewport looks over
# its shoulder at the machine front.  The pre-rotation robot spot only fixes
# the cluster centroid — the robot spawns on the pick stand (see below).
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
# ``_ROBOT_BEFORE_XY`` only sets the cluster centroid now; the robot itself
# spawns on the pick stand computed from the Product below.
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

# The robot spawns **on** the pick stand: the arm reaches the LoadingLine tray
# from frame one, so nothing has to teleport the pelvis at startup.
# ``stand = product - (x_b * forward + y_b * left)``; the same offsets are
# re-exported by action_provider/ces_grasp/constants.py.
ROBOT_STAND_YAW = _ROBOT_YAW_BEFORE + _CLUSTER_YAW  # 180 deg, faces -X at the CES front
PICK_STAND_X_B = 0.30  # product ahead of the robot
PICK_STAND_Y_B = -0.38  # product on the robot's right


def _stand_xy(
    target_xy: tuple[float, float], yaw_deg: float, x_b: float, y_b: float
) -> tuple[float, float]:
    rad = math.radians(yaw_deg)
    fwd = (math.cos(rad), math.sin(rad))
    left = (-math.sin(rad), math.cos(rad))
    return (
        target_xy[0] - (x_b * fwd[0] + y_b * left[0]),
        target_xy[1] - (x_b * fwd[1] + y_b * left[1]),
    )


PICK_STAND_XY = _stand_xy(
    (PRODUCT_POS[0], PRODUCT_POS[1]), ROBOT_STAND_YAW, PICK_STAND_X_B, PICK_STAND_Y_B
)

ROBOT_INIT_POS = (PICK_STAND_XY[0], PICK_STAND_XY[1], 0.8)
ROBOT_INIT_ROT = _yaw_quat(ROBOT_STAND_YAW)  # face -X

# +X/+Y nudges the table away from the LoadingLine tray (上料口).
_TABLE_XY_NUDGE = (0.45, 0.35)
TABLE_SPAWN_POS = (_TABLE_XY[0] + _TABLE_XY_NUDGE[0], _TABLE_XY[1] + _TABLE_XY_NUDGE[1], 0.0)
TABLE_SPAWN_ROT = _yaw_quat(_TABLE_YAW_BEFORE + _CLUSTER_YAW)

# 灰色托盘相对桌心 y-0.06，再往世界 -X 挪一点。放置站不跟这块托盘走。
PLACE_TRAY_SHIFT_X = -0.15
PLACE_TRAY_CENTER_XY = (
    TABLE_SPAWN_POS[0] + PLACE_TRAY_SHIFT_X,
    TABLE_SPAWN_POS[1] - 0.06,
)

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
    """Keep the HeavyDuty table and gray tote; hide cardboard crates."""
    del env_ids
    try:
        import omni.usd
    except ImportError:
        print("[ces_scene] omni.usd unavailable — cannot clean packing table")
        return

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return

    keep_name = "SM_HeavyDutyPackingTable_C02_01"
    n_clean = 0
    for i in range(env.num_envs):
        root = stage.GetPrimAtPath(f"/World/envs/env_{i}/PackingTable")
        if not root.IsValid():
            continue
        table_grp = stage.GetPrimAtPath(
            f"/World/envs/env_{i}/PackingTable/PackingTable_2/SM_CratePacking_Table_A1"
        )
        if table_grp.IsValid():
            for child in table_grp.GetChildren():
                if child.GetName() == keep_name:
                    continue
                _hide_prim_tree(child)
        n_clean += 1
    if n_clean:
        print(f"[ces_scene] packing table cleaned (table + gray tote) in {n_clean} env(s)")


def _nudge_prim_world(prim, d_world) -> None:
    """Add a local translate so the prim moves by ``d_world`` in world metres."""
    from pxr import Gf, Usd, UsdGeom

    parent = prim.GetParent()
    if parent.IsValid():
        parent_xf = UsdGeom.Xformable(parent).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        d_local = parent_xf.GetInverse().TransformDir(Gf.Vec3d(*d_world))
    else:
        d_local = Gf.Vec3d(*d_world)
    xf = UsdGeom.Xformable(prim)
    op = None
    for existing in xf.GetOrderedXformOps():
        if existing.GetOpName() == "xformOp:translate:placeNudge":
            op = existing
            break
    if op is None:
        op = xf.AddTranslateOp(opSuffix="placeNudge")
    op.Set(d_local)


def place_gray_tray_on_table(env, env_ids=None):
    """Sit ``container_h20`` on the tabletop, centered toward the place target."""
    del env_ids
    try:
        import omni.usd
        from pxr import Usd, UsdGeom
    except ImportError:
        print("[ces_scene] omni.usd unavailable — cannot place gray tray")
        return

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return
    bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    want_x = PLACE_TRAY_CENTER_XY[0]
    want_y = PLACE_TRAY_CENTER_XY[1]
    want_z0 = TABLE_TOP_Z + 0.002
    n_place = 0
    for i in range(env.num_envs):
        prim = stage.GetPrimAtPath(
            f"/World/envs/env_{i}/PackingTable/PackingTable_2/container_h20"
        )
        if not prim.IsValid():
            continue
        UsdGeom.Imageable(prim).MakeVisible()
        rng = bbox.ComputeWorldBound(prim).ComputeAlignedRange()
        if rng.IsEmpty():
            continue
        mn, mx = rng.GetMin(), rng.GetMax()
        cx = 0.5 * (float(mn[0]) + float(mx[0]))
        cy = 0.5 * (float(mn[1]) + float(mx[1]))
        z0 = float(mn[2])
        _nudge_prim_world(prim, (want_x - cx, want_y - cy, want_z0 - z0))
        n_place += 1
    if n_place:
        print(
            f"[ces_scene] gray tote on table xy=({want_x:.3f},{want_y:.3f}) "
            f"z0={want_z0:.3f} in {n_place} env(s)"
        )


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
    """Startup hook: hide walls, strip crates, seat the gray tote, tune grasp."""
    hide_warehouse_walls(env, env_ids)
    cleanup_packing_table(env, env_ids)
    place_gray_tray_on_table(env, env_ids)
    tune_product_grasp_physics(env, env_ids)


@configclass
class TableCESSceneCfgWH(InteractiveSceneCfg):
    """Warehouse room, scaled packing table with gray tote, CES, Product."""

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
