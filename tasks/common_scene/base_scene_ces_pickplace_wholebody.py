# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""CES LoadingLine 产品抓取、持物行走和放置任务场景。

场景布局：
* 包装桌沿 Z 缩放到 CES 上料线底部高度。
* 机器人、CES、产品和桌子使用固定世界坐标。
* 包装桌启动后只保留 HeavyDuty 桌和灰筐，并设置产品抓取物理参数。
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
# 高度参数，单位均为米。
# ---------------------------------------------------------------------------
CES_SPAWN_Z = 0.9610  # CES 生成高度。
TABLE_TOP_Z = 0.6373  # 灰筐所在桌面高度。
TABLE_SCALE_Z = 0.6411  # 包装桌 Z 缩放。
PRODUCT_DROP_Z = 0.32  # 产品放置的 Z 坐标
PLACE_TRAY_HEIGHT = 0.1026  # 灰筐随包装桌缩放后的高度。

def _yaw_quat(deg: float) -> tuple[float, float, float, float]:
    """把 Z-up yaw 角转换为 ``wxyz`` 四元数。"""
    rad = math.radians(deg % 360.0)
    return (math.cos(rad / 2.0), 0.0, 0.0, math.sin(rad / 2.0))


# ---------------------------------------------------------------------------
# XY / yaw 直接参数，单位为米和角度。
# ---------------------------------------------------------------------------
CES_SPAWN_POS = (-3.9569, -1.8217, CES_SPAWN_Z)
CES_SPAWN_ROT = _yaw_quat(360.0)

PRODUCT_POS = (-3.4870, -0.9502, 0.8208)
PRODUCT_ROT = _yaw_quat(450.0)

ROBOT_STAND_YAW = 180.0
ROBOT_INIT_POS = (-3.1870, -1.3302, 0.8)
ROBOT_INIT_ROT = _yaw_quat(ROBOT_STAND_YAW)  # 朝世界 -X。

TABLE_SPAWN_POS = (-2.0869, -0.3717, 0.0)
TABLE_SPAWN_ROT = _yaw_quat(180.0)

PLACE_TRAY_CENTER_XY = (-2.2369, -0.4757)
PLACE_TRAY_BOTTOM_Z = 0.6393

# 世界跟随相机位于机器人后方约 1.15 m，沿机器人 -X 朝向观察。
_WORLD_CAM_POS = (ROBOT_INIT_POS[0] + 1.15, ROBOT_INIT_POS[1] - 0.08, 2.30)
_WORLD_CAM_ROT = (0.5, -0.5, -0.5, 0.5)  # ROS 相机朝世界 -X。

_PACKING_ROOT = "PackingTable/PackingTable_2"
_PACKING_GROUP = f"{_PACKING_ROOT}/SM_CratePacking_Table_A1"
_PACKING_TABLE = "SM_HeavyDutyPackingTable_C02_01"
_GRAY_TRAY_PATH = f"{_PACKING_ROOT}/container_h20"
_CES_TRAY_PATH = "CESMachine/Root/LoadingLine/Tray_Assembly_01"
_PRODUCT_PATH = f"{_CES_TRAY_PATH}/Product"
# 物理属性配置。
PRODUCT_MASS_KG = 0.25
PAD_FRICTION = (12.0, 10.0)
PRODUCT_FRICTION = (0.80, 0.60)
TRAY_FRICTION = (0.15, 0.10)


def _stage_or_none(action: str):
    try:
        import omni.usd
    except ImportError:
        print(f"[ces_scene] omni.usd unavailable — cannot {action}")
        return None
    return omni.usd.get_context().get_stage()


def _env_path(env_i: int, rel_path: str) -> str:
    return f"/World/envs/env_{env_i}/{rel_path}"


def _env_prim(stage, env_i: int, rel_path: str):
    return stage.GetPrimAtPath(_env_path(env_i, rel_path))


def _hide_prim_tree(prim) -> None:
    from pxr import Usd, UsdGeom, UsdPhysics

    if not prim.IsValid():
        return
    UsdGeom.Imageable(prim).MakeInvisible()
    for child in Usd.PrimRange(prim):
        if child.HasAPI(UsdPhysics.CollisionAPI):
            attr = UsdPhysics.CollisionAPI(child).GetCollisionEnabledAttr()
            if attr:
                attr.Set(False)


def cleanup_packing_table(env, env_ids=None):
    """保留 HeavyDuty 桌和灰筐，隐藏桌面其余纸箱及其碰撞。"""
    del env_ids
    stage = _stage_or_none("clean packing table")
    if stage is None:
        return

    n_clean = 0
    for i in range(env.num_envs):
        root = _env_prim(stage, i, "PackingTable")
        if not root.IsValid():
            continue
        group = _env_prim(stage, i, _PACKING_GROUP)
        children = group.GetChildren() if group.IsValid() else ()
        for child in children:
            if child.GetName() != _PACKING_TABLE:
                _hide_prim_tree(child)
        n_clean += 1
    if n_clean:
        print(f"[ces_scene] packing table cleaned (table + gray tote) in {n_clean} env(s)")


def place_gray_tray_on_table(env, env_ids=None):
    """把 ``container_h20`` 的中心 XY 和底部 Z 移到固定世界坐标。"""
    del env_ids
    stage = _stage_or_none("place gray tray")
    if stage is None:
        return

    try:
        from pxr import Gf, Usd, UsdGeom
    except ImportError:
        print("[ces_scene] pxr unavailable — cannot place gray tray")
        return

    bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    want_x, want_y = PLACE_TRAY_CENTER_XY
    n_place = 0
    for i in range(env.num_envs):
        tray = _env_prim(stage, i, _GRAY_TRAY_PATH)
        if not tray.IsValid():
            continue
        UsdGeom.Imageable(tray).MakeVisible()

        rng = bbox.ComputeWorldBound(tray).ComputeAlignedRange()
        if rng.IsEmpty():
            continue
        mn, mx = rng.GetMin(), rng.GetMax()
        d_world = Gf.Vec3d(
            want_x - 0.5 * (float(mn[0]) + float(mx[0])),
            want_y - 0.5 * (float(mn[1]) + float(mx[1])),
            PLACE_TRAY_BOTTOM_Z - float(mn[2]),
        )
        parent_xf = UsdGeom.Xformable(tray.GetParent()).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()
        )
        tray_xf = UsdGeom.Xformable(tray)
        op = next(
            (
                op
                for op in tray_xf.GetOrderedXformOps()
                if op.GetOpName() == "xformOp:translate:placeNudge"
            ),
            None,
        )
        if op is None:
            op = tray_xf.AddTranslateOp(opSuffix="placeNudge")
        op.Set(parent_xf.GetInverse().TransformDir(d_world))
        n_place += 1
    if n_place:
        print(
            f"[ces_scene] gray tote on table xy=({want_x:.3f},{want_y:.3f}) "
            f"z0={PLACE_TRAY_BOTTOM_Z:.3f} in {n_place} env(s)"
        )


def _create_physics_material(stage, path: str, static: float, dynamic: float):
    """创建零恢复系数的 PhysX 材料。"""
    from pxr import UsdPhysics, UsdShade

    material = stage.DefinePrim(path, "Material")
    api = UsdPhysics.MaterialAPI.Apply(material)
    api.CreateStaticFrictionAttr(static)
    api.CreateDynamicFrictionAttr(dynamic)
    api.CreateRestitutionAttr(0.0)
    return UsdShade.Material(material)


def _env_physics_material(stage, env_i: int, name: str, friction: tuple[float, float]):
    return _create_physics_material(stage, _env_path(env_i, name), *friction)


def _bind_physics_material(
    prim,
    shade_mat,
    *,
    include_path: str | None = None,
    exclude_path: str | None = None,
) -> int:
    """只绑定碰撞体的 physics 材料，不覆盖 CAD 外观。"""
    from pxr import Usd, UsdPhysics, UsdShade

    if not prim.IsValid():
        return 0

    n = 0
    for child in Usd.PrimRange(prim):
        path = str(child.GetPath())
        if include_path and include_path not in path.lower():
            continue
        if exclude_path and exclude_path in path:
            continue
        if not child.HasAPI(UsdPhysics.CollisionAPI):
            continue
        api = UsdShade.MaterialBindingAPI.Apply(child)
        api.UnbindDirectBinding()
        api.Bind(
            shade_mat,
            bindingStrength=UsdShade.Tokens.strongerThanDescendants,
            materialPurpose="physics",
        )
        n += 1
    return n


def _set_product_mass(env) -> None:
    try:
        import torch

        view = env.scene["object"].root_physx_view
        masses = view.get_masses()
        masses[:] = PRODUCT_MASS_KG
        ids = torch.arange(masses.shape[0], device=masses.device)
        view.set_masses(masses, ids)
        print(f"[ces_scene] Product mass set to {PRODUCT_MASS_KG:.3f} kg")
    except Exception as exc:
        print(f"[ces_scene] Product mass write skipped: {exc}")


def tune_product_grasp_physics(env, env_ids=None):
    """设置 Product 质量，并分离指垫、产品和托盘的抓取摩擦。"""
    del env_ids
    _set_product_mass(env)

    stage = _stage_or_none("tune product grasp physics")
    if stage is None:
        return

    try:
        from pxr import UsdPhysics
    except ImportError:
        print("[ces_scene] pxr unavailable — cannot tune product grasp physics")
        return

    for i in range(env.num_envs):
        product_mat = _env_physics_material(
            stage, i, "ProductGraspMaterial", PRODUCT_FRICTION
        )
        pad_mat = _env_physics_material(stage, i, "Dex1PadGraspMaterial", PAD_FRICTION)
        tray_mat = _env_physics_material(stage, i, "TraySlipMaterial", TRAY_FRICTION)

        product = _env_prim(stage, i, _PRODUCT_PATH)
        if product.IsValid():
            UsdPhysics.MassAPI.Apply(product).CreateMassAttr(PRODUCT_MASS_KG)

        n_prod = _bind_physics_material(product, product_mat)
        n_tray = _bind_physics_material(
            _env_prim(stage, i, _CES_TRAY_PATH), tray_mat, exclude_path="/Product"
        )
        n_pad = _bind_physics_material(
            _env_prim(stage, i, "Robot"), pad_mat, include_path="right_hand"
        )
        pad_s, pad_d = PAD_FRICTION
        prod_s, prod_d = PRODUCT_FRICTION
        tray_s, tray_d = TRAY_FRICTION
        print(
            f"[ces_scene] friction pads={pad_s}/{pad_d} "
            f"part={prod_s}/{prod_d} tray={tray_s}/{tray_d} "
            f"product_cols={n_prod} tray_cols={n_tray} right_hand_cols={n_pad}"
        )


def ces_scene_startup(env, env_ids=None):
    cleanup_packing_table(env, env_ids)
    place_gray_tray_on_table(env, env_ids)
    tune_product_grasp_physics(env, env_ids)


@configclass
class TableCESSceneCfgWH(InteractiveSceneCfg):
    """包含仓库、缩放包装桌、灰筐、CES 设备和 Product 的基础场景。"""

    # 房间墙壁配置
    room_walls = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Room",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.0, 0.0], rot=[1.0, 0.0, 0.0, 0.0]),
        spawn=UsdFileCfg(
            usd_path=f"{project_root}/assets/objects/small_warehouse/small_warehouse_digital_twin.usd",
        ),
    )
    # 包装桌配置
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

    # CES 设备配置
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

    # Product 配置
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
    # 光源配置
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    # 越肩世界相机配置位于机器人后方，朝 -X 观察 CES 正面。
    world_camera = CameraBaseCfg.get_camera_config(
        prim_path="/World/PerspectiveCamera",
        pos_offset=_WORLD_CAM_POS,
        rot_offset=_WORLD_CAM_ROT,
    )
