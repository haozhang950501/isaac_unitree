# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""CES LoadingLine 产品抓取、持物行走和放置任务场景。

场景布局：

* 包装桌沿 Z 缩放到 CES 上料线底部高度，保留桌面的灰色
  ``container_h20`` 周转筐并隐藏纸箱。
* 机器人、CES 和桌子先组成紧凑簇，再绕簇的 XY 中心旋转 +180°，
  让默认视口从机器人身后看向设备正面，与 G1 跟随相机构图一致。
* ``object`` 直接包装 USD 内的 Product，使用 ``spawn=None``，不重复生成。
* 仓库墙体保持可见，不参与本次代码清理。
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
CES_SPAWN_Z = 0.9610
LOADING_LINE_BOTTOM_Z = 0.6173

_TABLE_LOCAL_HEIGHT = 0.9941
# LoadingLine 底面约 0.62 m；原始桌高 0.99 m 看着太高。
# +7 cm 时 05→15 会剐托盘；再降 2.5 cm 到 +2.0 cm。不要拉回 scale_z=1。
TABLE_TOP_EXTRA_Z = 0.020
TABLE_TOP_Z = LOADING_LINE_BOTTOM_Z + TABLE_TOP_EXTRA_Z
TABLE_SCALE_Z = TABLE_TOP_Z / _TABLE_LOCAL_HEIGHT  # ≈ 0.641
PRODUCT_DROP_Z = 0.32
# 包装桌自带灰筐 ``container_h20``，其世界高度随桌子使用同一 Z 缩放。
_PLACE_TRAY_LOCAL_HEIGHT = 0.16
PLACE_TRAY_HEIGHT = _PLACE_TRAY_LOCAL_HEIGHT * TABLE_SCALE_Z

# ---------------------------------------------------------------------------
# 先在旋转前坐标系摆放紧凑簇，再绕 XY 几何中心整体旋转 +180°。
#
# 旋转前机器人朝 +X 面向 CES，设备原点约 (-1.73, -1.33)，桌子在机器人右侧。
# 旋转后机器人朝 -X，默认视口越过机器人肩部观察设备正面。旋转前机器人
# 坐标只用于确定簇中心，真实机器人最终直接生成在下面计算的抓取站。
# ---------------------------------------------------------------------------
_PRODUCT_LOCAL_IDENTITY = (0.469837, 0.871558, -0.140226)
_PRODUCT_YAW_REL_CES = 90.0  # CES yaw=180° 时，产品世界 yaw=270°。

_CES_YAW_BEFORE = 180.0
_ROBOT_YAW_BEFORE = 0.0
_TABLE_YAW_BEFORE = 0.0
_CLUSTER_YAW = 180.0

_CES_BEFORE_XY = (-1.7302, -1.3284)
# 比旧站位 (-4.00, -0.80) 靠近 0.35 m，距离设备正面仍约 1.2 m。
_ROBOT_BEFORE_XY = (_CES_BEFORE_XY[0] - 1.92, _CES_BEFORE_XY[1] + 0.48)
# 桌子位于机器人右前方；横向距离由旧约 2.8 m 收紧到约 1.7 m。
_TABLE_BEFORE_XY = (_ROBOT_BEFORE_XY[0] + 0.50, _ROBOT_BEFORE_XY[1] - 1.70)

_CLUSTER_CX = (_ROBOT_BEFORE_XY[0] + _CES_BEFORE_XY[0] + _TABLE_BEFORE_XY[0]) / 3.0
_CLUSTER_CY = (_ROBOT_BEFORE_XY[1] + _CES_BEFORE_XY[1] + _TABLE_BEFORE_XY[1]) / 3.0


def _yaw_quat(deg: float) -> tuple[float, float, float, float]:
    """把 Z-up 偏航角转换为 ``wxyz`` 四元数。"""
    rad = math.radians(deg % 360.0)
    return (math.cos(rad / 2.0), 0.0, 0.0, math.sin(rad / 2.0))


def _rotate_xy(x: float, y: float, deg: float, cx: float, cy: float) -> tuple[float, float]:
    """把 XY 点绕指定中心逆时针旋转给定角度。"""
    rad = math.radians(deg)
    c, s = math.cos(rad), math.sin(rad)
    dx, dy = x - cx, y - cy
    return (cx + c * dx - s * dy, cy + s * dx + c * dy)


def _rz_xy(x: float, y: float, deg: float) -> tuple[float, float]:
    """把局部 XY 向量绕世界 Z 轴旋转，不附加平移。"""
    rad = math.radians(deg)
    c, s = math.cos(rad), math.sin(rad)
    return (c * x - s * y, s * x + c * y)


_CES_YAW = _CES_YAW_BEFORE + _CLUSTER_YAW
# ``_ROBOT_BEFORE_XY`` 只参与簇中心计算；机器人实际生成位置由 Product 推导。
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

# 机器人直接生成在抓取站，第一帧手臂就能触及上料托盘，不需要启动瞬移骨盆。
# 站位公式为 ``product - (x_b * 前向 + y_b * 左向)``，相同偏移由 CES 常量复用。
ROBOT_STAND_YAW = _ROBOT_YAW_BEFORE + _CLUSTER_YAW  # 180°，朝 -X 面向 CES 正面。
PICK_STAND_X_B = 0.30  # 产品位于机器人前方。
PICK_STAND_Y_B = -0.38  # 产品位于机器人右侧。


def _stand_xy(
    target_xy: tuple[float, float], yaw_deg: float, x_b: float, y_b: float
) -> tuple[float, float]:
    """按目标点在机体系中的前向/左向偏移反算骨盆世界 XY。"""
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
ROBOT_INIT_ROT = _yaw_quat(ROBOT_STAND_YAW)  # 朝世界 -X。

# +X/+Y nudges the table away from the LoadingLine tray (上料口).
# 2026-08-24：15 展开后世界 +Y 不够进灰筐。桌往 −Y 收 12 cm；放置站不跟走。
_TABLE_XY_NUDGE = (0.45, 0.23)  # 原值 (0.45, 0.35)，世界 Y 减少 0.12 m。
TABLE_SPAWN_POS = (_TABLE_XY[0] + _TABLE_XY_NUDGE[0], _TABLE_XY[1] + _TABLE_XY_NUDGE[1], 0.0)
TABLE_SPAWN_ROT = _yaw_quat(_TABLE_YAW_BEFORE + _CLUSTER_YAW)

# HeavyDuty 局部 Y ±0.381；container_h20 局部 Y 半宽 0.247。
# AABB 近沿对齐后再 +Y 收一点：整桌 AABB 比桌面可视边更靠机器人，否则筐沿会探出桌沿。
_TABLE_LOCAL_HALF_Y = 0.381
_TRAY_LOCAL_HALF_Y = 0.247
PLACE_TRAY_EDGE_INSET_Y = 0.03  # 世界 +Y，筐沿收回桌面
PLACE_TRAY_SHIFT_X = -0.15
PLACE_TRAY_SHIFT_Y = -_TABLE_LOCAL_HALF_Y + _TRAY_LOCAL_HALF_Y + PLACE_TRAY_EDGE_INSET_Y
PLACE_TRAY_CENTER_XY = (
    TABLE_SPAWN_POS[0] + PLACE_TRAY_SHIFT_X,
    TABLE_SPAWN_POS[1] + PLACE_TRAY_SHIFT_Y,
)

# 世界跟随相机位于机器人后方约 1.15 m，沿机器人 -X 朝向观察。
_WORLD_CAM_POS = (ROBOT_INIT_POS[0] + 1.15, ROBOT_INIT_POS[1] - 0.08, 2.30)
_WORLD_CAM_ROT = (0.5, -0.5, -0.5, 0.5)  # ROS 相机朝世界 -X。


def _hide_prim_tree(prim) -> None:
    """隐藏一个 prim，并关闭其整个子树中已应用的碰撞 API。"""
    from pxr import Usd, UsdGeom, UsdPhysics

    if not prim.IsValid():
        return
    UsdGeom.Imageable(prim).MakeInvisible()
    for p in Usd.PrimRange(prim):
        if p.HasAPI(UsdPhysics.CollisionAPI):
            attr = UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr()
            if attr:
                attr.Set(False)


def cleanup_packing_table(env, env_ids=None):
    """保留 HeavyDuty 桌和灰筐，隐藏桌面其余纸箱及其碰撞。"""
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
    """写入局部平移 op，使 prim 在世界系移动给定米制向量。"""
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
    """把 ``container_h20`` 放到桌面，并让近 Y 边保持设定内缩量。"""
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
    want_z0 = TABLE_TOP_Z + 0.002
    n_place = 0
    for i in range(env.num_envs):
        prefix = f"/World/envs/env_{i}/PackingTable/PackingTable_2"
        prim = stage.GetPrimAtPath(f"{prefix}/container_h20")
        table = stage.GetPrimAtPath(
            f"{prefix}/SM_CratePacking_Table_A1/SM_HeavyDutyPackingTable_C02_01"
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
        tray_half_y = 0.5 * (float(mx[1]) - float(mn[1]))
        want_y = PLACE_TRAY_CENTER_XY[1]
        if table.IsValid():
            table_rng = bbox.ComputeWorldBound(table).ComputeAlignedRange()
            if not table_rng.IsEmpty():
                table_min_y = float(table_rng.GetMin()[1])
                want_y = table_min_y + tray_half_y + PLACE_TRAY_EDGE_INSET_Y
        _nudge_prim_world(prim, (want_x - cx, want_y - cy, want_z0 - z0))
        n_place += 1
    if n_place:
        print(
            f"[ces_scene] gray tote on table xy=({want_x:.3f},{want_y:.3f}) "
            f"z0={want_z0:.3f} near-Y inset {PLACE_TRAY_EDGE_INSET_Y*1000:.0f}mm in {n_place} env(s)"
        )


def _create_physics_material(stage, path: str, static: float, dynamic: float):
    """在指定 USD 路径创建零恢复系数的 PhysX 材料并返回 Shade 包装。"""
    from pxr import UsdPhysics, UsdShade

    material_prim = stage.DefinePrim(path, "Material")
    material_api = UsdPhysics.MaterialAPI.Apply(material_prim)
    material_api.CreateStaticFrictionAttr(static)
    material_api.CreateDynamicFrictionAttr(dynamic)
    material_api.CreateRestitutionAttr(0.0)
    return UsdShade.Material(material_prim)


def _bind_physics_material(
    prim,
    shade_mat,
    *,
    include_path: str | None = None,
    exclude_path: str | None = None,
) -> int:
    """只按 physics purpose 绑定材料，不覆盖 CAD 视觉外观。

    默认 all-purpose 绑定会覆盖 OmniPBR，曾导致 ``Tray_Assembly_01``
    整体变白。可选路径过滤用于只选择右手碰撞体，或排除托盘内 Product。
    """
    from pxr import Usd, UsdPhysics, UsdShade

    n = 0
    for p in Usd.PrimRange(prim):
        path = str(p.GetPath())
        if include_path is not None and include_path not in path.lower():
            continue
        if exclude_path is not None and exclude_path in path:
            continue
        if p.HasAPI(UsdPhysics.CollisionAPI):
            api = UsdShade.MaterialBindingAPI.Apply(p)
            api.UnbindDirectBinding()
            api.Bind(
                shade_mat,
                bindingStrength=UsdShade.Tokens.strongerThanDescendants,
                materialPurpose="physics",
            )
            n += 1
    return n


def tune_product_grasp_physics(env, env_ids=None):
    """设置 Product 质量，并分离指垫、产品和托盘的抓取摩擦。

    摩擦合并模式为 max。如果 Product 本身使用 10 级高摩擦，它会粘在
    托盘凹槽中导致 Dex1 无法抬起；因此高摩擦只放在指垫，产品保持中等、
    托盘保持低摩擦。所有数值均属于已验证 Baseline。
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
        from pxr import UsdPhysics
    except ImportError:
        return

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return
    for i in range(env.num_envs):
        prefix = f"/World/envs/env_{i}"
        part_mat = _create_physics_material(
            stage,
            f"{prefix}/ProductGraspMaterial",
            part_mu_s,
            part_mu_d,
        )
        pad_mat = _create_physics_material(
            stage,
            f"{prefix}/Dex1PadGraspMaterial",
            pad_mu_s,
            pad_mu_d,
        )
        tray_mat = _create_physics_material(
            stage,
            f"{prefix}/TraySlipMaterial",
            tray_mu_s,
            tray_mu_d,
        )

        product = stage.GetPrimAtPath(
            f"{prefix}/CESMachine/Root/LoadingLine/Tray_Assembly_01/Product"
        )
        n_prod = _bind_physics_material(product, part_mat) if product.IsValid() else 0
        if product.IsValid():
            UsdPhysics.MassAPI.Apply(product).CreateMassAttr(mass_kg)

        tray = stage.GetPrimAtPath(
            f"{prefix}/CESMachine/Root/LoadingLine/Tray_Assembly_01"
        )
        n_tray = (
            _bind_physics_material(
                tray,
                tray_mat,
                exclude_path="/Product",
            )
            if tray.IsValid()
            else 0
        )

        robot = stage.GetPrimAtPath(f"{prefix}/Robot")
        n_pad = (
            _bind_physics_material(
                robot,
                pad_mat,
                include_path="right_hand",
            )
            if robot.IsValid()
            else 0
        )
        print(
            f"[ces_scene] friction pads={pad_mu_s}/{pad_mu_d} "
            f"part={part_mu_s}/{part_mu_d} tray={tray_mu_s}/{tray_mu_d} "
            f"product_cols={n_prod} tray_cols={n_tray} right_hand_cols={n_pad}"
        )


def ces_scene_startup(env, env_ids=None):
    """启动事件：清理纸箱、摆正灰筐，并写入抓取质量与摩擦。"""
    cleanup_packing_table(env, env_ids)
    place_gray_tray_on_table(env, env_ids)
    tune_product_grasp_physics(env, env_ids)


@configclass
class TableCESSceneCfgWH(InteractiveSceneCfg):
    """包含仓库、缩放包装桌、灰筐、CES 设备和 Product 的基础场景。"""

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

    # 越肩世界相机位于机器人后方，朝 -X 观察 CES 正面。
    world_camera = CameraBaseCfg.get_camera_config(
        prim_path="/World/PerspectiveCamera",
        pos_offset=_WORLD_CAM_POS,
        rot_offset=_WORLD_CAM_ROT,
    )
