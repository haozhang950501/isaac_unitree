# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""Fix product spawn pose / physics on the CES USD and write a new file.

Issues addressed:
1. product starts ~0.69 m below Station_02/Asm_070000 seat → falls to floor
2. product is nested under CESmachine (bad if machine also has RigidBodyAPI)
3. metersPerUnit=0.01 while geometry is metre-scale
4. PhysicsScene gravityDirection/magnitude unset / invalid
5. CESmachine missing kinematic RigidBodyAPI (only MassAPI)

Usage::

    /home/zh/isaacsim/python.sh tools/fix_ces_product_pose.py
    /home/zh/isaacsim/python.sh tools/fix_ces_product_pose.py \\
        --src assets/bozhon/CESmachine_pickabble.usd \\
        --dst assets/bozhon/CESmachine_pickabble.usd
"""
from __future__ import annotations

import argparse
import functools
import os
import shutil
import sys

print = functools.partial(print, flush=True)  # noqa: A001

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_BOZHON = os.path.join(_PROJECT_ROOT, "assets", "bozhon")
_DEFAULT_CES = os.path.join(_BOZHON, "CESmachine_pickabble.usd")

_simulation_app = None

PROD_PATH = "/Root/Root/CESmachine/product"
PROD_DST = "/Root/Root/product"
SEAT_PATH = "/Root/Root/CESmachine/CESmachine/Station_02/Asm_070000/N_070001"
CES_PATH = "/Root/Root/CESmachine"
SCENE_PATH = "/Root/Root/PhysicsScene"


def _boot_app(headless: bool = True):
    global _simulation_app
    try:
        from pxr import Usd  # noqa: F401

        return
    except ImportError:
        pass
    try:
        from isaaclab.app import AppLauncher

        _simulation_app = AppLauncher({"headless": headless, "enable_cameras": False}).app
        return
    except Exception as exc:  # noqa: BLE001
        print(f"[fix] AppLauncher unavailable ({exc}); trying SimulationApp")
    from isaacsim import SimulationApp

    _simulation_app = SimulationApp({"headless": headless})


def _close_app():
    global _simulation_app
    if _simulation_app is not None:
        _simulation_app.close()
        _simulation_app = None


def _parse_args():
    p = argparse.ArgumentParser(description="Fix CES product pose and physics")
    p.add_argument("--src", default=_DEFAULT_CES)
    p.add_argument("--dst", default=_DEFAULT_CES)
    p.add_argument("--clearance", type=float, default=0.001, help="gap above seat top (m)")
    p.add_argument("--product-mass", type=float, default=0.5)
    p.add_argument("--machine-mass", type=float, default=500.0)
    p.add_argument("--keep-nested", action="store_true", help="do not reparent product under /Root/Root")
    return p.parse_args()


def _bbox(prim, bbox_cache):
    rng = bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange()
    if rng.IsEmpty():
        return None
    mn, mx = rng.GetMin(), rng.GetMax()
    return (
        tuple(float(x) for x in mn),
        tuple(float(x) for x in mx),
        tuple(float(0.5 * (a + b)) for a, b in zip(mn, mx)),
    )


def _world_matrix(prim):
    from pxr import Usd, UsdGeom

    return UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())


def _set_world_translate(prim, world_t, UsdGeom, Gf):
    """Move prim so its world translation becomes world_t, keeping rot/scale."""
    parent = prim.GetParent()
    parent_world = (
        Gf.Matrix4d(_world_matrix(parent))
        if parent and parent.IsValid()
        else Gf.Matrix4d(1.0)
    )
    world_m = Gf.Matrix4d(_world_matrix(prim))
    # Replace translation, keep the upper 3x3.
    world_m.SetTranslateOnly(Gf.Vec3d(world_t))
    local_m = world_m * parent_world.GetInverse()
    xf = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()
    xf.AddTransformOp(UsdGeom.XformOp.PrecisionDouble).Set(local_m)
    return local_m.ExtractTranslation()


def _reparent_product(stage, Sdf, UsdGeom, Gf):
    src = stage.GetPrimAtPath(PROD_PATH)
    if not src or not src.IsValid():
        # already moved?
        dst = stage.GetPrimAtPath(PROD_DST)
        if dst and dst.IsValid():
            print(f"[fix] product already at {PROD_DST}")
            return dst
        raise RuntimeError(f"missing {PROD_PATH}")

    world_m = Gf.Matrix4d(_world_matrix(src))

    layer = stage.GetRootLayer()
    existing = stage.GetPrimAtPath(PROD_DST)
    if existing and existing.IsValid():
        stage.RemovePrim(PROD_DST)

    # Copy then remove (keeps subtree / physics APIs intact).
    ok = Sdf.CopySpec(layer, Sdf.Path(PROD_PATH), layer, Sdf.Path(PROD_DST))
    if not ok:
        raise RuntimeError(f"CopySpec failed {PROD_PATH} -> {PROD_DST}")
    stage.RemovePrim(PROD_PATH)
    dst = stage.GetPrimAtPath(PROD_DST)
    if not dst or not dst.IsValid():
        raise RuntimeError(f"reparent failed, missing {PROD_DST}")

    # Bake a single double-precision transform so world pose is unchanged under
    # the new parent (avoids quatd vs float precision conflicts on orient ops).
    parent_world = Gf.Matrix4d(_world_matrix(dst.GetParent()))
    local_m = world_m * parent_world.GetInverse()
    dst_xf = UsdGeom.Xformable(dst)
    dst_xf.ClearXformOpOrder()
    dst_xf.AddTransformOp(UsdGeom.XformOp.PrecisionDouble).Set(local_m)
    print(
        f"[fix] reparented {PROD_PATH} -> {PROD_DST} "
        f"world_t={world_m.ExtractTranslation()}"
    )
    return dst


def _place_on_seat(product, seat, bbox_cache, clearance, UsdGeom, Gf):
    bbox_cache.Clear()
    pb = _bbox(product, bbox_cache)
    sb = _bbox(seat, bbox_cache)
    if pb is None or sb is None:
        raise RuntimeError("empty bbox for product or seat")
    pmin, pmax, pc = pb
    smin, smax, sc = sb
    # Keep XY of product center; put bottom on seat top + clearance
    target_world = Gf.Vec3d(pc[0], pc[1], smax[2] + clearance + 0.5 * (pmax[2] - pmin[2]))
    # Current world center vs target: easier to shift by (target_z_bottom - current_bottom)
    world = _world_matrix(product)
    cur_t = world.ExtractTranslation()
    dz = (smax[2] + clearance) - pmin[2]
    new_t = Gf.Vec3d(cur_t[0], cur_t[1], cur_t[2] + dz)
    local_t = _set_world_translate(product, new_t, UsdGeom, Gf)
    bbox_cache.Clear()
    pb2 = _bbox(product, bbox_cache)
    print(
        f"[fix] place on seat: seat_topZ={smax[2]:.6f} product_bottom "
        f"{pmin[2]:.6f} -> {pb2[0][2]:.6f} dz={dz:.6f} local_t={local_t}"
    )
    return dz


def _author_physics(stage, product, args, UsdPhysics, PhysxSchema, Gf):
    # Physics scene
    ps = stage.GetPrimAtPath(SCENE_PATH)
    if ps and ps.IsValid():
        scene = UsdPhysics.Scene(ps)
        scene.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
        scene.CreateGravityMagnitudeAttr(9.81)
        try:
            PhysxSchema.PhysxSceneAPI.Apply(ps).CreateEnableCCDAttr(True)
        except Exception as exc:  # noqa: BLE001
            print(f"[fix] CCD attr skip: {exc}")
        print(f"[fix] PhysicsScene gravity=(0,0,-1)*9.81 CCD=on")

    # Machine kinematic RB
    ces = stage.GetPrimAtPath(CES_PATH)
    if ces and ces.IsValid():
        rb = UsdPhysics.RigidBodyAPI.Apply(ces)
        rb.CreateRigidBodyEnabledAttr(True)
        rb.CreateKinematicEnabledAttr(True)
        mass = UsdPhysics.MassAPI.Apply(ces)
        mass.CreateMassAttr(float(args.machine_mass))
        print(f"[fix] CESmachine kinematic RB mass={args.machine_mass}")

    # Product dynamic RB + mass
    rb = UsdPhysics.RigidBodyAPI.Apply(product)
    rb.CreateRigidBodyEnabledAttr(True)
    rb.CreateKinematicEnabledAttr(False)
    mass = UsdPhysics.MassAPI.Apply(product)
    mass.CreateMassAttr(float(args.product_mass))
    print(f"[fix] product dynamic RB mass={args.product_mass}")

    # Ensure product colliders stay enabled
    from pxr import UsdGeom

    n = 0
    for p in stage.Traverse():
        path = str(p.GetPath())
        if not path.startswith(str(product.GetPath())):
            continue
        if not p.IsA(UsdGeom.Mesh):
            continue
        if not p.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI.Apply(p)
            UsdPhysics.MeshCollisionAPI.Apply(p).CreateApproximationAttr("convexHull")
        UsdPhysics.CollisionAPI(p).CreateCollisionEnabledAttr(True)
        n += 1
    print(f"[fix] product mesh colliders ensured: {n}")


def main() -> int:
    args = _parse_args()
    src = os.path.abspath(args.src)
    dst = os.path.abspath(args.dst)
    if not os.path.isfile(src):
        print(f"[fix] source not found: {src}")
        return 1

    _boot_app(headless=True)
    from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics

    print(f"[fix] copying {src} -> {dst}")
    shutil.copyfile(src, dst)
    stage = Usd.Stage.Open(dst)
    if stage is None:
        print(f"[fix] failed to open {dst}")
        return 1

    old_mpu = UsdGeom.GetStageMetersPerUnit(stage)
    if abs(old_mpu - 1.0) > 1e-9:
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        print(f"[fix] metersPerUnit {old_mpu} -> 1.0")
    else:
        print("[fix] metersPerUnit already 1.0")
    print(f"[fix] upAxis={UsdGeom.GetStageUpAxis(stage)}")

    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
    )

    seat = stage.GetPrimAtPath(SEAT_PATH)
    if not seat or not seat.IsValid():
        print(f"[fix] seat missing: {SEAT_PATH}")
        return 1

    if args.keep_nested:
        product = stage.GetPrimAtPath(PROD_PATH)
        if not product or not product.IsValid():
            product = stage.GetPrimAtPath(PROD_DST)
        if not product or not product.IsValid():
            print("[fix] product prim missing")
            return 1
    else:
        product = _reparent_product(stage, Sdf, UsdGeom, Gf)

    _place_on_seat(product, seat, bbox_cache, args.clearance, UsdGeom, Gf)
    _author_physics(stage, product, args, UsdPhysics, PhysxSchema, Gf)

    # Verify
    bbox_cache.Clear()
    pb = _bbox(product, bbox_cache)
    sb = _bbox(seat, bbox_cache)
    gap = pb[0][2] - sb[1][2]
    print(
        f"[fix] verify productZ=[{pb[0][2]:.6f},{pb[1][2]:.6f}] "
        f"seatZ=[{sb[0][2]:.6f},{sb[1][2]:.6f}] gap(bottom-seatTop)={gap:.6f}"
    )
    if gap < -0.005:
        print("[fix] WARNING: product still intersecting seat significantly")
    elif gap > 0.05:
        print("[fix] WARNING: product floating high above seat")
    else:
        print("[fix] seat placement OK")

    n_rb = sum(1 for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI))
    print(f"[fix] rigidBodyAPIs={n_rb} product_path={product.GetPath()}")

    stage.GetRootLayer().Save()
    print(f"[fix] wrote {dst}")
    _close_app()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        import traceback

        traceback.print_exc()
        _close_app()
        raise
