# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""Move the CESmachine root Xform pivot to the geometry AABB center.

The centered USD keeps CAD origin via ``/CESmachine`` translate, so the Stage
gizmo sits outside the machine even though the AABB is at the world origin.
This script bakes that translate into the station children and clears the root
translate, without moving the mesh in world space.

Usage::

    /home/zh/isaacsim/python.sh tools/recenter_ces_pivot.py \\
        --src assets/bozhon/CESmachine_pickabble.usd
"""
from __future__ import annotations

import argparse
import functools
import os
import sys

print = functools.partial(print, flush=True)  # noqa: A001

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_SRC = os.path.join(_PROJECT_ROOT, "assets", "bozhon", "CESmachine_pickabble.usd")

_simulation_app = None


def _boot_app():
    global _simulation_app
    try:
        from pxr import Usd  # noqa: F401

        return
    except ImportError:
        pass
    from isaacsim import SimulationApp

    _simulation_app = SimulationApp({"headless": True})


def _close_app():
    global _simulation_app
    if _simulation_app is not None:
        _simulation_app.close()
        _simulation_app = None


def _get_translate_op(xformable, UsdGeom):
    for op in xformable.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            return op
    return None


def _get_transform_op(xformable, UsdGeom):
    for op in xformable.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTransform:
            return op
    return None


def _bake_offset_into_child(child, offset: "Gf.Vec3d", UsdGeom, Gf):
    """Left-multiply child local xform by Translate(offset)."""
    xformable = UsdGeom.Xformable(child)
    t_op = _get_translate_op(xformable, UsdGeom)
    m_op = _get_transform_op(xformable, UsdGeom)

    # USD uses row-vector convention: p_world = p_local * local * parent.
    # Old: world = M_child * Translate(root_T).  New root = I, so
    # M'_child = M_child * Translate(root_T).
    if m_op is not None and t_op is None:
        old = m_op.Get()
        if old is None:
            old = Gf.Matrix4d(1.0)
        bake = Gf.Matrix4d(1.0).SetTranslate(offset)
        m_op.Set(old * bake)
        return

    if t_op is not None:
        # If both translate and later ops exist, only adjusting translate is
        # correct when translate is applied first in xformOpOrder.
        cur = t_op.Get()
        if cur is None:
            cur = Gf.Vec3d(0, 0, 0)
        ops = xformable.GetOrderedXformOps()
        if ops and ops[0].GetOpType() == UsdGeom.XformOp.TypeTranslate:
            t_op.Set(Gf.Vec3d(cur) + offset)
            return
        # Translate is not first — fold via full local matrix instead.
        local = xformable.GetLocalTransformation()
        bake = Gf.Matrix4d(1.0).SetTranslate(offset)
        new_local = local * bake
        # Reset ops to a single transform matrix.
        xformable.ClearXformOpOrder()
        xf = xformable.AddTransformOp()
        xf.Set(new_local)
        return

    # No translate/transform yet — add translate and keep it first in op order.
    new_op = xformable.AddTranslateOp()
    new_op.Set(offset)
    ops = xformable.GetOrderedXformOps()
    if ops and ops[0].GetOpName() != new_op.GetOpName():
        ordered = [new_op] + [op for op in ops if op.GetOpName() != new_op.GetOpName()]
        xformable.SetXformOpOrder(ordered)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default=_DEFAULT_SRC)
    parser.add_argument(
        "--dst",
        default="",
        help="Output path (default: overwrite --src)",
    )
    args = parser.parse_args()
    src = os.path.abspath(args.src)
    dst = os.path.abspath(args.dst) if args.dst else src
    if not os.path.isfile(src):
        print(f"[pivot] missing {src}")
        return 1

    _boot_app()
    from pxr import Gf, Usd, UsdGeom

    if dst != src:
        import shutil

        shutil.copyfile(src, dst)
        print(f"[pivot] copied {src} -> {dst}")

    stage = Usd.Stage.Open(dst)
    root = stage.GetPrimAtPath("/CESmachine")
    inner = stage.GetPrimAtPath("/CESmachine/CESmachine")
    if not root or not inner:
        print("[pivot] expected /CESmachine/CESmachine")
        return 1

    bbox = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
    )
    xcache = UsdGeom.XformCache(Usd.TimeCode.Default())

    def report(tag):
        rng = bbox.ComputeWorldBound(root).ComputeAlignedRange()
        mn, mx = rng.GetMin(), rng.GetMax()
        c = Gf.Vec3d(0.5 * (mn[0] + mx[0]), 0.5 * (mn[1] + mx[1]), 0.5 * (mn[2] + mx[2]))
        o = xcache.GetLocalToWorldTransform(root).ExtractTranslation()
        print(
            f"[pivot] {tag}: origin=({o[0]:.6f},{o[1]:.6f},{o[2]:.6f}) "
            f"aabb_center=({c[0]:.6f},{c[1]:.6f},{c[2]:.6f}) "
            f"delta=({c[0]-o[0]:.6f},{c[1]-o[1]:.6f},{c[2]-o[2]:.6f})"
        )
        return c, Gf.Vec3d(o)

    c0, o0 = report("before")
    # Root currently carries the CAD centering translate; bake it into children.
    root_xf = UsdGeom.Xformable(root)
    t_op = _get_translate_op(root_xf, UsdGeom)
    if t_op is None:
        # Fall back: bake origin→center offset expressed in root parent space.
        offset = c0 - o0
        print(f"[pivot] no root translate op; baking origin_to_center={offset}")
    else:
        cur = t_op.Get()
        offset = Gf.Vec3d(cur) if cur is not None else (c0 - o0)
        print(f"[pivot] baking root translate {tuple(offset)} into station children")

    # Prefer baking onto the 10 station modules so BOTH /CESmachine and
    # /CESmachine/CESmachine have their origin at the geometry center.
    children = list(inner.GetChildren())
    if not children:
        print("[pivot] no children under inner assembly")
        return 1

    for child in children:
        if child.GetName() in ("Prototypes", "prototypes"):
            continue
        _bake_offset_into_child(child, offset, UsdGeom, Gf)
        print(f"[pivot]   +offset -> {child.GetPath()}")

    # Clear root translate (gizmo at geometry center when AABB is at origin).
    if t_op is not None:
        t_op.Set(Gf.Vec3d(0, 0, 0))
    else:
        root_xf.AddTranslateOp().Set(Gf.Vec3d(0, 0, 0))

    # Keep inner assembly identity (or clear stray translate).
    inner_xf = UsdGeom.Xformable(inner)
    inner_t = _get_translate_op(inner_xf, UsdGeom)
    if inner_t is not None:
        inner_t.Set(Gf.Vec3d(0, 0, 0))

    # Invalidate caches
    bbox.Clear()
    xcache.Clear()
    c1, o1 = report("after")
    geom_shift = (c1 - c0).GetLength()
    pivot_err = (c1 - o1).GetLength()
    print(f"[pivot] geometry world shift={geom_shift:.6e} m (want ~0)")
    print(f"[pivot] pivot-to-center error={pivot_err:.6e} m (want ~0)")

    stage.GetRootLayer().Save()
    print(f"[pivot] wrote {dst}")
    _close_app()
    return 0 if pivot_err < 1e-3 and geom_shift < 1e-3 else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        import traceback

        traceback.print_exc()
        _close_app()
        raise
