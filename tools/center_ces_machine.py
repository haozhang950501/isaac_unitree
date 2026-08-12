# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""Center CESmachine USD so its AABB center sits at the world origin.

CAD / STEP imports often keep the manufacturer's design-space origin, so the
geometry lands far from (0,0,0).  This script:

1. Opens the converted USD (default: ``assets/bozhon/CESmachine_pickabble.usd``)
2. Computes the world-space axis-aligned bounding box of the default prim
3. Applies an inverse translation on the root Xform so the AABB center is
   exactly at the world origin
4. Writes a new file (default: same path with ``_centered`` suffix), or
   ``--in-place`` overwrites the source

Usage (from Isaac Lab / Isaac Sim python)::

    /home/zh/isaacsim/python.sh tools/center_ces_machine.py
    /home/zh/isaacsim/python.sh tools/center_ces_machine.py \\
        --src assets/bozhon/CESmachine_pickabble.usd \\
        --dst assets/bozhon/CESmachine_centered.usd \\
        --mode center   # or: xy_center_z_bottom
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
_DEFAULT_SRC = os.path.join(_BOZHON, "CESmachine_pickabble.usd")
_DEFAULT_DST = os.path.join(_BOZHON, "CESmachine_centered.usd")

_simulation_app = None


def _boot_app(headless: bool = True):
    global _simulation_app
    try:
        from pxr import Usd  # noqa: F401
        return
    except ImportError:
        pass

    # Prefer Isaac Lab AppLauncher when available.
    try:
        from isaaclab.app import AppLauncher

        _simulation_app = AppLauncher({"headless": headless, "enable_cameras": False}).app
        return
    except Exception as exc:  # noqa: BLE001
        print(f"[center] AppLauncher unavailable ({exc}); trying SimulationApp")

    from isaacsim import SimulationApp

    _simulation_app = SimulationApp({"headless": headless})


def _close_app():
    global _simulation_app
    if _simulation_app is not None:
        _simulation_app.close()
        _simulation_app = None


def _parse_args():
    p = argparse.ArgumentParser(description="Center CESmachine USD at world origin")
    p.add_argument("--src", default=_DEFAULT_SRC)
    p.add_argument("--dst", default=_DEFAULT_DST)
    p.add_argument(
        "--mode",
        choices=("center", "xy_center_z_bottom"),
        default="center",
        help="center: AABB center -> (0,0,0); xy_center_z_bottom: XY center, Z min -> 0",
    )
    p.add_argument("--in-place", action="store_true", help="overwrite --src instead of writing --dst")
    return p.parse_args()


def _local_translate(xformable, Gf, UsdGeom):
    """Return current local translation as Gf.Vec3d, preferring xformOp:translate."""
    ops = xformable.GetOrderedXformOps()
    for op in ops:
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            val = op.Get()
            if val is None:
                return Gf.Vec3d(0, 0, 0)
            return Gf.Vec3d(val)
    # Fall back to decomposed local matrix.
    m = xformable.GetLocalTransformation()
    t = m.ExtractTranslation()
    return Gf.Vec3d(t)


def _set_local_translate(xformable, translate, UsdGeom, Gf):
    """Set / create xformOp:translate and keep it in xformOpOrder."""
    ops = xformable.GetOrderedXformOps()
    translate_op = None
    for op in ops:
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            translate_op = op
            break
    if translate_op is None:
        translate_op = xformable.AddTranslateOp()
    translate_op.Set(Gf.Vec3d(translate))


def main() -> int:
    args = _parse_args()
    src = os.path.abspath(args.src)
    dst = src if args.in_place else os.path.abspath(args.dst)

    if not os.path.isfile(src):
        print(f"[center] source not found: {src}")
        return 1

    _boot_app(headless=True)
    from pxr import Gf, Usd, UsdGeom

    if not args.in_place:
        print(f"[center] copying {src} -> {dst}")
        shutil.copyfile(src, dst)
    else:
        print(f"[center] editing in place: {src}")

    stage = Usd.Stage.Open(dst)
    if stage is None:
        print(f"[center] failed to open stage: {dst}")
        return 1

    print(f"[center] upAxis={UsdGeom.GetStageUpAxis(stage)}")
    print(f"[center] metersPerUnit={UsdGeom.GetStageMetersPerUnit(stage)}")

    root = stage.GetDefaultPrim()
    if not root:
        children = stage.GetPseudoRoot().GetChildren()
        if not children:
            print("[center] empty stage")
            return 1
        root = children[0]
        stage.SetDefaultPrim(root)
        print(f"[center] no defaultPrim; using {root.GetPath()}")

    print(f"[center] root={root.GetPath()} type={root.GetTypeName()}")

    # Ensure root is Xformable so we can translate it.
    if not root.IsA(UsdGeom.Xformable):
        # Wrap content under a new World xform if needed.
        print("[center] root is not Xformable; creating /World wrapper")
        world = UsdGeom.Xform.Define(stage, "/World")
        # Move existing root under /World when possible.
        # Safer approach: apply offset to first xformable child.
        xform_root = None
        for child in root.GetChildren():
            if child.IsA(UsdGeom.Xformable):
                xform_root = child
                break
        if xform_root is None:
            print("[center] no xformable prim found to translate")
            return 1
        root = xform_root
        stage.SetDefaultPrim(world.GetPrim())
        print(f"[center] translating {root.GetPath()}")

    xformable = UsdGeom.Xformable(root)
    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
    )

    def report(tag: str):
        bbox_cache.Clear()
        wb = bbox_cache.ComputeWorldBound(root)
        rng = wb.ComputeAlignedRange()
        mn, mx = rng.GetMin(), rng.GetMax()
        center = Gf.Vec3d(
            0.5 * (mn[0] + mx[0]),
            0.5 * (mn[1] + mx[1]),
            0.5 * (mn[2] + mx[2]),
        )
        size = Gf.Vec3d(mx[0] - mn[0], mx[1] - mn[1], mx[2] - mn[2])
        print(
            f"[center] {tag}: min=({mn[0]:.6f},{mn[1]:.6f},{mn[2]:.6f}) "
            f"max=({mx[0]:.6f},{mx[1]:.6f},{mx[2]:.6f}) "
            f"center=({center[0]:.6f},{center[1]:.6f},{center[2]:.6f}) "
            f"size=({size[0]:.6f},{size[1]:.6f},{size[2]:.6f})"
        )
        return mn, mx, center, size

    mn, mx, center, size = report("before")

    if args.mode == "center":
        delta = Gf.Vec3d(0, 0, 0) - center
    else:
        # XY center -> 0, Z min (feet) -> 0.
        delta = Gf.Vec3d(-center[0], -center[1], -float(mn[2]))

    cur_t = _local_translate(xformable, Gf, UsdGeom)
    new_t = cur_t + delta
    print(
        f"[center] mode={args.mode} delta=({delta[0]:.6f},{delta[1]:.6f},{delta[2]:.6f}) "
        f"translate {tuple(cur_t)} -> {tuple(new_t)}"
    )
    _set_local_translate(xformable, new_t, UsdGeom, Gf)

    mn2, mx2, center2, size2 = report("after")
    print(
        f"[center] residual center offset = "
        f"({center2[0]:.6e},{center2[1]:.6e},{center2[2]:.6e})"
    )

    stage.GetRootLayer().Save()
    print(f"[center] wrote {dst}")
    _close_app()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        _close_app()
        raise
