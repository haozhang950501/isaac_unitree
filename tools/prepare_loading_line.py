# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""Optimize LoadingLine USD: rename, center pivots, encapsulate trays.

CAD/HOOPS exports leave ``tn__...`` names and a design-space origin far from
(0,0,0). This script:

1. Flattens instanceable CAD leaves
2. Encapsulates each tray + on-tray products into a centered ``Tray_Assembly_*``
   Xform (world pose preserved; origin at AABB center for easy motion)
3. Renames remaining Xforms/Meshes to readable names
4. Centers the whole machine AABB at the world origin and bakes root translate
   so coordinate containers have no residual offset (pivot = geometry center)
5. Deactivates the internal ``Prototypes`` library

Usage::

    /home/zh/isaacsim/python.sh tools/prepare_loading_line.py
    /home/zh/isaacsim/python.sh tools/prepare_loading_line.py \\
        --src assets/bozhon/LoadingLine.usd \\
        --dst assets/bozhon/LoadingLine_opt.usd
"""
from __future__ import annotations

import argparse
import functools
import os
import re
import shutil
import sys

print = functools.partial(print, flush=True)  # noqa: A001

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_BOZHON = os.path.join(_PROJECT_ROOT, "assets", "bozhon")
_DEFAULT_SRC = os.path.join(_BOZHON, "LoadingLine.usd")
_DEFAULT_DST = os.path.join(_BOZHON, "LoadingLine_opt.usd")

_simulation_app = None

SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9_]+")
STATION_CODE_RE = re.compile(r"^tn__(\d{6})")

# Tray plate CAD codes -> assembly names. Products/fixtures whose AABB center
# sits on/above the plate (within expanded XY) are reparented under the tray.
TRAY_PLATE_CODES = {
    "060104": "Tray_Assembly_01",
    "060114": "Tray_Assembly_02",
}

# Prefer these readable names when the CAD code matches.
CODE_NAME_MAP = {
    "060101": "Base_Plate",
    "060102": "End_Support",
    "060103": "Carriage_Frame",
    "060104": "Tray_Plate",
    "060105": "Tray_Fixture_Bar",
    "060106": "Tray_Under_Block",
    "060107": "Side_Block",
    "060108": "Side_Stop",
    "060109": "Product_Rail",
    "060110": "Tray_Insert",
    "060111": "Clamp_Plate",
    "060112": "Clamp_Pad",
    "060113": "Shim_Plate",
    "060114": "Tray_Plate",
    "060115": "Product_Seat",
    "070000": "Product",
    "070005": "Product_Cover",
}

TOKEN_PATTERNS = [
    (re.compile(r"HGH\d+", re.I), "Rail_Guide"),
    (re.compile(r"SMCCY", re.I), "Cylinder"),
    (re.compile(r"SMCAS", re.I), "Fitting_AS"),
    (re.compile(r"SMCDM", re.I), "Sensor_DM"),
    (re.compile(r"SMCMGJ", re.I), "Bracket_MGJ"),
    (re.compile(r"SMCRJ", re.I), "Joint_RJ"),
    (re.compile(r"SGJ\d*", re.I), "Support_Post"),
    (re.compile(r"GXF\d*", re.I), "Guide_Block"),
    (re.compile(r"LPL\d*", re.I), "LPL_Part"),
    (re.compile(r"S150100", re.I), "Side_Rail"),
    (re.compile(r"KQ2H\w*", re.I), "Fitting_KQ2"),
    (re.compile(r"AirTAC\w*", re.I), None),
]

_SKIP_RENAME_TYPES = {"Material", "Shader", "MaterialBindingAPI", "Scope"}


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
        print(f"[ll] AppLauncher unavailable ({exc}); trying SimulationApp")
    from isaacsim import SimulationApp

    _simulation_app = SimulationApp({"headless": headless})


def _close_app():
    global _simulation_app
    if _simulation_app is not None:
        _simulation_app.close()
        _simulation_app = None


def _parse_args():
    p = argparse.ArgumentParser(description="Optimize LoadingLine USD")
    p.add_argument("--src", default=_DEFAULT_SRC)
    p.add_argument("--dst", default=_DEFAULT_DST)
    p.add_argument(
        "--xy-margin",
        type=float,
        default=0.025,
        help="XY margin (m) when collecting on-tray parts around a tray plate",
    )
    p.add_argument(
        "--z-below",
        type=float,
        default=0.002,
        help="Allow part Z-min this far below tray Z-min (m) still count as on-tray",
    )
    return p.parse_args()


def _strip_tn(name: str) -> str:
    return name[4:] if name.startswith("tn__") else name


def _cad_code(name: str) -> str | None:
    raw = _strip_tn(name)
    m = re.match(r"(\d{6})", raw)
    if m:
        return m.group(1)
    return None


def _sanitize(name: str, max_len: int = 48) -> str:
    name = SAFE_NAME_RE.sub("_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    if not name:
        name = "Part"
    if name[0].isdigit():
        name = "N_" + name
    return name[:max_len]


def _unique_name(desired: str, used: set[str]) -> str:
    base = _sanitize(desired)
    if base not in used:
        used.add(base)
        return base
    i = 2
    while f"{base}_{i}" in used:
        i += 1
    name = f"{base}_{i}"
    used.add(name)
    return name


def _extract_readable(name: str) -> str | None:
    code = _cad_code(name)
    if code and code in CODE_NAME_MAP:
        return CODE_NAME_MAP[code]
    raw = _strip_tn(name)
    m = re.search(r"solid(\d+)$", raw, re.I)
    if m:
        return f"Solid_{int(m.group(1))}"
    m = re.search(r"(\d+)mm", raw, re.I)
    if m and len(raw) < 24:
        return f"N_{m.group(1)}mm"
    for pattern, replacement in TOKEN_PATTERNS:
        m = pattern.search(raw)
        if not m:
            continue
        if replacement is not None:
            return _sanitize(replacement)
        return _sanitize(m.group(0)[:24])
    m = re.match(r"(\d{6})", raw)
    if m:
        return f"Asm_{m.group(1)}"
    # Truncated leftover vendor hashes from a previous pass — re-token.
    for pattern, replacement in TOKEN_PATTERNS:
        m = pattern.search(name)
        if m and replacement is not None:
            return _sanitize(replacement)
    head = raw.split("_")[0]
    if 3 <= len(head) <= 24 and re.search(r"[A-Za-z]{2,}", head):
        return _sanitize(head)
    return None


def _should_rename_prim(prim, UsdGeom) -> bool:
    name = prim.GetName()
    if name in ("Looks", "Materials", "Shader", "PreviewSurface", "Diffuse", "Prototypes"):
        return False
    if str(prim.GetPath()).startswith("/__Prototype"):
        return False
    t = prim.GetTypeName()
    if t in _SKIP_RENAME_TYPES:
        return False
    if prim.IsA(UsdGeom.Xform) or prim.IsA(UsdGeom.Mesh) or t in ("", "Xform"):
        return True
    return False


def _flatten_instances(stage) -> int:
    targets = [p for p in stage.Traverse() if p.IsInstanceable()]
    for prim in targets:
        prim.SetInstanceable(False)
    print(f"[ll] flattened instanceable prims: {len(targets)}")
    return len(targets)


def _bbox_range(prim, bbox_cache):
    rng = bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange()
    if rng.IsEmpty():
        return None
    return rng


def _aabb_center(rng):
    from pxr import Gf

    mn, mx = rng.GetMin(), rng.GetMax()
    return Gf.Vec3d(0.5 * (mn[0] + mx[0]), 0.5 * (mn[1] + mx[1]), 0.5 * (mn[2] + mx[2]))


def _world_matrix(prim):
    from pxr import Usd, UsdGeom

    return UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())


def _set_local_matrix(prim, local_m, UsdGeom):
    xf = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()
    xf.AddTransformOp(UsdGeom.XformOp.PrecisionDouble).Set(local_m)


def _store_cad_name(prim, old_name: str):
    try:
        data = prim.GetCustomData()
        data = dict(data) if data else {}
        data["cadName"] = old_name
        prim.SetCustomData(data)
    except Exception:  # noqa: BLE001
        pass


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


def _bake_offset_into_child(child, offset, UsdGeom, Gf):
    """Left-multiply child local xform by Translate(offset) (row-vector USD)."""
    xformable = UsdGeom.Xformable(child)
    t_op = _get_translate_op(xformable, UsdGeom)
    m_op = _get_transform_op(xformable, UsdGeom)

    if m_op is not None and t_op is None:
        old = m_op.Get()
        if old is None:
            old = Gf.Matrix4d(1.0)
        bake = Gf.Matrix4d(1.0).SetTranslate(offset)
        m_op.Set(old * bake)
        return

    if t_op is not None:
        cur = t_op.Get()
        if cur is None:
            cur = Gf.Vec3d(0, 0, 0)
        ops = xformable.GetOrderedXformOps()
        if ops and ops[0].GetOpType() == UsdGeom.XformOp.TypeTranslate:
            t_op.Set(Gf.Vec3d(cur) + offset)
            return
        local = xformable.GetLocalTransformation()
        bake = Gf.Matrix4d(1.0).SetTranslate(offset)
        xformable.ClearXformOpOrder()
        xformable.AddTransformOp().Set(local * bake)
        return

    new_op = xformable.AddTranslateOp()
    new_op.Set(offset)
    ops = xformable.GetOrderedXformOps()
    if ops and ops[0].GetOpName() != new_op.GetOpName():
        ordered = [new_op] + [op for op in ops if op.GetOpName() != new_op.GetOpName()]
        xformable.SetXformOpOrder(ordered)


def _find_tray_plates(asm):
    plates = []
    for child in asm.GetChildren():
        code = _cad_code(child.GetName())
        if code in TRAY_PLATE_CODES:
            plates.append((child, TRAY_PLATE_CODES[code]))
    plates.sort(key=lambda item: item[1])
    return plates


def _collect_on_tray_parts(asm, tray_prim, bbox_cache, xy_margin: float, z_below: float):
    """Parts whose center XY is in tray AABB (expanded) and not clearly below tray."""
    tray_rng = _bbox_range(tray_prim, bbox_cache)
    if tray_rng is None:
        return []
    tmin, tmax = tray_rng.GetMin(), tray_rng.GetMax()
    members = []
    for child in asm.GetChildren():
        if child.GetPath() == tray_prim.GetPath():
            members.append(child)
            continue
        # Skip other tray plates and huge base/rail spans.
        code = _cad_code(child.GetName())
        if code in TRAY_PLATE_CODES:
            continue
        rng = _bbox_range(child, bbox_cache)
        if rng is None:
            continue
        mn, mx = rng.GetMin(), rng.GetMax()
        sx, sy, sz = mx[0] - mn[0], mx[1] - mn[1], mx[2] - mn[2]
        # Skip long rails / base that span the whole line.
        if sy > 0.35 or sx > 0.35:
            continue
        # Tall posts under the line (legs) — not on tray.
        if sz > 0.12 and mn[2] < tmin[2] - 0.05:
            continue
        cx = 0.5 * (mn[0] + mx[0])
        cy = 0.5 * (mn[1] + mx[1])
        if not (
            tmin[0] - xy_margin <= cx <= tmax[0] + xy_margin
            and tmin[1] - xy_margin <= cy <= tmax[1] + xy_margin
        ):
            continue
        # Must sit on/above the tray (allow tiny penetration / shim below).
        if mx[2] < tmin[2] - z_below:
            continue
        if mn[2] < tmin[2] - 0.03 and mx[2] <= tmin[2] + 0.005:
            # Entirely under the tray plate → fixed structure, skip.
            continue
        members.append(child)
    return members


def _union_aabb(prims, bbox_cache):
    from pxr import Gf

    mn = None
    mx = None
    for prim in prims:
        rng = _bbox_range(prim, bbox_cache)
        if rng is None:
            continue
        rmin, rmax = rng.GetMin(), rng.GetMax()
        if mn is None:
            mn = Gf.Vec3d(rmin)
            mx = Gf.Vec3d(rmax)
        else:
            mn = Gf.Vec3d(min(mn[0], rmin[0]), min(mn[1], rmin[1]), min(mn[2], rmin[2]))
            mx = Gf.Vec3d(max(mx[0], rmax[0]), max(mx[1], rmax[1]), max(mx[2], rmax[2]))
    if mn is None:
        return None

    class _R:
        def GetMin(self_inner):
            return mn

        def GetMax(self_inner):
            return mx

    return _R()


def _reparent_under(stage, child, new_parent_path, Sdf, UsdGeom, Gf):
    """Move prim under new_parent, preserving world transform."""
    world_m = Gf.Matrix4d(_world_matrix(child))
    old_path = child.GetPath()
    new_path = Sdf.Path(new_parent_path).AppendChild(child.GetName())
    layer = stage.GetRootLayer()

    if stage.GetPrimAtPath(new_path):
        # Name collision — rename destination.
        parent = stage.GetPrimAtPath(new_parent_path)
        used = {c.GetName() for c in parent.GetChildren()}
        new_name = _unique_name(child.GetName(), used)
        new_path = Sdf.Path(new_parent_path).AppendChild(new_name)

    ok = Sdf.CopySpec(layer, old_path, layer, new_path)
    if not ok:
        raise RuntimeError(f"CopySpec failed {old_path} -> {new_path}")
    stage.RemovePrim(old_path)
    dst = stage.GetPrimAtPath(new_path)
    if not dst or not dst.IsValid():
        raise RuntimeError(f"reparent missing {new_path}")

    parent = dst.GetParent()
    parent_world = Gf.Matrix4d(_world_matrix(parent)) if parent and parent.IsValid() else Gf.Matrix4d(1.0)
    local_m = world_m * parent_world.GetInverse()
    _set_local_matrix(dst, local_m, UsdGeom)
    return dst


def _encapsulate_trays(stage, asm_path, args, Usd, UsdGeom, Sdf, Gf):
    asm = stage.GetPrimAtPath(asm_path)
    if not asm:
        raise RuntimeError(f"missing {asm_path}")

    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
    )

    plates = _find_tray_plates(asm)
    if not plates:
        print("[ll] WARNING: no tray plates found (060104/060114)")
        return []

    created = []
    for tray_prim, asm_name in plates:
        bbox_cache.Clear()
        # Re-resolve tray after previous reparents (path may be unchanged — still top-level).
        tray_prim = stage.GetPrimAtPath(tray_prim.GetPath())
        if not tray_prim or not tray_prim.IsValid():
            # Try find by CAD code among current children.
            code_target = None
            for code, name in TRAY_PLATE_CODES.items():
                if name == asm_name:
                    code_target = code
                    break
            for c in asm.GetChildren():
                if _cad_code(c.GetName()) == code_target:
                    tray_prim = c
                    break
        if not tray_prim or not tray_prim.IsValid():
            print(f"[ll] skip missing tray for {asm_name}")
            continue

        members = _collect_on_tray_parts(asm, tray_prim, bbox_cache, args.xy_margin, args.z_below)
        if tray_prim not in members:
            members.insert(0, tray_prim)

        union = _union_aabb(members, bbox_cache)
        if union is None:
            print(f"[ll] empty bbox for {asm_name}")
            continue
        center = _aabb_center(union)
        print(f"[ll] {asm_name}: {len(members)} prims, center={tuple(center)}")
        for m in members:
            print(f"[ll]   + {m.GetName()[:70]}")

        # Create tray assembly Xform at AABB center (under assembly).
        tray_path = f"{asm_path}/{asm_name}"
        if stage.GetPrimAtPath(tray_path):
            stage.RemovePrim(tray_path)
        tray_xf = UsdGeom.Xform.Define(stage, tray_path)
        # Parent is asm; set local translate = center in parent space.
        parent_world = Gf.Matrix4d(_world_matrix(asm))
        world_m = Gf.Matrix4d(1.0).SetTranslate(center)
        local_m = world_m * parent_world.GetInverse()
        _set_local_matrix(tray_xf.GetPrim(), local_m, UsdGeom)

        # Reparent members (snapshot paths first).
        member_paths = [m.GetPath() for m in members]
        for path in member_paths:
            prim = stage.GetPrimAtPath(path)
            if not prim or not prim.IsValid():
                continue
            _store_cad_name(prim, prim.GetName())
            _reparent_under(stage, prim, tray_path, Sdf, UsdGeom, Gf)

        created.append(tray_path)

        # Verify tray origin ≈ AABB center and geometry did not jump.
        bbox_cache.Clear()
        tray_prim = stage.GetPrimAtPath(tray_path)
        rng = _bbox_range(tray_prim, bbox_cache)
        c2 = _aabb_center(rng)
        origin = Gf.Vec3d(_world_matrix(tray_prim).ExtractTranslation())
        print(
            f"[ll] {asm_name} verify origin={tuple(origin)} aabb_center={tuple(c2)} "
            f"err={(c2 - origin).GetLength():.3e}"
        )
    return created


def _plan_and_apply_renames(stage, root_path, asm_path, Sdf, UsdGeom):
    """Rename all Xform/Mesh descendants under asm (and tray children)."""
    layer = stage.GetRootLayer()
    applied = 0
    failed = 0

    for _round in range(8):
        targets = []
        for prim in stage.Traverse():
            path = str(prim.GetPath())
            if not path.startswith(asm_path + "/") and path != asm_path:
                continue
            if not _should_rename_prim(prim, UsdGeom):
                continue
            name = prim.GetName()
            if name.startswith("Tray_Assembly_"):
                continue  # keep our encapsulation names
            # Rename CAD hashes, long names, and still-mangled vendor leftovers.
            needs = (
                name.startswith("tn__")
                or _cad_code(name) is not None
                or len(name) > 28
                or bool(re.search(r"[A-Za-z]{3,}\d{4,}", name))  # e.g. SMCAS1201...
            )
            if not needs:
                continue
            desired = _extract_readable(name)
            if desired is None:
                desired = "Mesh" if prim.IsA(UsdGeom.Mesh) else "Part"
            if desired == name:
                continue
            targets.append(prim)

        if not targets:
            break
        targets.sort(key=lambda p: p.GetPath().pathElementCount, reverse=True)
        round_applied = 0
        for prim in targets:
            prim = stage.GetPrimAtPath(prim.GetPath())
            if not prim or not prim.IsValid():
                continue
            parent = prim.GetParent()
            if not parent:
                continue
            used = {c.GetName() for c in parent.GetChildren()}
            used.discard(prim.GetName())
            desired = _extract_readable(prim.GetName()) or (
                "Mesh" if prim.IsA(UsdGeom.Mesh) else "Part"
            )
            # Keep Product / Tray_Plate uniqueness with suffixes.
            new = _unique_name(desired, used)
            if new == prim.GetName():
                continue
            _store_cad_name(prim, prim.GetName())
            edit = Sdf.BatchNamespaceEdit()
            edit.Add(Sdf.NamespaceEdit.Rename(prim.GetPath(), new))
            if layer.Apply(edit):
                applied += 1
                round_applied += 1
            else:
                failed += 1
                if failed <= 8:
                    print(f"[ll] rename FAILED {prim.GetPath()} -> {new}")
        if round_applied == 0:
            break

    print(f"[ll] renames applied={applied} failed={failed}")
    return applied, failed


def _center_machine(stage, root_path, Usd, UsdGeom, Gf):
    """Move AABB center to world origin, then bake root translate into children.

    After this, ``/LoadingLine`` and the inner assembly have identity transforms
    (no offset), while the gizmo / origin sits at the geometry AABB center.
    """
    root = stage.GetPrimAtPath(root_path)
    asm = None
    for child in root.GetChildren():
        if child.GetName() not in ("Prototypes", "prototypes", "Looks", "Materials"):
            if child.IsA(UsdGeom.Xform):
                asm = child
                break
    if asm is None:
        raise RuntimeError("missing inner assembly under root")

    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
    )
    rng = _bbox_range(root, bbox_cache)
    center = _aabb_center(rng)
    delta = Gf.Vec3d(0, 0, 0) - center
    print(f"[ll] center delta={tuple(delta)} (AABB center -> origin)")

    # Apply centering as a pure root translate (CAD roots are typically identity).
    root_xf = UsdGeom.Xformable(root)
    root_xf.ClearXformOpOrder()
    root_xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(delta)

    bbox_cache.Clear()
    rng2 = _bbox_range(root, bbox_cache)
    c2 = _aabb_center(rng2)
    print(f"[ll] after translate center={tuple(c2)}")

    offset = delta
    print(f"[ll] baking root offset {tuple(offset)} into assembly children")
    for child in list(asm.GetChildren()):
        if child.GetName() in ("Prototypes", "prototypes"):
            continue
        _bake_offset_into_child(child, offset, UsdGeom, Gf)

    # Clear root and inner assembly transforms — containers have no offset.
    root_xf.ClearXformOpOrder()
    root_xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(0, 0, 0))

    inner_xf = UsdGeom.Xformable(asm)
    inner_t = _get_translate_op(inner_xf, UsdGeom)
    if inner_t is not None:
        inner_t.Set(Gf.Vec3d(0, 0, 0))
    else:
        # Keep existing ops but ensure no stray translate; identity is fine.
        local = inner_xf.GetLocalTransformation()
        if not Gf.IsClose(local, Gf.Matrix4d(1.0), 1e-9):
            # Bake any residual inner transform into its children, then clear.
            inner_tvec = Gf.Vec3d(local.ExtractTranslation())
            if inner_tvec.GetLength() > 1e-9:
                for child in list(asm.GetChildren()):
                    if child.GetName() in ("Prototypes", "prototypes"):
                        continue
                    _bake_offset_into_child(child, inner_tvec, UsdGeom, Gf)
            inner_xf.ClearXformOpOrder()
            inner_xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(0, 0, 0))

    bbox_cache.Clear()
    xcache = UsdGeom.XformCache(Usd.TimeCode.Default())
    rng3 = _bbox_range(root, bbox_cache)
    c3 = _aabb_center(rng3)
    o3 = Gf.Vec3d(xcache.GetLocalToWorldTransform(root).ExtractTranslation())
    print(
        f"[ll] final origin={tuple(o3)} aabb_center={tuple(c3)} "
        f"pivot_err={(c3 - o3).GetLength():.3e}"
    )
    return c3, o3


def _verify(stage, root_path, UsdGeom, tray_paths):
    print("[ll] === top-level under assembly ===")
    root = stage.GetPrimAtPath(root_path)
    asm = None
    for c in root.GetChildren():
        if c.GetName() not in ("Prototypes", "prototypes", "Looks", "Materials"):
            asm = c
            break
    if asm:
        for c in asm.GetChildren():
            print(f"[ll]   {c.GetName()}")
            if c.GetName().startswith("Tray_Assembly"):
                for gc in c.GetChildren():
                    print(f"[ll]     - {gc.GetName()}")

    n_tn = sum(
        1
        for p in stage.Traverse()
        if p.GetName().startswith("tn__") and str(p.GetPath()).startswith(root_path)
    )
    print(f"[ll] tn__ remaining under root: {n_tn}")
    for path in tray_paths:
        prim = stage.GetPrimAtPath(path)
        if prim and prim.IsValid():
            print(f"[ll] tray ok: {path} children={len(prim.GetChildren())}")
        else:
            # Path may have shifted if asm renamed — search by name.
            name = path.rsplit("/", 1)[-1]
            found = [p for p in stage.Traverse() if p.GetName() == name]
            if found:
                print(f"[ll] tray ok: {found[0].GetPath()} children={len(found[0].GetChildren())}")
            else:
                print(f"[ll] WARNING missing tray {path}")


def main() -> int:
    args = _parse_args()
    src = os.path.abspath(args.src)
    dst = os.path.abspath(args.dst)
    if not os.path.isfile(src):
        print(f"[ll] source not found: {src}")
        return 1

    _boot_app(headless=True)
    from pxr import Gf, Sdf, Usd, UsdGeom

    print(f"[ll] copying {src} -> {dst}")
    shutil.copyfile(src, dst)
    stage = Usd.Stage.Open(dst)
    if stage is None:
        print(f"[ll] failed to open {dst}")
        return 1

    print(f"[ll] upAxis={UsdGeom.GetStageUpAxis(stage)} mpu={UsdGeom.GetStageMetersPerUnit(stage)}")
    root = stage.GetDefaultPrim()
    if not root:
        print("[ll] no defaultPrim")
        return 1
    root_path = str(root.GetPath())
    # Inner assembly is typically /LoadingLine/LoadingLine
    asm = None
    for c in root.GetChildren():
        if c.GetName() == root.GetName() and c.IsA(UsdGeom.Xform):
            asm = c
            break
    if asm is None:
        for c in root.GetChildren():
            if c.IsA(UsdGeom.Xform) and c.GetName() not in ("Prototypes", "prototypes"):
                asm = c
                break
    if asm is None:
        print("[ll] missing inner assembly")
        return 1
    asm_path = str(asm.GetPath())
    print(f"[ll] root={root_path} asm={asm_path}")

    _flatten_instances(stage)

    # 1) Encapsulate trays BEFORE renames (match CAD codes).
    tray_paths = _encapsulate_trays(stage, asm_path, args, Usd, UsdGeom, Sdf, Gf)

    # 2) Rename remaining tn__ / long CAD names.
    _plan_and_apply_renames(stage, root_path, asm_path, Sdf, UsdGeom)

    # Refresh tray paths after renames (Tray_Assembly_* names are kept).
    tray_paths = [
        str(p.GetPath())
        for p in stage.Traverse()
        if p.GetName().startswith("Tray_Assembly_")
    ]

    # 3) Center machine + clear container offsets.
    _center_machine(stage, root_path, Usd, UsdGeom, Gf)

    # Hide CAD prototype library.
    for proto_name in ("Prototypes", "prototypes"):
        proto = stage.GetPrimAtPath(f"{root_path}/{proto_name}")
        if proto and proto.IsValid():
            proto.SetActive(False)
            print(f"[ll] deactivated {proto.GetPath()}")

    stage.GetRootLayer().Save()
    print(f"[ll] wrote {dst}")
    _verify(stage, root_path, UsdGeom, tray_paths)
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
