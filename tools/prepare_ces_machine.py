# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""Rename CES machine assembly prims and author kinematic rigid-body colliders.

Reads the project CES USD (default ``assets/bozhon/CESmachine_pickabble.usd``),
shortens HOOPS ``tn__...`` names into readable assembly names, attaches a single
kinematic rigid body on the machine root, and enables mesh-simplification
colliders on prototype meshes (skipping tiny fasteners).

Usage::

    /home/zh/isaacsim/python.sh tools/prepare_ces_machine.py
    /home/zh/isaacsim/python.sh tools/prepare_ces_machine.py \\
        --src assets/bozhon/CESmachine_pickabble.usd \\
        --dst assets/bozhon/CESmachine_physics.usd
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
_DEFAULT_SRC = os.path.join(_BOZHON, "CESmachine_pickabble.usd")
_DEFAULT_DST = os.path.join(_BOZHON, "CESmachine_physics.usd")

_simulation_app = None

# ---------------------------------------------------------------------------
# Top-level / special-case rename tables (matched against original CAD names)
# ---------------------------------------------------------------------------

# Order matters: first matching predicate wins among siblings under /CESmachine/CESmachine.
TOP_MODULE_RULES = [
    (lambda n: n.startswith("tn__010000"), "Station_01_Main"),
    (lambda n: n.startswith("tn__020000"), "Station_02"),
    (lambda n: n.startswith("tn__030000XYZR") or "XYZR" in n, "Stage_XYZR"),
    (lambda n: "ZKN77W800H1200D600" in n, "Cabinet_ZKN77"),
    (lambda n: n.startswith("tn__040000"), "Station_04"),
    (lambda n: n.startswith("tn__050000"), "Stage_05_XYZ"),
    (lambda n: n.startswith("tn__060000"), "Station_06"),
    (lambda n: n.startswith("tn__080000"), "Station_08"),
    # Two nearly-identical valve manifolds; assigned A/B by encounter order.
    (lambda n: n.startswith("tn__xaUov") or n.startswith("tn___1_"), "Valve_Block"),
]

STAGE_XYZR_CHILD_RULES = [
    (lambda n: "032000Y" in n or re.search(r"032000Y", n), "Axis_Y"),
    (lambda n: "031000X" in n or re.search(r"031000X", n), "Axis_X"),
    (lambda n: "033000ZR" in n or "033000" in n, "Axis_ZR"),
    (lambda n: "034000" in n, "Axis_Aux"),
]

STAGE_05_CHILD_RULES = [
    (lambda n: "050300Z" in n or "050300" in n, "Axis_Z"),
    (lambda n: "050200X" in n or "050200" in n, "Axis_X"),
    (lambda n: "050100Y" in n or "050100" in n, "Axis_Y"),
]

# Readable token patterns extracted from HOOPS-mangled names (first match wins).
TOKEN_PATTERNS = [
    (re.compile(r"(HUAKUAI)", re.I), "Slider"),
    (re.compile(r"(BENTI)", re.I), "Body"),
    (re.compile(r"(PCB)", re.I), "PCB"),
    (re.compile(r"GB_FASTENER_SCREWS[_\w]*|HSHCSM\d+X\d+", re.I), "Screw_HSHCS"),
    (re.compile(r"GB_WOOD_SCREWS[_\w]*|WOOD_SCREWS", re.I), "Screw_Wood"),
    (re.compile(r"CROSSRECESSED[_\w]*|CrossRecessed", re.I), "Screw_Cross"),
    (re.compile(r"(ISO4762M\d+x\d+)", re.I), None),  # keep captured group
    (re.compile(r"(AirTAC\w*)", re.I), None),
    (re.compile(r"(Panasonic\w*)", re.I), None),
    (re.compile(r"(SMCDM\w+|SMCAS\w+|SMCD\w+)", re.I), None),
    (re.compile(r"(SY\d+[\w]*)", re.I), None),
    (re.compile(r"(BZ\d+AirTAC\w*|BZ\d+\w*)", re.I), None),
    (re.compile(r"(HGH\d+\w*)", re.I), None),
    (re.compile(r"(DJA\d+\w*)", re.I), None),
    (re.compile(r"(SLCM\d+\w*|CopyofSLCM\w*)", re.I), None),
    (re.compile(r"(EWAC\w*)", re.I), None),
    (re.compile(r"(JCRB\w*)", re.I), None),
    (re.compile(r"(PMY\d+\w*)", re.I), None),
    (re.compile(r"(YWMD\w*)", re.I), None),
    (re.compile(r"(LPL\d+\w*)", re.I), None),
    (re.compile(r"(HSV\d+\w*)", re.I), None),
    (re.compile(r"(GAFC\w*)", re.I), None),
    (re.compile(r"(SGJ\w*)", re.I), None),
    (re.compile(r"(CX\d+\w*)", re.I), None),
    (re.compile(r"(BS\d+\w*)", re.I), None),
    (re.compile(r"(ZKN77\w*)", re.I), None),
    (re.compile(r"(MS\d+)", re.I), None),
]

STATION_CODE_RE = re.compile(r"^tn__(\d{6})")
SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9_]+")
TINY_DIAGONAL_M = 0.008  # 8 mm
DEFAULT_MASS_KG = 500.0


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
        print(f"[ces] AppLauncher unavailable ({exc}); trying SimulationApp")
    from isaacsim import SimulationApp

    _simulation_app = SimulationApp({"headless": headless})


def _close_app():
    global _simulation_app
    if _simulation_app is not None:
        _simulation_app.close()
        _simulation_app = None


def _parse_args():
    p = argparse.ArgumentParser(description="Rename CES machine and author physics")
    p.add_argument("--src", default=_DEFAULT_SRC)
    p.add_argument("--dst", default=_DEFAULT_DST)
    p.add_argument("--mass", type=float, default=DEFAULT_MASS_KG)
    p.add_argument(
        "--tiny-diagonal",
        type=float,
        default=TINY_DIAGONAL_M,
        help="Skip colliders whose AABB diagonal is smaller than this (metres)",
    )
    p.add_argument(
        "--no-skip-tiny",
        action="store_true",
        help="Author colliders on every mesh, including fasteners",
    )
    p.add_argument(
        "--flatten-instances",
        action="store_true",
        help="Force SetInstanceable(False) on all instanceable prims before physics",
    )
    return p.parse_args()


def _strip_tn(name: str) -> str:
    return name[4:] if name.startswith("tn__") else name


def _sanitize(name: str, max_len: int = 48) -> str:
    name = SAFE_NAME_RE.sub("_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    if not name:
        name = "Part"
    if name[0].isdigit():
        name = "N_" + name
    return name[:max_len]


def _extract_readable(name: str) -> str | None:
    raw = _strip_tn(name)
    # solidN / Mesh-like CAD tessellation leaves
    m = re.search(r"solid(\d+)$", raw, re.I)
    if m:
        return f"Solid_{int(m.group(1))}"
    m = re.match(r"(M\d{1,2})(?:_|$)", raw, re.I)
    if m and len(raw) < 80:
        return _sanitize(m.group(1))
    m = re.match(r"(\d+mm)", raw, re.I)
    if m:
        return _sanitize(m.group(1))
    for pattern, replacement in TOKEN_PATTERNS:
        m = pattern.search(raw)
        if not m:
            continue
        if replacement is not None:
            return _sanitize(replacement)
        return _sanitize(m.group(1))
    # Station-like numeric prefix with trailing short code
    m = re.match(r"(\d{6,})([A-Za-z]{1,6})?", raw)
    if m and len(raw) < 40:
        return _sanitize(raw.split("_")[0])
    # Leading short index like "1", "2" under a part → Geom_N
    m = re.match(r"^(\d{1,3})(?:_|$)", raw)
    if m:
        return f"Geom_{int(m.group(1)):02d}"
    # Short-ish CAD names without huge hash tails
    head = raw.split("_")[0]
    if head.upper() in {"NONE", "NULL", "N", "X", "Y", "Z"}:
        return None
    if 3 <= len(head) <= 32 and re.search(r"[A-Za-z]", head):
        if not re.fullmatch(r"[A-Za-z0-9]{10,}", head) or re.search(r"\d{3,}", head):
            if re.search(r"[A-Za-z]{2,}", head):
                return _sanitize(head)
    return None


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


def _match_rule(name: str, rules) -> str | None:
    for pred, new_name in rules:
        try:
            if pred(name):
                return new_name
        except Exception:  # noqa: BLE001
            continue
    return None


_SKIP_RENAME_TYPES = {
    "Material",
    "Shader",
    "MaterialBindingAPI",
    "Scope",
}


def _should_rename_prim(prim, UsdGeom) -> bool:
    name = prim.GetName()
    if name in ("Looks", "Materials", "Shader", "PreviewSurface", "Diffuse"):
        return False
    if str(prim.GetPath()).startswith("/__Prototype"):
        return False
    t = prim.GetTypeName()
    if t in _SKIP_RENAME_TYPES:
        return False
    # Rename Xforms and Meshes; skip material binding scopes under parts.
    if prim.IsA(UsdGeom.Xform) or prim.IsA(UsdGeom.Mesh) or t in ("", "Xform"):
        return True
    return False


def _flatten_instances(stage) -> int:
    """Expand point-instanced CAD leaves so meshes become editable prims."""
    count = 0
    # Collect first — mutating during Traverse is unsafe.
    targets = [p for p in stage.Traverse() if p.IsInstanceable()]
    for prim in targets:
        prim.SetInstanceable(False)
        count += 1
    print(f"[ces] flattened instanceable prims: {count}")
    return count


def _plan_renames(stage, Usd, UsdGeom):
    """Return list of (old_path, new_name) deepest-first, plus top-level report.

    Call after ``_flatten_instances`` so geometry lives in the main tree and
    prototype authoring restrictions no longer apply.
    """
    asm_path = "/CESmachine/CESmachine"
    asm = stage.GetPrimAtPath(asm_path)
    if not asm:
        raise RuntimeError(f"missing assembly root {asm_path}")

    renames = []  # (Sdf.Path, new_name)
    report_top = []
    valve_idx = 0

    top_used: set[str] = set()
    top_map = {}
    for child in asm.GetChildren():
        old = child.GetName()
        mapped = _match_rule(old, TOP_MODULE_RULES)
        if mapped == "Valve_Block":
            valve_idx += 1
            mapped = f"Valve_Block_{chr(ord('A') + valve_idx - 1)}"
        if mapped is None:
            mapped = _extract_readable(old) or f"Module_{len(report_top)+1:02d}"
        new = _unique_name(mapped, top_used)
        renames.append((child.GetPath(), new))
        report_top.append((old, new))
        top_map[old] = new

    def walk(prim, special_rules=None):
        used: set[str] = set()
        children = [c for c in prim.GetChildren() if _should_rename_prim(c, UsdGeom)]
        # Still recurse into Looks? No — skip those branches entirely.
        other_children = [
            c
            for c in prim.GetChildren()
            if c.GetName() not in ("Looks", "Materials")
            and c.GetTypeName() not in _SKIP_RENAME_TYPES
        ]
        planned = []
        for idx, child in enumerate(children):
            old = child.GetName()
            desired = None
            if special_rules is not None:
                desired = _match_rule(old, special_rules)
            if desired is None:
                desired = _extract_readable(old)
            if desired is None:
                code = STATION_CODE_RE.match(old)
                if code:
                    desired = f"Asm_{code.group(1)}"
                elif child.IsA(UsdGeom.Mesh):
                    desired = f"Mesh_{idx+1:02d}"
                else:
                    desired = f"Part_{idx+1:03d}"
            new = _unique_name(desired, used)
            planned.append((child, new))

        for child, new in planned:
            if child.GetName() != new:
                renames.append((child.GetPath(), new))
            child_rules = None
            if new == "Stage_XYZR" or "030000XYZR" in child.GetName():
                child_rules = STAGE_XYZR_CHILD_RULES
            elif new == "Stage_05_XYZ" or child.GetName().startswith("tn__050000"):
                child_rules = STAGE_05_CHILD_RULES
            walk(child, child_rules)

        # Recurse into non-renamed structural children (e.g. already-short names)
        planned_paths = {c.GetPath() for c, _ in planned}
        for child in other_children:
            if child.GetPath() in planned_paths:
                continue
            if not _should_rename_prim(child, UsdGeom):
                continue
            walk(child, None)

    for child in asm.GetChildren():
        mapped = top_map[child.GetName()]
        special = None
        if mapped == "Stage_XYZR":
            special = STAGE_XYZR_CHILD_RULES
        elif mapped == "Stage_05_XYZ":
            special = STAGE_05_CHILD_RULES
        walk(child, special)

    renames.sort(key=lambda item: item[0].pathElementCount, reverse=True)
    seen_paths = set()
    unique = []
    for path, name in renames:
        key = str(path)
        if key in seen_paths or path.name == name:
            continue
        seen_paths.add(key)
        unique.append((path, name))
    return unique, report_top


def _store_cad_name(prim, old_name: str):
    try:
        data = prim.GetCustomData()
        data = dict(data) if data else {}
        data["cadName"] = old_name
        prim.SetCustomData(data)
    except Exception:  # noqa: BLE001
        # Prototypes / locked specs — ignore.
        pass


def _apply_renames(stage, renames, Sdf):
    layer = stage.GetRootLayer()
    for path, new_name in renames:
        prim = stage.GetPrimAtPath(path)
        if prim and prim.IsValid() and prim.GetName() != new_name:
            _store_cad_name(prim, prim.GetName())

    batch = Sdf.BatchNamespaceEdit()
    for path, new_name in renames:
        batch.Add(Sdf.NamespaceEdit.Rename(path, new_name))
    ok = layer.Apply(batch)
    if ok:
        print(f"[ces] BatchNamespaceEdit applied {len(renames)} renames")
        return len(renames), 0

    print("[ces] batch rename failed; falling back to sequential deepest-first")
    applied = 0
    failed = 0
    for path, new_name in renames:
        prim = stage.GetPrimAtPath(path)
        if not prim or not prim.IsValid():
            failed += 1
            continue
        if prim.GetName() == new_name:
            continue
        # Avoid colliding with an existing sibling.
        parent = prim.GetParent()
        if parent and parent.GetChild(new_name):
            used = {c.GetName() for c in parent.GetChildren()}
            new_name = _unique_name(new_name, used)
        edit = Sdf.BatchNamespaceEdit()
        edit.Add(Sdf.NamespaceEdit.Rename(prim.GetPath(), new_name))
        if layer.Apply(edit):
            applied += 1
        else:
            failed += 1
            if failed <= 10:
                print(f"[ces] rename FAILED: {prim.GetPath()} -> {new_name}")
    return applied, failed


def _cleanup_tn_names(stage, Sdf, UsdGeom) -> tuple[int, int]:
    """Second pass: rename any remaining ``tn__*`` prims under the assembly."""
    layer = stage.GetRootLayer()
    applied = 0
    failed = 0
    asm_prefix = "/CESmachine/CESmachine/"

    # Iterate until stable — ancestor renames invalidate deeper cached paths.
    for _round in range(6):
        targets = [
            p
            for p in stage.Traverse()
            if p.GetName().startswith("tn__") and str(p.GetPath()).startswith(asm_prefix)
        ]
        if not targets:
            break
        targets.sort(key=lambda p: p.GetPath().pathElementCount, reverse=True)
        round_applied = 0
        for prim in targets:
            prim = stage.GetPrimAtPath(prim.GetPath())
            if not prim or not prim.IsValid() or not prim.GetName().startswith("tn__"):
                continue
            parent = prim.GetParent()
            if not parent:
                continue
            used = {c.GetName() for c in parent.GetChildren()}
            used.discard(prim.GetName())
            desired = _extract_readable(prim.GetName())
            if desired is None:
                desired = "Mesh" if prim.IsA(UsdGeom.Mesh) else "Part"
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
        if round_applied == 0:
            break
    print(f"[ces] tn__ cleanup applied={applied} failed={failed}")
    return applied, failed


def _mesh_diagonal(mesh_prim, UsdGeom, bbox_cache) -> float:
    try:
        rng = bbox_cache.ComputeWorldBound(mesh_prim).ComputeAlignedRange()
        if rng.IsEmpty():
            return 0.0
        mn, mx = rng.GetMin(), rng.GetMax()
        dx, dy, dz = mx[0] - mn[0], mx[1] - mn[1], mx[2] - mn[2]
        return (dx * dx + dy * dy + dz * dz) ** 0.5
    except Exception:  # noqa: BLE001
        return 0.0


def _author_physics(stage, args, Usd, UsdGeom, UsdPhysics, PhysxSchema):
    root = stage.GetPrimAtPath("/CESmachine")
    if not root:
        raise RuntimeError("missing /CESmachine")

    rb = UsdPhysics.RigidBodyAPI.Apply(root)
    rb.CreateRigidBodyEnabledAttr(True)
    rb.CreateKinematicEnabledAttr(True)
    mass_api = UsdPhysics.MassAPI.Apply(root)
    mass_api.CreateMassAttr(float(args.mass))
    print(f"[ces] RigidBody kinematic on /CESmachine mass={args.mass}")

    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
    )

    meshes = [p for p in stage.Traverse() if p.IsA(UsdGeom.Mesh)]
    authored = 0
    skipped_tiny = 0
    for mesh in meshes:
        diag = _mesh_diagonal(mesh, UsdGeom, bbox_cache)
        if not args.no_skip_tiny and diag < args.tiny_diagonal:
            skipped_tiny += 1
            continue
        try:
            UsdPhysics.CollisionAPI.Apply(mesh)
            UsdPhysics.CollisionAPI(mesh).CreateCollisionEnabledAttr(True)
            UsdPhysics.MeshCollisionAPI.Apply(mesh).CreateApproximationAttr("meshSimplification")
            try:
                pcol = PhysxSchema.PhysxCollisionAPI.Apply(mesh)
                pcol.CreateRestOffsetAttr(0.0)
                pcol.CreateContactOffsetAttr(0.002)
            except Exception:  # noqa: BLE001
                pass
            authored += 1
        except Exception as exc:  # noqa: BLE001
            print(f"[ces] collider skip {mesh.GetPath()}: {exc}")

    print(
        f"[ces] colliders authored={authored} skipped_tiny={skipped_tiny} "
        f"(threshold={args.tiny_diagonal} m) meshes_seen={len(meshes)}"
    )
    return authored, skipped_tiny


def _verify(stage, Usd, UsdGeom, UsdPhysics, report_top):
    print("[ces] === top-level rename map ===")
    for old, new in report_top:
        print(f"[ces]   {new:20s}  <=  {old[:80]}")

    asm = stage.GetPrimAtPath("/CESmachine/CESmachine")
    if asm:
        print("[ces] === children of /CESmachine/CESmachine ===")
        for c in asm.GetChildren():
            print(f"[ces]   {c.GetName()}")
            if c.GetName() in ("Stage_XYZR", "Stage_05_XYZ"):
                for gc in c.GetChildren():
                    print(f"[ces]     - {gc.GetName()}")

    n_rb = sum(1 for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI))
    n_col = sum(1 for p in stage.Traverse() if p.HasAPI(UsdPhysics.CollisionAPI))
    n_tn = sum(
        1
        for p in stage.Traverse()
        if p.GetName().startswith("tn__")
        and str(p.GetPath()).startswith("/CESmachine/CESmachine")
    )
    n_mesh = sum(1 for p in stage.Traverse() if p.IsA(UsdGeom.Mesh))
    print(f"[ces] rigidBodyAPIs={n_rb} collisionAPIs={n_col} meshes={n_mesh} tn__remaining={n_tn}")

    root = stage.GetPrimAtPath("/CESmachine")
    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
    )
    try:
        rng = bbox_cache.ComputeWorldBound(root).ComputeAlignedRange()
        mn, mx = rng.GetMin(), rng.GetMax()
        cx = 0.5 * (mn[0] + mx[0])
        cy = 0.5 * (mn[1] + mx[1])
        cz = 0.5 * (mn[2] + mx[2])
        print(
            f"[ces] bbox center=({cx:.6f},{cy:.6f},{cz:.6f}) "
            f"size=({mx[0]-mn[0]:.3f},{mx[1]-mn[1]:.3f},{mx[2]-mn[2]:.3f})"
        )
        if abs(cx) > 1e-3 or abs(cy) > 1e-3 or abs(cz) > 1e-3:
            print("[ces] WARNING: machine no longer centered at origin")
        else:
            print("[ces] center check OK")
    except Exception as exc:  # noqa: BLE001
        print(f"[ces] bbox check skipped: {exc}")


def main() -> int:
    args = _parse_args()
    src = os.path.abspath(args.src)
    dst = os.path.abspath(args.dst)
    if not os.path.isfile(src):
        print(f"[ces] source not found: {src}")
        return 1

    _boot_app(headless=True)
    from pxr import PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics

    print(f"[ces] copying {src} -> {dst}")
    shutil.copyfile(src, dst)
    stage = Usd.Stage.Open(dst)
    if stage is None:
        print(f"[ces] failed to open {dst}")
        return 1

    print(f"[ces] upAxis={UsdGeom.GetStageUpAxis(stage)} mpu={UsdGeom.GetStageMetersPerUnit(stage)}")

    # Expand CAD instances so meshes are real prims (needed for rename + physics).
    _flatten_instances(stage)

    renames, report_top = _plan_renames(stage, Usd, UsdGeom)
    print(f"[ces] planned renames: {len(renames)}")
    applied, failed = _apply_renames(stage, renames, Sdf)
    print(f"[ces] renames applied={applied} failed={failed}")

    # Best-effort cleanup of remaining tn__ leaves under the assembly tree.
    # Deep CAD solid leaves that are still composed from /CESmachine/Prototypes
    # may refuse NamespaceEdit; those keep their CAD hashes.
    _cleanup_tn_names(stage, Sdf, UsdGeom)

    authored, skipped = _author_physics(stage, args, Usd, UsdGeom, UsdPhysics, PhysxSchema)

    # Hide the internal CAD prototype library in the Stage window.
    for proto_name in ("Prototypes", "prototypes"):
        proto = stage.GetPrimAtPath(f"/CESmachine/{proto_name}")
        if proto and proto.IsValid():
            proto.SetActive(False)
            print(f"[ces] deactivated {proto.GetPath()} (internal CAD library)")

    stage.GetRootLayer().Save()
    print(f"[ces] wrote {dst}")

    _verify(stage, Usd, UsdGeom, UsdPhysics, report_top)
    _close_app()
    return 0 if authored > 0 else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        import traceback

        traceback.print_exc()
        _close_app()
        raise
