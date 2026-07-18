
# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""Open the tray USD standalone and dump per-mesh LOCAL bounding boxes to a file.

These give us the handle offsets relative to the tray origin (the USD geometry
is baked in the tray local frame; the world placement is applied by the task
config at runtime). Run:

    python tools/inspect_tray_geom.py
"""
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ["PROJECT_ROOT"] = project_root

from isaaclab.app import AppLauncher
app_launcher = AppLauncher({"headless": True, "enable_cameras": False})
simulation_app = app_launcher.app

import numpy as np
from pxr import Usd, UsdGeom

out_path = "/tmp/tray_handles.txt"
lines = []
def w(s):
    lines.append(str(s))

usd_path = os.path.join(project_root, "assets/bozhon/tray_fixture_isaac45_dynamic.usd")
w(f"opening {usd_path}")
stage = Usd.Stage.Open(usd_path)
bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])

meshes = []
for prim in stage.Traverse():
    if prim.IsA(UsdGeom.Mesh):
        b = bbox_cache.ComputeWorldBound(prim)
        r = b.ComputeAlignedRange()
        mn, mx = r.GetMin(), r.GetMax()
        c = ((mn[0]+mx[0])/2, (mn[1]+mx[1])/2, (mn[2]+mx[2])/2)
        s = (mx[0]-mn[0], mx[1]-mn[1], mx[2]-mn[2])
        meshes.append((prim.GetName(), c, s, (tuple(mn), tuple(mx))))

amn = np.array([1e9]*3); amx = -amn
for _, _, _, (mn, mx) in meshes:
    amn = np.minimum(amn, mn); amx = np.maximum(amx, mx)
w(f"{len(meshes)} meshes")
w(f"overall min={amn.tolist()}")
w(f"overall max={amx.tolist()}")
w(f"overall size={(amx-amn).tolist()} center={((amn+amx)/2).tolist()}")

def dump(title, key, rev):
    w(f"\n== {title} ==")
    for name, c, s, _ in sorted(meshes, key=key, reverse=rev)[:10]:
        w(f"  {name:16s} center=({c[0]:+.4f},{c[1]:+.4f},{c[2]:+.4f}) size=({s[0]:.4f},{s[1]:.4f},{s[2]:.4f})")

dump("max +X", lambda m: m[1][0], True)
dump("min -X", lambda m: m[1][0], False)
dump("max +Y", lambda m: m[1][1], True)
dump("min -Y", lambda m: m[1][1], False)
dump("max +Z (tops / handles)", lambda m: m[1][2], True)

with open(out_path, "w") as f:
    f.write("\n".join(lines) + "\n")
print("WROTE", out_path)
simulation_app.close()
