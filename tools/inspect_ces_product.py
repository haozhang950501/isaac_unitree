# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""Measure Product / tray / table AABBs and pelvis-relative grasp height.

Run::

    python tools/inspect_ces_product.py --device cuda:0
"""
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ["PROJECT_ROOT"] = project_root

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-Move-CES-Product-G129-Dex1-Wholebody")
parser.add_argument("--settle_steps", type=int, default=80)
AppLauncher.add_app_launcher_args(parser)
cli = parser.parse_args()
cli.headless = True

app_launcher = AppLauncher(cli)
simulation_app = app_launcher.app

import math

import torch

import gymnasium as gym
import tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

from tasks.common_scene.base_scene_ces_pickplace_wholebody import (
    PRODUCT_POS,
    ROBOT_INIT_POS,
    TABLE_SPAWN_POS,
    TABLE_TOP_Z,
)

STAND_PELVIS_Z = 0.755
TCP_LOCAL = (0.0, 0.115, 0.0)


def _aabb(prim, bbox_cache):
    rng = bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange()
    if rng.IsEmpty():
        return None
    mn, mx = rng.GetMin(), rng.GetMax()
    return (
        (float(mn[0]), float(mn[1]), float(mn[2])),
        (float(mx[0]), float(mx[1]), float(mx[2])),
    )


def _print_aabb(label, box):
    if box is None:
        print(f"[inspect] {label}: EMPTY")
        return
    (x0, y0, z0), (x1, y1, z1) = box
    print(
        f"[inspect] {label}: "
        f"x[{x0:.4f},{x1:.4f}] y[{y0:.4f},{y1:.4f}] z[{z0:.4f},{z1:.4f}]  "
        f"size=({x1-x0:.4f},{y1-y0:.4f},{z1-z0:.4f}) m  "
        f"center=({0.5*(x0+x1):.4f},{0.5*(y0+y1):.4f},{0.5*(z0+z1):.4f})"
    )


env_cfg = parse_env_cfg(cli.task, device=cli.device, num_envs=1)
for cam in ("front_camera", "left_wrist_camera", "right_wrist_camera", "robot_camera", "world_camera"):
    if hasattr(env_cfg.scene, cam):
        setattr(env_cfg.scene, cam, None)
if hasattr(env_cfg.observations.policy, "camera_image"):
    env_cfg.observations.policy.camera_image = None

env = gym.make(cli.task, cfg=env_cfg).unwrapped
env.sim.reset()
env.reset()

obj = env.scene["object"]
robot = env.scene["robot"]

with torch.inference_mode():
    for _ in range(cli.settle_steps):
        robot.set_joint_position_target(robot.data.default_joint_pos)
        env.scene.write_data_to_sim()
        env.sim.step(render=False)
        env.scene.update(dt=env.physics_dt)

print("\n[inspect] authored constants")
print(f"  PRODUCT_POS     {tuple(round(x, 4) for x in PRODUCT_POS)}")
print(f"  ROBOT_INIT_POS  {tuple(round(x, 4) for x in ROBOT_INIT_POS)}")
print(f"  TABLE_SPAWN_POS {tuple(round(x, 4) for x in TABLE_SPAWN_POS)}")
print(f"  TABLE_TOP_Z     {TABLE_TOP_Z:.4f}")

obj_pos = obj.data.root_pos_w[0].cpu()
obj_quat = obj.data.root_quat_w[0].cpu()
root_pos = robot.data.root_pos_w[0].cpu()
print("\n[inspect] live poses")
print(f"  Product  pos={obj_pos.tolist()} quat={obj_quat.tolist()}")
print(f"  Pelvis   pos={root_pos.tolist()}")
print(f"  Product z - pelvis z = {float(obj_pos[2] - root_pos[2]):.4f} m")
print(f"  Product z - STAND_PELVIS_Z = {float(obj_pos[2] - STAND_PELVIS_Z):.4f} m")
print(f"  Table top - pelvis z = {TABLE_TOP_Z - float(root_pos[2]):.4f} m")

import omni.usd
from pxr import Gf, Usd, UsdGeom

stage = omni.usd.get_context().get_stage()
bbox = UsdGeom.BBoxCache(0, [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy])


def _world_xf(prim):
    xf = UsdGeom.Xformable(prim)
    if not xf:
        return None, None
    mat = xf.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    t = mat.ExtractTranslation()
    r = mat.ExtractRotation().GetQuaternion()
    img, real = r.GetImaginary(), r.GetReal()
    pos = (float(t[0]), float(t[1]), float(t[2]))
    # Gf quaternion is (imaginary, real); Isaac uses (w, x, y, z)
    quat = (float(real), float(img[0]), float(img[1]), float(img[2]))
    return pos, quat


print("\n[inspect] prims named 'Product' under CESMachine")
ces_root = stage.GetPrimAtPath("/World/envs/env_0/CESMachine")
product_prims = []
if ces_root.IsValid():
    for p in Usd.PrimRange(ces_root):
        if p.GetName() == "Product":
            product_prims.append(p)
else:
    print("[inspect] CESMachine missing")

if not product_prims:
    print("[inspect] no prim named Product")

for p in product_prims:
    path = str(p.GetPath())
    pos, quat = _world_xf(p)
    box = _aabb(p, bbox)
    print(f"\n[inspect] {path}")
    print(f"  type={p.GetTypeName()}  specifier={p.GetSpecifier()}")
    if pos is not None:
        print(f"  world pos  = ({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f})")
        print(f"  world quat = ({quat[0]:.4f}, {quat[1]:.4f}, {quat[2]:.4f}, {quat[3]:.4f})")
    _print_aabb("  AABB", box)

# Also dump Root children so we can see Product vs LoadingLine siblings.
print("\n[inspect] CESMachine/Root children")
root = stage.GetPrimAtPath("/World/envs/env_0/CESMachine/Root")
if root.IsValid():
    for child in root.GetChildren():
        print(f"  - {child.GetName():20s}  {child.GetTypeName():12s}  {child.GetPath()}")

paths = {
    "Root_Product": "/World/envs/env_0/CESMachine/Root/Product",
    "Tray_Product": "/World/envs/env_0/CESMachine/Root/LoadingLine/Tray_Assembly_01/Product",
    "Tray_Assembly_01": "/World/envs/env_0/CESMachine/Root/LoadingLine/Tray_Assembly_01",
    "LoadingLine": "/World/envs/env_0/CESMachine/Root/LoadingLine",
    "CESMachine": "/World/envs/env_0/CESMachine",
    "PackingTable": "/World/envs/env_0/PackingTable",
    "HeavyDutyTable": (
        "/World/envs/env_0/PackingTable/PackingTable_2/SM_CratePacking_Table_A1/"
        "SM_HeavyDutyPackingTable_C02_01"
    ),
}

print("\n[inspect] world AABBs")
boxes = {}
for name, path in paths.items():
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        print(f"[inspect] {name}: missing {path}")
        continue
    box = _aabb(prim, bbox)
    boxes[name] = box
    _print_aabb(name, box)

# Prefer the Root/Product Xform the user selected in the stage tree.
prod_box = boxes.get("Root_Product") or boxes.get("Tray_Product")
if prod_box is not None:
    (x0, y0, z0), (x1, y1, z1) = prod_box
    sx, sy, sz = x1 - x0, y1 - y0, z1 - z0
    pinch_w = min(sx, sy)
    long_w = max(sx, sy)
    contact_gap = max(0.012, pinch_w - 0.004)
    q_closed = (0.050 - contact_gap) / 2.0
    q_open = -0.010
    print("\n[inspect] grasp hints (from preferred Product AABB)")
    print(f"  pinch width (min XY) = {pinch_w*1000:.1f} mm")
    print(f"  long axis   (max XY) = {long_w*1000:.1f} mm")
    print(f"  height               = {sz*1000:.1f} mm")
    print(f"  suggested gripper_open   = {q_open:.4f}  (gap ~{1000*(0.050-2*q_open):.0f} mm)")
    print(f"  suggested gripper_closed = {q_closed:.4f}  (gap ~{1000*contact_gap:.0f} mm)")
    print(f"  TCP_LOCAL                = {TCP_LOCAL}")
    cx, cy, cz = 0.5 * (x0 + x1), 0.5 * (y0 + y1), 0.5 * (z0 + z1)
    print(f"  AABB center              = ({cx:.4f}, {cy:.4f}, {cz:.4f})")
    z_b_pick = cz - STAND_PELVIS_Z
    z_b_place = float(TABLE_TOP_Z + 0.5 * sz - STAND_PELVIS_Z)
    print(f"  z_b pick  (AABB center vs stand pelvis) = {z_b_pick:.4f} m")
    print(f"  z_b place (table + half height)         = {z_b_place:.4f} m")

table_box = boxes.get("HeavyDutyTable") or boxes.get("PackingTable")
if table_box is not None:
    (x0, y0, z0), (x1, y1, z1) = table_box
    print("\n[inspect] place region (table XY, z around table top + product half-height)")
    print(f"  xy=[{x0:.3f},{x1:.3f}] x [{y0:.3f},{y1:.3f}]")
    print(f"  recommended place XY = ({0.5*(x0+x1):.3f}, {0.5*(y0+y1):.3f})")

print("\n[inspect] done")
env.close()
simulation_app.close()
sys.exit(0)
