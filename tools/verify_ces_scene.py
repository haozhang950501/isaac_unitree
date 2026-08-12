# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""Camera-free acceptance checks for the CES product pick-and-place scene.

Verifies:

1. Product prim is found as the ``object`` rigid body and has non-zero mass.
2. Product settles without flying off the LoadingLine tray.
3. ``reset_scene_to_default`` restores Product near its authored pose.

Run::

    python tools/verify_ces_scene.py --device cuda:0
"""
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ["PROJECT_ROOT"] = project_root

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-Move-CES-Product-G129-Dex1-Wholebody")
parser.add_argument("--settle_steps", type=int, default=200)
AppLauncher.add_app_launcher_args(parser)
cli = parser.parse_args()
cli.headless = True

app_launcher = AppLauncher(cli)
simulation_app = app_launcher.app

import math

import torch

import gymnasium as gym
import tasks  # noqa: F401
from isaaclab.envs.mdp import reset_scene_to_default
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

from tasks.common_scene.base_scene_ces_pickplace_wholebody import PRODUCT_POS, PRODUCT_ROT

PASS, FAIL = "PASS", "FAIL"
results = []


def check(label: str, ok: bool, detail: str):
    results.append((PASS if ok else FAIL, label, detail))
    print(f"[verify] {PASS if ok else FAIL}  {label}: {detail}")


def quat_angle_deg(q_a: torch.Tensor, q_b: torch.Tensor) -> float:
    dot = float(torch.abs((q_a * q_b).sum()))
    return math.degrees(2.0 * math.acos(min(1.0, dot)))


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

print(f"\n[verify] task = {cli.task}")
mass = float(obj.root_physx_view.get_masses()[0].sum())
print(f"[verify] object mass = {mass:.4f} kg")
check("mass", mass > 0.05, f"{mass:.4f} kg (expect ~0.25)")

pos0 = obj.data.root_pos_w.clone()
quat0 = obj.data.root_quat_w.clone()
print(f"[verify] object initial pos={pos0[0].cpu().numpy().tolist()} quat={quat0[0].cpu().numpy().tolist()}")

target = torch.tensor(PRODUCT_POS, device=pos0.device, dtype=pos0.dtype)
err0 = float(torch.norm(pos0[0] - target)) * 1000.0
check("spawn_pose", err0 < 30.0, f"Product vs authored {err0:.1f} mm (expect <30)")

with torch.inference_mode():
    for _ in range(cli.settle_steps):
        robot.set_joint_position_target(robot.data.default_joint_pos)
        env.scene.write_data_to_sim()
        env.sim.step(render=False)
        env.scene.update(dt=env.physics_dt)

pos1 = obj.data.root_pos_w.clone()
quat1 = obj.data.root_quat_w.clone()
drift_mm = float(torch.norm(pos1 - pos0)) * 1000.0
ang = quat_angle_deg(quat0[0], quat1[0])
check("settle", drift_mm < 20.0 and ang < 15.0, f"drift={drift_mm:.1f} mm ang={ang:.1f} deg")

with torch.inference_mode():
    reset_scene_to_default(env, torch.arange(env.num_envs, device=env.device))
    env.scene.write_data_to_sim()
    env.sim.step(render=False)
    env.scene.update(dt=env.physics_dt)

pos2 = obj.data.root_pos_w.clone()
quat2 = obj.data.root_quat_w.clone()
reset_err = float(torch.norm(pos2[0] - target)) * 1000.0
target_q = torch.tensor(PRODUCT_ROT, device=quat2.device, dtype=quat2.dtype)
reset_ang = quat_angle_deg(quat2[0], target_q)
check("reset", reset_err < 30.0 and reset_ang < 20.0, f"err={reset_err:.1f} mm ang={reset_ang:.1f} deg")

print("\n[verify] summary")
n_fail = sum(1 for s, _, _ in results if s == FAIL)
for s, label, detail in results:
    print(f"  {s}  {label}: {detail}")
print(f"[verify] {'ALL PASS' if n_fail == 0 else f'{n_fail} FAILED'}")

env.close()
simulation_app.close()
sys.exit(0 if n_fail == 0 else 1)
