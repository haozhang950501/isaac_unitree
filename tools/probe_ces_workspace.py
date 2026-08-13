# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""Measure the right-arm TCP reachable envelope in the pelvis frame.

The base is pinned so the map is independent of walking.  ArmDiffIK (with TCP
offset) drives a grid of targets; residual position error is printed as a
sweet-spot map for CES stand-pose planning::

    stand_xy = target_xy - (x_b * forward + y_b * left)

Run::

    python tools/probe_ces_workspace.py --device cuda:0
    python tools/probe_ces_workspace.py --device cuda:0 --quick
"""
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ["PROJECT_ROOT"] = project_root

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-Move-CES-Product-G129-Dex1-Wholebody")
parser.add_argument("--quick", action="store_true", help="5x3 grid, fewer IK steps")
parser.add_argument("--ik_steps", type=int, default=80)
parser.add_argument("--z_b", type=float, default=0.066, help="pelvis-frame TCP height (pick)")
AppLauncher.add_app_launcher_args(parser)
cli = parser.parse_args()
cli.headless = True

app_launcher = AppLauncher(cli)
simulation_app = app_launcher.app

import torch

import gymnasium as gym
import tasks  # noqa: F401
from isaaclab.utils.math import quat_apply
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

from action_provider.manip_common import ArmDiffIK

TCP_LOCAL = (0.0, 0.115, 0.0)
RIGHT_ARM = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]
EE_BODY = "right_hand_base_link"


env_cfg = parse_env_cfg(cli.task, device=cli.device, num_envs=1)
for cam in ("front_camera", "left_wrist_camera", "right_wrist_camera", "robot_camera", "world_camera"):
    if hasattr(env_cfg.scene, cam):
        setattr(env_cfg.scene, cam, None)
if hasattr(env_cfg.observations.policy, "camera_image"):
    env_cfg.observations.policy.camera_image = None

env = gym.make(cli.task, cfg=env_cfg).unwrapped
env.sim.reset()
env.reset()

robot = env.scene["robot"]
device = env.device
ik = ArmDiffIK(
    robot,
    RIGHT_ARM,
    EE_BODY,
    device,
    tcp_offset=TCP_LOCAL,
    w_pos=1.0,
    w_rot=0.3,
    max_iters=6,
)

# pin wherever the robot currently stands
pin_pose = robot.data.root_state_w[:, 0:7].clone()
pin_vel = torch.zeros(1, 6, device=device)
right_ids = ik.joint_ids
right_default = robot.data.default_joint_pos[:, right_ids].clone()

if cli.quick:
    xs = [0.32, 0.38, 0.44, 0.50]
    ys = [-0.35, -0.25, -0.15]
    n_steps = min(cli.ik_steps, 40)
else:
    xs = [0.28, 0.34, 0.40, 0.46, 0.52]
    ys = [-0.40, -0.32, -0.24, -0.16, -0.08]
    n_steps = cli.ik_steps


def reach(x_b: float, y_b: float, z_b: float, quat=None) -> tuple[float, int]:
    target_b = torch.tensor([[x_b, y_b, z_b]], device=device)
    target_w = pin_pose[:, 0:3] + quat_apply(pin_pose[:, 3:7], target_b)
    robot.write_joint_state_to_sim(right_default, torch.zeros_like(right_default), joint_ids=right_ids)
    for _ in range(n_steps):
        robot.write_root_pose_to_sim(pin_pose)
        robot.write_root_velocity_to_sim(pin_vel)
        q_des = ik.solve(target_w, quat)
        full = robot.data.joint_pos.clone()
        full[:, right_ids] = q_des
        robot.set_joint_position_target(full)
        env.scene.write_data_to_sim()
        env.sim.step(render=False)
        env.scene.update(dt=env.physics_dt)
    err = ik.position_error_norm(target_w)
    nlim = ik.joints_at_limit()
    return err, nlim


print(f"\n[ws] pelvis-frame TCP grid, z_b={cli.z_b:.3f} m, ik_steps={n_steps}")
print("[ws] error in mm; * = joint at travel limit")
print("     y\\x  " + "  ".join(f"{x:+.2f}" for x in xs))
best = None
grid = {}
with torch.inference_mode():
    for y in ys:
        row = []
        for x in xs:
            err, nlim = reach(x, y, cli.z_b, None)
            grid[(x, y)] = (err, nlim)
            flag = "*" if nlim else " "
            row.append(f"{err*1000:5.1f}{flag}")
            if best is None or err < best[0]:
                best = (err, x, y, nlim)
        print(f"    {y:+.2f}  " + "  ".join(row))

print(
    f"\n[ws] best position-only: x_b={best[1]:+.2f} y_b={best[2]:+.2f} "
    f"err={best[0]*1000:.1f} mm  at_limit={best[3]}"
)

# 6-DoF top-down at the best cell (hand +Y down)
from isaaclab.utils.math import quat_from_matrix

x_axis = torch.tensor([[1.0, 0.0, 0.0]], device=device)
y_axis = torch.tensor([[0.0, 0.0, -1.0]], device=device)
z_axis = torch.cross(x_axis, y_axis, dim=-1)
rot = torch.stack((x_axis[0], y_axis[0], z_axis[0]), dim=-1).unsqueeze(0)
# columns = hand axes in world, but target is pelvis-aligned world after pin.
# pin yaw may not be identity; build the quat in WORLD by rotating pelvis axes.
fwd = quat_apply(pin_pose[:, 3:7], torch.tensor([[1.0, 0.0, 0.0]], device=device))
left = quat_apply(pin_pose[:, 3:7], torch.tensor([[0.0, 1.0, 0.0]], device=device))
down = torch.tensor([[0.0, 0.0, -1.0]], device=device)
# jaw along pelvis -Y (to the right): -left
jaw = -left
jaw[:, 2] = 0
jaw = jaw / torch.clamp(jaw.norm(dim=-1, keepdim=True), min=1e-6)
z_h = torch.cross(jaw, down, dim=-1)
z_h = z_h / torch.clamp(z_h.norm(dim=-1, keepdim=True), min=1e-6)
y_h = torch.cross(z_h, jaw, dim=-1)
rot_w = torch.stack((jaw[0], y_h[0], z_h[0]), dim=-1).unsqueeze(0)
grasp_q = quat_from_matrix(rot_w)

with torch.inference_mode():
    err6, nlim6 = reach(best[1], best[2], cli.z_b, grasp_q)
print(f"[ws] best cell with top-down quat: err={err6*1000:.1f} mm  at_limit={nlim6}")

# approach / lift at the same XY
with torch.inference_mode():
    for dz, label in ((0.10, "approach +0.10"), (0.15, "lift +0.15"), (-0.20, "place ~table")):
        err_z, nlim_z = reach(best[1], best[2], cli.z_b + dz, None)
        flag = " AT LIMIT" if nlim_z else ""
        print(f"[ws]   {label}: error {err_z*1000:5.1f} mm{flag}")

print(
    "\n[ws] stand_xy = target_xy - (x_b * forward + y_b * left)  "
    f"with x_b={best[1]:.2f}, y_b={best[2]:.2f}"
)
print("[ws] done")
env.close()
simulation_app.close()
sys.exit(0)
