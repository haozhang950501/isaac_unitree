
# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""Fast, camera-free harness to exercise the autonomous tray grasp provider.

It builds the real task scene (minus the RTX cameras, which are slow to warm up
and irrelevant for kinematics), drives :class:`TrayGraspActionProvider` for a
number of control cycles and logs the tray height, the robot base pose and the
per-hand IK tracking error so the behaviour can be tuned without the GUI.

    python tools/test_tray_grasp.py --steps 1800 --device cuda:0
"""
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ["PROJECT_ROOT"] = project_root

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-Move-Cylinder-G129-Dex1-Wholebody")
parser.add_argument("--steps", type=int, default=1800)
parser.add_argument("--log_every", type=int, default=25)
AppLauncher.add_app_launcher_args(parser)
cli = parser.parse_args()
cli.headless = True

app_launcher = AppLauncher(cli)
simulation_app = app_launcher.app

import torch
import types
import gymnasium as gym
import tasks
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from dds.dds_create import create_dds_objects
from action_provider.action_provider_tray_grasp import TrayGraspActionProvider

# ---- minimal args namespace the provider / DDS layer expect ---------------
args = types.SimpleNamespace(
    task=cli.task,
    robot_type="g129",
    enable_dex1_dds=True,
    enable_dex3_dds=False,
    enable_inspire_dds=False,
    enable_wholebody_dds=True,
    model_path="assets/model/policy.onnx",
    device=cli.device,
)

env_cfg = parse_env_cfg(cli.task, device=cli.device, num_envs=1)
# strip cameras (and the camera observation) so we don't need the RTX pipeline
for cam in ("front_camera", "left_wrist_camera", "right_wrist_camera",
            "robot_camera", "world_camera"):
    if hasattr(env_cfg.scene, cam):
        setattr(env_cfg.scene, cam, None)
if hasattr(env_cfg.observations.policy, "camera_image"):
    env_cfg.observations.policy.camera_image = None

env = gym.make(cli.task, cfg=env_cfg).unwrapped
env.sim.reset()
env.reset()

# register DDS objects (g129 / dex1 / run_command) required by the provider
create_dds_objects(args, env)

provider = TrayGraspActionProvider(env, args)

tray = env.scene["tray_fixture"]
robot = env.scene["robot"]
z0 = float(tray.data.root_pos_w[0, 2].item())
print(f"[test] initial tray height = {z0:.3f} m")

with torch.inference_mode():
    for i in range(cli.steps):
        provider.get_action(env)
        if i % cli.log_every == 0:
            tz = float(tray.data.root_pos_w[0, 2].item())
            bp = robot.data.root_pos_w[0].cpu().numpy()
            _, yaw = provider.get_base_pose_w()
            print(f"[test] step={i:4d} phase={provider.sm.phase.name:8s} "
                  f"base=({bp[0]:+.2f},{bp[1]:+.2f},{bp[2]:+.2f}) yaw={yaw:+.2f} "
                  f"tray_z={tz:.3f} (dz={tz - z0:+.3f})")
        if not simulation_app.is_running():
            break

tz = float(tray.data.root_pos_w[0, 2].item())
print(f"[test] FINAL tray height = {tz:.3f} m (lifted {tz - z0:+.3f} m), phase={provider.sm.phase.name}")
provider.cleanup()
simulation_app.close()
