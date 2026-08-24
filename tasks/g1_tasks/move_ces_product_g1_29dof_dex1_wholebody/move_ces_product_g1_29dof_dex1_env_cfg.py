# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""G1 29-DoF + Dex1 whole-body environment: pick Product off the CES LoadingLine
tray with the right arm and place it on the adjacent packing table."""
import torch

import isaaclab.envs.mdp as base_mdp
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from . import mdp

from tasks.common_config import CameraPresets, G1RobotPresets  # isort: skip
from tasks.common_event.event_manager import SimpleEvent, SimpleEventManager
from tasks.common_scene.base_scene_ces_pickplace_wholebody import (  # isort: skip
    ROBOT_INIT_POS,
    ROBOT_INIT_ROT,
    TableCESSceneCfgWH,
    ces_scene_startup,
)

##
# Scene definition
##


@configclass
class ObjectTableSceneCfg(TableCESSceneCfgWH):
    """Adds the robot, cameras and contact sensors to the CES product scene."""

    # Compact cluster, facing -X toward CES after the +180° world yaw.
    robot: ArticulationCfg = G1RobotPresets.g1_29dof_dex1_wholebody(
        init_pos=ROBOT_INIT_POS,
        init_rot=ROBOT_INIT_ROT,
    )

    contact_forces = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*", history_length=10, track_air_time=True, debug_vis=False
    )

    front_camera = CameraPresets.g1_front_camera()
    left_wrist_camera = CameraPresets.left_gripper_wrist_camera()
    right_wrist_camera = CameraPresets.right_gripper_wrist_camera()
    robot_camera = CameraPresets.g1_world_camera()


##
# MDP settings
##


@configclass
class ActionsCfg:
    """Direct joint position control over the whole articulation."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=1.0, use_default_offset=True
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        robot_joint_state = ObsTerm(func=mdp.get_robot_boy_joint_states)
        robot_gipper_state = ObsTerm(func=mdp.get_robot_gipper_joint_states)
        camera_image = ObsTerm(func=mdp.get_camera_image)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class TerminationsCfg:
    """Episode ends if Product falls onto the warehouse floor."""

    object_dropped = DoneTerm(func=mdp.object_dropped)


@configclass
class RewardsCfg:
    reward = RewTerm(func=mdp.compute_reward, weight=1.0)


@configclass
class EventCfg:
    """Startup: seat tote and grasp physics. Reset: restore robot + Product."""

    ces_scene_startup = EventTerm(
        func=ces_scene_startup,
        mode="startup",
    )
    reset_scene = EventTerm(
        func=mdp.reset_scene_to_default,
        mode="reset",
    )


@configclass
class MoveCESProductG129Dex1WholebodyEnvCfg(ManagerBasedRLEnvCfg):
    """Environment configuration for the CES LoadingLine Product pick-and-place task."""

    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(
        num_envs=1,
        env_spacing=4.0,
        replicate_physics=True,
    )

    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events = EventCfg()
    commands = None
    rewards: RewardsCfg = RewardsCfg()
    curriculum = None

    def __post_init__(self):
        """Post initialization."""
        self.decimation = 4
        # walk to CES → grasp → lift → walk to table → place → retract
        self.episode_length_s = 120.0

        self.sim.dt = 0.005
        self.scene.contact_forces.update_period = self.sim.dt
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625

        # Same floor friction as Move-Cylinder wholebody.  Product grasp friction
        # is set on the part itself in ces_scene_startup — do not make the floor sticky
        # or the gait plants a foot and pitches over backwards.
        self.sim.physics_material.static_friction = 1.0
        self.sim.physics_material.dynamic_friction = 1.0
        self.sim.physics_material.friction_combine_mode = "max"
        self.sim.physics_material.restitution_combine_mode = "max"

        self.event_manager = SimpleEventManager()

        # Product sits in a tight tray pocket — restore exact default pose.
        self.event_manager.register(
            "reset_object_self",
            SimpleEvent(
                func=lambda env: base_mdp.reset_root_state_uniform(
                    env,
                    torch.arange(env.num_envs, device=env.device),
                    pose_range={},
                    velocity_range={},
                    asset_cfg=SceneEntityCfg("object"),
                )
            ),
        )

        self.event_manager.register(
            "reset_all_self",
            SimpleEvent(
                func=lambda env: base_mdp.reset_scene_to_default(
                    env, torch.arange(env.num_envs, device=env.device)
                )
            ),
        )
