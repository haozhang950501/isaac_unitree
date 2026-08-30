# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""G1 29DoF + Dex1 CES Wholebody 环境配置。

机器人使用右臂从 LoadingLine 托盘抓取 Product，持物行走到相邻包装桌，
再把产品释放到灰筐。该文件只配置 Isaac Lab 场景、MDP 和仿真参数，
实际动作顺序由 ``action_provider/ces_grasp`` 状态机负责。
"""
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
# 场景定义
##


@configclass
class ObjectTableSceneCfg(TableCESSceneCfgWH):
    """在 CES 基础场景上增加 G1、四路相机和全身接触传感器。"""

    # 机器人直接生成在抓取站；场景整体旋转后朝世界 -X 面向 CES。
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
# MDP 配置
##


@configclass
class ActionsCfg:
    """对整台机器人使用带默认偏置的直接关节位置控制。"""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=1.0, use_default_offset=True
    )


@configclass
class ObservationsCfg:
    """策略观测分组；保持原项目三类观测独立返回，不拼接 Tensor。"""

    @configclass
    class PolicyCfg(ObsGroup):
        """G1 关节、Dex1 夹爪与相机图像组成的策略观测组。"""
        robot_joint_state = ObsTerm(func=mdp.get_robot_boy_joint_states)
        robot_gipper_state = ObsTerm(func=mdp.get_robot_gipper_joint_states)
        camera_image = ObsTerm(func=mdp.get_camera_image)

        def __post_init__(self) -> None:
            """关闭观测扰动，并保留各观测项原有结构。"""
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class TerminationsCfg:
    """Product 掉到仓库地面附近时结束 episode。"""

    object_dropped = DoneTerm(func=mdp.object_dropped)


@configclass
class RewardsCfg:
    """使用 CES 离散放置/掉落奖励，权重保持 1.0。"""
    reward = RewTerm(func=mdp.compute_reward, weight=1.0)


@configclass
class EventCfg:
    """启动时整理场景物理，reset 时恢复机器人和 Product 默认状态。"""

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
    """CES LoadingLine Product 取放任务的完整 Isaac Lab 环境配置。"""

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

    def __post_init__(self) -> None:
        """写入控制频率、PhysX、地面摩擦和手动 reset 事件。"""
        self.decimation = 4
        # 120 秒覆盖钉盆抓取、抬起、持物换站、放置和收臂的完整最坏时长。
        self.episode_length_s = 120.0

        self.sim.dt = 0.005
        self.scene.contact_forces.update_period = self.sim.dt
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625

        # 地面摩擦与 Move-Cylinder Wholebody 相同。产品抓取摩擦由启动事件
        # 单独写入；不能把地面调得过粘，否则落脚会被钉住并向后倾倒。
        self.sim.physics_material.static_friction = 1.0
        self.sim.physics_material.dynamic_friction = 1.0
        self.sim.physics_material.friction_combine_mode = "max"
        self.sim.physics_material.restitution_combine_mode = "max"

        self.event_manager = SimpleEventManager()

        # Product 位于紧凑托盘槽中，reset 必须恢复精确默认位姿，不能随机化。
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
