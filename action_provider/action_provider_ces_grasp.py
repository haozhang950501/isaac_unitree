# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""CES Baseline 动作提供器：钉盆抓取、持物行走和钉盆放置。

该类把状态机命令转换为完整关节目标，并复用 Wholebody 策略生成腿部动作。
它同时维护策略观测/动作历史、手臂限速、Dex1 PD 接触夹持和 root pin。
每次控制调用仍推进四个 0.005 秒物理子步，保持原来的 50 Hz 控制契约。
"""
from __future__ import annotations

import logging

import torch

from action_provider.action_provider_wh_dds import DDSRLActionProvider
from action_provider.ces_grasp import constants as C
from action_provider.ces_grasp.state_machine import (
    CesCommand,
    CesPickPlacePhase,
    CesPickPlaceStateMachine,
)
from action_provider.ces_grasp.ik_solver import ArmDiffIK


logger = logging.getLogger("ces")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
if logger.level == logging.NOTSET:
    logger.setLevel(logging.INFO)
logger.propagate = False


class CESGraspActionProvider(DDSRLActionProvider):
    """把 CES 状态机、Wholebody 步态策略和 Isaac Lab 仿真串成单一控制器。"""

    def __init__(self, env, args_cli):
        """解析关节索引，配置 IK/Dex1，并初始化状态机与运行缓存。"""
        super().__init__(env, args_cli)
        self.name = "CESGrasp"
        self._decimation = C.CONTROL_DECIMATION
        self.dt = float(self._decimation * env.physics_dt)
        self.ik = ArmDiffIK(
            env.scene["robot"],
            C.RIGHT_ARM_JOINTS,
            C.EE_BODY,
            env.device,
            tcp_offset=C.TCP_LOCAL,
            max_delta_per_step=0.03,
            w_pos=1.0,
            w_rot=0.45,
            max_iters=8,
            pos_tol=0.005,
        )
        self.fsm = CesPickPlaceStateMachine(
            self,
            speed_scale=getattr(args_cli, "ces_pick_speed", None),
        )
        robot = env.scene["robot"]
        self._right_arm_idx = list(self.ik.joint_ids)
        left_ids, _ = robot.find_joints(C.LEFT_ARM_JOINTS, preserve_order=True)
        self._left_arm_idx = list(left_ids)
        rg_ids, _ = robot.find_joints(C.RIGHT_GRIPPER_JOINTS, preserve_order=True)
        lg_ids, _ = robot.find_joints(C.LEFT_GRIPPER_JOINTS, preserve_order=True)
        self._right_grip_idx = list(rg_ids)
        self._left_grip_idx = list(lg_ids)
        self._configure_dex1_pd(robot)
        self._lock_arm_idx = torch.tensor(
            self._right_arm_idx + self._left_arm_idx,
            dtype=torch.long,
            device=env.device,
        )
        self._right_arm_default = robot.data.default_joint_pos[:, self._right_arm_idx].clone()
        self._left_arm_default = robot.data.default_joint_pos[:, self._left_arm_idx].clone()
        self._leg_default = robot.data.default_joint_pos[:, self.action_to_indices].clone()
        self._reset_provider_runtime()
        logger.info(
            f"[CESGrasp] Baseline Smooth V1 carry-walk "
            f"pick_stand={tuple(round(x, 3) for x in C.PICK_STAND_XY)} "
            f"place_stand={tuple(round(x, 3) for x in C.PLACE_STAND_XY)} "
            f"(no TCP weld; Dex1 PD squeeze + pad friction)"
        )

    def _configure_dex1_pd(self, robot) -> None:
        """把已验证的 Dex1 PD 和力矩上限同时写入执行器与仿真关节。"""
        hands = robot.actuators.get("hands")
        if hands is not None:
            if torch.is_tensor(hands.stiffness):
                hands.stiffness[:] = C.DEX1_STIFFNESS
                hands.damping[:] = C.DEX1_DAMPING
            else:
                hands.stiffness = C.DEX1_STIFFNESS
                hands.damping = C.DEX1_DAMPING
            logger.info("[CESGrasp] Dex1 hand PD kp=1800 kd=30")

        grip_ids = torch.tensor(
            self._right_grip_idx,
            dtype=torch.long,
            device=self.env.device,
        )
        joint_count = len(self._right_grip_idx)
        shape = (1, joint_count)
        robot.write_joint_stiffness_to_sim(
            torch.full(shape, C.DEX1_STIFFNESS, device=self.env.device),
            joint_ids=grip_ids,
        )
        robot.write_joint_damping_to_sim(
            torch.full(shape, C.DEX1_DAMPING, device=self.env.device),
            joint_ids=grip_ids,
        )
        robot.write_joint_effort_limit_to_sim(
            torch.full(shape, C.DEX1_EFFORT_LIMIT, device=self.env.device),
            joint_ids=grip_ids,
        )

    def _reset_provider_runtime(self) -> None:
        """统一初始化动作提供器的可变状态，供构造和任务重置共同调用。"""
        self._walk_cmd = [0.0, 0.0, 0.0, C.WALK_HEIGHT]
        self._walk_prime: list[float] | None = None
        self._grip_cmd: float | None = None
        self._last_policy_legs = None
        self._was_walking = False
        self._q_right = self._right_arm_default[0].clone()
        self._squeeze = False
        self._err_logged = False
        self._err_t = -10.0

    def reset_task(self) -> None:
        """把状态机、手臂、夹爪和步态历史恢复到可重新抓取的初始状态。"""
        self.fsm.reset()
        self._reset_provider_runtime()
        logger.info("[CESGrasp] task reset")

    def reset_walk_filter(self) -> None:
        """释放骨盆前清零机体系步态滤波状态。"""
        self._walk_cmd = [0.0, 0.0, 0.0, C.WALK_HEIGHT]

    def prime_walk_filter(self, command) -> None:
        """预载步态滤波：释放骨盆的第一帧就直接给这个机体系指令，不从零 ramp。

        pick 结束后立刻满幅 S（-vx），避免等站稳再发时抱件前倾撞上 CES。
        """
        self._walk_prime = [float(command[index]) for index in range(4)]

    def sync_right_arm_target(self, arm_q: torch.Tensor) -> None:
        """在 Walk→Pin 交接时让手臂限速器从实时到站关节姿态继续。"""
        self._q_right = arm_q.clone()

    def _begin_walk_policy(self, command: list[float]) -> None:
        """清空站立历史，并给骨盆与夹持产品施加首帧反向速度。

        策略观测是 10 帧堆叠。Pick 阶段填入的全部是零速度，如果直接释放
        骨盆，第一次推理仍把机器人判断成站立，机器人会先朝 CES 收集一两步。
        清空观测和动作历史后，堆叠的每一帧都从反向命令开始；速度 kick 则
        防止持物质心在第一帧未钉盆时向前倒。
        """
        self.actor_obs_buffer.reset()
        self.action_buffer.reset()
        zeros = torch.zeros(
            self.num_envs,
            self.num_actions_all,
            dtype=torch.float,
            device=self.env.device,
        )
        self.action_buffer.compute(zeros)
        self._kick_walk_start_velocity(command[0])

    def _kick_walk_start_velocity(self, vx_body: float):
        """把机体系前向速度转换到世界系，并立即写给骨盆和夹持产品。"""
        robot = self.env.scene["robot"]
        yaw = self.get_heading()
        fwd, _left = C.forward_left(yaw)
        vx_w = float(vx_body) * fwd[0]
        vy_w = float(vx_body) * fwd[1]
        dtype = robot.data.root_pos_w.dtype
        vel = torch.zeros(1, 6, device=self.env.device, dtype=dtype)
        vel[0, 0] = vx_w
        vel[0, 1] = vy_w
        robot.write_root_velocity_to_sim(vel)
        if self._squeeze:
            obj = self.env.scene["object"]
            ov = torch.zeros(1, 6, device=self.env.device, dtype=obj.data.root_pos_w.dtype)
            ov[0, 0] = vx_w
            ov[0, 1] = vy_w
            obj.write_root_velocity_to_sim(ov)
        logger.info(
            f"[CESGrasp] walk start kick vx_b={vx_body:+.2f} "
            f"world=({vx_w:+.2f},{vy_w:+.2f}) (no forward collect)"
        )

    def _filter_walk_command(self, target) -> list[float]:
        """限制步态指令变化率，避免普通阶段切换产生速度阶跃。"""
        target = [float(target[i]) for i in range(4)]
        limits = (C.WALK_VX_ACCEL, C.WALK_VY_ACCEL, C.WALK_WZ_ACCEL)
        for i, rate in enumerate(limits):
            delta = target[i] - self._walk_cmd[i]
            max_delta = rate * self.dt
            self._walk_cmd[i] += max(-max_delta, min(max_delta, delta))
        self._walk_cmd[3] = target[3]
        return list(self._walk_cmd)

    # 以下方法组成 FSM 使用的运行上下文接口；状态机不直接读取仿真内部对象。
    @property
    def device(self):
        """向 FSM 暴露环境运行设备。"""
        return self.env.device

    def get_base_pose_w(self):
        """返回所有环境的机器人根节点世界位置和四元数。"""
        robot = self.env.scene["robot"]
        return robot.data.root_pos_w, robot.data.root_quat_w

    def get_object_pose_w(self):
        """返回所有环境的 Product 根节点世界位置和四元数。"""
        obj = self.env.scene["object"]
        return obj.data.root_pos_w, obj.data.root_quat_w

    def get_right_arm_q(self):
        """返回右臂七关节实时位置的副本，避免状态机原地修改仿真数据。"""
        robot = self.env.scene["robot"]
        return robot.data.joint_pos[:, self._right_arm_idx].clone()

    def _slew_arm(self, q_tgt: torch.Tensor) -> torch.Tensor:
        """限制右臂单控制周期关节变化量，夹持行走和钉盆阶段使用慢速档。"""
        # 夹持冻臂时慢跟；抬起走规划关节轨迹，不能限太死。
        slow = self._squeeze and self.fsm.phase in (
            CesPickPlacePhase.CARRY,
            CesPickPlacePhase.GOTO_PLACE,
            CesPickPlacePhase.PLACE_HOLD,
        )
        lim = C.ARM_SLEW_RAD_LIFT if slow else C.ARM_SLEW_RAD
        dq = q_tgt - self._q_right
        dq = torch.clamp(dq, -lim, lim)
        self._q_right = self._q_right + dq
        return self._q_right

    def get_product_aabb_center_w(self):
        """返回 Product 渲染/碰撞几何的世界 AABB 中心及根节点姿态。

        Product 的 USD 根节点 pivot 不在实际几何中心，抓取点必须以 AABB
        为准。若 USD prim 或包围盒不可用，则安全回退到刚体根节点位置。
        """
        import omni.usd
        from pxr import UsdGeom

        obj = self.env.scene["object"]
        _, quat = self.get_object_pose_w()
        path = obj.cfg.prim_path.replace("env_.*", "env_0")
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            return obj.data.root_pos_w.clone(), quat
        cache = UsdGeom.BBoxCache(
            0, [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy]
        )
        rng = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        if rng.IsEmpty():
            return obj.data.root_pos_w.clone(), quat
        mn, mx = rng.GetMin(), rng.GetMax()
        center = torch.tensor(
            [[0.5 * (mn[0] + mx[0]), 0.5 * (mn[1] + mx[1]), 0.5 * (mn[2] + mx[2])]],
            device=self.device,
            dtype=obj.data.root_pos_w.dtype,
        )
        return center, quat

    def get_heading(self) -> float:
        """返回第一个环境的骨盆世界偏航角。"""
        h = self.env.scene["robot"].data.heading_w
        return float(h[0].item()) if h.dim() else float(h.item())

    def stance_tilt(self) -> float:
        """用机体系投影重力的水平分量表示站立倾斜程度。"""
        g = self.env.scene["robot"].data.projected_gravity_b[0]
        return float(torch.sqrt(g[0] * g[0] + g[1] * g[1]).item())

    def is_standing(self) -> bool:
        """按现有倾斜、速度、高度阈值判断机器人是否稳定站立。"""
        robot = self.env.scene["robot"]
        tilt = self.stance_tilt()
        ang = robot.data.root_ang_vel_w[0]
        lin = robot.data.root_lin_vel_w[0]
        z = float(robot.data.root_pos_w[0, 2])
        yaw_rate = float(ang[2].abs().item())
        xy_speed = float(torch.sqrt(lin[0] * lin[0] + lin[1] * lin[1]).item())
        return (
            tilt < C.STAND_TILT_MAX
            and yaw_rate < C.STAND_YAW_RATE_MAX
            and xy_speed < C.STAND_XY_SPEED_MAX
            and 0.68 < z < 0.88
        )

    def _pin_root_pose(self, root_pin):
        """每个物理子步写回指定骨盆位姿并清零速度，不移动产品。"""
        robot = self.env.scene["robot"]
        pose = robot.data.root_state_w[:, 0:7].clone()
        position, quaternion = root_pin
        pose[0, :3] = pose.new_tensor(position)
        pose[0, 3:7] = pose.new_tensor(quaternion)
        vel = torch.zeros(1, 6, device=self.env.device, dtype=pose.dtype)
        robot.write_root_pose_to_sim(pose)
        robot.write_root_velocity_to_sim(vel)

    def _stop_held_product(self):
        """Walk 交接 root pin 时只刹停一次产品速度，不修改产品位姿。"""
        obj = self.env.scene["object"]
        vel = torch.zeros(1, 6, device=self.env.device, dtype=obj.data.root_pos_w.dtype)
        obj.write_root_velocity_to_sim(vel)

    def _write_locked_upper_body(self, full_action: torch.Tensor):
        """只用运动学方式锁定双臂；Dex1 继续走 PD 以产生真实垫面摩擦。

        如果把夹爪关节也直接写入状态，指垫会瞬移穿过产品，破坏接触夹持。
        """
        robot = self.env.scene["robot"]
        idx = self._lock_arm_idx
        pos = full_action.index_select(0, idx).unsqueeze(0)
        vel = torch.zeros_like(pos)
        robot.write_joint_state_to_sim(pos, vel, joint_ids=idx)

    def _resolve_walk_policy(self, command: CesCommand) -> torch.Tensor | None:
        """处理步态交接、滤波和观测历史，返回本帧策略腿部动作。"""
        walk_active = command.walk is not None
        if command.root_pin is not None and self._was_walking and self._squeeze:
            self._stop_held_product()
        self._was_walking = walk_active

        target = command.walk or (0.0, 0.0, 0.0, C.WALK_HEIGHT)
        if walk_active:
            # Pick→后退及转弯→前进必须直接跨过零速区，否则策略会停步。
            if self._walk_prime is not None:
                self._walk_cmd = list(self._walk_prime)
                self._walk_prime = None
                self._begin_walk_policy(self._walk_cmd)
            elif (
                self._walk_cmd[0] * target[0] < 0.0
                and abs(target[0]) >= C.WALK_POLICY_VX_DEADBAND
            ):
                self._walk_cmd = [float(target[index]) for index in range(4)]
            else:
                self._filter_walk_command(target)
            return self.run_policy(self._walk_cmd)

        if self._walk_prime is not None:
            self._walk_cmd = list(self._walk_prime)
        else:
            self.reset_walk_filter()
        # 钉盆期间策略不下发腿部动作，但仍推进观测堆叠，保持历史时间一致。
        self.compute_observations(self._walk_cmd)
        return None

    def _right_arm_target(
        self,
        command: CesCommand,
        action_dtype: torch.dtype,
    ) -> torch.Tensor:
        """把关节轨迹或笛卡尔命令转换为经过限速的右臂关节目标。"""
        if command.arm_q is not None:
            if self.fsm.phase is CesPickPlacePhase.DESCEND:
                logger.error(
                    "[ces_verify] arm_q hard-set during DESCEND (40 must be q_ref only)"
                )
            return self._slew_arm(command.arm_q.to(action_dtype))
        if command.tcp_pos is not None:
            try:
                desired = self.ik.solve(
                    command.tcp_pos,
                    command.tcp_quat,
                    q_ref=command.arm_q_ref,
                )
                return self._slew_arm(desired[0])
            except Exception as error:
                if not self._err_logged:
                    self._err_logged = True
                    logger.exception("[%s] IK failed: %s", self.name, error)
                return self._q_right
        if self._squeeze:
            return self._q_right

        self._q_right = self._right_arm_default[0].clone()
        return self._q_right

    def _compose_joint_targets(
        self,
        command: CesCommand,
        policy_action: torch.Tensor | None,
    ) -> torch.Tensor:
        """组装腿、腰、双臂的完整关节目标，暂不写入夹爪。"""
        full_action = self._full_action_buf
        full_action.zero_()
        if policy_action is not None:
            full_action[self.action_to_indices] = policy_action.reshape(-1)
        elif self._squeeze and self._last_policy_legs is not None:
            # 仅到站钉盆时保持最后一步腿姿；行走期间不改腰部，避免路线越走越歪。
            full_action[self.action_to_indices] = self._last_policy_legs
        else:
            full_action[self.action_to_indices] = self._leg_default[0]
        full_action[self.waist_to_all_indices] = self.default_waist_positions[0]
        full_action[self._left_arm_idx] = self._left_arm_default[0]
        full_action[self._right_arm_idx] = self._right_arm_target(
            command,
            full_action.dtype,
        )
        return full_action

    def _finalize_action(
        self,
        command: CesCommand,
        policy_action: torch.Tensor | None,
        full_action: torch.Tensor,
    ) -> bool:
        """更新夹持状态与策略历史，写入夹爪目标并返回是否运动学锁臂。"""
        if command.gripper >= C.GRIPPER_CLOSED - 0.002:
            self._squeeze = True
            if self._grip_cmd is None:
                self._grip_cmd = C.GRIPPER_CLOSED
        if self.fsm.phase in (
            CesPickPlacePhase.RELEASE,
            CesPickPlacePhase.RETRACT,
            CesPickPlacePhase.SETTLE,
        ):
            self._squeeze = False
            self._grip_cmd = None

        history_action = full_action.clone()
        if policy_action is None:
            # 策略历史保存的是原始腿部偏移，不是钉盆时使用的默认偏置关节目标。
            history_action[self.action_to_indices] = 0.0
        delayed_actions = self.advance_action_history(history_action)
        if policy_action is not None:
            self.apply_delayed_policy_legs(full_action, delayed_actions)
            self._last_policy_legs = full_action[self.action_to_indices].clone()

        # 夹爪必须最后写，防止腿部延迟历史或整机 action 覆盖接触夹持力。
        if self._squeeze and self._grip_cmd is not None:
            full_action[self._right_grip_idx] = self._grip_cmd
        else:
            full_action[self._right_grip_idx] = command.gripper
        full_action[self._left_grip_idx] = C.GRIPPER_OPEN

        lock_upper = (
            command.arm_q is not None
            or command.tcp_pos is not None
            or self._squeeze
        )
        # 夹持后只能让手臂走 PD；直接写关节状态会让指垫瞬移离开产品。
        return lock_upper and not self._squeeze

    def _advance_simulation(
        self,
        command: CesCommand,
        full_action: torch.Tensor,
        kinematic_arm: bool,
    ) -> None:
        """固定推进四个物理子步，并在每个子步重复 root pin。"""
        robot = self.env.scene["robot"]
        for _ in range(self._decimation):
            if command.root_pin is not None:
                self._pin_root_pose(command.root_pin)
            robot.set_joint_position_target(full_action)
            if kinematic_arm:
                self._write_locked_upper_body(full_action)
            self.env.scene.write_data_to_sim()
            self.env.sim.step(render=False)
            if kinematic_arm:
                self._write_locked_upper_body(full_action)
            self.env.scene.update(dt=self.env.physics_dt)

        self.env.sim.render()
        self.env.observation_manager.compute()

    def get_action(self, env):
        """执行一帧 CES 控制；返回值遵循动作提供器接口并保持为 ``None``。"""
        del env
        try:
            command = self.fsm.step()
            if self.fsm.phase is CesPickPlacePhase.SETTLE:
                self._err_logged = False
            policy_action = self._resolve_walk_policy(command)
            full_action = self._compose_joint_targets(command, policy_action)
            kinematic_arm = self._finalize_action(
                command,
                policy_action,
                full_action,
            )
            self._advance_simulation(command, full_action, kinematic_arm)
        except Exception as error:
            elapsed = float(getattr(self.fsm, "t", 0.0))
            if (not self._err_logged) or (elapsed - self._err_t > 2.0):
                self._err_logged = True
                self._err_t = elapsed
                logger.exception(
                    "[%s] CES grasp action failed: %s",
                    self.name,
                    error,
                )
        return None
