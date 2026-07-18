# unitree_sim_isaaclab 项目记忆

更新时间：2026-07-18（新增自主托盘抓取提升功能）（新增自主托盘抓取提升功能）

## 1. 项目概况

- 项目路径：`/home/zh/unitree_sim_isaaclab`
- 系统：Ubuntu 22.04，Linux 5.15
- Conda 环境：`unitree_sim_env`
- Isaac Sim：4.5
- Isaac Lab：本机路径 `/home/zh/IsaacLab`
- GPU：NVIDIA A10 24 GB
- 当前主要任务：`Isaac-Move-Cylinder-G129-Dex1-Wholebody`
- 机器人：Unitree G1 29DoF、Dex1 二指夹爪、Wholebody 可移动配置

Wholebody 任务会自动使用 `dds_wholebody` 动作源，并启用整机 DDS 控制。

## 2. 主要目录结构

```text
unitree_sim_isaaclab/
├── sim_main.py                 # 项目主入口、环境创建、控制循环及手动仿真控制
├── robots/                     # G1/H1-2 机器人 USD 与执行器配置
├── tasks/
│   ├── common_config/          # 机器人和相机公共配置
│   ├── common_scene/           # 房间、桌子、操作物等公共场景
│   ├── common_observations/    # 机器人、夹爪及相机观测
│   ├── common_event/           # 重置事件
│   └── g1_tasks/               # G1 任务注册和环境配置
├── assets/
│   └── bozhon/                 # 本次导入的托盘资产
├── action_provider/            # DDS、回放及策略动作来源
├── layeredcontrol/             # 控制器和仿真步进
├── dds/                        # DDS 通信
├── teleimager/                 # 图像发布子模块
└── tools/                      # 数据、奖励和 USD 相关工具
```

## 3. 当前托盘资产

已提交的自包含资产：

```text
assets/bozhon/tray_fixture_isaac45_baked.usd
assets/bozhon/tray_fixture_isaac45_dynamic.usd
```

用途：

- `tray_fixture_isaac45_baked.usd`：单位和原点已烘焙，适合静态/运动学场景。
- `tray_fixture_isaac45_dynamic.usd`：动态刚体版本，包含质量和凸包碰撞，用于重力及交互仿真。

原始资产存在以下兼容问题，因此不能直接在 Isaac Sim 4.5 中使用：

- USD crate 含中文 Prim 名，旧版 OpenUSD 无法解析。
- 引用了三个缺失的外部 Payload。
- 原始单位为毫米，直接引用会得到约 `575 × 460 × 47 m` 的错误尺寸。
- 原型最初位于默认 Prim 外，引用时只出现 `TrayFixture` 根节点而没有可见网格。

修复后资产：

- 包含 57 个网格。
- 实际尺寸约 `0.575 × 0.460 × 0.047 m`。
- 单位、中心和最低点已经直接烘焙到顶点，不依赖根节点缩放。
- 动态版本使用每个部件的凸包碰撞。

## 4. 当前托盘场景配置

配置文件：

```text
tasks/g1_tasks/move_cylinder_g1_29dof_dex1_wholebody/
└── move_cylinder_g1_29dof_dex1_hw_env_cfg.py
```

当前参数：

```text
Prim 路径：/World/envs/env_.*/TrayFixture
USD：assets/bozhon/tray_fixture_isaac45_dynamic.usd
位置：(-2.35644, -3.60, 0.80)
旋转：(0.7071068, 0, 0, 0.7071068)，即绕 Z 轴 90°
质量：3.0 kg
运动学：关闭
重力：开启
初始线速度/角速度：0
碰撞：开启
contact_offset：0.005
rest_offset：0.0
```

桌面高度约为 `z=0.794 m`，托盘最低点初始设置为 `z=0.80 m`，因此启动时位于桌面上方约 6 mm，并会在物理开始后轻微稳定到桌面。

环境物理配置：

```text
sim.dt：0.005 s
decimation：4
静摩擦：1.0
动摩擦：1.0
```

## 5. 环境和任务注册

任务注册文件：

```text
tasks/g1_tasks/move_cylinder_g1_29dof_dex1_wholebody/__init__.py
```

Gym 任务名：

```text
Isaac-Move-Cylinder-G129-Dex1-Wholebody
```

机器人预设：

```text
tasks/common_config/robot_configs.py
G1RobotPresets.g1_29dof_dex1_wholebody()
```

机器人底层 USD 配置：

```text
robots/unitree.py
G129_CFG_WITH_DEX1_WHOLEBODY
```

## 6. Conda 环境进入流程

打开新终端后执行：

```bash
source /home/zh/miniconda3/etc/profile.d/conda.sh
conda activate unitree_sim_env
cd /home/zh/unitree_sim_isaaclab
```

可选检查：

```bash
which python
nvidia-smi
```

Python 应指向：

```text
/home/zh/miniconda3/envs/unitree_sim_env/bin/python
```

## 7. 推荐启动命令

### 手动开始、暂停和重复重置

```bash
python sim_main.py \
  --device cuda:0 \
  --enable_cameras \
  --task Isaac-Move-Cylinder-G129-Dex1-Wholebody \
  --robot_type g129 \
  --enable_dex1_dds \
  --manual_sim_control
```

手动控制模式启动后默认暂停，终端命令如下：

```text
回车或 s：开始/继续
p：暂停
r：重置整个场景并暂停
q：正常退出
```

重复验证流程：

```text
输入 r → 场景恢复初始状态并暂停
按回车 → 再次开始仿真
```

### 自动开始

不添加 `--manual_sim_control` 时，环境创建完成后会自动进行物理步进：

```bash
python sim_main.py \
  --device cuda:0 \
  --enable_cameras \
  --task Isaac-Move-Cylinder-G129-Dex1-Wholebody \
  --robot_type g129 \
  --enable_dex1_dds
```

GPU 参数应使用 `cuda:0`，不是 `gpu`。

## 8. 正常关闭流程

优先在启动项目的终端中：

```text
手动控制模式输入 q
```

或按：

```text
Ctrl+C
```

如果窗口关闭后仍有残留进程：

```bash
pgrep -af sim_main.py
kill <PID>
```

仅在普通终止无效时使用：

```bash
kill -9 <PID>
```

建议激活 Conda 环境后直接运行 `python sim_main.py`，不要通过额外后台包装器启动，否则关闭窗口时更容易残留子进程。

## 9. Isaac Sim 界面行为

本项目通过 `sim_main.py` 以 Isaac Lab standalone 模式启动，不是普通 Isaac Sim 编辑器工作流。

- 物理时间轴主要由 Python 控制器驱动。
- 默认模式启动后自动步进，因此不依赖界面播放按钮。
- `--manual_sim_control` 模式由终端命令控制开始、暂停和重置。
- 只关闭渲染窗口不一定会关闭 DDS、图像服务和 Python 主进程，应使用终端命令退出。

## 10. 2026-07-18 完成的修改

1. 将 Bozhon 托盘加入 G1 Dex1 Wholebody 场景。
2. 修复原始 USD 的中文 Prim、外部 Payload、默认 Prim 和毫米单位兼容问题。
3. 将托盘网格展开并烘焙为 Isaac Sim 4.5 可用的自包含 USD。
4. 配置动态刚体、3 kg 质量、重力、凸包碰撞和接触偏移。
5. 将托盘放置在桌面初始位置，并绕 Z 轴旋转 90°。
6. 实测托盘自由落体：120 个物理步中，根节点高度从约 `1.199 m` 降到 `0.794 m`，确认刚体、重力和桌面碰撞有效。
7. 为 `sim_main.py` 增加 `--manual_sim_control`：
   - 初始暂停；
   - 可开始、暂停；
   - 可重复重置场景；
   - 可从终端正常退出。
8. 修复 `teleimager` 读取 YAML 时未指定编码的问题，统一使用 UTF-8，避免相机配置中的 Unicode 字符阻止物理控制循环启动。
9. 验证 GPU 启动设备为 `cuda:0`，NVIDIA A10 工作正常。

## 10b. 自主托盘抓取提升功能（2026-07-18 新增）

目标：运行仿真后，宇树 G1 自主行走靠近桌子 → 双手移动到托盘两侧把手 → 提起托盘到一定高度。

### 启动命令

```bash
python sim_main.py \
  --device cuda:0 \
  --enable_cameras \
  --task Isaac-Move-Cylinder-G129-Dex1-Wholebody \
  --robot_type g129 \
  --enable_dex1_dds \
  --auto_tray_grasp \
  --manual_sim_control
```

新增 CLI 开关 `--auto_tray_grasp`：仅对含 `Wholebody` 的任务生效，会把 `action_source` 切到 `tray_grasp`，并启用 `use_rl_action_mode`（provider 自己步进仿真）。

### 新增/修改文件

```text
action_provider/action_provider_tray_grasp.py   # 主 provider，继承 DDSRLActionProvider
action_provider/tray_grasp/__init__.py
action_provider/tray_grasp/ik_solver.py          # 差分/阻尼最小二乘 IK + FK 读取
action_provider/tray_grasp/interpolation.py      # 笛卡尔插补 + slerp + smoothstep
action_provider/tray_grasp/state_machine.py      # 有限状态机
action_provider/create_action_provider.py        # 注册 "tray_grasp" 分支
sim_main.py                                       # 新增 --auto_tray_grasp
tools/inspect_tray_geom.py                        # 离线检查托盘把手几何
tools/test_tray_grasp.py                          # 无相机快速测试脚本（本机 headless 渲染极慢，不实用）
```

### 架构要点

- 复用整机 RL 策略（`assets/model/policy.onnx`）维持腿+腰平衡与行走；手臂由 IK、行走速度与夹爪开合由状态机自主生成（不再走 DDS 遥操作）。
- provider 每次 `get_action` 内部步进 4 个物理子步（control_dt = 4×0.005 = 0.02 s），返回 None。
- 末端坐标系：`left_hand_base_link`(body idx 37)、`right_hand_base_link`(idx 38)。
- 左右臂各 7 关节，`find_joints(preserve_order=True)` 解析。
- IK 严格按 Isaac Lab 官方浮动基座约定：`get_jacobians()` 是世界系；浮动基座 body 行索引不减 1、关节列 +6；用 `R(base)⁻¹` 把世界系 Jacobian 转到根系，再与根系位姿误差配对做 DLS。目前仅位置 3-DoF 跟踪（保持默认朝向）。

### 托盘把手坐标

托盘 USD 局部系（已烘焙）两端竖板即把手：局部 `(±0.2725, 0, 0.03)`。运行时用托盘 `root_pos_w/root_quat_w` 变换到世界系。当前场景下：
- 左把手（世界）≈ `(-2.356, -3.327, 0.829)`
- 右把手（世界）≈ `(-2.356, -3.872, 0.829)`
- 机器人面向 +X，世界 Y 较大者分给左手。

### 状态机阶段

`SETTLE → WALK → APPROACH → DESCEND → GRASP → LIFT → HOLD`（`state_machine.py::GraspPhase`）。

WALK 为三段式路径导航（避免 locomotion 策略对侧移 vy 跟踪差的问题）：
1. `WALK0/GOTO_WP1`：unicycle（转向+前进）走到站位正后方 `approach_run=0.4 m` 的中间点，前进速度带下限 `nav_vx_floor` 克服步态死区。
2. `WALK1/TURN`：原地转正到朝向 +X。
3. `WALK2/GOTO_STAND`：面向 +X 直线前进到站位（末段不依赖侧移）。
每段有超时兜底，保证一定推进到伸手阶段。

站位 = 两把手中点沿 -X 退 `stand_distance=0.48 m`，面向 +X。

### 当前状态与已知问题

- ✅ 能自主行走到桌前站位、双手抬起伸向两侧把手、执行下探/抓取/提升的完整动作序列。
- ❌ 夹爪未能真正夹住并提起托盘（dex1 两指行程小 + 位置-only IK 未对齐把手朝向，抓取接触不稳）。

### 后续可调参数（都在 `state_machine.py` 构造函数）

- 站位/路径：`stand_distance`、`approach_run`、`nav_kp/nav_kyaw/nav_vmax/nav_vx_floor`、各段超时 `t_goto_wp1/t_turn/t_goto_stand`。
- 抓取：`grasp_offset_w`、`approach_offset_w`、`lift_offset_w`、`approach_time/descend_time/grasp_time/lift_time`。
- 夹爪：`gripper_open=0.033` / `gripper_closed=-0.02`。
- 提升抓取成功率的后续方向：给 IK 加朝向控制（让指面对准把手薄板）、增大夹爪闭合力/行程、下探更贴合把手、抓取后短暂停顿再提升。

## 11. Git 归档

主仓库提交：

```text
move_dev_18 新增自主托盘抓取提升功能（IK/插补/状态机/provider）
9d68545 导入tray_asset_718.
```

远程仓库：`https://github.com/haozhang950501/isaac_unitree`（remote 名 `github`）。

teleimager 子模块提交：

```text
a7e0619 导入tray_asset_718.
```

此前存档提交：

```text
48fde75 Archive updated Python bytecode.
```

注意：`teleimager` 是 Git 子模块，当前修复提交在 detached HEAD 上。若要把主仓库推送到远程并在其他机器可靠克隆，需要先把子模块提交 `a7e0619` 推送到有权限访问的 teleimager 远程分支，再推送主仓库。

## 12. 已知事项

- 动态托盘的某个复杂网格可能出现“GPU-compatible convex mesh cooking failed，fallback to CPU”的警告；当前刚体和桌面碰撞验证正常，该警告不会阻止基础刚体仿真。
- WebRTC 服务依赖本机相关组件；WebRTC 启动错误不会否定本地 Isaac Sim 窗口中的物理验证。
- `assets/` 默认被 `.gitignore` 忽略，本次两个托盘 USD 已通过强制添加进入 Git。
- 项目历史中存在被 Git 跟踪的 `__pycache__/*.pyc`，运行 Python 后可能再次出现字节码变更。
