# unitree_sim_isaaclab 项目记忆

更新时间：2026-08-13（CESMachine 场景搭建完成并归档）

## 1. 项目概况

| 项 | 值 |
|---|---|
| 路径 | `/home/zh/unitree_sim_isaaclab` |
| 系统 | Ubuntu 22.04 / Linux 5.15 |
| Conda | `unitree_sim_env` |
| Isaac Sim | 4.5 |
| Isaac Lab | `/home/zh/IsaacLab` |
| GPU | NVIDIA A10 24 GB |
| **当前主任务** | `Isaac-Move-CES-Product-G129-Dex1-Wholebody` |
| 机器人 | Unitree G1 29DoF + Dex1 + Wholebody |

**本阶段进展**：CES + LoadingLine + 单桌仿真/训练场景已搭好并通过 `verify_ces_scene`；托盘双手 / 治具取放代码与资产已清理。**自主抓取 FSM 尚未接入**。

## 2. 环境与启动

```bash
source /home/zh/miniconda3/etc/profile.d/conda.sh
conda activate unitree_sim_env
cd /home/zh/unitree_sim_isaaclab
# 勿 unset PYTHONPATH（conda activate.d 会注入 Isaac Sim 路径）
```

### GUI（CES 场景）

```bash
python sim_main.py \
  --device cuda:0 \
  --enable_cameras \
  --task Isaac-Move-CES-Product-G129-Dex1-Wholebody \
  --robot_type g129 \
  --enable_dex1_dds \
  --manual_sim_control
```

可选：`--viewport_camera front_cam|perspective|none`（默认不切相机）。

手动控制：`回车/s` 开始，`p` 暂停，`r` 重置，`q` 退出。

### Headless 验收

```bash
python tools/verify_ces_scene.py --device cuda:0
```

---

## 3. 当前任务：CES LoadingLine 产品取放

目标（计划）：右手 Dex1 从
`LoadingLine/Tray_Assembly_01/Product` 上方接近 → 慢降 → 夹住 → 提升 →
退到桌边 → 放到唯一一张 packing table。

### 3.1 已完成

- [x] CES 资产放进 `assets/bozhon/CESmachine_pickabble.usd`
- [x] 场景：`base_scene_ces_pickplace_wholebody.py`（CES + 缩放单桌 + 隐藏墙/杂物）
- [x] Gym 任务注册：`Isaac-Move-CES-Product-G129-Dex1-Wholebody`
- [x] Product 以 `RigidObjectCfg(spawn=None)` 挂到 USD 内动态刚体
- [x] `verify_ces_scene`：质量 / settle / reset 通过
- [x] 清理 tray_grasp / material_pick_place 及旧托盘·治具 USD
- [x] DiffIK + 插补保留在 `action_provider/manip_common/`

### 3.2 未完成（下次）

- [ ] 自主抓取 FSM（复用 `manip_common`）
- [ ] 按 Product 可达包络定站位 / 放置点
- [ ] 行走导航（walk / snap）
- [ ] 需要时再补 RL 奖励 / 终止

### 3.3 关键路径

```text
tasks/common_scene/base_scene_ces_pickplace_wholebody.py
tasks/g1_tasks/move_ces_product_g1_29dof_dex1_wholebody/
  ├── __init__.py   # Gym id 必须含 Wholebody
  ├── move_ces_product_g1_29dof_dex1_env_cfg.py
  └── mdp/
assets/bozhon/CESmachine_pickabble.usd
tools/verify_ces_scene.py
tools/center_ces_machine.py / prepare_ces_machine.py /
      fix_ces_product_pose.py / recenter_ces_pivot.py /
      prepare_loading_line.py   # 资产离线处理，默认读写 assets/bozhon/
action_provider/manip_common/   # ArmDiffIK + CartesianInterpolator
```

### 3.4 场景几何

```text
ces_machine spawn ≈ (-1.730, -1.328, 0.961)，yaw = +180°
Product 世界位姿  (-2.2, -2.2, 0.821)
LoadingLine 底高  z = 0.6173
packing_table     spawn≈(-3.70, -3.60, 0)，scale_z≈0.621 → 桌面齐 LoadingLine
机器人初始        (-4.00, -0.80, 0.8)，朝 +X
```

Product prim：

```text
/World/envs/env_.*/CESMachine/Root/LoadingLine/Tray_Assembly_01/Product
```

注意：`ces_machine` 用 `AssetBaseCfg` 整包加载；**不要**在 CES 根再加
`rigid_props`（会与内部 Product 刚体冲突）。

### 3.5 验收（verify_ces_scene）

- mass ≈ 0.25 kg
- spawn / settle / reset 误差亚毫米级

---

## 4. Dex1 夹爪速查

两指 `right_hand_Joint1_1` / `right_hand_Joint2_1`：**q 增大则闭合**。

```text
gap ≈ 0.050 - 2*q  (m)
gripper_open   = -0.010  → ~70 mm
gripper_closed =  0.016  → ~18 mm
TCP_LOCAL      = (0, 0.115, 0)  # right_hand_base_link → 指垫中点
STAND_PELVIS_Z = 0.755 m        # 实测站立骨盆高（非命令 0.8）
```

---

## 5. 已清理（2026-08-13）

已删除、不可再启动：

- `action_provider/{tray_grasp,material_grasp,action_provider_tray_grasp,action_provider_material_grasp}`
- `tasks/.../base_scene_material_pickplace_wholebody.py`
- `tasks/g1_tasks/move_material_g1_29dof_dex1_wholebody/`
- 相关 tools；`sim_main` 中 `--auto_tray_grasp` / `--auto_material_pick_place` / `--station_mode`
- `assets/bozhon/` 内托盘 / 治具 USD

`assets/bozhon/` **仅保留** `CESmachine_pickabble.usd`。

---

## 6. 仓库结构（当前）

```text
unitree_sim_isaaclab/
├── sim_main.py
├── memory.md
├── robots/
├── tasks/
│   ├── common_scene/
│   │   ├── base_scene_ces_pickplace_wholebody.py   # 当前主场景
│   │   ├── base_scene_pickplace_cylindercfg*.py
│   │   ├── base_scene_pickplace_redblock.py
│   │   ├── base_scene_stack_rgyblock.py
│   │   └── base_scene_pick_redblock_into_drawer.py
│   └── g1_tasks/
│       ├── move_ces_product_g1_29dof_dex1_wholebody/  # 当前主任务
│       ├── move_cylinder_g1_29dof_*_wholebody/
│       └── pick_place_* / stack_* / pick_redblock_* ...
├── assets/bozhon/
│   └── CESmachine_pickabble.usd
├── action_provider/
│   ├── create_action_provider.py   # dds / dds_wholebody / replay
│   ├── manip_common/               # DiffIK + 插补（待接 CES FSM）
│   ├── action_provider_wh_dds.py
│   ├── action_provider_dds.py
│   └── action_provider_replay.py
├── tools/
│   ├── verify_ces_scene.py
│   ├── center_ces_machine.py
│   ├── prepare_ces_machine.py
│   ├── fix_ces_product_pose.py
│   ├── recenter_ces_pivot.py
│   ├── prepare_loading_line.py
│   └── ...
├── layeredcontrol/  dds/  teleimager/
└── ...
```

## 7. Git / 杂项

- 远程：`https://github.com/haozhang950501/isaac_unitree`（remote 名 `github`）
- `assets/` 在 `.gitignore` 中；CES USD 需 `git add -f`
- `teleimager` 为子模块
- 关闭仿真：终端 `q` / Ctrl+C；必要时 `pgrep -af sim_main.py`
