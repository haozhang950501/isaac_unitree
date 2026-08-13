# unitree_sim_isaaclab 项目记忆

更新时间：2026-08-13（CES pick&place 基本功能：snap 换站 + Dex1 摩擦抓取）

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

**本阶段进展**：右手 Dex1 从 LoadingLine 托盘夹 Product → 胸口高度抬起 → **snap 定位换站** → 放到 packing table。夹取靠垫面摩擦，不焊 TCP。走路换站仍未接通。

---

## 2. 环境与启动

```bash
source /home/zh/miniconda3/etc/profile.d/conda.sh
conda activate unitree_sim_env
cd /home/zh/unitree_sim_isaaclab
# 勿 unset PYTHONPATH（conda activate.d 会注入 Isaac Sim 路径）
```

### GUI（CES 自主取放，当前用 snap）

```bash
python sim_main.py \
  --device cuda:0 \
  --enable_cameras \
  --task Isaac-Move-CES-Product-G129-Dex1-Wholebody \
  --robot_type g129 \
  --enable_dex1_dds \
  --auto_ces_pick_place \
  --station_mode snap \
  --manual_sim_control
```

改代码后必须关仿真重开（不会热更新）。端口占用：`fuser -k 60000/tcp`。WebRTC / `image_server` 报错可忽略。

手动控制：`回车/s` 开始，`p` 暂停，`r` 重置（机器人 + Product 回托盘 + FSM 清零），`q` 退出。

只抓起不放置：`--ces_stop_after lift`。

可选：`--viewport_camera front_cam|perspective|none`。

### Headless 验收 / 几何

```bash
python tools/verify_ces_scene.py --device cuda:0
python tools/inspect_ces_product.py --device cuda:0
python tools/probe_ces_workspace.py --device cuda:0 --quick
```

---

## 3. 今日开发总结（pick&place 基本功能）

### 3.1 已跑通的流程

```text
SETTLE → GOTO_PICK(snap) → UNFOLD → APPROACH → DESCEND → GRASP
→ LIFT(~8 cm) → CARRY/HOLD(冻臂、PD 夹紧)
→ GOTO_PLACE(snap 到桌边) → PLACE_APPROACH → PLACE_DESCEND → RELEASE → RETRACT
```

- 伸手：DiffIK + 分段插补；伸手阶段钉盆 + 右臂 `write_joint_state`（否则默认垂臂把臂拉回去）。
- 夹住后：手臂改走 PD（不再瞬移关节），Dex1 只走 PD 闭合，靠垫面摩擦把件提起。
- 换站：snap 瞬移骨盆；仅当骨盆真的跳了才把零件跟着平移（不是每帧焊 TCP）。
- `r`：`env.reset()` + `reset_object_self` 把 Product 写回托盘，并 `CESGrasp.reset_task()`。

### 3.2 踩过的坑（不要再走）

| 现象 | 原因 / 结论 |
|---|---|
| `tcp_err` 锁死 ~569 mm，APPROACH 几十秒不切 | `interp.step(dt)` 里 `dt` 未定义；空路径 `IndexError`；异常被吞掉后超时永远不跑 |
| 伸手 PD 无效、臂一直垂着 | `use_rl_action_mode` 跳过 `env.step`；隐式 PD 把臂拉回默认垂臂。伸手时必须 `write_joint_state` |
| 夹持后退摔倒 / 弹飞 | 全身策略按**垂臂站姿**训练。夹持前伸是 OOD：负 vx 只后仰不迈步；钉盆+策略蹬腿会物理爆炸 |
| 假迈步 overlay | 用户否定；键盘 `S` 能退是因为手臂垂着 |
| 焊 TCP 提起 | 看起来像粘住；用户要求垫面摩擦 |
| 闭合崩飞 | 夹爪运动学锁进网格 + `GRIPPER_CLOSED=0.026` 穿模 + 零件 μ=10 和托盘粘死 |
| 一侧手指卡槽 | TCP 偏到上料口一侧；`GRASP_INSET` 收到 0.020，Z 再抬高 |
| 夹不起来 | 零件 μ 不能和垫面一样高（combine=max 会粘托盘）；提升时不能 `write_joint_state` 把垫面瞬移开 |

### 3.3 关键路径

```text
tasks/common_scene/base_scene_ces_pickplace_wholebody.py
tasks/g1_tasks/move_ces_product_g1_29dof_dex1_wholebody/
action_provider/manip_common/          # ArmDiffIK + CartesianInterpolator
action_provider/ces_grasp/             # FSM + 站位常量
action_provider/action_provider_ces_grasp.py
tools/inspect_ces_product.py
tools/probe_ces_workspace.py
```

---

## 4. 场景几何

```text
三者先收紧，再绕 XY 质心 yaw +180°（启动即可从机器人背后正视 CES）：
ces_machine spawn ≈ (-3.957, -1.822, 0.961)，yaw = 0°
Product 世界位姿  ≈ (-3.487, -0.950, 0.821)，yaw = +90°
LoadingLine 底高  z = 0.6173
packing_table     ≈ (-2.087, -0.252, 0)，yaw = +180°，scale_z≈0.621
                  （相对上料口 +0.45 X / +0.35 Y）
机器人初始        ≈ (-2.037, -2.302, 0.8)，朝 -X
PerspectiveCamera 在机器人身后约 1.15 m，朝 -X 看向 CES 正面
```

Product prim：

```text
/World/envs/env_.*/CESMachine/Root/LoadingLine/Tray_Assembly_01/Product
AABB ≈ 36 × 138.5 × 25.5 mm（世界 X 为夹持短边）
```

注意：`ces_machine` 用 `AssetBaseCfg` 整包加载；**不要**在 CES 根再加 `rigid_props`。

---

## 5. 关键参数（当前可用）

### 5.1 Dex1 / TCP

```text
TCP_LOCAL         = (0, 0.115, 0)   # right_hand_base_link → 指垫中点
GRIPPER_OPEN      = -0.010          # gap ≈ 70 mm
GRIPPER_CLOSED    =  0.019          # gap ≈ 12 mm；q 增大则闭合
手 PD             kp=1800  kd=30    # CESGrasp 启动时写入
夹爪不 write_joint_state（穿模会崩件）
```

### 5.2 抓取几何

```text
jaw 沿世界 X，手指（手 +Y）朝世界下
GRASP_INSET       = 0.020           # 沿 -X 收进抽屉，避免 +X 指卡上料口槽
GRASP_SHIFT_Y     = 0.0
GRASP_Z_OFFSET    ≈ 0.025           # AABB 中心 + 半高 12.75 mm + 间隙 12 mm
APPROACH_HEIGHT   = 0.080
APPROACH_STANDOFF = 0.18
LIFT_HEIGHT       = 0.08            # 胸口高度，不要过顶
RIGHT_ARM_READY   = (0.40, -0.42, 0.18, 1.20, 0.0, 0.95, 0.0)
ARM_SLEW_RAD      = 0.080           # 伸手
ARM_SLEW_RAD_LIFT = 0.012           # 夹持提升，垫面跟着走
```

### 5.3 站位

```text
STAND_PELVIS_Z = 0.755
pick  x_b=0.30  y_b=-0.38  yaw=π     stand ≈ (-3.19, -1.33)
place x_b=0.46  y_b=-0.18  yaw=π/2   stand ≈ (-2.27, -0.77)
PLACE_TARGET_XY = 桌中心 y-0.06 ≈ (-2.087, -0.312)  # 产品往桌内，避免半截露沿
PLACE_Z = TABLE_TOP_Z + 0.018 ≈ 0.635
```

### 5.4 物理（摩擦拆开，combine=max）

```text
Product 质量     0.25 kg
垫面 μ           12.0 / 10.0
零件 μ           0.80 / 0.60     # 不能和垫面一样高，否则粘托盘
托盘 μ           0.15 / 0.10
恢复系数         0
```

0.5 kg 实测夹不起来。零件 μ=10 会和托盘粘死。

### 5.5 时间

```text
UNFOLD 3.2 s  ORIENT 2.2  SLIDE 2.4  DESCEND 2.0
GRASP  3.0 s  LIFT 3.2    CARRY 0.6  HOLD 0.3
PLACE_APPROACH 2.8  PLACE_DESCEND 3.5
```

---

## 6. 后续：walk 换站（未完成）

当前 `--station_mode walk` 在 FSM 里会被改成 snap（`walk is disabled`）。

### 6.1 为什么现在不能走

全身 RL 策略按**默认垂臂站姿**训练。夹持后右臂前伸是 OOD：

- 负 `vx` 过小 → 只后仰不迈步然后倒下
- `vx` 提到 0.25–0.28 仍弹飞 / 后仰踢腿
- 钉盆 + 策略蹬腿 → 物理爆炸
- `vx=0` 松钉让策略「接腿」= 已知摔倒模式
- 观测欺骗（假垂臂 / 假重力）会自相矛盾
- 键盘 `S` 能退，是因为手臂垂着、指令斜坡、策略一直控腿
- 用户不要滑移、不要假迈步 overlay

### 6.2 下次若做 walk，约束

- [ ] 夹持姿态下策略仍能站稳（可能要冻腿为默认站姿、只让策略在垂臂时走路，或换/微调策略）
- [ ] 夹爪全程 PD 闭合，产品靠摩擦跟着走，**不要焊 TCP**
- [ ] 抬臂保持胸口高度，不要过顶
- [ ] 命令机体系 `[vx,vy,wz]`；夹持后不要倒车硬退
- [ ] 路点绕开 CES（`PLACE_VIA_XY ≈ (-2.55, -1.25)`，x > -2.95）
- [ ] 从 pick 站走到 place 站过程中不摔倒、件不掉
- [ ] GUI 确认 spawn→抓取站、抓稳→桌边 两段走路

参考（垂臂、不钉盆）任务：`Isaac-Move-Cylinder-G129-Dex1-Wholebody`。

---

## 7. 已清理（2026-08-13）

已删除、不可再启动：

- `action_provider/{tray_grasp,material_grasp,action_provider_tray_grasp,action_provider_material_grasp}`
- `tasks/.../base_scene_material_pickplace_wholebody.py`
- `tasks/g1_tasks/move_material_g1_29dof_dex1_wholebody/`
- 相关 tools；`sim_main` 中 `--auto_tray_grasp` / `--auto_material_pick_place`
- `assets/bozhon/` 内托盘 / 治具 USD

`assets/bozhon/` **仅保留** `CESmachine_pickabble.usd`。

---

## 8. 仓库结构（当前）

```text
unitree_sim_isaaclab/
├── sim_main.py
├── memory.md
├── robots/
├── tasks/
│   ├── common_scene/
│   │   ├── base_scene_ces_pickplace_wholebody.py   # 当前主场景
│   │   └── ...
│   └── g1_tasks/
│       └── move_ces_product_g1_29dof_dex1_wholebody/
├── assets/bozhon/
│   └── CESmachine_pickabble.usd
├── action_provider/
│   ├── create_action_provider.py
│   ├── manip_common/
│   ├── ces_grasp/
│   └── action_provider_ces_grasp.py
├── tools/
│   ├── verify_ces_scene.py
│   ├── inspect_ces_product.py
│   └── probe_ces_workspace.py
├── layeredcontrol/  dds/  teleimager/
└── ...
```

## 9. Git / 杂项

- 远程：`https://github.com/haozhang950501/isaac_unitree`（remote 名 `github`）
- `assets/` 在 `.gitignore` 中；CES USD 需 `git add -f`
- `teleimager` 为子模块
- 关闭仿真：终端 `q` / Ctrl+C；必要时 `pgrep -af sim_main.py`
