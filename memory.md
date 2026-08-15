# unitree_sim_isaaclab 项目记忆

更新时间：2026-08-15（Pick 优化完成：v2 路点抓取已跑通，放置仍为抬起姿态 IK + 高处松爪）

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

**本阶段进展**：右手 Dex1 从 LoadingLine 托盘夹起 Product（v2 自然路点，已稳定）→ 胸口高度抬起 → **snap 定位换站** → 灰筐上方松爪自由落下。夹取靠垫面摩擦，不焊 TCP。走路换站仍未接通。

**2026-08-15 Pick 优化完成**：默认 `ces_pick_natural_v2`。00→05→10→20→25→30 关节插补；30 后锁 XY、只落 Z；40 只作 `q_ref`。05/10 改为胸口前伸（避开抽屉）。抬起一次 IK 后关节回放，夹持中不每帧 DiffIK。放置实验（收肘/反放 40→10）已全部撤销。

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
  --manual_sim_control \
  --ces_use_joint_waypoints \
  --ces_waypoint_set ces_pick_natural_v2
```

自然路点默认 `ces_pick_natural_v2`；可用 `--ces_waypoint_set ces_pick_natural_v1` 回退。旧笛卡尔伸手：去掉 `--ces_use_joint_waypoints`。

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
开关开：SETTLE → GOTO_PICK → UNFOLD(00→30 关节) → DESCEND(锁XY、slerp朝向、落Z、q_ref 30→40) → GRASP
开关关：SETTLE → GOTO_PICK → UNFOLD → APPROACH → DESCEND → GRASP
之后相同：LIFT(Z+8 cm，世界 Y−6 cm) → CARRY/HOLD → GOTO_PLACE(snap) → PLACE_APPROACH(抬起姿态 IK，多半只落 Z) → RELEASE → RETRACT
```

- 伸手：DiffIK + 分段插补；伸手阶段钉盆 + 右臂 `write_joint_state`（否则默认垂臂把臂拉回去）。
- 夹住后：手臂改走 PD（不再瞬移关节），Dex1 只走 PD 闭合，靠垫面摩擦把件提起。
- 换站：snap 瞬移骨盆；仅当骨盆真的跳了才把零件跟着平移（不是每帧焊 TCP）。
- 放置：不到桌面接触。snap 后从抬起姿态 IK（锁朝向，`q_ref=lift`；已在桌上方则只落 Z），TCP 停在灰筐沿上方约 8 cm，冻臂后张开 Dex1，产品自由落。
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
| 放置贴桌手腕剧抖 | `PLACE_Z` 过低，TCP/垫面磕桌面；DiffIK 在接触区振荡。改为桌面上 10 cm 松爪自由落下 |
| 改 GRASP_SHIFT_Y / 追 AABB XY | 30 之后硬纠 TCP 会让腕肘大角度甩转。下降只锁 30 的 XY、只落 Z |
| 30 后立刻切 `top_down_grasp_quat` | 第一帧姿态突变。应在 DESCEND 内从 q_now slerp 到产品朝向 |
| 夹太深咬凹槽沿 | `GRASP_Z_CLEARANCE` 12→22 mm；最终 TCP Z 见 §10 |

### 3.3 关键路径

```text
tasks/common_scene/base_scene_ces_pickplace_wholebody.py
tasks/g1_tasks/move_ces_product_g1_29dof_dex1_wholebody/
action_provider/manip_common/          # ArmDiffIK + CartesianInterpolator
action_provider/ces_grasp/             # FSM + 站位常量
action_provider/ces_grasp/pose_library.py
action_provider/ces_grasp/poses/ces_pick_natural_v1/
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
GRASP_Z_OFFSET    ≈ 0.035           # AABB 中心 + 半高 12.75 mm + 间隙 22 mm
APPROACH_HEIGHT   = 0.080
APPROACH_STANDOFF = 0.18
LIFT_HEIGHT       = 0.08            # 胸口高度，不要过顶
LIFT_SHIFT_Y      = -0.06           # pick yaw=π 时世界 -Y，躲开抽屉沿
RIGHT_ARM_READY   = (0.40, -0.42, 0.18, 1.20, 0.0, 0.95, 0.0)
ARM_SLEW_RAD      = 0.080           # 伸手
ARM_SLEW_RAD_LIFT = 0.012           # 仅 carry/hold/goto_place；lift 走规划轨迹
```

### 5.3 站位

```text
STAND_PELVIS_Z = 0.755
pick  x_b=0.30  y_b=-0.38  yaw=π     stand ≈ (-3.19, -1.33)
place x_b=0.46  y_b=-0.18  yaw=π/2   stand ≈ (-2.27, -0.77)
PLACE_TARGET_XY = 桌中心 y-0.06 ≈ (-2.087, -0.312)  # 产品往桌内，避免半截露沿
PLACE_Z = TABLE_TOP_Z + PLACE_TRAY_HEIGHT + 0.08
# 灰筐 container_h20 留在桌上；不要贴桌 IK，会抖
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
路点：00→05 1.1  05→10 0.85  10→20 1.5  20→25 1.3  25→30 1.1
DESCEND 1.1  GRASP 1.0  GRASP_WAIT_MAX 0.6  LIFT 2.2
CARRY 0.6  HOLD 0.3  PLACE_APPROACH 2.8  PLACE_DESCEND 跳过  RELEASE 0.8  RETRACT 0.8
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

### 6.3 HOLD 快照（给 walk 到 place 的双足控制）

HOLD 在 **pick 站**冻臂，尚未 snap。走路换站应保持这套右臂 q 和夹爪闭合，件靠摩擦跟着走。

**站位 / TCP（v2 基线 lift_done 实测，后续 05/10 胸口改过，数量级仍可用）：**

| 量 | 值 |
|---|---|
| 骨盆 | `(-3.187, -1.330, 0.755)`，yaw=π |
| Dex1 TCP 世界 | `(-3.512, -0.981, 0.964)` |
| Dex1 TCP 骨盆 | `(0.325, -0.349, 0.210)` |
| 相对夹取 | 世界 Z+8 cm，世界 Y−6 cm（`LIFT_SHIFT_Y`） |

**右臂关键 q**（`RIGHT_ARM_JOINTS`：pitch, roll, yaw, elbow, wr, wp, wy）

| 来源 | q |
|---|---|
| 30 交接实测（HOLD 同族，抬起前） | `[-1.421, -1.254, +1.612, +1.031, +0.289, -0.077, +1.466]` |
| HOLD / `_carry_arm_q` | 抬起结束冻住：一次 IK（Z+0.08、Y−0.06）后的关节回放终点；腕保持抓取朝向 |
| 当前 authored 05（伸手，不是 HOLD） | `[-0.55, -0.15, +0.22, +0.90, +0.20, +0.10, +0.35]` |
| 当前 authored 10 | `[-0.70, -0.22, +0.38, +0.75, +0.32, -0.05, +0.55]` |

下次跑通 HOLD 时看日志：`[ces_fsm] HOLD tcp_w=... tcp_b=... q=[...]`，用新数替换上表。

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
│   ├── ces_grasp/                  # FSM + poses/ces_pick_natural_v1、v2
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

---

## 10. 自然路点 / URDF-viz 目标（2026-08-15）

开关：`--ces_use_joint_waypoints`。JSON 按**关节名**匹配，忽略 `viewer_urdf`。不改 DDS 下标。

抓取站骨盆 `(-3.187, -1.330, 0.755)`，yaw=π。产品中心 ≈ `(-3.487, -0.950, 0.821)`。

**Dex1 TCP（垫面中点，骨盆系）——给 URDF-viz 推掌心 / q：**

| 阶段 | Dex1 TCP | XR 掌 (`right_hand_palm_joint`) |
|---|---|---|
| 30 预抓悬停 | `(0.320, -0.380, 0.181)` | `(0.320, -0.380, 0.296)` |
| 40 最终夹取 | `(0.320, -0.380, 0.101)` | `(0.320, -0.380, 0.216)` |

世界系夹取点：`(-3.507, -0.950, 0.856)`（中心 X−20 mm，Z+35 mm）。

姿态：Dex1 +X=夹爪=世界 +X=骨盆 −X；+Y=手指朝下=骨盆 −Z。XR 掌 +X=手指下，+Z=夹爪沿骨盆 −X。

运行时 30 后**锁 XY、只落 Z**，朝向 2 s slerp 到产品夹持姿态；40 只作 `q_ref`。

### 姿态调优 V2（本地已完成）

- 姿态目录：`action_provider/ces_grasp/poses/ces_pick_natural_v2/`。
- 关节路点：`00→05→10→20→25→30`，按 `trajectory_manifest.json` 的时长做 smoothstep 插补。
- `05_forward_reach` 在进入 10 前已基本达到工作高度，05→10 的 XR 掌心 Z 波动约 `0.8 mm`，用于降低右前方抽屉碰撞风险。
- `30_pre_grasp_vertical` 是关节路点与真实 TCP/IK 的交接姿态；最终位置精度由 IsaacLab 中的 `right_hand_base_link + TCP_LOCAL` 决定，不使用 XR 掌心 FK 代替。
- `40_grasp_posture_ref` 只作动态零空间 `q_ref`，不得作为 `arm_q` 硬下发；三个腕关节与 30 完全相同，肩肘参考变化最大约 `0.57 rad`。
- 本地已完成 `00→05→10→20→25→30→40` 的 urdf-viz 连续回放；这不是 IsaacLab 碰撞、接触或抓取成功验证。

**当前 v2 伸手 05/10（胸口前伸，已在 IsaacLab 跑通抓取）：**

| wp | q (pitch,roll,yaw,elb,wr,wp,wy) |
|---|---|
| 05 | -0.55, -0.15, +0.22, +0.90, +0.20, +0.10, +0.35 |
| 10 | -0.70, -0.22, +0.38, +0.75, +0.32, -0.05, +0.55 |

20/25/30/40 authored q 未改。右移仍在 10→20。

### 阿里云 IsaacLab 验证（2026-08-15，Pick 已完成）

命令（无 `--manual_sim_control`，`--ces_stop_after lift`）：

```bash
python sim_main.py --device cuda:0 --enable_cameras \
  --task Isaac-Move-CES-Product-G129-Dex1-Wholebody \
  --robot_type g129 --enable_dex1_dds --auto_ces_pick_place \
  --station_mode snap --ces_use_joint_waypoints \
  --ces_waypoint_set ces_pick_natural_v2 --ces_stop_after lift \
  --viewport_camera none
```

- [x] 路点 `00→05→10→20→25→30`，`30→40` 只作 `q_ref`
- [x] 05/10 胸口前伸，避开抽屉；抓取稳定
- [x] 抬起一次 IK + 关节回放，夹持不抖
- [x] 40 未作为 `arm_q` 硬下发

**路点实际 q（与 authored 一致，限位余量 rad / 相对上一点 dmax）：**

| wp | q (pitch,roll,yaw,elb,wr,wp,wy) | margin | dmax |
|---|---|---|---|
| 00 | +0.350,-0.180,+0.000,+0.870,0,0,0 | 1.224 | — |
| 05 | -1.249,-1.458,+1.246,+0.120,+0.408,-0.452,+0.985 | 0.630 | 1.599 |
| 10 | -1.370,-1.407,+1.360,+0.107,+0.346,-0.142,+0.865 | 0.750 | 0.310 |
| 20 | -1.283,-1.459,+1.392,+0.551,+0.725,-0.123,+0.820 | 0.792 | 0.444 |
| 25 | -1.446,-1.620,+1.507,+0.541,+0.286,-0.125,+1.240 | 0.375 | 0.439 |
| 30 | -1.421,-1.254,+1.612,+1.031,+0.289,-0.077,+1.466 | 0.149 | 0.490 |
| 40 q_ref | -1.149,-0.908,+1.434,+1.601,+0.289,-0.077,+1.466 | — | 0.570（仅肩肘） |

00→10 无碰撞报错/FAILED；30 腕限位最紧 0.149 rad。日志未见腕解翻转。

**TCP（骨盆系）**

| 时刻 | tcp_b | vs 目标 | err mm (X,Y,Z) |
|---|---|---|---|
| 30 交接 | (0.294,-0.384,0.167) | (0.320,-0.380,0.181) | **-25.8, -3.9, -13.6** |
| 下降结束 | (0.322,-0.366,0.130) | (0.320,-0.380,0.101) | +2.1, **+14.0, +28.9** |
| 抬起后 | (0.325,-0.349,0.210) | — | 世界 Z 0.885→0.964（约 +8 cm） |

判据：30 的 Y 合格（3.9 mm），Z 超差（13.6>5）。下降目标 Z=0.101 未到（停在 0.130，世界 0.885，规划 0.846，`tcp_err=51 mm`）。hold_xy 漂 33.2 mm（slerp 朝向时 IK 带偏了 XY）。

**场景行：** `AABB=(-3.4870,-0.9609,0.8110) grasp=(-3.5070,-0.9609,0.8458)`  
AABB Y 比 `PRODUCT_POS` 偏 −11 mm。夹爪闭合、站稳、抬臂完成；lift 中 `tcp_err` 25→81 mm（目标点上移，件是否跟着需看 GUI）。未根据 XR 代理 X 改 q。

**后续（不要用 XR X 误差改 q）：**
1. 30 的 Z 再抬约 14 mm（骨盆 0.167→0.181），X 可记但先不动。
2. DESCEND 更贴规划 Z（0.846 世界 / 0.101 骨盆），并抑制 slerp 时的 XY 漂移。
3. GUI 确认件是否随抬升；再决定是否只给 CES DESCEND 加局部分轴权重。
