# unitree_sim_isaaclab 项目记忆

更新时间：2026-08-24，CES `Baseline_done` 代码已完成单主线精简与模块化。

## 1. 当前任务

| 项 | 当前值 |
|---|---|
| 主任务 | `Isaac-Move-CES-Product-G129-Dex1-Wholebody` |
| 机器人 | Unitree G1 29DoF + Dex1 + Wholebody |
| 路点组 | 仅 `ces_pick_smooth_v1` |
| 换站方式 | 仅 whole-body walk |
| 任务终点 | 完整 place，15 松爪后收回 05 |
| 夹持方式 | Dex1 PD + 垫面摩擦，不焊 TCP |

唯一运行链路：

`SETTLE → GOTO_PICK → UNFOLD → DESCEND → GRASP → LIFT → RETURN_HOME → CARRY → GOTO_PLACE → PLACE_HOLD → PLACE_APPROACH → RELEASE → RETRACT → DONE`

对应动作：

`00→10→20→30` → 40 仅作动态 `q_ref` → 抓取/抬起 → `40(live)→30→20→05` → 立即后退 → 转弧右转 → 前进 → 钉实际骨盆位姿 → `05→15` → 松爪 → `15→05`。

## 2. 启动

```bash
python sim_main.py \
  --device cuda:0 \
  --enable_cameras \
  --task Isaac-Move-CES-Product-G129-Dex1-Wholebody \
  --robot_type g129 \
  --enable_dex1_dds \
  --auto_ces_pick_place \
  --station_mode walk \
  --manual_sim_control \
  --ces_use_joint_waypoints \
  --ces_waypoint_set ces_pick_smooth_v1 \
  --ces_pick_speed 1.5
```

兼容参数仍可保留在旧脚本中，但只接受 Baseline 值：

- `--station_mode walk`
- `--ces_stop_after place`
- `--ces_waypoint_set ces_pick_smooth_v1`
- `--ces_use_joint_waypoints` 可写可不写，运行时始终启用关节路点

`--ces_pick_speed` 默认 1.5，限制在 `[0.25, 3.0]`。它只缩放 UNFOLD、LIFT、RETURN_HOME 和清单 Place 关节段的时间，不改路点 q 或曲线形状。DESCEND 和 GRASP 不缩放。

手动控制：`回车/s` 开始，`p` 暂停，`r` 重置，`q` 退出。代码不会热更新，修改后需要重启仿真。

## 3. Pick / Return / Place 合约

### Pick

- 正向硬下发：`00→10→20→30`，插值为 `monotone_cubic_hermite`。
- 30 到位后先在当前高度把夹爪偏航对齐世界 ±X，再锁住世界 XY 下降。
- 40 永远只是下降阶段的动态零空间 `q_ref`，不能作为 `arm_q` 下发。
- 抓住后只求解一次抬升 IK，随后用关节插补抬起，避免摩擦夹持时逐帧 DiffIK 抖动。

### Return

- 逻辑路径：`40(live)→30→20→05`。
- 40 使用抬起后的实时 q；`40→30` 独立 smoothstep，避免硬发 authored 40。
- `30→20→05` 用单调三次 Hermite；20 是退出抽屉边缘的必要路点。
- 清单原始时长：`0.8 / 1.2 / 3.0 s`。

### Place

- 到站后钉住实际到达的骨盆 XY/Z 和完整四元数，手臂/夹爪保持 PD，不运行步态。
- `PLACE_HOLD` 固定 0.45 s，然后走清单关节段 `05→15`。
- 15 到位立即松爪；没有 Z-only IK、笛卡尔 XY 目标或 pose 25。
- 松爪 0.8 s 后按 `15→05` 收臂，收臂总时长 3.2 s。

当前人工姿态：

```text
05 = [-0.04000009, -0.34000003, 0.52000004, -0.7499997, -0.34, -0.09, 0.90999967]
15 = [-1.2266918, -0.2318979, 1.4931674, 1.2198853, -0.042756237, 0.005500171, 1.1593101]
```

最终 q 由人工姿态文件管理，不应为代码整理而改动。

## 4. Walk 约束

- whole-body 指令为机体系 `[vx, vy, wz, height]`。
- Pick 完成第一帧立即发 `vx=-0.45`，调用 `prime_walk_filt`。
- 同时清空 10 帧 actor observation/action history，并给骨盆和夹持产品相同的反向世界速度 kick；否则策略会先朝 CES 跨两步。
- 路线保持「后退 → 边退边右转 → 正向进站」，固定幅值和死区不变。
- 桌前 keep-out 是独立闩锁；触发后永久停止并从当前位置放置。
- 持续倾斜或行走超时进入 FAILED，夹爪保持闭合。

## 5. 场景与物理

- 机器人 spawn 在抓取站：约 `(-3.187, -1.330, 0.8)`，yaw=π。
- Pick 控制时骨盆钉在 `STAND_PELVIS_Z=0.755`。
- 桌面 `TABLE_TOP_EXTRA_Z=0.020`，桌面约 0.6373 m。
- 产品质量 0.25 kg。
- 摩擦：Dex1 pads `12/10`，Product `0.8/0.6`，LoadingLine tray `0.15/0.10`。
- 仓库墙壁保持显示；默认视角为 `PerspectiveCamera_robot`。
- 抽屉运行时改色钩子已删除：该钩子对 RTX 实例未生效，也不参与任务物理。

这些数值属于 Baseline，不因代码重构调整。

## 6. 当前代码结构

```text
action_provider/ces_grasp/
├── fsm_types.py       # 阶段和命令数据
├── state_machine.py   # 共享状态、reset、异常保护、阶段分发
├── fsm_pick.py        # Pick / Grasp / Lift / Return
├── fsm_walk.py        # Carry walk / keep-out / live pelvis pin
├── fsm_place.py       # Place hold / 05→15 / Release / 15→05
├── navigation.py      # 机体系分段步行规划
├── pose_library.py    # 仅加载 Smooth V1
├── constants.py       # Baseline 参数
└── poses/ces_pick_smooth_v1/
```

已删除：笛卡尔 Pick/Place、snap 换站、`stop_after=lift`、V1/V2 回退、Z-only Place、无效场景改色和未引用函数。

## 7. 本地代码检查

当前无 torch/Isaac Lab 的 Windows 环境只执行 CPU/static 检查：

```bash
python -m unittest \
  tools.test_ces_smooth_pick \
  tools.test_ces_walk_navigation \
  tools.test_ces_state_machine
```

检查覆盖：

- Smooth V1 清单、实际 05/15 q、插值连续性和速度缩放
- 40 只能通过 `arm_q_ref` 下发
- `40(live)→30→20→05` 返回路径
- 第一帧反向 walk、路线规划、停止线和 keep-out
- 实际骨盆位姿钉住、15 松爪、无 PLACE_DESCEND
- Python 语法、JSON 解析和旧分支残留搜索

这些检查不是 Isaac Sim / 阿里云物理验收，不据此声称抓取、摩擦、平衡或放置物理结果。
