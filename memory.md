# unitree_sim_isaaclab 项目记忆

更新时间：2026-08-29，CES `Baseline_done` 已完成单主线深度精简。

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
  --manual_sim_control \
  --ces_pick_speed 1.5
```

CES 专用参数只保留 `--auto_ces_pick_place` 和 `--ces_pick_speed`。
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

- Pick 使用场景初始骨盆位姿；到站后捕获并钉住实际骨盆 XYZ 和完整四元数，手臂/夹爪保持 PD，不运行步态。
- Walk→Pin 只在交接帧把产品速度刹停一次，不修改产品位姿。
- `PLACE_HOLD` 固定 0.45 s，然后走清单关节段 `05→15`。
- 15 到位立即松爪；没有 Z-only IK、笛卡尔 XY 目标或 pose 25。
- 松爪 0.8 s 后按 `15→05` 收臂，收臂总时长 3.2 s。

当前人工姿态：

```text
05 = [-0.04000009, -0.34000003, 0.52000004, -0.7499997, -0.34, -0.09, 0.90999967]
15 = [-1.2266918, -0.2318979, 1.4931674, 1.2198853, -0.042756237, 0.005500171, 1.1593101]
```

最终 q 由单一运行清单管理，不应为代码整理而改动。

## 4. Walk 约束

- whole-body 指令为机体系 `[vx, vy, wz, height]`。
- Pick 完成第一帧立即发 `vx=-0.45`，调用 `prime_walk_filt`。
- 同时清空 10 帧 actor observation/action history，并给骨盆和夹持产品相同的反向世界速度 kick；否则策略会先朝 CES 跨两步。
- 路线保持「后退 → 边退边右转 → 正向进站」，固定幅值和死区不变。
- 桌前 keep-out 是独立闩锁；触发后永久停止并从当前位置放置。
- 持续倾斜或行走超时进入 FAILED，夹爪保持闭合。

## 5. 场景与物理

- 机器人 spawn 在抓取站：约 `(-3.187, -1.330, 0.8)`，yaw=π。
- Pick 控制时骨盆钉在场景 `ROBOT_INIT_POS/ROBOT_INIT_ROT`，不再另设站位高度缓存。
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
├── interpolation.py   # torch.lerp / bisect / Hermite / Isaac Lab slerp
├── ik_solver.py       # CES 专用 DLS IK 和动态 q_ref 零空间控制
├── pose_library.py    # 校验并加载单一 Smooth V1 运行清单
├── constants.py       # Baseline 参数
└── poses/ces_pick_smooth_v1/trajectory_manifest.json
```

已删除：笛卡尔 Pick/Place、换站瞬移与产品搬运、旧兼容参数、通用 `manip_common` 抽象、V1/V2 回退、Z-only Place、无效场景改色和未引用函数。

## 7. 本地代码检查

临时 CPU 回归代码已在全部通过后按计划删除。删除前检查覆盖：

- Smooth V1 清单、实际 05/15 q、插值连续性和速度缩放
- 40 只能通过 `arm_q_ref` 下发
- `40(live)→30→20→05` 返回路径
- 第一帧反向 walk、路线规划、停止线和 keep-out
- 实际骨盆位姿钉住、15 松爪、无旧版笛卡尔下降阶段
- Python 语法、JSON 解析、模块导入和旧分支残留搜索

这些检查不是 Isaac Sim / 阿里云物理验收，不据此声称抓取、摩擦、平衡或放置物理结果。
