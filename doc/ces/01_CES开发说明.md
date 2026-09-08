# CES 抓取任务开发说明

> 文档口径：CES Fix Baseline（2026-09-08，最终冻结）
> 核对日期：2026-09-08
> 任务 ID：`Isaac-Move-CES-Product-G129-Dex1-Wholebody`
> 状态：完整 Pick→Walk→Place 已通过 Isaac Sim 验证；除明确的安全缺陷修复外，不再直接修改本 Baseline。

## 1. 文档目的与证据口径

本文说明 CES LoadingLine Product 的“抓取 → 持物行走 → 放置”功能是怎样一步步开发出来的，以及当前代码真正采用了什么方案。

结论来自三类证据：

1. 当前工作区源码和根目录 `memory.md`：用于描述**现在实际运行的行为**。
2. `git log`、提交说明和历史 diff：用于还原**开发先后顺序**。
3. 历史 Memory 记录：用于解释当时为什么改方向、哪些验证只做到 CPU/URDF-viz、哪些经过了 Isaac Sim 物理调试。

“反推”不等于臆测。下文把内容分为：

- **当前事实**：可由当前工作区源码和 `memory.md` 直接确认。
- **历史事实**：可由提交或 Memory 记录确认。
- **工程推断**：根据相邻提交的代码变化解释动机，会明确写成“可以推断”。

## 2. 项目结构与 CES 所在位置

仓库总体上是一个 Isaac Lab/Isaac Sim 仿真应用：

```text
isaac_unitree/
├── sim_main.py                         # 全项目启动入口、CLI、主循环
├── tasks/                              # Gym/Isaac Lab 任务、场景、MDP
├── action_provider/                    # 动作源和控制适配器
├── layeredcontrol/                     # 同步控制循环
├── robots/                             # G1/H1 机器人 USD 与执行器配置
├── dds/                                # DDS 状态、命令、奖励、重置通道
├── teleimager/                         # 相机/WebRTC 图像服务
├── assets/                             # 机器人、CES、桌、仓库、策略模型
├── tools/                              # 通用工具
└── memory.md                           # 当前项目行为基线和验收边界
```

CES 直接相关目录：

```text
action_provider/
├── action_provider_ces_grasp.py        # 状态机命令落到仿真与 Wholebody 策略
├── action_provider_wh_dds.py           # CES 复用的步态策略、观测历史、动作延迟
├── create_action_provider.py           # 选择 ces_grasp 动作提供器
└── ces_grasp/
    ├── constants.py                    # Baseline 参数、固定站位与路线
    ├── fsm_types.py                    # 阶段和单帧命令协议
    ├── state_machine.py                # 状态、分发、reset、异常保护
    ├── fsm_pick.py                     # 展臂、下降、闭爪、抬起、回胸
    ├── fsm_walk.py                     # 持物步行、到站、桌前保护
    ├── fsm_place.py                    # 05→15、松爪、15→05
    ├── interpolation.py                # smoothstep、Hermite、SLERP
    ├── ik_solver.py                    # 右臂全位姿 DLS IK + 零空间 q_ref
    ├── navigation.py                   # BACKOFF→TURN→APPROACH
    ├── grasp_yaw.py                    # 夹持轴对齐世界 ±X
    ├── pose_library.py                 # 唯一轨迹清单加载与合约校验
    └── poses/ces_pick_smooth_v1/
        └── trajectory_manifest.json    # 7 组 q、3 条路径、时长、插值方法

tasks/
├── common_scene/base_scene_ces_pickplace_wholebody.py
└── g1_tasks/move_ces_product_g1_29dof_dex1_wholebody/
    ├── __init__.py                     # Gym 任务注册
    ├── move_ces_product_g1_29dof_dex1_env_cfg.py
    └── mdp/{observations,rewards,terminations}.py
```

关键资产：

| 资产 | 用途 |
|---|---|
| `assets/bozhon/CESmachine_pickabble.usd` | CES 设备、LoadingLine、Tray 和内嵌 Product |
| `assets/robots/g1-29dof_wholebody_dex1/g1_29dof_with_dex1_rev_1_0.usd` | G1 29DoF + Dex1 |
| `assets/objects/PackingTable/PackingTable.usd` | 放置桌和灰色周转筐 |
| `assets/objects/small_warehouse/small_warehouse_digital_twin.usd` | 仓库环境 |
| `assets/model/policy.onnx` | Wholebody 双足策略 |

## 3. 当前 Baseline 到底做什么

当前唯一链路是：

```text
SETTLE → GOTO_PICK → UNFOLD → DESCEND → GRASP → LIFT
       → RETURN_HOME → CARRY → GOTO_PLACE → PLACE_HOLD
       → PLACE_APPROACH → RELEASE → RETRACT → DONE
```

对应动作语义：

```text
00→10→20→30
→ 40 仅作为下降 IK 的动态 q_ref
→ 闭爪、一次性 IK 规划抬升、关节插值抬起
→ 40(live)→30→20→05
→ 后退→后退右转弧→前进
→ 捕获并钉住到站实际骨盆位姿
→ 05→15→松爪→15→05
```

三个最容易误解的点：

1. `40_grasp_posture_ref` **不是**硬下发关节姿态，只进入 IK 的零空间项。
2. 当前 Place **没有** Z-only IK，也没有笛卡尔 XY 修正，只有人工批准的 `05→15` 关节段。
3. 持物靠 Dex1 PD 与接触摩擦，不焊接 TCP，也不直接搬动 Product 位姿。

## 4. 根据 Memory 与提交历史反推的开发过程

### 4.1 阶段一：先把 CES 场景变成独立任务（2026-08-13）

提交：`618fc90 搭建 CESMachine 取放场景并清理旧托盘/治具任务`

完成内容：

- 注册 `Isaac-Move-CES-Product-G129-Dex1-Wholebody`。
- 引入 `CESmachine_pickabble.usd`，直接包装其内部 Product，而不是再生成一个重复物体。
- 建立 CES、机器人、包装桌、灰筐和仓库的场景组合。
- 清理早期 tray/fixture 任务残留，把 DiffIK/插值作为后续 FSM 的技术准备。

可以推断，这一步的核心目标是先固定“任务边界”和“场景坐标系”。没有这一步，后续抓取点、站位和导航目标都没有稳定参考。

### 4.2 阶段二：实现第一版完整 Pick/Place FSM（2026-08-13）

提交：`34bb2d2 pick&place基本功能开发`

首次出现：

- `action_provider_ces_grasp.py`；
- `action_provider/ces_grasp/constants.py`；
- 大型单文件 `state_machine.py`；
- CES 奖励、终止和物理参数调整。

这一阶段先证明“状态机能把动作串起来”，代码体量大、职责集中，是功能优先的原型期。

随后 `e974aba` 把放置改为高处松爪自由落下，用来规避贴桌时手腕抖动。这说明早期 Place 更依赖笛卡尔/接触调试，还没有形成最终的人工 pose 15 方案。

### 4.3 阶段三：从直接动作转为人工路点 + IK 的混合方案（2026-08-15）

提交：

- `6875c87 自然轨迹优化`
- `e1550b8 姿态调优V2`
- `01c99d5 Pick优化完成`

开发思想发生了关键变化：

- 远离 Product 的大幅展臂，用人工设计的关节路点完成；
- 靠近 Product 的最后下降，用真实 TCP 全位姿 IK 完成；
- pose 40 只给肩肘一个姿态倾向，不直接覆盖任务空间目标。

当时先有 `ces_pick_natural_v1`，再有 `ces_pick_natural_v2`。V2 曾采用 `00→05→10→20→25→30`，并保留 V1/旧 FSM 回退。当前代码已删除这些兼容分支，只保留 Smooth V1。

为什么采用混合方案：

- 纯关节路点可保证中远段外形自然、可人工审查；
- 纯 IK 容易在冗余自由度上选择不自然肘形；
- 最终落点必须适应 Product 实际位置，不能只靠离线 q；
- 零空间 q_ref 可以在不破坏 TCP 目标的前提下引导肘部。

### 4.4 阶段四：发现 Wholebody 命令是机体系，开始修导航（2026-08-16 至 08-19）

提交：

- `8756f2f 导航开发待测试`
- `634f9a8 pick0.1_619`
- `157d2cc pick+walk_done`

历史问题是把 `[vx, vy, wz, height]` 当成世界系命令。实际它是机器人机体系：

$$
\mathbf v_w=R_z(\psi)\mathbf v_b
$$

机器人抓取站偏航为 $\psi=\pi$，所以机体系 `vx>0` 对应世界 `-X`。如果按世界 `+X` 的直觉直接发正 `vx`，视觉上会像“走反了”。

导航从通用“转向后前进”逐渐收敛到当前实测有效的固定三段路线：

1. 机体系负 `vx` 后退；
2. 继续后退并叠加右转 `wz`，画转弧；
3. 转正后机体系正 `vx` 前进到放置站。

原因是 Wholebody 策略存在明显死区：小速度、纯 `vy` 或纯 `wz` 很可能只晃不迈步。因此不能用常规比例控制在终点附近无限降速，而要用固定幅值 + 死区 + 提前松指令补偿滑行。

### 4.5 阶段五：确定 Smooth Pick 和 50 Hz 插补（2026-08-21）

提交：

- `521c145 smooth Pick design`
- `984f178 Pick 提速 + 压缩启动等待 + 修 WebRTC 报错`

Smooth V1 固定为 `00→10→20→30`，使用单调三次 Hermite。它解决的是“经过多个手工路点时速度连续、每个关节不在相邻路点间过冲”。

速度倍率只改时间：

$$
T_i'=\max(T_{\min}, T_i/s)
$$

其中 $s$ 是 `--ces_pick_speed`。因此路点 q 和曲线几何形状不变，未触发最小时长时速度约乘 $s$。

当前默认 $s=1.5$，限制在 `[0.25, 3.0]`。DESCEND 和 GRASP 不缩放，因为前者关系到对位精度，后者关系到真实闭爪接触时间。

### 4.6 阶段六：把返回路径改成实时 40 回胸（2026-08-22 至 08-23）

提交：

- `904f219 优化 Pick 逆序回收到胸前姿态`
- `c29e812 手臂回收轨迹去除冗余姿态`

关键设计：返回第一点不能硬发 authored pose 40，因为下降/抓取/抬起后的真实 q 已经由 IK 和接触决定。正确起点是抬起后的实时 q：

```text
40(live) → 30 → 20 → 05
```

当前实现把第一段 `live→30` 单独用 smoothstep，后续 `30→20→05` 用 Hermite。这样既消除交接阶跃，又保留人工批准的回胸外形。

历史 Memory 中曾有 `40(live)→30→05` 的较短版本；当前 Baseline 又加入 pose 20，因为它承担退出抽屉边缘的安全过渡。判断当前行为必须以最新 `memory.md` 和 manifest 为准。

### 4.7 阶段七：完善 Walk→Pin→Place 全链路（2026-08-23 至 08-24）

提交：

- `9fe9634 walk优化`
- `c1f8add pick+place 优化`
- `dbdca84 walk到站钉live quat稳住夹持，下一步重做05/15`
- `86f948e 更新 Pick/Place 05 和 15 位姿`
- `4064859 归档 CES pick-walk-place 几何与走位 baseline`

到站后的策略从“继续依赖步态站稳”改成：

1. 导航停止并确认 `is_standing()`；
2. 捕获实时骨盆位置和完整四元数；
3. 后续每个物理子步把骨盆写回该位姿，速度归零；
4. 只在 Walk→Pin 交接帧把 Product 速度刹停一次，不修改 Product 位姿；
5. 手臂和 Dex1 继续通过 PD 保持真实接触。

这解决了放置时底盘余晃把夹持物甩动的问题，同时保留了“产品靠摩擦跟随”的物理真实性。

### 4.8 阶段八：修正下降前夹爪偏航，完成人工 pose 15（2026-08-24）

提交：

- `daf5409 pick_baseline_ok：DESCEND 先把夹爪偏航对齐世界 X 再落 Z`
- `4ac4306 15pose_left：桌面抬高、松爪贴筐底；15 姿态留下待重做`
- `cbe27f8 手动调整15位姿完成`
- `738d372 Baseline_done`

pose 30 的夹持轴与世界 X 约差 68°。如果一边下落一边大幅扭腕，容易在托盘附近造成碰撞或 IK 抖动。最终方案是：

1. 保持 pose 30 的世界 XY/Z；
2. 用 SLERP 把夹持轴转到距离最近的世界 `+X` 或 `-X`；
3. 姿态对齐后保持 XY 和姿态，只下降 Z；
4. pose 40 始终只作为零空间 q_ref。

pose 15 则由人工在 URDF-viz 中最终确认。当前 q 已集中到单一 manifest，不再分散在多个 JSON 文件里。

### 4.9 阶段九：Baseline 冻结后做结构和运行时重构（2026-08-29 至 08-30）

提交：

- `cdbf3ca refactor: simplify CES baseline state machine`
- `7e6b914 refactor: optimize CES baseline runtime`
- `69bf52a refactor: streamline CES baseline pipeline`

三轮重构没有改变 Baseline 合约，主要做了：

- 把超大状态机拆成 `fsm_pick/fsm_walk/fsm_place` mixin；
- 删除 Natural V1/V2、snap、通用 station mode、Z-only Place 等旧分支；
- 把 CES 专用插值和 IK 收进 `action_provider/ces_grasp`；
- 把七组 q、三条路径和时长合并到 schema 2 manifest；
- 改成显式 root pin；
- 复用 Tensor、索引和动作缓冲，减少控制循环分配；
- 删除完成阶段性验证后不再保留的临时 CES 测试/工具。

当前代码因此是“单任务、单清单、单路线”的专用 Baseline，而不是一个通用抓取框架。

### 4.10 阶段十：Place 收尾修复并冻结 Fix Baseline（2026-09-08）

场景参数整理后，Place 的诊断日志仍引用 `TABLE_TOP_Z` 和 `PLACE_TRAY_HEIGHT`，但 `constants.py` 一度漏导入这两个常量。异常恰好发生在 RELEASE 满 0.8 s、准备进入 RETRACT 时，使状态机停留在 RELEASE；旧异常兜底又没有继续返回到站骨盆的 `root_pin`，因此机器人解除固定后表现为收尾失稳。

最终修复与验收结论：

- 恢复两个场景常量导入；
- 将 `_log_place_result()` 的采样、计算和格式化全部隔离，诊断失败不再影响 RELEASE→RETRACT→DONE 主链；
- Place 各阶段的顶层异常兜底继续返回 `_place_lock_pose`，非关键异常不会解除到站骨盆固定；
- Isaac Sim 已验证 RELEASE 正常进入 RETRACT 和 DONE，收尾不再失稳；
- 深夹实验曾把 `GRASP_Z_CLEARANCE` 从 0.022 降到 0.007，确认会使指垫穿入 Product 凹槽；隔离 worktree 中的世界 `-Y` 横移实验也未可靠避开凹槽，两者均已否决并移除；
- 最终抓取参数恢复为 `GRASP_Z_CLEARANCE=0.022`、`GRASP_SHIFT_Y=0.0`，`TCP_LOCAL=(0,0.115,0)` 保持不变。

以上状态定义为 CES Fix Baseline。后续新抓取几何、回缩策略或场景适配必须以独立任务、分支或 manifest 实验，不得静默改写本 Baseline。

## 5. 当前每个阶段的输入、动作和退出条件

| 阶段 | 主要输入 | 输出控制 | 退出条件 |
|---|---|---|---|
| `SETTLE` | 机器人站立状态、Product AABB | 钉初始骨盆、张爪 | 0.3 s 且站稳，或短超时 |
| `GOTO_PICK` | 站立判定 | 继续钉盆、张爪 | 连续站稳 0.2 s；6 s 超时放行 |
| `UNFOLD` | 实时右臂 q、manifest | `arm_q`：00→10→20→30 | 关节轨迹完成 |
| `DESCEND` | 实时 TCP、目标 TCP、30/40 q | `tcp_pos/tcp_quat` + `arm_q_ref` | 笛卡尔轨迹完成或超时 |
| `GRASP` | 下降末端实时 q | 固定 `arm_q`，smoothstep 闭爪 | 0.6 s |
| `LIFT` | 一次 IK 得到的抬升 q | 关节插值抬起，夹爪闭合 | 轨迹完成或超时 |
| `RETURN_HOME` | 抬起后 live q、30/20/05 | live→30 smoothstep，30→20→05 Hermite | 完成或超时 |
| `CARRY` | 05 与闭爪状态 | 首帧直接 `vx=-0.45`，预载滤波 | 立即转 `GOTO_PLACE` |
| `GOTO_PLACE` | 骨盆 x/y/yaw、倾角 | Wholebody 固定幅值导航 | 停止线后站稳；倾倒/超时失败 |
| `PLACE_HOLD` | 到站 live 骨盆、live 手臂 | 钉盆、闭爪、保持 05 | 0.45 s |
| `PLACE_APPROACH` | live q、pose 15 | `arm_q`：05→15 | 完成或超时 |
| `RELEASE` | pose 15 | 钉盆、张爪、保持 15 | 0.8 s |
| `RETRACT` | live q、carry q/05 | `arm_q`：15→05，张爪 | 完成或超时 |
| `DONE` | 最终 q、到站骨盆 | 持续钉盆、保持 05 | 外部 reset/退出 |
| `FAILED` | 是否已抓到产品 | 步态归零；已抓到则闭爪保持 | 外部 reset/退出 |

## 6. 核心数学原理

### 6.1 世界系、机体系与站位

机器人偏航为 $\psi$ 时，世界系前向和左向单位向量为：

$$
\begin{bmatrix}x_w\\y_w\end{bmatrix}
=
R_z(\psi)
\begin{bmatrix}x_b\\y_b\end{bmatrix},
\qquad
R_z(\psi)=
\begin{bmatrix}
\cos\psi&-\sin\psi\\
\sin\psi&\cos\psi
\end{bmatrix}
$$

所以机体系 $+X$ 轴和 $+Y$ 轴在世界系中的投影分别是：

$$
\mathbf f=(\cos\psi,\sin\psi),\qquad
\mathbf l=(-\sin\psi,\cos\psi)
$$

若目标在机器人机体系中应位于“前方 $x_b$、左方 $y_b$”，骨盆站位为：

$$
\mathbf p_{stand}=\mathbf p_{target}-x_b\mathbf f-y_b\mathbf l
$$

这是早期调站位时使用的几何关系。当前源码为了可读性，抓取站和放置站已经写成最终世界坐标常量：抓取站跟随 `ROBOT_INIT_POS`，放置站为 `PLACE_STAND_XY=(-2.2669,-0.7717)`。`forward_left()` 仍保留给抓取 inset、导航误差和速度 kick 使用。

场景里的 yaw 角最终要写成 Isaac 使用的 `wxyz` 四元数。绕世界 Z 轴旋转 $\psi$ 等价于轴角 $(\hat z,\psi)$：

$$
q_z(\psi)=
\left(\cos\frac{\psi}{2},0,0,\sin\frac{\psi}{2}\right)
$$

这就是 `_yaw_quat(deg)` 的来源。代码里先把角度转弧度，并对 360 取模；因此 `360°` 与 `0°` 的四元数相同，`450°` 等价于 `90°`。这里保留 `450°` 这种写法，是为了让读者知道 Product 相对 CES 多转了 90°。

### 6.2 抓取点

以 Product 的世界 AABB 中心 $\mathbf p_{aabb}$ 为基准：

$$
\mathbf p_g=\mathbf p_{aabb}
+d_{in}\mathbf f(\psi_{pick})
+(0,\Delta y,h_{half}+h_{clear})
$$

当前 $d_{in}=0.020$ m、$\Delta y=0$、$h_{half}=0.01275$ m、$h_{clear}=0.022$ m。抓取站 $\psi_{pick}=\pi$，所以推进抽屉方向是世界 `-X`。

使用 AABB 而不是 USD 根节点，是因为 Product 根 pivot 不在几何中心。

`h_{clear}=0.022` 是冻结值：继续减小该值会让 TCP 沿世界 `-Z` 深入，已实测造成指垫与 Product 凹槽干涉。`TCP_LOCAL=(0,0.115,0)` 是手部局部坐标系中的工具点标定，其中的 Y 不是世界 Y，不能用来代替世界横向避障。

### 6.3 smoothstep

归一化时间 $\tau\in[0,1]$：

$$
s(\tau)=3\tau^2-2\tau^3
$$

这个三次多项式不是随手选的。假设 $s(\tau)=a\tau^3+b\tau^2+c\tau+d$，希望它满足：

$$
s(0)=0,\quad s(1)=1,\quad s'(0)=0,\quad s'(1)=0
$$

代入可得：

$$
d=0,\quad c=0,\quad a+b=1,\quad 3a+2b=0
$$

解出：

$$
a=-2,\qquad b=3
$$

所以得到 $s(\tau)=3\tau^2-2\tau^3$。它的作用很朴素：段首段尾速度为零，避免单段切换时突然给手臂或夹爪一个速度阶跃。

两点插值：

$$
\mathbf q(\tau)=(1-s)\mathbf q_0+s\mathbf q_1
$$

实际代码会先用累计时长找到当前段：

$$
\tau=\operatorname{clip}
\left(
\frac{t-t_i}{T_i},0,1
\right)
$$

再把 $\tau$ 送进 `ease_in_out()`，最后用 `torch.lerp()` 做线性混合。它用于闭爪、笛卡尔分段位置、live→30、05→15 和 15→05。

### 6.4 单调三次 Hermite

第 $i$ 段长度 $h_i$，割线斜率：

$$
\boldsymbol\delta_i=\frac{\mathbf q_{i+1}-\mathbf q_i}{h_i}
$$

若相邻割线同号，内部路点斜率用加权调和平均；不同号则置零：

$$
\mathbf m_i=
\frac{w_1+w_2}{w_1/\boldsymbol\delta_{i-1}+w_2/\boldsymbol\delta_i},
\quad
w_1=2h_i+h_{i-1},\quad w_2=h_i+2h_{i-1}
$$

端点斜率为零。段内：

$$
\mathbf q(\tau)=h_{00}\mathbf q_i+h_{10}h_i\mathbf m_i
+h_{01}\mathbf q_{i+1}+h_{11}h_i\mathbf m_{i+1}
$$

其中：

$$
h_{00}=2\tau^3-3\tau^2+1,\quad
h_{10}=\tau^3-2\tau^2+\tau
$$

$$
h_{01}=-2\tau^3+3\tau^2,\quad
h_{11}=\tau^3-\tau^2
$$

代码按关节分量判断同号，因此保持各关节局部形状，不在相邻人工路点之间制造额外峰值。

Hermite 段的推导思路是：每一段都用一个三次多项式，同时满足段首/段尾的位置和速度四个约束：

$$
q_i(0)=q_i,\quad q_i(1)=q_{i+1},\quad
\frac{dq_i}{dt}(0)=m_i,\quad
\frac{dq_i}{dt}(1)=m_{i+1}
$$

因为代码里的自变量是归一化 $\tau$，真实时间导数和归一化导数差一个 $h_i$：

$$
\frac{dq}{d\tau}=h_i\frac{dq}{dt}
$$

所以基函数里会出现 $h_i m_i$ 和 $h_i m_{i+1}$。如果某个关节在相邻两段先增后减，代码把中间斜率置零，让它在路点处自然停一下，而不是穿过去制造过冲。这就是“单调”的工程含义：它保护每个关节分量的局部形状，但不保证整体末端路径无碰撞，也不是 C2 最小加速度轨迹。

### 6.5 四元数 SLERP

两单位四元数夹角为 $\Omega$ 时：

$$
q(\tau)=
\frac{\sin((1-\tau)\Omega)}{\sin\Omega}q_0+
\frac{\sin(\tau\Omega)}{\sin\Omega}q_1
$$

当前调用 Isaac Lab 的 `quat_slerp`。外层仍先用 smoothstep 重参数化 $\tau$，所以姿态在段首尾平滑起停。

四元数不能直接按分量 lerp 后拿来当姿态轨迹，因为单位四元数落在四维单位球面 $S^3$ 上。直接线性插值会偏离球面，再归一化也会让角速度不均匀。SLERP 的思路是在 $S^3$ 的大圆上走最短弧。

先算点积：

$$
\cos\Omega=q_0\cdot q_1
$$

如果点积为负，通常把 $q_1$ 取反，因为 $q$ 和 $-q$ 表示同一个空间姿态，但取反后会走较短的球面弧。随后按角度比例插值：

$$
q(h)=
\frac{\sin((1-h)\Omega)}{\sin\Omega}q_0+
\frac{\sin(h\Omega)}{\sin\Omega}q_1
$$

CES 里 $h=s(\tau)$，所以“姿态路径”仍是四元数球面最短弧，“时间速度”则被 smoothstep 重新参数化，在段首段尾放慢。下降前夹爪偏航对齐就是用这个机制：保持 TCP 位置不动，只沿姿态球面把夹持轴转到世界 `+X/-X`。

### 6.6 DLS 全位姿 IK 与零空间 q_ref

位置误差和轴角姿态误差拼成：

$$
\mathbf e=W
\begin{bmatrix}
\mathbf p^*-\mathbf p\\
\boldsymbol\phi(R^{-1}R^*)
\end{bmatrix},
\qquad J_w=WJ
$$

阻尼最小二乘任务解：

$$
\Delta\mathbf q_{task}=J_w^T(J_wJ_w^T+\lambda^2I)^{-1}\mathbf e
$$

这个式子来自线性化 IK。当前关节为 $\mathbf q$，小关节增量和 TCP 误差近似满足：

$$
\mathbf e\approx J\Delta\mathbf q
$$

普通最小二乘希望最小化 $\|J\Delta\mathbf q-\mathbf e\|^2$，但接近奇异位形时会让 $\Delta\mathbf q$ 变得很大。DLS 在目标函数里加一项关节增量惩罚：

$$
\min_{\Delta\mathbf q}
\left\|J_w\Delta\mathbf q-\mathbf e\right\|^2
+\lambda^2\left\|\Delta\mathbf q\right\|^2
$$

对 $\Delta\mathbf q$ 求导并令导数为零，可以得到常见正规方程：

$$
(J_w^TJ_w+\lambda^2I)\Delta\mathbf q=J_w^T\mathbf e
$$

CES 是 7 关节右臂跟踪 6 维 TCP 任务。代码使用等价的右伪逆形式：

$$
\Delta\mathbf q
=J_w^T(J_wJ_w^T+\lambda^2I)^{-1}\mathbf e
$$

好处是求解的是 $6\times6$ 系统，而不是 $7\times7$ 系统；代码对应 `system = jacobian @ jacobian_t + damping`，然后用 `torch.linalg.solve(system, error)`，不显式求矩阵逆。

动态 pose 40 只进入零空间：

$$
J_w^+=J_w^T(J_wJ_w^T+\lambda^2I)^{-1}
$$

$$
N=I-J_w^+J_w
$$

$$
\Delta\mathbf q=\Delta\mathbf q_{task}
+N k_{ref}(\mathbf q_{ref}-\mathbf q)
$$

随后逐关节限幅并裁到关节上下限。因为 $N$ 投影到任务雅可比零空间，q_ref 只影响冗余姿态，不应破坏 TCP 主任务；数值阻尼和线性化误差意味着这是近似关系，不是严格解析解。

权重矩阵 $W$ 的作用是把“位置误差”和“姿态误差”的重要性放到同一个方程里。当前 CES 用位置权重 1.0、姿态权重 0.45，意思是抓取下降时宁可姿态稍软，也要优先把 TCP 位置压准。求解后还会做两层保护：

$$
\Delta\mathbf q\leftarrow
\operatorname{clip}(g\Delta\mathbf q,-\Delta q_{max},\Delta q_{max})
$$

$$
\mathbf q\leftarrow
\operatorname{clip}(\mathbf q+\Delta\mathbf q,\mathbf q_{min},\mathbf q_{max})
$$

同一个控制帧内部最多迭代 8 次，但不重新读取 PhysX 位姿和 Jacobian，而是用一阶近似更新误差：

$$
\mathbf e\leftarrow\mathbf e-J_w\Delta\mathbf q
$$

这是性能和精度的折中：每帧只读一次仿真状态，下一帧再根据新物理状态重新线性化。

### 6.7 TCP 偏移后的雅可比

末端刚体原点到 TCP 的向量为 $\mathbf r$，平移点后的线速度雅可比：

$$
J_{v,tcp}=J_{v,ee}-[\mathbf r]_\times J_\omega
$$

其中 $[\mathbf r]_\times$ 是反对称叉乘矩阵。当前 TCP 局部偏移为 `(0, 0.115, 0)`。

推导从刚体上一点的速度开始。若 EE 原点线速度为 $\mathbf v_{ee}$，角速度为 $\boldsymbol\omega$，TCP 相对 EE 原点的向量为 $\mathbf r$，则：

$$
\mathbf v_{tcp}=\mathbf v_{ee}+\boldsymbol\omega\times\mathbf r
$$

反对称矩阵定义为：

$$
[\mathbf r]_\times=
\begin{bmatrix}
0&-r_z&r_y\\
r_z&0&-r_x\\
-r_y&r_x&0
\end{bmatrix}
$$

它满足：

$$
[\mathbf r]_\times\boldsymbol\omega=\mathbf r\times\boldsymbol\omega
$$

而 $\boldsymbol\omega\times\mathbf r=-(\mathbf r\times\boldsymbol\omega)$，所以：

$$
\mathbf v_{tcp}=\mathbf v_{ee}-[\mathbf r]_\times\boldsymbol\omega
$$

把速度写成雅可比形式 $\mathbf v_{ee}=J_v\dot q$、$\boldsymbol\omega=J_\omega\dot q$，就得到：

$$
J_{v,tcp}=J_v-[\mathbf r]_\times J_\omega
$$

`ArmDiffIK._jacobian_b()` 先把 PhysX 给的世界雅可比旋到机器人基座系，再用这个式子把 EE 原点雅可比改成 TCP 点雅可比。

### 6.8 导航误差

目标点与当前点差 $\Delta\mathbf p=\mathbf p_t-\mathbf p$。固定行进轴：

$$
\mathbf a=d(\cos\psi_t,\sin\psi_t),\quad d\in\{-1,+1\}
$$

沿程剩余和侧向误差：

$$
e_{remain}=\Delta\mathbf p\cdot\mathbf a
$$

$$
e_{lat}=\Delta x(-\sin\psi)+\Delta y\cos\psi
$$

朝向误差：

$$
e_\psi=\operatorname{wrap}(\psi_t-\psi)
$$

控制量不与误差成比例，而是按误差符号选择固定幅值。迟滞规则为：未激活时超过阈值 $e_0$ 才进入，已激活后降到 $0.4e_0$ 才退出。

转弧理论半径：

$$
R=\frac{|v|}{|\omega|}=\frac{0.45}{1.20}=0.375\text{ m}
$$

所以 `WALK_TURN_LEAD` 用该半径再加 0.20 m，提前结束直线后退。

### 6.9 步态指令滤波与特例

普通阶段按变化率限制：

$$
u_{k+1}=u_k+\operatorname{clip}(u^*-u_k,-a\Delta t,a\Delta t)
$$

但 Pick→Walk 第一帧和 `vx` 正负切换会直接跨过死区，不能从 0 缓慢爬升，否则策略只晃动而不迈步。

### 6.10 Dex1 PD 与摩擦夹持

概念上的关节 PD：

$$
\tau=K_p(q_d-q)-K_d\dot q,
\qquad |\tau|\le \tau_{max}
$$

当前运行时写入 $K_p=1800$、$K_d=30$、$\tau_{max}=80$。接触摩擦遵循库仑约束：

$$
|F_t|\le \mu F_n
$$

指垫静/动摩擦 `12/10`，Product `0.8/0.6`，Tray `0.15/0.10`，合并模式为 `max`。高摩擦只放指垫，不放 Tray，避免产品粘在槽中抬不起来。

## 7. 时间、频率与动作执行

环境物理步长：

$$
\Delta t_{physics}=0.005\text{ s}=200\text{ Hz}
$$

CES 每次 `get_action()` 固定推进 4 个物理子步：

$$
\Delta t_{control}=4\times0.005=0.020\text{ s}=50\text{ Hz}
$$

`RobotController` 外层默认 `step_hz=100`，但 CES 设置 `use_rl_action_mode=True`：动作提供器内部已经推进仿真，外层不会再调用 `env.step()`，从而避免一帧推进两次。

默认速度倍率 1.5 下，manifest 时长变为：

| 路径 | 原始时长 | 实际时长 |
|---|---:|---:|
| 00→10 | 1.75 s | 1.167 s |
| 10→20 | 2.07 s | 1.380 s |
| 20→30 | 1.20 s | 0.800 s |
| live 40→30 | 0.80 s | 0.533 s |
| 30→20 | 1.20 s | 0.800 s |
| 20→05 | 3.00 s | 2.000 s |
| 05→15 | 3.20 s | 2.133 s |

单次抬升从 2.2 s 缩到约 1.467 s。`0.4 s` 最小时长保护在这些默认段上未触发。

## 8. 场景与物理 Baseline 参数

| 项目 | 当前值 | 说明 |
|---|---:|---|
| CES 生成位置 | `(-3.9569, -1.8217, 0.9610)` | 世界坐标常量 |
| Product 初始位置 | `(-3.4870, -0.9502, 0.8208)` | 世界坐标常量 |
| 机器人初始骨盆 | `(-3.1870, -1.3302, 0.8)` | 直接生成在抓取站 |
| 抓取站 yaw | $\pi$ | 面向世界 `-X` |
| 包装桌位置 | `(-2.0869, -0.3717, 0.0)` | 世界坐标常量 |
| 灰筐中心 XY | `(-2.2369, -0.4757)` | 启动事件把筐移到该位置 |
| 灰筐底部 Z | `0.6393 m` | 固定世界高度 |
| 放置站 yaw | $\pi/2$ | 面向世界 `+Y` |
| 放置站 XY | `(-2.2669, -0.7717)` | 直接常量，不再运行时反算 |
| 桌面 Z | `0.6373 m` | 固定最终值 |
| 包装桌 Z 缩放 | `0.6411` | 保留 4 位小数已足够 |
| 灰筐高度 | `0.1026 m` | 随桌子缩放后的最终值 |
| Product 质量 | `0.25 kg` | 启动事件写入 |
| Product 掉落阈值 | `0.32 m` | 奖励与终止共用 |
| 右夹爪开/闭 q | `-0.010 / 0.019 rad` | q 增大表示闭合 |
| 抓取抬升 | `+0.08 m` | 同时世界 Y 偏 `-0.06 m` |
| 物理 dt | `0.005 s` | 200 Hz |
| 控制 dt | `0.020 s` | 50 Hz |

## 9. 启动与人工操作

推荐命令：

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

`--auto_ces_pick_place` 会自动：

- 启用 Wholebody DDS/策略路径；
- 启用 Dex1 通道；
- 把动作源改成内部 `ces_grasp`；
- 让 `RobotController` 进入“动作提供器自行推进仿真”的模式。

手动控制：

- 回车或 `s`：开始/继续；
- `p`：暂停；
- `r`：恢复机器人和 Product 默认状态、reset CES 状态机，然后暂停；
- `q`：退出。

代码不会热更新，修改 Python/JSON 后应重启仿真。

## 10. 验证分层与当前可信边界

### 10.1 层 1：静态与 CPU 结构检查

可以确认：

- manifest schema、关节顺序、路径长度和时长合法；
- 40 只能通过 `arm_q_ref`；
- Hermite/smoothstep 数学连续性；
- 导航坐标投影、停止线和迟滞逻辑；
- Python 语法、JSON 可解析、导入关系。

当前仓库在最终重构后删除了阶段性 CES 测试文件，所以以后修改应重新补充可长期保留的回归测试，而不能只依赖 `memory.md` 中“曾经通过 50 项”的记录。

### 10.2 层 2：URDF-viz 姿态/轨迹预览

可以确认：

- 人工 q 是否在关节限位内；
- 关节轨迹外形、速度连续性、最终读回；
- 手臂几何观感。

不能确认：

- Product、指垫、托盘的接触摩擦；
- G1 双足平衡；
- Isaac PhysX 穿透/碰撞；
- 真实 TCP 标定误差。

### 10.3 层 3：Isaac Sim/Isaac Lab 物理验收

应实际观察：

- 下降前 yaw 对齐是否不撞 Tray；
- 夹爪闭合后 Product 是否稳定抬起；
- Walk 第一帧是否确实后退，不向 CES 收集；
- 转弧与停止线是否留有桌前余量；
- Walk→Pin 时 Product 是否没有被甩出；
- 05→15 是否进入灰筐可接受区域；
- RELEASE 后 Product 是否落在筐内；
- 15→05 是否不刮桌/筐。

只有这一层通过，才能声称“抓取、摩擦、行走平衡和放置物理成功”。

## 11. 当前实现的已知边界

1. **单环境假设**：AABB 查询把 prim 路径固定到 `env_0`，任务配置也是 `num_envs=1`。
2. **奖励不等于精确入筐**：奖励判断 Product 是否在包装桌矩形和高度范围内，不验证是否真正落在灰筐内部。
3. **FSM 的 DONE 不等于奖励成功**：DONE 表示控制流程完成；最终落点只写 DEBUG 日志，不作为状态转换门槛。
4. **相机不参与抓取闭环**：相机观测用于图像共享/外部查看，抓取点来自仿真 AABB，不是 CV 检测。
5. **导航是专用固定路线**：它不是通用路径规划器，换场景坐标后必须重新核对路线和安全线。
6. **姿态由人工所有**：05/15 等最终 q 不应在“代码整理”中自动优化或重写。
7. **异常以保守保持为主**：状态机/动作提供器记录异常并给安全保持命令，但不会自动重新规划抓取。
8. **CLI 帮助文字有一处滞后**：`sim_main.py` 的 `--ces_pick_speed` 帮助只写了 unfold/lift/return-home；当前 `state_machine.py` 实际还会缩放 manifest Place 关节段 `05→15`。`15→05` retract 仍固定为 3.2 s。

## 12. 后续开发建议与变更流程

本节适用于新版本探索，不表示允许直接改动 Fix Baseline。冻结范围包括抓取点 `GRASP_Z_CLEARANCE=0.022`、`GRASP_SHIFT_Y=0.0`、`TCP_LOCAL=(0,0.115,0)`，以及现有人工 q、路线、物理参数和 Place 控制链。只有明确的安全缺陷或回归修复，经过静态检查与 Isaac Sim 全链路验收后，才可更新 Baseline。

如果要修改 CES，建议按以下顺序：

1. 先同步远端、确认工作区干净、阅读最新 `memory.md`。
2. 明确改的是哪一层：场景几何、人工 q、插值、IK、导航、接触物理或启动链路。
3. 若改人工 pose，先在 URDF-viz 预览，由人确认 q，再只更新 manifest 对应数组。
4. 若改路线，先用纯数学测试验证世界/机体系投影、阶段切换、停止线和 keep-out。
5. 若改 IK，至少验证 40 没有进入 `arm_q`，以及 TCP offset 雅可比符号。
6. 若改速度，只改 `--ces_pick_speed` 或时长，不能顺手改变 q。
7. 做 Python 编译、JSON/schema、公式级单元测试和 `git diff --check`。
8. 最后在 Isaac Sim 做接触、碰撞、摩擦、平衡和放置验收。
9. 把“静态通过”和“物理通过”分开记录，不互相替代。

## 13. 关联文档

- [02_CES架构设计.md](02_CES架构设计.md)：系统边界、组件关系、状态机、控制权和安全架构。
- [03_CES模块设计与核心代码解析.md](03_CES模块设计与核心代码解析.md)：逐文件逐函数说明、公式与核心实现解读。
- [04_项目代码理解阅读推荐.md](04_项目代码理解阅读推荐.md)：建议阅读顺序、代码文件路线和数学基础清单。
