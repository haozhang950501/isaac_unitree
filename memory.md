# unitree_sim_isaaclab 项目记忆

更新时间：2026-08-24 晚 **`Baseline_done`**。CES pick → walk → place 闭环已归档。

**下一阶段 TODO：简化和优化代码，不改现有功能。** 路点 q、桌高、钉盆、夹爪、walk 指令、Place 在 15 松爪，一律先不动。见 §12。

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

**本阶段进展**：右手 Dex1 从 LoadingLine 托盘夹起 Product（默认 `ces_pick_smooth_v1`）→ 胸口高度抬起 → `40(live)→30→20→05` 收到胸前 → HOLD 后 `walk` 真实走到放置站 → `05→15` 人工关节姿态 → **15 到位即松爪自由落体** → `15→05` 收臂。夹取靠垫面摩擦，不焊 TCP。不再做 Place Z-only IK（会碰灰筐沿）。

**当前状态（更新至 2026-08-24）**：

- 机器人 **spawn 就在抓取站**（`ROBOT_INIT_POS = PICK_STAND_XY`），启动不再瞬移。
- 抬起后阶段 `RETURN_HOME`（枚举名保留）：Smooth v1 按独立回收清单 `40(live)→30→20→05_chest_carry` 收到胸前，夹爪保持闭合，到05后再 walk；旧 V1/V2 回退组仍保持逆序回00。
- **Walk 换站已跑通**：「后退 → 边退边右转（转弧）→ 前进」，固定幅值 + 死区（§6.2）。Isaac Sim 实测后退、右转、走到放置站范围都 OK。
- **pick→walk 禁止先往前凑步**（2026-08-24）：只发满幅 S 不够。松钉第一帧必须 `actor_obs_buffer.reset()` + 骨盆/产品后退 kick，否则策略会先朝 CES 走两步再后退。见 §6.6。
- **Place**：05→15 用清单关节插补；**15 到位即松爪**（`PLACE_RELEASE_FROM_15=True`），然后 `15→05` 收臂。不再做 Z-only 下降。pose 25 已取消，见 §11。
- **walk→place 夹持已稳住**（2026-08-24 傍晚 Isaac Sim）：到站钉实际 XY/Z **加上走路残留的完整四元数**，手臂只 PD、夹爪锁 `0.019`、钉盆时不跑步态。旧的纯 yaw 钉盆会把夹爪甩开。见 §6.7。
- **Pick 已归档 `pick_baseline_ok`（`822b4e2`）**：DESCEND 先在 30 高度 slerp 夹爪偏航到世界 ±X，再锁 XY 只落 Z。Isaac Sim 确认夹爪沿产品短边夹。
- **桌面相对 LoadingLine 底 +2.0 cm**（2026-08-24 晚）：05→15 剐托盘，从 +4.5 cm 再降 2.5 cm，`TABLE_TOP_EXTRA_Z=0.020`，桌面约 0.64 m。完整 0.99 m 太高已否。收臂仍是 15→05。
- **15 放置姿态**：`5c51680` 手动调整后纳入本 baseline（见 §11 当前 q）。不要为了外观再改钉盆 / 夹爪 / 桌高。
- **场景收尾**：仓库四周墙壁默认显示；启动默认 `PerspectiveCamera_robot`（`--viewport_camera robot`）。
- **抽屉仍白**：LoadingLine 与机身壳不是同一套 CAD 材质，运行时改色套不上 RTX 实例。**不影响抓取，本 baseline 不修。**
- 导航仍含 900 组撞桌扫描（§6.2 末）。

**`Baseline_done`。** 下一阶段只做代码简化/优化，不改行为。

**2026-08-15 Pick 优化完成**：默认 `ces_pick_natural_v2`。00→05→10→20→25→30 关节插补；30 后锁 XY、只落 Z；40 只作 `q_ref`。05/10 改为胸口前伸（避开抽屉）。抬起一次 IK 后关节回放，夹持中不每帧 DiffIK。放置实验（收肘/反放 40→10）已全部撤销。

**2026-08-21 Smooth Pick design**：用户在 `urdf_pose_toolkit` 中确认 `00→10→20→30→40` 视觉效果。项目新增 `ces_pick_smooth_v1` 并设为默认；运行时关节硬轨迹只有 `00→10→20→30`，`30→40` 仍只作下降 IK 的动态 `q_ref`。正向 `05_forward_reach/25` 从新组移除但 V2 文件原样保留；00/10/20/30/40 的 q 与 V2 完全一致。关节插值改为清单可选的 `monotone_cubic_hermite`。**正向和当时的回00动作已在 Isaac Sim 实跑确认完全 OK 且丝滑。**

**2026-08-22～23 回收历史**：第一版胸前 05 为 `[-0.7999999,-0.24,0.59999996,0.09000012,0.36,-0.13,0.72999984]`；回收曾走 `40→30→20→10→05`，随后去掉中间 20/10 改成 `40(live)→30→05`。阿里云实测证明直达路径会让夹持产品刮抽屉边缘，因此该直达方案已被 2026-08-24 版本替代。

**2026-08-24 回收 V2（历史 q，已被本轮定稿替代）**：用户曾在 `urdf_pose_toolkit/poses/ces_return_to_chest_v2/` 保存 05，q=`[-0.75999993,-0.26,+0.58,-0.68999976,-0.34,-0.05000001,+0.08999996]`，URDF 关节限位最小余量 `0.3572 rad`。正式路径保持 `40(live)→30→20→05`：40 使用抬起后实时 q，不硬下发 authored q_ref；`40(live)→30` 单独用 smoothstep 前导段；到 30 后再按单调三次 Hermite `30→20→05` 退出抽屉。路径与时长不变，只是终点 05 已在本轮再次替换。

**2026-08-24 05 / 15**：05 q=`[-0.04000009,-0.34000003,+0.52000004,-0.7499997,-0.34,-0.09,+0.90999967]`，walk 持物可用。15 经 `5c51680` 手动调整后纳入 `Baseline_done`，见 §11。

**2026-08-24 pick→walk 先往前两步**：阿里云复测，发了满幅 `vx=-0.45` 仍先朝 CES 走两步再后退。不是没发 S，也不是该再等站稳。根因是 10 帧观测栈在 pick 阶段全是 `vx=0`，策略把松钉当成站立起步。修法见 §6.6，**不要回退成等站稳**。

**2026-08-21 Pick 提速**：动作没问题但太慢，加 `--ces_pick_speed` 时间缩放，默认 **1.5×**。只除 `UNFOLD(00→30)` / `LIFT` / `RETURN_HOME` 三段关节轨迹的**时长**，路点 q 和曲线形状一字不动（均匀时间缩放 = 同一条曲线、速度整体 ×1.5）。`DESCEND` / `GRASP` 故意不缩放。pick 段 18.4 s → 14.0 s（−24%）。详见 §10.5。

**2026-08-21 启动等待压缩**：按 `s` 到右臂起动 **2.1 s → 0.7 s**。链路上没有隐藏的预热 / reset / 等 DDS，等待全是 `SETTLE_TIME` / `STAND_MIN_TIME` / `STAND_STABLE_TIME` 三个计时器；而这两个阶段骨盆被 `_apply_snap` 钉死且速度清零，`is_standing()` 第一帧即真，属于纯空烧。只改计时门槛，判定逻辑和 6 s 超时兜底都没动。详见 §2.1。

**2026-08-21 WebRTC 报错修复**：`image_server` 每 10 s 刷一对 `Publisher failed to start (Timeout)` + `WebRTC Thread Error: [Errno 2]`，根因是缺 TLS 证书。生成自签名证书到 `~/.config/xr_teleoperate/`（该路径优先级高于子模块内的默认路径，**不弄脏子模块**）。该报错在独立 daemon 线程里，从未影响主循环性能。详见 §2.2。

---

## 2. 环境与启动

```bash
source /home/zh/miniconda3/etc/profile.d/conda.sh
conda activate unitree_sim_env
cd /home/zh/unitree_sim_isaaclab
# 勿 unset PYTHONPATH（conda activate.d 会注入 Isaac Sim 路径）
```

### GUI（CES 自主取放）

Walk 换站已跑通，日常验证就用 `--station_mode walk`；`snap` 只在需要排除行走因素、单独调放置动作时用。

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

默认 `ces_pick_smooth_v1`；可用 `--ces_waypoint_set ces_pick_natural_v2` 回退到已在 IsaacLab 跑通过的 V2，或用 `ces_pick_natural_v1` 回退旧姿态。旧笛卡尔伸手：去掉 `--ces_use_joint_waypoints`。

`--ces_pick_speed` 是 pick 手臂的时间缩放（默认 1.5，范围 clamp 在 `[0.25, 3.0]`）。它**只除时长、不动路点**，`1.0` 就是原速。启动日志的 `[ces_fsm] drop-place ...` 行会打印 `arm_speed=`、正向 `seg_s=`、逆向 `return=` / `return_s=`（以及 manifest 原始值），照着调即可。见 §10.5。

Pick 站定位始终用 snap（spawn 就在抓取站），只有夹稳后的 `GOTO_PLACE` 才释放骨盆、调用全身行走策略。

改代码后必须关仿真重开（不会热更新）。端口占用：`fuser -k 60000/tcp`。

手动控制：`回车/s` 开始，`p` 暂停，`r` 重置（机器人 + Product 回托盘 + FSM 清零），`q` 退出。

按 `s` 到右臂起动约 **0.7 s**（原 2.1 s），见 §2.1。`image_server` 的 WebRTC 报错已修，见 §2.2。

### 2.1 按 `s` 之后的启动延迟（2026-08-21 已优化）

链路上**没有**隐藏的预热 / reset / 等 DDS：按 `s` 只是把 `simulation_paused` 置 False（`sim_main.py:597`），下一帧就进 `controller.step()`。等待全部来自状态机三个计时器：

| 常量 | 原值 | 现值 | 作用 |
|---|---|---|---|
| `SETTLE_TIME` | 1.0 | **0.3** | SETTLE 退出的硬计时 |
| `STAND_MIN_TIME` | 0.6 | **0.2** | GOTO_PICK 最小定位时间 |
| `STAND_STABLE_TIME` | 0.5 | **0.2** | `is_standing()` 需连续满足的时长 |
| **合计** | **2.1 s** | **0.7 s** | 之后进 UNFOLD，右臂首次运动 |

**为什么能砍**：这两个阶段 `_apply_snap`（`action_provider_ces_grasp.py:202-218`）每帧把骨盆写成 `STAND_PELVIS_Z=0.755`、速度清零，所以 `is_standing()` 的四项（tilt / yaw_rate / xy_speed / `0.68<z<0.88`）第一帧就全满足 —— 原来是在等一个早已成立的条件。

**没有削弱安全性**：只改了计时门槛，`is_standing()` 判定本身没动。真站不稳仍会一直等，最坏由 `_navigate` 的 `t>6.0s` 超时兜底（`state_machine.py:380`）。**若改成让机器人自己走到抓取站、或 spawn 时会晃，把这三个值调回 1.0/0.6/0.5。**

### 2.2 `image_server` 的 WebRTC 报错（2026-08-21 已修）

症状：每 10 秒重复一对

```text
image_server.py:521  Unexpected error in publish: Publisher failed to start (Timeout)
image_server.py:460  WebRTC Thread Error: [Errno 2] No such file or directory
```

**根因**：`WebRTC_PublisherThread.run()` 建 TLS 服务器时 `ssl_context.load_cert_chain(CERT_PEM_PATH, KEY_PEM_PATH)`（`image_server.py:438`）找不到证书 → 抛 `FileNotFoundError`（第二条）。异常发生在 `self._start_event.set()`（:441）之前，于是 `wait_for_start(timeout=10.0)`（:504）干等 10 秒超时 → `ConnectionError`（第一条）。失败后 publisher 不入缓存，下次 publish 重来 → **正好 10 秒一轮**。

**不影响仿真**：`_webrtc_pub` 在独立 daemon 线程（`image_server.py:1457`），10 秒阻塞卡的是它自己，主循环不受影响。

**修法**（已执行）：生成自签名证书到 `~/.config/xr_teleoperate/`。`image_server.py:64-68` 的查找顺序是 `$XR_TELEOP_CERT` → `~/.config/xr_teleoperate/` → `teleimager/`，所以放用户目录**不会弄脏子模块、不动仓库任何文件**：

```bash
mkdir -p ~/.config/xr_teleoperate
LAN_IP=$(hostname -I | awk '{print $1}')
openssl req -x509 -newkey rsa:2048 -nodes \
  -keyout ~/.config/xr_teleoperate/key.pem \
  -out   ~/.config/xr_teleoperate/cert.pem \
  -days 3650 -subj "/CN=localhost" \
  -addext "subjectAltName=DNS:localhost,IP:127.0.0.1,IP:$LAN_IP"
chmod 600 ~/.config/xr_teleoperate/key.pem
```

证书 10 年有效。删掉这两个文件即可回到报错状态。WebRTC 三路端口 60001/60002/60003（`teleimager/cam_config_server.yaml`），ZMQ 55555-55557 一直是好的。自签名证书浏览器会提示不安全，属正常，手动信任即可。

只抓起不放置：`--ces_stop_after lift`。

可选：`--viewport_camera robot|perspective|front_cam|none`。默认 `robot`（`PerspectiveCamera_robot` 跟随机位）。仓库四周墙壁默认显示。

### Headless 验收 / 几何

```bash
python tools/verify_ces_scene.py --device cuda:0
python tools/inspect_ces_product.py --device cuda:0
python tools/probe_ces_workspace.py --device cuda:0 --quick
```

---

## 3. 流程与踩坑总表

### 3.1 已跑通的流程

```text
开关开：SETTLE → GOTO_PICK → UNFOLD(00→30 关节) → DESCEND(锁XY、slerp朝向、落Z、q_ref 30→40) → GRASP
开关关：SETTLE → GOTO_PICK → UNFOLD → APPROACH → DESCEND → GRASP
之后相同：LIFT(Z+8 cm，世界 Y−6 cm) → RETURN_HOME(40 live→30→20→05胸前) → CARRY/HOLD
        → GOTO_PLACE(walk 三段：后退→转弧→前进，到线停死，保持05)
        → PLACE_HOLD(钉实际骨盆 live quat，冻05，夹爪不动，约 0.45 s)
        → PLACE_APPROACH(关节05→15) → RELEASE(15到位即松爪自由落体) → RETRACT(15→05)
```

- 伸手：DiffIK + 分段插补；伸手阶段钉盆 + 右臂 `write_joint_state`（否则默认垂臂把臂拉回去）。
- 夹住后：手臂改走 PD（不再瞬移关节），Dex1 只走 PD 闭合，靠垫面摩擦把件提起。
- 换站：`walk` 走真实步态（§6）；`snap` 仅作对照，瞬移骨盆时才把零件跟着平移（不是每帧焊 TCP）。
- 放置：`05→15` 到位即松爪自由落体，然后 `15→05` 收臂。不再做 Z 下降。
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
| 放置贴桌手腕剧抖 | `PLACE_Z` 过低，TCP/垫面磕桌面；DiffIK 在接触区振荡。改为灰筐沿上方 8 cm（`PLACE_RELEASE_ABOVE_TABLE`）松爪自由落下 |
| 放置时肩腕大角度乱转 | 位置-only DiffIK 只约束"到哪"，不约束"怎么转过去"，关节窗口也只能限范围。**位置精度要 IK，姿态自然度要关节路点** —— 待办见 §11 |
| 改 GRASP_SHIFT_Y / 追 AABB XY | 30 之后硬纠 TCP 会让腕肘大角度甩转。下降只锁 30 的 XY、只落 Z |
| 30 后立刻切 `top_down_grasp_quat` | 第一帧姿态突变。应在 DESCEND 内从 q_now slerp 到产品朝向 |
| 夹太深咬凹槽沿 | `GRASP_Z_CLEARANCE` 12→22 mm；最终 TCP Z 见 §10 |
| 发了指令机器人不动（转向 / 侧移） | **只有 `vx` 能触发步态**。`vy`/`wz` 只能给已经在走的步态转向，自己起不了步。任何纠偏都必须骑在非零 `vx` 上，见 §6.1 |
| 纠偏窗口比纠偏死区还紧 | 死锁：误差低于 `align_yaw`/`lateral_tol` 就不发指令，却又不判到位，机器人无限踱步。到位窗必须 ≥ 纠偏进入阈值 |
| 某一轴幅值贴着死区 | 实机跟踪稍弱就整条被死区吃掉，该轴纠偏彻底失效（`WALK_VY=0.30` vs 死区 0.25 就是）。留 ≥1.5 倍余量 |
| **靠近桌子时"继续纠偏姿态"→ 撞桌摔倒** | 因为纠偏需要非零 `vx`，"继续纠偏"就等于"继续往前开"。障碍物前**必须闩锁停死**，姿态残差交给手臂，别用走位去修 |
| `stop_margin` 小于滑行量 | 必然冲过目标。`vx` 不能降到死区以下 → 没有缓刹，只能提前松手。余量按日志 `coast = margin - along` 反推，别猜 |
| 只测标称工况就以为安全 | 撞桌是"歪起步 + 跟踪失配 + 指令滞后"叠加出来的。必须做组合扫描（见 `test_no_approach_ever_reaches_the_table`），单点全过但 42/900 会撞 |
| 最关键的"停"只写在规划里 | 三次撞桌有两次是规划分支漏了停止条件。要另加一条只看位姿的硬性禁入闩锁兜底 |
| **pick 完满幅 S 仍先往前走两步再后退** | 指令已经是 `-vx`，错在观测栈：pick 钉盆时每帧 `compute_observations([0,0,0,0.8])`，10 帧历史全是站立。松钉后策略先按站立起步朝 CES 收两步，等栈里换上后退命令才真退。**再等站稳会更差**（那是 08-23 已经修掉的前倾）。必须 `reset` 观测/动作栈 + 骨盆与产品写后退世界速度。见 §6.6 |
| **walk 到站后钉盆，件飞掉、空爪合死** | `_apply_snap` 把骨盆写成**纯 yaw**，走路残留俯仰被瞬间掰平，夹爪甩一下把摩擦抓打断。要钉**到位时的完整四元数**，手臂只 PD、夹爪目标不变。见 §6.7 |
| **到站不定盆、冻腿做 05→15** | 人后仰，件仍掉 |
| **到站不定盆、05→15 还跑站立策略** | 钉盆+策略会爆炸；不定盆+策略伸手会整机飞起 |
| **夹持时每帧用 live 腰覆盖行走策略** | walk 上身越走越歪、件中途掉。走路腰必须仍是默认指令，只在到站钉盆后才冻腿 |

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
packing_table     ≈ (-2.087, -0.372, 0)，yaw = +180°，scale_z≈0.641
                  （桌面 z≈0.637 = LoadingLine 底 0.617 + 2.0 cm）
                  （相对上料口 +0.45 X / +0.23 Y；2026-08-24 桌 −Y 12 cm，放置站不跟）
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
ARM_SLEW_RAD      = 0.080           # 伸手；/dt(0.02) → 下发速度天花板 4.0 rad/s
ARM_SLEW_RAD_LIFT = 0.012           # 仅 carry/hold/goto_place；lift 走规划轨迹
PICK_SPEED_SCALE  = 1.5             # pick 手臂时间缩放默认值，CLI 可覆盖
PICK_SPEED_MAX    = 3.0             # 上限；4.5× 处才会撞 slew，留了余量
PICK_SEGMENT_MIN_TIME = 0.40        # 缩放后单段时长下限，防止压成阶跃
05_chest_carry q = (-0.04000009,-0.34000003,+0.52000004,-0.74999970,-0.34000000,-0.09000000,+0.90999967)
```

### 5.3 站位

```text
SETTLE_TIME       = 0.3             # 原 1.0；snap 钉盆下 is_standing 第一帧即真
STAND_MIN_TIME    = 0.2             # 原 0.6
STAND_STABLE_TIME = 0.2             # 原 0.5；三者合计 2.1s → 0.7s，见 §2.1
STAND_PELVIS_Z = 0.755
pick  x_b=0.30  y_b=-0.38  yaw=π     stand ≈ (-3.19, -1.33)
place x_b=0.46  y_b=-0.18  yaw=π/2   stand ≈ (-2.27, -0.77)
PLACE_TARGET_XY = 桌心 −X 15 cm、近沿对齐后再 +Y 3 cm ≈ (-2.237, -0.476)
# 放置站仍按桌 Y 偏移前的位置算 (-2.267, -0.772)，不跟着桌子走
PLACE_Z = TABLE_TOP_Z + PLACE_TRAY_HEIGHT + 0.08
PLACE_FINAL_TCP_Z = TABLE_TOP_Z + 0.025   # 筐底上方约 25 mm，一丝自由落体
# 灰筐 container_h20 留在桌上；不要贴桌 IK，会抖
# Place 在 15 松爪，Z-only 下降默认关闭
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
DESCEND 0.70  yaw-align 0.35  GRASP 0.60  GRASP_WAIT_MAX 0.25  LIFT 2.2
CARRY 0.6  HOLD 0.3  PLACE_HOLD 0.45  PLACE_APPROACH(05→15) 3.2/速度倍率
RELEASE 0.8（15 到位即松爪；PLACE_DESCEND 默认跳过）  RETRACT 15→05 3.2
```

---

## 6. Walk 换站（2026-08-19 完成，Isaac Sim 已走到放置站）

`--station_mode walk` 不再回退到 snap。Pick 完成进入 HOLD 后，FSM 按 `CARRY_WALK_LEGS` 三段走；到达后钉住**实际到达位姿**继续放置，不做大幅末端瞬移。

### 6.1 核心约束：只有 `vx` 能触发步态

- 行走策略对小速度指令**不迈步**：键盘 WASD 点动机器人不动，必须长按把指令斜坡拉起来才走。
- 所以导航**不能用比例控制**（`gain * 距离` 在靠近目标时必然衰减到死区以下，机器人只会原地晃）。
- 现在所有平移/转向都是**固定幅值 + 死区**：要么发满幅指令，要么发 0，像"按住键再松手"。
- **只有 `vx` 能触发步态 —— 这是最硬的一条**（2026-08-19 二次实测）。先发现纯偏航 `(0,0,wz)` 站着不动；改完又发现纯侧移 `(0,+0.30,0)` 一样站着不动 —— 日志 `mode=forward_align cmd_b=(+0.00,+0.30,+0.00)` 连发 18 s，`xy` 和 `lat=+0.53` 一个数都没变。
  - 结论：`vy` / `wz` 都**只能给一个已经由 `vx` 驱动起来的步态转向**，自己不能起步。
  - 所以规则是：**平移段发出去的指令里 `vx` 永不为零**。侧向纠偏只能做成**斜行**（`vx` + `vy` 同时给），不存在"停下来横挪"这种动作。单测 `test_no_command_steers_without_driving` 守这条。
  - 推论（血的教训）：**"继续纠偏" 等价于 "继续往前开"**。所以障碍物前面不能靠纠偏收敛，只能到线停死，见 §6.2 最后一段。
- 每个轴的幅值都要**离死区留够余量**，否则实机跟踪稍弱就整条被吃掉。旧 `WALK_VY=0.30` 对 0.25 的死区只剩 20% 余量（`vx` 有 50%、`wz` 有 200%），是最先失效的一轴 —— 表现为 `lat` 冻住不动、机器人在放置站前来回踱步。现在 `WALK_VY=0.40`，单测 `test_every_axis_keeps_room_above_the_dead_band` 要求 ≥1.5 倍死区。
- 键盘真实上限（`send_commands_keyboard.py` 的 `ranges`）：`x_vel (-0.6, 1.0)`、`y_vel ±0.5`、**`yaw_vel ±1.57`**。以前写"Z/X 长按约 0.7~1.0"是错的。
- 幅值在 `constants.py`：`WALK_VX=0.45`（上限 75%）、`WALK_VY=0.40`（上限 80%）、`WALK_WZ=1.20`（上限 76%）—— 三轴同比例。旧 `WALK_WZ=0.70` 只有 45%，策略吃不动。历史上 0.25–0.28 的 `vx` 只后仰不迈步（见 6.3），走不动就先加这里。
- **姿态窗口不能比纠偏死区还紧**：纠偏只在误差**超过** `align_yaw`/`lateral_tol` 时才发指令，低于它一点都不发。曾把窗口设成 0.16 rad 而 `align_yaw=0.20`，卡在"9.3° 纠不动、又差 0.13° 不达标"上无限踱步。现在 `WALK_YAW_ARRIVE_FINAL = WALK_ALIGN_YAW`，单测 `test_pose_windows_stay_inside_their_fix_bands` 守这条。

### 6.2 路线：后退 → 边退边右转（转弧）→ 前进

机器人在 pick 站 `(-3.187, -1.330)`、`yaw=π` 朝世界 `-X`，所以**后退就是走世界 +X**（后退不改变 Y，对齐用的是 X）。

| 段 | kind | 内容 | 终点 |
|---|---|---|---|
| ① backoff | reverse | 负 `vx` 退出 CES，距入弧点 `WALK_TURN_PREVIEW` 起预转 | `(-2.842, -1.330)` |
| ② turn_to_table | turn | **继续后退**（`vx=-0.45`）+ 右转 `wz=-1.20`，`yaw π → π/2` | X 落在放置站进入线 `x≈-2.267`，正对桌子 |
| ③ approach_place | forward | 正 `vx` 走世界 `+Y`，途中**斜行**纠侧偏，到停止线就停死 | `PLACE_STAND_XY (-2.267, -0.772)` 前约 `stop_margin` |

- 转弧半径 `R = WALK_TURN_VX / WALK_WZ = 0.45/1.20 = 0.375 m`。`WALK_TURN_LEAD = R + 0.20`：理论少退一个半径，再提前 20 cm，抵消 wz 爬升和滑行把世界 X 走出进入线。落点 X 偏了只调 lead。
- 距 backoff 目标 `WALK_TURN_PREVIEW=0.25 m` 就开始带 `wz`（mode `reverse_preturn`），转完时 X 应对齐放置站，第③段主要走世界 +Y，少用侧向 −X。
- **转弧带的是后退不是前进**：后退是唯一实测能迈步的模式；而且弧终点被推到 `y=-1.705`，第③段留 0.93 m 行程，`stop_margin` 才有作用空间。若改成前进转弧，转完只剩 0.18 m 到桌前，而桌沿余量只有约 0.02 m，一过冲就撞桌。
- 转弧全程 x 单调增大，是**远离** CES 的方向；最靠近 CES 的时刻就是弧起点（现在约 `x=-2.84`）。
- `WALK_BACKOFF_TRIM` 已归零：它和 `WALK_TURN_LEAD` 同轴同向，"宁可退不够"的作用由 lead 承担，两个一起留会退得太少、贴着 CES 转弧。
- **pick→walk 起步**：CARRY 立刻满幅 S（`prime_walk_filt`），并且必须 `_begin_walk_policy` 清空 10 帧站立观测 + 后退 kick。只发满幅 S 仍会先往前走两步，见 §6.6。
- **最后一段只走世界 +Y**：右转结束后只发机体系 `+vx`（W），这就是世界 +Y。不再发 `vy` 纠世界 X，也不再 `forward_realign` 的 −vx。X 残差留给手臂。
- 漂移保护：转弧走了 `WALK_TURN_MAX_DRIFT=0.70 m` 还没转到位，说明策略没吃下 `wz`，此时把幅值提到 `WALK_WZ_MAX=1.55` 并**把弧反向**（mode 记作 `turn_pinned`，日志打醒目告警），边把漂移走回去边继续试。**注意这里不能砍掉 `vx`**：`vx` 是唯一能触发步态的轴，归零只会原地冻住、白等 60 s 超时。
- `WALK_YAW_ARRIVE=0.40` 是**转向的 stop_margin**（0.15 → 0.25 → 0.40）：实测在 0.25 处松手后还会多转约 14°，即余转合计约 28°，所以提前到约 23° 松手。落点 `head=70°`（目标 90°）就是松手太晚的证据。
- 灰色托盘中心 = `PLACE_TARGET_XY`（桌心 `PLACE_TRAY_SHIFT_X=-0.15`，近沿与桌 AABB 对齐后再 `PLACE_TRAY_EDGE_INSET_Y=+0.03`，避免筐沿探出桌面）。**放置站不跟桌子 −Y 走**，仍为 `(-2.267, -0.772)`。
- 走位交点由 `build_carry_route` 从**放置站**算出，挪托盘不会改路线。
- 进度按**行走轴投影**算（不是欧氏距离），侧向漂移不会让某段转不完。
- 平移中的纠偏也是固定幅值 + 带回差的死区：朝向差 > `WALK_ALIGN_YAW=0.20` 才发 `wz`，> `WALK_REALIGN_YAW=0.60` 就放弃行进轴、**带着 `turn_vx` 弧线转正**（realign 以前发纯偏航，同样不起步，一起改了）；侧向差 > `WALK_LATERAL_TOL=0.10` 才发 `vy` 侧移。`align_yaw` 从 0.30 收到 0.20 是因为转弧收尾会留约 14° 残差，0.30（17°）够不到它 —— 日志里 `err=+14deg` 而 `cmd_b` 第三位一直是 0，机器人就那么歪着走完最后一段。
- **最后一段：停止线是唯一不可谈判的条件**（`WalkLeg.lateral_first=True`）。这条经历了三次撞桌才定下来，务必别再往回改：
  1. 第一版 `_leg_done` 只看行走轴投影，Y 一到就宣布到站，且到位后前进指令不停 → 偏在桌子一侧撞桌。
  2. 于是加了"侧向/朝向没到位就不算到站"。但**所有纠偏都需要非零 `vx`**，所以"继续纠偏"等于"继续往前开"—— 朝向差一点点就能让机器人一路开进桌子。实测机器人摔在地上。
  3. 现在：**`remaining <= stop_margin` 就地停死并闩锁**（mode `forward_stopped`），侧向/朝向还差多少都不管。姿态窗口 `WALK_LATERAL_ARRIVE` / `WALK_YAW_ARRIVE_FINAL` **只用于报告**（`WalkStep.on_target`，不达标打 off-target 告警），不再参与判停。
  - **闩锁之后不许倒车重试**。倒出去再进来意味着继续在桌边走动，而出来那一下的滑行正是撞桌的原因。旧的 `forward_recover` / `forward_align_back` / `WALK_OVERSHOOT_TOL` / `WALK_REAPPROACH_RUNWAY` 全部删掉了。
  - 侧向纠偏只发生在**进站途中**（mode `forward_align`，斜行 `vx`+`vy`）。**绝不能把 `vx` 归零**：那样发出 `(0,+0.30,0)`，机器人在放置站前一动不动 18 s。
  - 姿态窗口仍**不能小于对应纠偏的进入阈值**，否则永远报不出 on_target。
- **`stop_margin` 必须 ≥ 滑行量，这是最关键的一个数**。`vx=0.45` 松手后滑行约 0.25 m，而旧的 `WALK_STOP_MARGIN_PLACE=0.20` 比滑行量还小 → 到 0.20 才停必然冲过站点。又因为 `vx` 低于死区根本不迈步，**没法缓刹**，唯一手段就是提前松手。现在 `0.30`。
  - 到站日志直接算出 **`coast = margin - along`** 和 `arm_reach`。**下次实机跑完就按 `coast + 0.05` 重设这个余量**，别再猜。
  - `along` 必须 >0（停短）。Place 15 后只跟 Z、不追灰筐中心；下降时 XY 可随肩膀偏移。安全上仍是停短优先，绝不能越过撞桌。
  - `X_B_PLACE=0.46 m` 现在只用于放置站几何和日志，不再作为 Place IK 的 X 目标。等实测 `coast` 出来仍应收紧停车余量，减少放置 XY 漂移。
- **硬性禁入闩锁（兜底，不依赖规划逻辑）**：`_table_keepout_hit` 只看骨盆相对放置站的投影，越过 `WALK_TABLE_AHEAD_OF_STAND(0.02) - WALK_TABLE_SAFE(0.06)` 就无条件零指令 + 告警 + 就地放置。前两次撞桌都是**规划分支漏了停止条件**，所以最关键的那个"停"不能只写在规划里。
- 走路时不调用 `_apply_snap`、不移动 Product root；右臂保持 `_carry_arm_q`、Dex1 保持 `GRIPPER_CLOSED`，件仍只靠垫面摩擦。
- 到站稳定 `WALK_ARRIVE_HOLD=0.35 s` 后锁实际骨盆位姿，日志打 `dxy/dyaw` + **分解到行走轴的 `along`（+=停短）和侧向 `side`（−=偏左）** + `coast` + `arm_reach`。旧故障会打成 `along=-150mm side=-300mm`。
- 走路到站后锁实际骨盆位姿；Place 不追世界灰筐中心。15 后 IK 只跟 Z，肩膀可转，X/Y 允许偏移。旧记录里的 `PLACE_HOLD_XY_M_WALK` 常量代码里并不存在。

纯 CPU 测试：`python -m unittest tools.test_ces_walk_navigation`（27 项）。死区模型是"|vx|<0.30 / |vy|<0.25 / |wz|<0.40 不动"，外加最关键的一条：**`vx` 低于死区时整条指令作废**（`vy`/`wz` 再大也不动），用来复现两次真实故障 —— 原地纯偏航转不动、放置站前纯侧移冻住 18 s。`test_body_reverse_at_pick_yaw_is_world_plus_x` 守住 pick 朝向 π 时 S 必须是世界 +X。

**`test_no_approach_ever_reaches_the_table` 是这里最重要的一个用例**：900 组组合扫描（进站距离 × 侧偏 × 朝向差 × 偏航增益 × 侧移增益 × 指令滞后），断言骨盆**任何时刻**都不许到达桌沿。教训是**单点工况测不出撞桌**：前两次我都只测了标称路线加一两个位姿，全过；一上组合扫描立刻暴露 42/900 撞桌。杀死机器人的是"歪着起步 + 跟踪失配 + 指令滞后"三者叠加，必须一起扫。

改完的效果（同一 900 组）：
- 撞桌 **42 → 0** 组。**其中光是"停止线闩锁"就把 42 组降到 0**，`stop_margin` 0.20→0.30 是再加一层余量（最大越过量 +105mm → -2mm，离桌沿从 15mm 拉到 112mm）。
- `yaw_gain` 0.7–1.5、`lat_gain` 0.7–1.0、滞后 0.25–0.45 s 全覆盖。
- 停短范围 0~30 cm，宽度几乎全来自**滑行量未知**（滞后 0.25 vs 0.45）。实机滑行是个固定值，跑一次按日志的 `coast` 就能把这个带压窄。

### 6.3 Walk 的五次实测故障（按时间顺序，每一次都推翻了上一次的假设）

全身 RL 策略按**默认垂臂站姿**训练，夹持后右臂前伸是 OOD，所以只能靠实测逼近：

1. **负 `vx` 过小 → 只后仰不迈步然后倒下**。靠 `WALK_VX=0.45` 解决（历史上 0.25–0.28 仍弹飞 / 后仰踢腿）。
2. **原地纯 `wz` 不转**：后退到位后 settle 0.5 s 再发 `(0,0,-0.70)`，人停住仍朝柜子。三因叠加：① 0.70 只有键盘上限 1.57 的 45%；② `vx=vy=0` 时策略退化成静止站立；③ 转向前还先 settle 成静止。改法：`wz=1.20` + 转向骑在后退指令上画弧 + 取消转向前的 settle。
3. **纯侧移也不迈步**：为了"先侧移上线"把 `vx` 归零，发出 `(0,+0.30,0)`，机器人在放置站前站了 18 s 一动不动。**这一条才定死了"只有 `vx` 能触发步态"**（§6.1）。
4. **为了摆正姿态一路开进桌子并摔倒**：上一条的直接后果 —— 既然纠偏必须带 `vx`，"姿态没到位就不算到站"就等于"没摆正就一直往前开"；再加上 `stop_margin(0.20) < 滑行量(0.25)`，必然冲过站点。改法：停止线闩锁 + 硬性禁入兜底（§6.2）。
5. **满幅 S 仍先往前走两步再后退**（2026-08-24）：08-23 只修了指令滤波，没修 10 帧 `vx=0` 观测栈。策略把松钉当成站立起步，先朝 CES 收两步。详见 §6.6。

其他仍然成立的约束：钉盆 + 策略蹬腿会物理爆炸；`vx=0` 松钉让策略"接腿"是已知摔倒模式；观测欺骗（假垂臂 / 假重力）会自相矛盾；不要滑移、不要假迈步 overlay。键盘 `S` 能退是因为手臂垂着、指令有斜坡、策略一直控腿。**不要用假垂臂观测来消 pick→walk 的前两步。**

### 6.4 Walk 验收清单

- [x] 命令机体系 `[vx,vy,wz]`；固定幅值 + 死区，不用比例控制
- [x] 路线不穿 CES：后退（世界 +X）离开机器后才转身；转弧 x 单调增大，也是远离 CES
- [x] 后退段真的迈步，能停到 backoff 点
- [x] 右转 90°：`vx=-0.45` + `wz=-1.20` 转弧、转向前不 settle
- [x] 侧向纠偏必须斜行（`vx` 永不为零）
- [x] 靠近桌子就停死不再前进：停止线闩锁 + 硬性禁入兜底，900 组扫描 0 撞桌
- [x] 走到放置站范围（Isaac Sim 2026-08-19；2026-08-23 再验：转完直接 W 能走到 place）
- [x] spawn 直接在抓取站，不再瞬移
- [x] 夹爪全程 PD 闭合，产品靠摩擦跟着走，不焊 TCP
- [x] 2026-08-23 walk优化：转弧→前进不停、直接发 W；pick 完立刻满幅 S，不等站稳；walk→place 夹爪目标锁死、到站钉实际骨盆 Z；托盘 −X 15 cm、站位不动
- [x] 2026-08-24 pick→walk 直接后退：清空 10 帧站立观测 + 骨盆/产品后退 kick（§6.6）。**满幅 S 单独不够。**
- [x] 2026-08-24 walk→place 夹持：到站钉 live quat，手臂 PD、夹爪锁死、钉盆不跑步态（§6.7）。Isaac Sim 确认件不再掉。
- [x] 2026-08-24 `pick_baseline_ok`：DESCEND 先 yaw 对齐世界 X 再落 Z；夹爪沿产品短边夹。
- [x] 2026-08-24 桌面相对 LoadingLine 底 +2.0 cm（05→15 剐托盘后连降）。完整桌高 0.99 m 太高，不要再拉回。
- [x] **人工重设计 05 胸前姿态**：walk 持物可用。
- [x] **15 放置终点纳入 baseline**（`5c51680` 手动调整）。15 到位即松爪，不再 Z-only 下降。
- [ ] **代码简化 / 优化（不改功能）**：见 §12。
- [ ] 按日志 `coast` 把 `WALK_STOP_MARGIN_PLACE` 从 0.30 收到 `coast + 0.05`，压掉最坏情况 75 cm 的前伸
- [ ] 到站 `dxy` 落在 ~10 cm 内、`along>0`、`on_target` 不报 off-target

### 6.5 HOLD 快照（给 walk 到 place 的双足控制）

HOLD 在 **pick 站**冻臂，尚未 snap。走路换站应保持这套右臂 q 和夹爪闭合，件靠摩擦跟着走。

**站位 / TCP（v2 基线 lift_done 实测；2026-08-24 新05接入后的真实 HOLD TCP 待重跑更新）：**

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
| HOLD / `_carry_arm_q`（当前 Smooth v1） | `05_chest_carry = [-0.04000009, -0.34000003, +0.52000004, -0.74999970, -0.34000000, -0.09000000, +0.90999967]` |
| V2 authored 05（历史伸手路点，不是当前 HOLD 05） | `[-0.55, -0.15, +0.22, +0.90, +0.20, +0.10, +0.35]` |
| 当前 authored 10 | `[-0.70, -0.22, +0.38, +0.75, +0.32, -0.05, +0.55]` |

下次跑通 HOLD 时看日志：`[ces_fsm] HOLD tcp_w=... tcp_b=... q=[...]`，用新数替换上表。

参考（垂臂、不钉盆）任务：`Isaac-Move-Cylinder-G129-Dex1-Wholebody`。

### 6.6 pick→walk 必须直接后退，禁止先往前凑步（2026-08-24，非常重要）

用户要求：pick 一结束就直接后退离开 CES，**不要先往前走两步**。

#### 两次修法，不要搞混

| 日期 | 现象 | 当时以为的原因 | 真正有效的修法 | 不够的修法 |
|---|---|---|---|---|
| 2026-08-23 | 抱件往前栽进 CES | 等站稳再松钉、指令从 0 ramp | `prime_walk_filt`：CARRY 立刻切 GOTO_PLACE，第一帧满幅 `vx=-0.45`，不等 `CARRY_TIME` / `is_standing` | 再加站稳等待（会更差） |
| 2026-08-24 | **已经满幅 S，仍先往前走两步再后退** | 以为 08-23 的修法被改回去了，或该再等站稳 | **清空观测栈 + 后退速度 kick**（下面） | 只发满幅 S；等站稳；加大 `WALK_VX`；假垂臂观测 |

08-23 修的是**指令通道**（滤波从 0 ramp / 过 0 停步）。08-24 修的是**策略看到的历史**。两条要同时在，缺一不可。

#### 根因

全身策略输入是 **10 帧堆叠观测**（`CircularBuffer max_len=10`，`dt=0.02` → 0.2 s）。内容含 `command=[vx,vy,wz,height]`。

pick 全程钉盆、`policy_active=False`，每帧仍调用 `compute_observations([0,0,0,0.8])`。所以松钉前 10 帧全是「站立、无行走命令」。

CARRY 虽然第一帧就把 `cmd` 设成 `[-0.45,0,0,0.8]` 并 `run_policy`，但栈里仍是 9 帧零命令 + 1 帧后退。策略把它当成**从站立起步**：先朝面对方向（CES，世界 −X）收 1～2 步，等历史换成后退命令才真退。这两步看起来像「先往前走再后退」，指令日志却一直是 `cmd_b=(-0.45,0,0)` —— 所以会误判成没发 S。

新 05 贴胸、产品更居中，松钉后质心更容易正前方栽，把这个「收集步」放大了。旧 05 偏右时没这么显眼，所以 08-23 会以为满幅 S 已经够。

DelayBuffer 默认 `time_lag=0`，**不是**动作延迟导致的。不要往动作 delay 上查。

#### 禁止再走的路

- **不要等站稳再发 S**。钉盆时 `is_standing()` 第一帧就真；等完再松钉 + 从 0 ramp = 08-23 已经修掉的前倾栽 CES。
- **不要钉盆同时跑策略**（钉盆 + 策略蹬腿会物理爆炸）。
- **不要伪造垂臂 / 重力观测**来骗策略后退（自相矛盾）。
- **不要因为「已经满幅 S 了」就以为起步没问题**。看的是观测栈，不是当前 `cmd`。

#### 正确修法（已写入代码）

松钉、第一次 `policy_active` 时调用 `_begin_walk_policy`（`action_provider_ces_grasp.py`）：

1. `actor_obs_buffer.reset()`：丢掉 pick 阶段 10 帧 `vx=0`。随后 `run_policy` 的第一次 `append` 会把整栈填成当前姿态 + 后退命令，策略从第一帧就看见「一直在下后退令」。
2. `action_buffer.reset()` 再 `compute(zeros)`：last_action 也从站立零动作起，不要带着 pick 阶段塞进去的腿目标。
3. `_kick_walk_start_velocity(vx_body)`：按当前 heading 把机体系 `vx` 写成世界速度，`write_root_velocity_to_sim` 给骨盆；若正在夹持，产品写**同一**世界速度，避免件被瞬间甩脱。pick yaw=π、`vx_b=-0.45` → 世界 `(+0.45, 0)`，即离开 CES 的 +X。

触发点：`prime_walk_filt` 被消费的那一帧（CARRY→GOTO_PLACE，`guide=True`）。之后滤波仍拉满，不从 0 ramp。

#### 日志（关仿真重开后必须看到）

```text
[ces_fsm] pick done — S backoff now (flush stand obs, kick reverse)
[CESGrasp] walk start kick vx_b=-0.45 world=(+0.45,-0.00) (no forward collect)
```

验收：pick 结束骨盆应立刻往世界 +X 走，不应先靠近 CES。若 kick 行的 `world.x` 为负，说明 heading 不是 π 或符号写反，先查 yaw，不要改路点 q。

#### 代码位置

```text
action_provider/action_provider_ces_grasp.py   # _begin_walk_policy, _kick_walk_start_velocity
action_provider/ces_grasp/state_machine.py     # CARRY 立刻 prime + GOTO_PLACE
tools/test_ces_walk_navigation.py              # test_body_reverse_at_pick_yaw_is_world_plus_x
```

### 6.7 walk→place：钉盆但不要拧夹爪（2026-08-24 傍晚，Isaac Sim 已确认件不掉）

Pick 和 walk 全程摩擦夹持是稳的。件在**到站切 place** 时飞掉，空爪再被 PD 收到 `0.019`。

#### 根因（按实测顺序，不要回退）

| 试法 | 结果 | 原因 |
|---|---|---|
| 到站立刻 `_apply_snap`（纯 yaw + 可选 `STAND_PELVIS_Z`） | 件飞 / 夹爆 | 走路残留俯仰被写成 yaw-only，夹爪甩一下，接触断 |
| 到站后改持物 kp=400、锁 live 夹爪 q | **pick 夹不起来** | 抬起前闭合力不够。已全部改回 `0.019` + `kp=1800` |
| 到站不定盆、冻最后一帧腿做 05→15 | 人后仰，件仍掉 | 伸手改质心，冻腿顶不住 |
| 到站不定盆、05→15 还跑 `vx=0` 策略 | 整机飞起 | 与「钉盆+策略爆炸」同类；不定盆伸手策略也 OOD |
| 夹持时每帧把 live 腰写进 `full_action` | **walk 上身歪、件中途掉** | 覆盖了行走策略的腰。走路必须仍用 `default_waist` |

#### 正确修法（已写入，Isaac Sim 确认）

到站（`WALK_ARRIVE_HOLD` 之后）：

1. 锁**实际**骨盆 `xy / z / 完整四元数`（`snap_quat`），**不要** `yaw_quat(heading)`。
2. `PLACE_HOLD` 约 `WALK_PLACE_HOLD_TIME=0.45 s`：右臂冻在当时 live q（贴近 05），夹爪仍锁 `GRIPPER_CLOSED=0.019`，`kp=1800`。
3. 钉盆时 **guide=False**，不跑步态；腿用走路最后一帧 `_last_policy_legs`，腰仍是默认值。
4. 夹持中手臂只 PD，禁止 `write_joint_state` 手臂/夹爪。
5. 然后 05→15，骨盆一直钉着同一 live quat。

日志应看到：`pin live pelvis quat, arm PD, gripper locked, no gait`。

#### 禁止再走的路

- 不要为了「place 时人别倒」就在钉盆期间开策略。
- 不要为了「别夹爆」就在抓取阶段降 kp / 锁 live 夹爪 q。
- 不要在 walk 期间用 live 腰覆盖策略。
- 不要把产品焊到 TCP。

#### 代码位置

```text
action_provider/ces_grasp/state_machine.py     # PLACE_HOLD, _place_lock_quat, _place_body_cmd
action_provider/action_provider_ces_grasp.py   # _apply_snap(..., quat=), 钉盆不改夹爪/手臂运动学锁
action_provider/ces_grasp/constants.py         # WALK_PLACE_HOLD_TIME
```

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
- 本阶段归档 commit：`Baseline_done`
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

### 当前默认：Smooth Pick v1（2026-08-24）

- 姿态目录：`action_provider/ces_grasp/poses/ces_pick_smooth_v1/`。
- 正向 q 保持 V2 已验证的 `00/10/20/30/40` 不变，正向只去掉冗余 `05_forward_reach/25`；逆向终点使用用户 2026-08-24 重新设计的 `05_chest_carry`。
- 运行时硬下发：`00→10→20→30`；时长 `1.75 / 2.07 / 1.20 s`。`30→40` 的 `1.20 s` 仍是动态 `q_ref` 参考，不把 40 当 `arm_q`。
- `trajectory_manifest.json::return_path`：逻辑 `40→30→20→05`；40 使用抬起后的实时 q，不硬下发 q_ref 文件。原始时长 `0.8 / 1.2 / 3.0 s`，默认 1.5× 后 `0.53 / 0.80 / 2.00 s`。单调三次 Hermite 经 20 退出抽屉，再连续收到新 05。
- `pose_library.py` 优先读取独立 `return_path`；旧 V1/V2 没有该字段时，兼容回退为“实时 q→正向路点逆序→00”。
- 默认已同时改到 `sim_main.py --ces_waypoint_set` 与 `constants.py::WAYPOINT_SET_DEFAULT`；V2 / V1 均保留，加载未声明插值的旧清单仍用原 `segment_smoothstep`，兼容旧行为。
- 本地 `urdf-viz` 已完整预览正向 `00→10→20→30→40`，并由用户完成新版逆向 `30→20→05` 和 Place `05→15` 设计；CPU 合约测试命令：

```bash
python -m unittest tools.test_ces_smooth_pick -v
```

- **Isaac Sim 验收边界**：阿里云实测已证明旧 `30→05` 会刮抽屉；本轮新的 `30→20→05`、05 持物 walk 和 `05→15` 目前只完成 URDF-viz 设计和代码接入。仍需重新证明产品不刮边、不掉落、不碰胸、walk 平衡稳定，以及 Place 下降/松爪不碰托盘沿。

### 10.5 Pick 提速：`--ces_pick_speed`（2026-08-21）

**当前受 `--ces_pick_speed` 影响的关节段：**

| 阶段 | authored | 默认 1.5× 后 |
|---|---|---|
| UNFOLD `00→10→20→30` | 5.27 s | 3.51 s |
| LIFT | 2.2 s | 1.47 s |
| RETURN_HOME `40(live)→30→20→05` | 5.00 s | 3.33 s |
| **上述三段合计** | **12.47 s** | **8.31 s** |

SETTLE / GOTO_PICK / DESCEND / GRASP 等物理或稳定等待不按这个倍率缩放。每段缩放后仍受 `PICK_SEGMENT_MIN_TIME=0.40 s` 下限保护。

**机制**：`scale_segment_times(durations, scale, min_time)`（`manip_common/interpolation.py`）把每段时长除以倍率。**均匀时间缩放不改关节空间曲线**，只把每个速度乘以倍率 —— 所以 URDF-viz 里确认过的姿态、`monotone_cubic_hermite` 的连续性、不越界性质全部原样成立。状态机在 `__init__` 里一次性算好 `_joint_segment_times / _return_segment_times / _lift_time / _wp_lead_in_time`，轨迹和超时判定共用同一套数，不会出现"超时先于轨迹完成触发"。

**为什么 DESCEND / GRASP 不缩放**：DESCEND 是锁 XY 只落 Z 的 DiffIK 段，压快会让 TCP 跟不上，`_start_grasp` 冻结的实际 Z 就偏高 → 抓空；GRASP 是夹爪闭合的物理时间，和轨迹无关。SETTLE / GOTO_PICK / CARRY / HOLD 是平衡策略的稳定等待，同理不动。

**上限依据**：`_slew_arm` 每步把关节增量截断在 `ARM_SLEW_RAD = 0.080 rad`，控制周期 `dt = 4 × 0.005 = 0.02 s`，所以下发速度天花板是 **4.0 rad/s**。2026-08-24 路径 authored 峰值为正向 `0.887 rad/s`、逆向 `0.734 rad/s`；默认 1.5× 的逆向峰值约 `1.10 rad/s`，仍有足够余量。另一半限制来自产品只靠垫面摩擦夹持，LIFT / RETURN_HOME 太快仍可能甩脱。

**调法**：`--ces_pick_speed 2.0` 等。启动日志打印 `arm_speed=x1.50` 和 `seg_s=1.17/1.38/0.80 (authored 1.75/2.07/1.20)`。**如果提速后掉件或抓偏，先降的是这个倍率，不要动路点 q。**

- **待办：1.5× + 新05 的物理验收**。重点看40(live)→30是否跳变、30→20是否真正把产品带离抽屉边缘、20→05时产品是否碰胸/滑动、walk时新贴胸持物姿态是否扰动平衡。CPU 侧只证明“清单正确、曲线不越界、峰值速度仍在限幅内”。

### 姿态调优 V2（历史已完成，可回退）

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

（Pick 已完成并归档，上面这些是当时的标定记录，保留作为几何依据；不要再用 XR 掌心 X 误差反推 q。）

---

## 11. Place：人工 05→15，15 到位松爪（`Baseline_done`）

### 11.1 当前关节路径

Place 只保留两个人工关节姿态，不使用 25。walk 到站钉 live quat 后 `05→15`，**15 到位即松爪自由落体**，再 `15→05` 收臂。`PLACE_RELEASE_FROM_15=True`；Z-only IK 代码仍留着但默认不跑（下降会碰灰筐沿）。

```text
05_chest_carry = (-0.04000009, -0.34000003, +0.52000004, -0.74999970, -0.34000000, -0.09000000, +0.90999967)
15_place_forward_release = (-1.2266918, -0.2318979, +1.4931674, +1.2198853, -0.042756237, +0.005500171, +1.1593101)
  ↑ 5c51680 手动调整后纳入 Baseline_done
```

正式文件：`ces_pick_smooth_v1/05_chest_carry.json` 与 `15_place_forward_release.json`。

`trajectory_manifest.json::place_path` 定义 `05→15`，原始时长 `3.2 s`；默认 1.5× 后约 `2.13 s`。`WALK_PLACE_HOLD_TIME=0.45`。

**本 baseline 约束：**

- 不要为了够筐去改钉盆、夹爪 kp、桌高，或重新加入世界 X/Y IK。
- 05 仍是 walk 全程持物姿态，也是回收终点；改 05 等于同时改 walk 平衡。
- 桌面相对 LoadingLine 底 +2.0 cm（`TABLE_TOP_EXTRA_Z=0.020`）。完整桌高 0.99 m 用户否决，不要再拉回去。

### 11.2 默认不再做 Z-only 下降

Z-only IK（`PLACE_DESCEND`）保留为开关：`PLACE_RELEASE_FROM_15=False` 才会走。默认 15 松爪后收臂。

### 11.3 验证边界

**Isaac Sim 已确认（截至 2026-08-24 晚 `Baseline_done`）**：pick（含 yaw 对齐、缩短等待）→ walk（直接后退）→ 到站钉 live quat → PLACE_HOLD 0.45 s → 05→15 → 松爪自由落体 → 15→05。墙壁显示；默认 `PerspectiveCamera_robot`。

抽屉外观仍白，不影响任务。

---

## 12. 下一阶段 TODO：简化 / 优化代码（不改功能）

`Baseline_done` 之后**只做重构**，行为必须与本版一致。不要借重构改路点、桌高、夹爪、walk 指令或 Place 时序。

建议范围（可拆小 PR）：

- **状态机**：`state_machine.py` 约 1850 行，Pick / Walk / Place 缠在一起。按阶段拆文件或拆方法，枚举与转移表保持不变。
- **死开关与遗留路径**：`PLACE_DESCEND`、笛卡尔 Place、旧 V1/V2 回退、`hide_warehouse_walls`、抽屉 `recolor_loading_line_drawer`（改色未生效，可删或挪到明确的可选钩子）。删前确认 CLI / manifest 没有依赖。
- **常量与注释**：`constants.py` / `memory.md` 里过时的 Z-only、15 占位、桌高实验记录收到附录，运行时只留当前值。
- **场景启动**：`ces_scene_startup` 里清箱子 / 放灰筐 / 摩擦 / 墙壁 分函数已有，整理重复的 `Usd.PrimRange` 绑定。
- **测试**：现有 `tools/test_ces_*.py` 对齐「15 松爪、无 PLACE_DESCEND」的默认路径；不要为了绿测去改仿真行为。

验收：同一启动命令，pick-walk-place 时序、路点 q、桌高、钉盆规则与本 commit 一致。
