# K1 机器人点球射门 — 强化学习训练管线 设计与实现

---

## 目录

1. [背景：这个项目是什么](#1-背景这个项目是什么)
2. [核心概念：强化学习是什么](#2-核心概念强化学习是什么)
3. [整体架构：三层结构](#3-整体架构三层结构)
4. [场景设计：点球大战的物理世界](#4-场景设计点球大战的物理世界)
5. [机器人模型：K1 双足人形机器人](#5-机器人模型k1-双足人形机器人)
6. [观测空间：机器人"看到"什么](#6-观测空间机器人看到什么)
7. [动作空间：机器人"做什么"](#7-动作空间机器人做什么)
8. [PD 控制器：从数字到力矩](#8-pd-控制器从数字到力矩)
9. [奖励函数：机器人"学到什么"](#9-奖励函数机器人学到什么)
10. [终止条件：回合何时结束](#10-终止条件回合何时结束)
11. [回合重置：如何重新开始](#11-回合重置如何重新开始)
12. [代码结构：文件清单与职责](#12-代码结构文件清单与职责)
13. [MotrixLab 注册机制](#13-motrixlab-注册机制)
14. [训练配置：PPO 算法参数](#14-训练配置ppo-算法参数)
15. [运行方式：如何启动训练](#15-运行方式如何启动训练)
16. [当前状态与下一步](#16-当前状态与下一步)

---

## 1. 背景：这个项目是什么

### 1.1 sim-soccer2 项目概况

**sim-soccer2** 是一个**双足机器人足球仿真与决策控制**项目。目标机器人是 **Unitree K1**（22 自由度双足人形机器人）和 **Pi Plus**（20 自由度人形机器人）。

项目的核心工作方式：

```
人类编写的规则程序 (decider/)
        ↓  通过 ZMQ 发送速度指令 [vx, vy, ω]
物理仿真引擎 (Sim Manager / MotrixSim)
        ↓  神经网络把速度指令转换成关节动作
机器人模型在 3D 世界里走动、踢球
```

### 1.2 为什么要做强化学习

当前的机器人决策完全由人工编写的规则控制（`decider/` 目录下的 Python 状态机）。人类告诉机器人"球在你的左前方，所以向左转，然后前进"。这种方式的局限性：

| 规则控制 (当前) | 强化学习 (目标) |
|---|---|
| 人类穷举所有情况 | AI 自己试错发现最优策略 |
| 遇到新情况容易出错 | 泛化能力强 |
| 难以优化细节（如射门角度） | 能学到精妙的射门技巧 |
| 一个场景写一套规则 | 一个模型适应多种场景 |

本项目选择**点球射门**作为第一个 RL 任务，因为它场景简单、目标明确、容易验证。

### 1.3 本 RL.md 文档的目的

本文档记录了从零开始在 sim-soccer2 项目中构建强化学习训练管线的**完整设计与实现**。目标读者是希望理解这个系统的任何人——无论你是否有 RL 背景。

---

## 2. 核心概念：强化学习是什么

### 2.1 强化学习的直觉

强化学习（Reinforcement Learning, RL）就是让 AI 通过**试错**来学习。想象训练一只狗：

- 狗做了一个动作（坐下）
- 你给它零食（奖励）
- 狗学会了"坐下 → 有零食吃"

RL 完全一样，只不过"狗"是一个神经网络，而"零食"是一个数学函数。

### 2.2 核心循环：Agent 与环境

```
      ┌──────────────────────────┐
      │      Agent (神经网络)      │
      │  输入：obs (我看到了什么)    │
      │  输出：action (我要做什么)   │
      └────────┬─────────┬───────┘
               │         │
          obs  │         │  action
               │         │
      ┌────────┴─────────┴───────┐
      │     Environment (物理仿真) │
      │  接收 action，推演物理      │
      │  返回 obs, reward, done   │
      └──────────────────────────┘
```

这个循环每秒执行 50 次（50Hz 控制频率）：
1. **Agent** 接收观察 `obs`（57 个数字，描述机器人状态和球的位直）
2. **Agent** 输出动作 `action`（12 个数字，控制 12 个腿部关节）
3. **Environment** 在物理引擎中执行这些关节动作，推进 0.02 秒
4. **Environment** 返回新的观察 `obs`、奖励 `reward`、是否结束 `done`

### 2.3 PPO 算法（Proximal Policy Optimization）

本项目使用 **PPO** 算法，它是最主流的强化学习算法之一。核心思想：

- **Policy（策略网络）**：决定"在某个状态下应该做什么动作"。输入是 57 维观察，输出是 12 维动作。
- **Critic（价值网络）**：预测"在当前状态下，未来能获得多少奖励"。用于指导 Policy 优化。
- **训练过程**：收集一批经验 → 评估好坏 → 更新网络 → 重复。每次更新只做"小幅调整"（Proximal），避免学坏了。

PPO 的优势：稳定、易于调参、适合各种机器人任务。

### 2.4 大规模并行训练

一个关键技巧：**同时运行成百上千个仿真环境**。

```
环境 1: 机器人尝试踢球 → 摔倒
环境 2: 机器人尝试踢球 → 踢偏了
环境 3: 机器人尝试踢球 → 进球了！
...
环境 2048: 机器人尝试踢球 → 正在尝试
```

每个环境独立运行，收集所有环境的经验，一次性更新神经网络。这样数据效率高，训练速度快。

本项目设计支持 **256~2048** 个并行环境。

---

## 3. 整体架构：三层结构

```
┌──────────────────────────────────────────────────┐
│  scripts/train.py  (训练入口)                      │
│  - 解析命令行参数                                   │
│  - 创建 PPO Trainer                               │
│  - 调用 trainer.train()                           │
└────────────────────┬─────────────────────────────┘
                     │
┌────────────────────┴─────────────────────────────┐
│  motrix_rl/  (RL 算法层)                          │
│  - SKRL/RSLRL PPO 算法实现                        │
│  - RslrlNpEnvWrap: 把环境适配到 RSLRL 接口         │
│  - 模型保存 / TensorBoard 日志                     │
└────────────────────┬─────────────────────────────┘
                     │
┌────────────────────┴─────────────────────────────┐
│  motrix_envs/  (环境层) ← 本次主要实现              │
│  - PenaltyShootoutEnv(NpEnv)                     │
│  - reset(): 重置场景，球放罚球点                    │
│  - step(action): 执行动作，推进物理                 │
│  - _get_obs(): 构建 57 维观察                     │
│  - _compute_reward(): 计算奖励                    │
└────────────────────┬─────────────────────────────┘
                     │
┌────────────────────┴─────────────────────────────┐
│  MotrixSim  (物理引擎层, C++ compiled)             │
│  - 加载 MJCF 场景 XML                             │
│  - 物理步进 (500Hz)                                │
│  - 碰撞检测、传感器读数                              │
│  - 批量并行 (num_envs 个独立物理世界)                │
└──────────────────────────────────────────────────┘
```

**关键设计原则**：不修改物理引擎和决策器代码（`simulation/motrixsim/` 和 `decider/` 完全不动），只在 MotrixLab 框架内新建环境。

---

## 4. 场景设计：点球大战的物理世界

### 4.1 坐标系与场地布局

```
                    球门 (x=4.5)
              +-------+-------+
              |   |   |   |   |
              |   |   |   |   |  ← 高 1.8m
              |   |   |   |   |
              +-------+-------+  ← 宽 1.9m
                    ⬆
                   球   罚球点 (x=3.0, y=0, z=0.11)
                    ⬆
                   🦿   机器人 (x=2.5, y=0, z=0.55)
                    0

  场地球场: 10m × 7m  (x: -5.0 到 5.0, y: -3.5 到 3.5)
  罚球点距离球门: 1.5m  (球门在 x=4.5, 罚球点在 x=3.0)
  机器人初始位置: 罚球点后方 0.5m
  机器人朝向: +x 方向 (面向球门)
```

### 4.2 场景 XML 文件

场景定义在 `scene_penalty_shootout.xml` 中，包含：

```xml
<mujoco model="K1 penalty shootout">
  <!-- 1. 物理参数 -->
  <option timestep="0.005"/>  <!-- 500Hz 物理步进 -->

  <!-- 2. 地面 -->
  <geom name="ground" type="plane" rgba="0.18 0.45 0.18 1"/>
  <!-- 边界墙 (防止球滚太远) -->
  <geom name="wall_left" type="box" .../>

  <!-- 3. 球门 -->
  <geom name="crossbar" type="cylinder" .../>  <!-- 横梁 -->
  <geom name="post_left" type="cylinder" .../>  <!-- 左门柱 -->
  <geom name="post_right" type="cylinder" .../> <!-- 右门柱 -->

  <!-- 4. 足球 (自由运动物体) -->
  <body name="ball" pos="3.0 0 0.11">
    <joint name="ball-root" type="free"/>  <!-- free joint: 6 自由度 -->
    <geom name="ball" type="sphere" size="0.11"/>  <!-- 半径 11cm -->
  </body>

  <!-- 5. K1 机器人 (22 关节, 完整运动学树) -->
  <body name="Trunk" pos="2.5 0 0.55">
    <joint name="world_joint" type="free"/>
    <site name="imu"/>  <!-- IMU 传感器安装点 -->
    <!-- 头部 (2 关节) -->
    <!-- 左臂 (4 关节) -->
    <!-- 右臂 (4 关节) -->
    <!-- 左腿 (6 关节) -->
    <!-- 右腿 (6 关节) -->
  </body>

  <!-- 6. 执行器 (只控制 12 个腿关节) -->
  <actuator>
    <motor name="Left_Hip_Pitch" joint="Left_Hip_Pitch" forcerange="-30 30"/>
    <motor name="Left_Hip_Roll"  joint="Left_Hip_Roll"  forcerange="-35 35"/>
    <!-- ... 共 12 个腿部电机 ... -->
  </actuator>

  <!-- 7. 传感器 -->
  <sensor>
    <framequat name="orientation" .../>      <!-- 姿态四元数 -->
    <gyro name="angular-velocity" .../>       <!-- 角速度 -->
    <velocimeter name="local_linvel" .../>    <!-- 线速度 -->
  </sensor>
</mujoco>
```

**为什么只控制 12 个腿关节？** 因为手臂和头部对踢球没有帮助。让神经网络只控制必要的关节，降低了学习难度（12 维动作 vs 22 维动作）。手臂和头部关节仍然存在于物理模型中（保持质量和惯性分布正确），但不在控制范围内，保持默认角度不动。

---

## 5. 机器人模型：K1 双足人形机器人

### 5.1 K1 机器人简介

K1 是 Unitree 公司的双足人形机器人。在本项目中：

- **自由度**：22 个旋转关节（头 2 + 左臂 4 + 右臂 4 + 左腿 6 + 右腿 6）
- **控制自由度**：12 个（只控制腿部，每腿 6 关节）
- **身高**：约 1.1m（Trunk 中心离地 0.55m 起）
- **体重**：约 13.3kg（含所有部件）

### 5.2 腿部关节详解

每条腿 6 个关节，共 12 个受控关节：

```
髋部 (Hip)
  ├── Hip_Pitch   (俯仰, 前后摆动)   范围: [-3.0, 2.21] rad
  ├── Hip_Roll    (翻滚, 左右摆动)   范围: [-0.4, 1.57] rad
  └── Hip_Yaw     (偏航, 旋转)       范围: [-1.0, 1.0] rad
膝部 (Knee)
  └── Knee_Pitch  (俯仰, 弯曲)       范围: [0, 2.23] rad
踝部 (Ankle)
  ├── Ankle_Pitch (俯仰, 前后倾)     范围: [-0.87, 0.345] rad
  └── Ankle_Roll  (翻滚, 左右倾)     范围: [-0.345, 0.345] rad
```

**默认站立姿态**（default_angles）：

| 关节 | 默认角度 | 含义 |
|------|---------|------|
| Hip_Pitch | -0.2 rad | 髋略后倾 |
| Hip_Roll | 0.0 rad | 髋不侧倾 |
| Hip_Yaw | 0.0 rad | 腿不旋转 |
| Knee_Pitch | 0.4 rad | 膝微弯 |
| Ankle_Pitch | -0.25 rad | 踝略前倾 |
| Ankle_Roll | 0.0 rad | 脚底水平 |

这个默认姿态是机器人站立时的自然姿势。动作值表示**相对于默认姿态的偏移**。

### 5.3 传感器

K1 机器人的躯干（Trunk）上安装了一个 IMU（惯性测量单元），提供：

- **姿态四元数** (`orientation`): 4 个数字，描述躯干在空间中的朝向
- **角速度** (`angular-velocity`): 3 个数字，绕 x/y/z 轴的旋转速度 (rad/s)
- **线速度** (`local_linvel`): 3 个数字，在机器人自身坐标系下的移动速度 (m/s)

这些传感器数据被用来计算观测中的重力方向、角速度和线速度。

---

## 6. 观测空间：机器人"看到"什么

### 6.1 什么是观测空间

观测空间（Observation Space）是神经网络在每一步接收到的**所有信息**的集合。设计观测空间是 RL 中最关键的决策之一：

- **太少**：AI 不知道球在哪，学不会踢球
- **太多**：AI 需要更多数据来过滤噪音，训练更慢

### 6.2 观测向量（57 维）

```
观测向量 (57 个 float32 数字):

┌──────┬──────┬──────────────────────────────────────┐
│ 索引  │ 维度  │ 内容                                   │
├──────┼──────┼──────────────────────────────────────┤
│ 0:3  │  3   │ 重力方向 (体坐标系)                      │
│      │      │ "机器人身体感受到的重力方向"               │
│      │      │ 站立时 = [0, 0, -1], 摔倒时 = 其他方向   │
├──────┼──────┼──────────────────────────────────────┤
│ 3:6  │  3   │ 角速度 (gyro)                         │
│      │      │ "机器人身体转动的速度"                   │
│      │      │ [绕x轴, 绕y轴, 绕z轴] (rad/s)           │
├──────┼──────┼──────────────────────────────────────┤
│ 6:9  │  3   │ 线速度 (local_linvel)                  │
│      │      │ "机器人在自己坐标系下的移动速度"           │
│      │      │ [前后, 左右, 上下] (m/s)                │
├──────┼──────┼──────────────────────────────────────┤
│ 9:21 │  12  │ 关节位置差                             │
│      │      │ 当前关节角度 - 默认站立角度               │
│      │      │ "腿偏离站立姿势了多少"                    │
├──────┼──────┼──────────────────────────────────────┤
│21:33 │  12  │ 关节速度                               │
│      │      │ 12 个腿部关节的旋转速度 (rad/s)          │
├──────┼──────┼──────────────────────────────────────┤
│33:45 │  12  │ 上一帧动作                             │
│      │      │ 上个时间步输出的 12 个关节偏移值          │
│      │      │ 帮助网络理解动作的时间连续性              │
├──────┼──────┼──────────────────────────────────────┤
│45:48 │  3   │ 球相对机器人位置 (体坐标系)               │
│      │      │ "球在我身体的哪个方向"                   │
│      │      │ [前方, 左侧, 上方] (m)                  │
├──────┼──────┼──────────────────────────────────────┤
│48:51 │  3   │ 球速度 (世界坐标系)                     │
│      │      │ "球正在以多快速度飞向哪里"               │
│      │      │ [vx, vy, vz] (m/s)                    │
├──────┼──────┼──────────────────────────────────────┤
│51:54 │  3   │ 球门相对机器人位置 (体坐标系)             │
│      │      │ "球门在我身体的哪个方向"                  │
│      │      │ [前方, 左侧, 上方] (m)                  │
├──────┼──────┼──────────────────────────────────────┤
│54:57 │  3   │ 球到球门方向 (世界坐标系, 归一化)         │
│      │      │ "从球到球门中心的方向向量"               │
│      │      │ 长度为 1 的单位向量                     │
└──────┴──────┴──────────────────────────────────────┘
```

### 6.3 关键设计：体坐标系

为什么要把球和球门的位置转换成**体坐标系**（body frame）？

- **世界坐标系**：[x, y, z] 相对于场地的原点
- **体坐标系**：[前方, 左侧, 上方] 相对于机器人自己

```
世界坐标: 球在 (3.5, 0, 0.11)     ← AI 还需要知道自己在哪才能计算
体坐标:   球在 "前方 0.5m, 左 0.1m" ← AI 直接知道球相对于自己的位置
```

体坐标使用四元数旋转计算：

```python
# 球的世界位置 - 机器人的世界位置 = 相对世界位置
ball_rel_world = ball_world_pos - robot_world_pos
# 用机器人姿态四元数的逆旋转相对位置 → 体坐标
ball_rel_body = quaternion.rotate_inverse(robot_quat, ball_rel_world)
```

这样无论机器人朝向哪个方向，"球在正前方"总是表示同一个含义。

### 6.4 归一化与安全处理

```python
# 避免极端值破坏训练
obs = np.clip(obs, -100.0, 100.0)
# 把 NaN 和 Inf 替换为 0
obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
```

---

## 7. 动作空间：机器人"做什么"

### 7.1 什么是动作空间

动作空间（Action Space）定义了神经网络可以输出的**控制信号**的范围和含义。

### 7.2 动作空间设计（12 维）

```
动作空间: gym.spaces.Box(-1, 1, (12,))

12 个浮点数，每个在 [-1, 1] 范围内:

索引 0:  Left_Hip_Pitch    左髋俯仰  [-1, 1] → [-0.25, +0.25] rad
索引 1:  Left_Hip_Roll     左髋翻滚  [-1, 1] → [-0.25, +0.25] rad
索引 2:  Left_Hip_Yaw      左髋偏航  [-1, 1] → [-0.25, +0.25] rad
索引 3:  Left_Knee_Pitch   左膝俯仰  [-1, 1] → [-0.25, +0.25] rad
索引 4:  Left_Ankle_Pitch  左踝俯仰  [-1, 1] → [-0.25, +0.25] rad
索引 5:  Left_Ankle_Roll   左踝翻滚  [-1, 1] → [-0.25, +0.25] rad
索引 6:  Right_Hip_Pitch   右髋俯仰  [-1, 1] → [-0.25, +0.25] rad
索引 7:  Right_Hip_Roll    右髋翻滚  [-1, 1] → [-0.25, +0.25] rad
索引 8:  Right_Hip_Yaw     右髋偏航  [-1, 1] → [-0.25, +0.25] rad
索引 9:  Right_Knee_Pitch  右膝俯仰  [-1, 1] → [-0.25, +0.25] rad
索引 10: Right_Ankle_Pitch 右踝俯仰  [-1, 1] → [-0.25, +0.25] rad
索引 11: Right_Ankle_Roll  右踝翻滚  [-1, 1] → [-0.25, +0.25] rad
```

### 7.3 物理含义

动作的物理含义分三步理解：

**第一步：动作 → 关节目标位置**

```
joint_target = action * action_scale + default_angle

例如:
  Left_Knee_Pitch 动作为 +1.0
  → target = +1.0 * 0.25 + 0.4 = 0.65 rad  (膝盖更弯)
  Left_Knee_Pitch 动作为 -1.0
  → target = -1.0 * 0.25 + 0.4 = 0.15 rad  (膝盖更直)
```

**第二步：关节目标位置 → 力矩（PD 控制器）**

见下一章。

**第三步：力矩 → 关节运动 + 物理推演**

力矩施加到物理引擎，关节开始转动。机器人可能会踢到球，球飞向球门。

---

## 8. PD 控制器：从数字到力矩

### 8.1 为什么需要 PD 控制器

神经网络输出的数字（如 `+0.5`）不是直接设置为关节角度，也不是直接控制电机功率。我们需要一个**控制器**把神经网络的目标位置转换为物理力矩。

### 8.2 PD 控制公式

PD 控制（比例-微分控制）是最经典的机器人控制方法：

```
τ = kp * (θ_target - θ_current) - kd * θ̇_current

其中:
  τ           = 力矩 (Nm, 牛顿·米)，施加到关节
  kp          = 比例增益 (刚度)，越大关节越"硬"
  θ_target    = 目标角度 = action * action_scale + default_angle
  θ_current   = 当前实际角度
  θ̇_current   = 当前关节速度
  kd          = 微分增益 (阻尼)，越大关节越"黏"
```

**直觉理解**：
- kp 项：像一个弹簧，把关节拉向目标角度。偏得越远，拉得越用力。
- kd 项：像一个阻尼器，阻止关节运动太快。速度越快，阻力越大。

### 8.3 K1 的 PD 参数

| 关节 | kp (刚度) | kd (阻尼) | 力矩上限 |
|------|----------|----------|---------|
| Hip_Pitch, Hip_Roll, Hip_Yaw | 200.0 | 5.0 | ±40 Nm |
| Knee_Pitch | 200.0 | 5.0 | ±40 Nm |
| Ankle_Pitch, Ankle_Roll | 50.0 | 1.0 | ±40 Nm |

踝关节的刚度较低，因为脚需要一定的柔顺性来适应地面。

力矩最终被裁剪到 `[-40, 40]` Nm 范围内，防止损坏电机（在真实机器人上）或仿真不稳定。

### 8.4 代码实现

```python
def _compute_torques(self, actions: np.ndarray, data: mtx.SceneData) -> np.ndarray:
    # 1. 缩放动作
    actions_scaled = actions * 0.25  # action_scale

    # 2. 获取当前关节状态
    dof_pos = self.get_dof_pos(data)  # 12 个关节的当前位置
    dof_vel = self.get_dof_vel(data)  # 12 个关节的当前速度

    # 3. 计算力矩
    torques = self.kps * (actions_scaled + self.default_angles - dof_pos) \
              - self.kds * dof_vel

    # 4. 裁剪到安全范围
    torques = np.clip(torques, -40.0, 40.0)

    return torques
```

---

## 9. 奖励函数：机器人"学到什么"

### 9.1 奖励函数的设计哲学

奖励函数是 RL 中**最重要的设计**。它告诉神经网络"什么行为是好的，什么行为是坏的"。

设计原则：
- **稀疏奖励** (如进球 +1000) 告诉 AI 终极目标
- **稠密奖励** (如球朝球门移动 +2.0) 帮助 AI 在早期探索中找到正确方向
- **惩罚项** (如动作抖动 -0.01) 让行为更平滑、更自然

### 9.2 奖励函数详解

```
R_total = R_goal + R_ball_toward_goal + R_ball_speed + R_alive + R_action_rate + R_joint_limit
```

#### 9.2.1 R_goal — 进球奖励（稀疏 +1000）

```python
R_goal = 1000.0  if 球进入球门 else 0.0
```

这是最大的奖励，也是 AI 的终极目标。只有当球完全越过球门线、在门柱之间、低于横梁时才算进球。

#### 9.2.2 R_ball_toward_goal — 球朝球门运动（稠密 +2.0）

```python
goal_dir = normalize(球门中心 - 球位置)  # 单位向量，指向球门
vel_dir = normalize(球速度)             # 单位向量，球运动方向
R = clamp(dot(vel_dir, goal_dir), 0, 1) * 2.0
```

物理含义：如果球正向球门飞去（点积 = 1），得到最大奖励 +2.0。如果球横向或远离球门运动（点积 ≤ 0），奖励为 0。

这个奖励帮助 AI 在早期探索中快速发现"把球踢向球门"是好行为。

#### 9.2.3 R_ball_speed — 球速奖励（稠密 +1.0）

```python
target_speed = 3.0  # m/s 理想球速
R = exp(-0.5 * ((target_speed - actual_speed) / 1.0)^2) * 1.0
```

这是一个**高斯函数**：当球速接近 3.0 m/s 时奖励最大（约为 +1.0）；球太快或太慢奖励都会降低。

为什么选 3.0 m/s？点球中适中的球速既能进球，又不会飞过横梁。

#### 9.2.4 R_alive — 存活奖励（稠密 +0.5）

```python
R = 0.5  # 每一步（只要机器人没摔倒）
```

这个常数奖励鼓励机器人**维持站立并持续尝试**，而不是过早放弃。

#### 9.2.5 R_action_rate — 动作平滑惩罚（稠密 -0.01）

```python
R = -0.01 * sum((action_current - action_previous)^2)
```

物理含义：如果前后两帧的动作变化很大（突然剧烈改变关节目标），会被惩罚。这鼓励平滑、自然的动作。

#### 9.2.6 R_joint_limit — 关节极限惩罚（稠密 -0.1）

```python
R = -0.1 * sum(max(pos_low - dof_pos, 0) + max(dof_pos - pos_high, 0))
```

物理含义：如果任何关节超出其物理极限范围，会受到惩罚。这鼓励 AI 在关节允许的范围内运动。

### 9.3 奖励设计总结

| 奖励项 | 类型 | 权重 | 范围 | 作用 |
|--------|------|------|------|------|
| goal | 稀疏 | +1000.0 | {0, 1000} | 终极目标 |
| ball_toward_goal | 稠密 | +2.0 | [0, 2] | 引导球向球门 |
| ball_speed | 稠密 | +1.0 | [0, 1] | 鼓励大力射门 |
| alive | 稠密 | +0.5 | {0.5} | 维持生存 |
| action_rate | 稠密 | -0.01 | [-0.12, 0] | 动作平滑 |
| joint_limit | 稠密 | -0.1 | ≤ 0 | 避免超限 |

每步总奖励范围约 **[-0.2, +3.5]**（不含 goal），进球时额外 +1000。

---

## 10. 终止条件：回合何时结束

### 10.1 终止（terminated）— 成功或失败

```python
terminated = any([
    goal_scored,     # ✅ 球进入球门 → 成功
    out_of_bounds,   # ❌ 球出界 → 失败
    robot_fallen,    # ❌ 机器人摔倒 → 失败
    base_too_low,    # ❌ 身体高度 < 0.3m → 失败
])
```

#### 进球判断

```python
x >= 4.5                    # 球越过球门线
abs(y) <= 0.95              # 球在门柱之间 (宽度 1.9m / 2)
0 <= z <= 1.8               # 球低于横梁且在草地以上
```

#### 出界判断

```python
abs(x) > 5.0                # 球超出场地 x 边界
abs(y) > 3.5                # 球超出场地 y 边界
```

#### 摔倒判断

```python
trunk_ground_contact        # 躯干触地 (碰撞检测)
base_z < 0.3                # 躯干高度过低
```

### 10.2 截断（truncated）— 超时

```python
truncated = (steps >= max_episode_steps)
max_episode_steps = max_episode_seconds / ctrl_dt = 10.0 / 0.02 = 500
```

即使机器人没有摔倒、球没有出界，如果 10 秒（500 步）内没进球，回合也会被截断。

### 10.3 terminated vs truncated 的区别

这是 RL 中的标准区分：

- **terminated**：环境状态决定了不能再继续（进球、摔倒、出界）
- **truncated**：人为设定时间到了（超时）

训练算法会对这两种情况做不同处理：
- terminated：这个状态之后没有未来（终结状态）
- truncated：这个状态之后还有未来，只是被截断了

---

## 11. 回合重置：如何重新开始

### 11.1 重置流程

每当回合结束（terminated 或 truncated），环境会自动调用 `reset()`：

```
1. 物理引擎完全重置
   data.reset(model)               # 清除所有运动状态

2. 设置机器人姿态
   位置: (2.5, 0, 0.55) + noise   # 罚球点后方 0.5m
   朝向: 面向前方 (+x)             # 四元数 [0,0,0,1]
   关节: default_angles ± 0.05 rad # 站立姿势 + 小随机扰动

3. 设置球位置
   位置: (3.0, 0, 0.11) + noise   # 罚球点
   速度: [0, 0, 0, 0, 0, 0]       # 静止

4. 初始化环境信息
   current_actions = zeros          # 上一帧无动作
   last_actions = zeros
   goal_scored = False
   ...

5. 正运动学更新
   model.forward_kinematic(data)    # 确保所有位姿一致

6. 返回初始观测
   obs = _get_obs(data, info)       # 57 维观测向量
```

### 11.2 随机化（课程学习预留）

```python
# 默认关闭，可通过配置开启
randomize_ball_position: bool = False    # 球在罚球点 ±0.2m 范围内随机
randomize_robot_position: bool = False   # 机器人在初始位置 ±0.1m 范围内随机
```

随机化是课程学习的基础。初期训练 → 固定罚球点；后期训练 → 随机位置 → 泛化能力更强。

### 11.3 批量重置

MotrixLab 的 `NpEnv` 基类会智能处理批量重置：

```python
# 假设 2048 个并行环境
# 第 1 个环境: 机器人还在尝试踢球 (不重置)
# 第 2 个环境: 进球了! (需要重置)
# 第 3 个环境: 出界了 (需要重置)
# ...
# 第 2048 个环境: 还在进行中 (不重置)

# NpEnv._reset_done_envs() 会自动:
#   1. 找出所有 done=True 的环境
#   2. 只为这些环境调用 reset()
#   3. 更新对应的 obs 和 info
```

---

## 12. 代码结构：文件清单与职责

### 12.1 新增文件

```
simulation/MotrixLab-main/
├── motrix_envs/src/motrix_envs/locomotion/k1_penalty/
│   ├── __init__.py                          # 模块初始化，触发注册装饰器
│   ├── cfg.py                                # 环境配置 (dataclass)
│   ├── k1_penalty_np.py                      # 环境类 PenaltyShootoutEnv
│   └── xmls/
│       └── scene_penalty_shootout.xml        # MuJoCo 物理场景定义
│
└── motrix_rl/src/motrix_rl/tasks/
    └── k1_penalty.py                         # RL 训练超参数 (SKRL + RSLRL)
```

### 12.2 修改文件

```
motrix_envs/src/motrix_envs/locomotion/__init__.py  → +1 行: from . import k1_penalty
motrix_rl/src/motrix_rl/tasks/__init__.py           → +1 行: k1_penalty,
```

### 12.3 各文件详细职责

| 文件 | 行数 | 职责 |
|------|------|------|
| `scene_penalty_shootout.xml` | 245 | MJCF 物理场景：机器人模型、球、球门、地面、传感器 |
| `cfg.py` | 160 | 配置 dataclass：PD 参数、奖励权重、场地尺寸、初始状态 |
| `k1_penalty_np.py` | ~550 | 环境核心：reset/step/obs/reward/termination 全部逻辑 |
| `k1_penalty.py` | 65 | RL 超参数：网络结构 [512,256,128]、学习率 3e-4、PPO 参数 |

### 12.4 关键类和方法

```python
class PenaltyShootoutEnv(NpEnv):
    """K1 点球射门 RL 环境"""

    def __init__(self, cfg, num_envs=1):
        """初始化：加载场景、创建 PD 增益、构建执行器-关节索引映射"""

    def apply_action(self, actions, state):
        """应用动作：神经网络输出 → PD 力矩 → 设置执行器"""

    def update_state(self, state):
        """更新状态：计算观察、奖励、终止条件"""

    def reset(self, data):
        """重置环境：球放罚球点、机器人放球后方、清空状态"""

    # ---- 内部方法 ----
    def _get_obs(self, data, info) -> np.ndarray:
        """构建 57 维观察向量"""

    def _compute_torques(self, actions, data) -> np.ndarray:
        """PD 控制：动作 → 关节力矩"""

    def _compute_rewards(self, data, info) -> dict:
        """计算 6 项奖励分量"""

    def _is_goal(self, ball_pos) -> np.ndarray:
        """判断球是否进球门"""

    def _is_out_of_bounds(self, ball_pos) -> np.ndarray:
        """判断球是否出界"""
```

---

## 13. MotrixLab 注册机制

### 13.1 什么是注册

MotrixLab 使用装饰器（decorator）注册环境。注册后，可以通过名字字符串（如 `"k1-penalty-shootout"`）来创建环境，而不需要直接导入类。

### 13.2 三层注册

```
第一层: 环境配置注册
@registry.envcfg("k1-penalty-shootout")
@dataclass
class K1PenaltyShootoutEnvCfg(EnvCfg):
    model_file = ".../scene_penalty_shootout.xml"
    sim_dt = 0.005
    ...

第二层: 环境类注册
@registry.env("k1-penalty-shootout", sim_backend="np")
class PenaltyShootoutEnv(NpEnv):
    ...

第三层: RL 训练超参数注册
@rlcfg("k1-penalty-shootout")
@dataclass
class K1PenaltyRslrlPpo(RslrlCfg):
    ...
```

### 13.3 注册触发链

```
motrix_envs/__init__.py
  → from . import locomotion       # 导入 locomotion 包
    → locomotion/__init__.py
      → from . import k1_penalty   # 导入 k1_penalty 包
        → k1_penalty/__init__.py
          → from . import k1_penalty_np  # 执行 @registry.env 装饰器

motrix_rl/__init__.py
  → (类似链) 执行 @rlcfg 装饰器
```

### 13.4 如何使用

```python
# 通过名字创建环境
from motrix_envs import registry
env = registry.make("k1-penalty-shootout", num_envs=2048)

# 列出所有已注册环境
registry.list_registered_envs()
# ['cartpole', 'pendulum', ..., 'k1-penalty-shootout', ...]
```

---

## 14. 训练配置：PPO 算法参数

### 14.1 网络结构

```
Policy Network (Actor) — 决定做什么动作:
  输入: 57 维观测
  ├── Linear(57 → 512) + ELU 激活
  ├── Linear(512 → 256) + ELU 激活
  ├── Linear(256 → 128) + ELU 激活
  └── Linear(128 → 12)  → 12 维动作 (Gaussian 分布)

Value Network (Critic) — 预测未来能得多少分:
  输入: 57 维观测
  ├── Linear(57 → 512) + ELU 激活
  ├── Linear(512 → 256) + ELU 激活
  ├── Linear(256 → 128) + ELU 激活
  └── Linear(128 → 1)  → 1 个标量值

ELU: Exponential Linear Unit, 激活函数
        { x          if x > 0
elu(x) = {
        { α(e^x - 1) if x ≤ 0
```

### 14.2 PPO 超参数

| 参数 | 值 | 含义 |
|------|-----|------|
| learning_rate | 3e-4 | 每次更新网络的学习步长 |
| num_learning_epochs | 5 | 每批数据重复学习的次数 |
| num_mini_batches | 4 | 每 epoch 分成多少小批 |
| entropy_coef | 0.01 | 熵正则化系数（鼓励探索） |
| gamma (discount_factor) | 0.99 | 未来奖励的折扣率 |
| lam (GAE λ) | 0.95 | 广义优势估计的平滑参数 |
| num_steps_per_env | 24 | 每轮每个环境执行的步数 |
| max_iterations | 3000 | 总训练轮数 |

### 14.3 训练规模

| 场景 | num_envs | 总步数 (approx) |
|------|----------|----------------|
| 快速调试 | 1-16 | ~72,000 |
| 小规模训练 | 256 | ~18,432,000 |
| 标准训练 | 1024 | ~73,728,000 |
| 大规模训练 | 2048 | ~147,456,000 |

---

## 15. 运行方式：如何启动训练

### 15.1 前置条件

```bash
# 1. 确认在项目根目录
cd /opt/sim_soccer2/simulation/MotrixLab-main

# 2. 安装依赖 (Python 3.10 环境)
uv sync --all-packages --all-extras

# 3. 确认 motrixsim 可用
python -c "import motrixsim; print('OK')"
```

### 15.2 验证环境注册

```bash
uv run python -c "
from motrix_envs import registry
assert 'k1-penalty-shootout' in registry.list_registered_envs()
print('环境注册成功')

env = registry.make('k1-penalty-shootout', num_envs=1)
print(f'观测空间: {env.observation_space}')   # Box(-inf, inf, (57,), float32)
print(f'动作空间: {env.action_space}')        # Box(-1.0, 1.0, (12,), float32)
"
```

### 15.3 环境 Smoke Test

```bash
uv run python -c "
import numpy as np
from motrix_envs import registry

env = registry.make('k1-penalty-shootout', num_envs=1)
state = env.init_state()
print(f'初始观测: shape={state.obs.shape}, NaN={np.any(np.isnan(state.obs))}')

for i in range(100):
    actions = np.random.uniform(-1, 1, (1, 12)).astype(np.float32)
    state = env.step(actions)
print(f'100 步后: reward={state.reward[0]:.3f}')
print('环境 smoke test 通过')
"
```

### 15.4 启动训练

```bash
# 使用 RSLRL (PyTorch)
uv run scripts/train.py \
  --env k1-penalty-shootout \
  --rllib rslrl \
  --num-envs 256

# 使用 SKRL
uv run scripts/train.py \
  --env k1-penalty-shootout \
  --rllib skrl \
  --num-envs 256 \
  --render

# 调试模式（少量环境、短训练）
uv run scripts/train.py \
  --env k1-penalty-shootout \
  --rllib rslrl \
  --num-envs 4 \
  --render
```

### 15.5 查看训练结果

```bash
# TensorBoard 可视化
uv run tensorboard --logdir runs/k1-penalty-shootout

# 查看 checkpoint
ls runs/k1-penalty-shootout/rslrl/

# 评估训练好的模型
uv run scripts/play.py \
  --env k1-penalty-shootout \
  --policy runs/k1-penalty-shootout/rslrl/.../model_3000.pt
```

### 15.6 3D 可视化（无需训练）

```bash
# 只看环境渲染，不使用策略
uv run scripts/view.py --env k1-penalty-shootout
```

---

## 16. 当前状态与下一步

### 16.1 已完成

| 任务 | 状态 |
|------|------|
| 场景 XML（K1 机器人 + 球 + 球门 + 传感器）| 完成 |
| 环境配置 dataclass | 完成 |
| PenaltyShootoutEnv 环境类（继承 NpEnv）| 完成 |
| 57 维观测空间 | 完成 |
| 12 维动作空间 + PD 控制 | 完成 |
| 6 项奖励函数 | 完成 |
| 4 种终止条件 + 超时截断 | 完成 |
| 回合重置（含批量重置） | 完成 |
| MotrixLab 三层注册 | 完成 |
| SKRL + RSLRL 训练配置 | 完成 |
| 环境创建 + 100 步随机动作测试 | 通过 |
| Python 语法编译检查 | 通过 |

### 16.2 当前环境限制

- **本地缺少 RSLRL/SKRL 包**：本机没有 `rsl_rl` 和 `skrl` Python 包（需通过 `uv sync` 在 Python 3.10 环境安装），因此无法在本机执行完整训练
- **本地 CUDA 不可用**：NVIDIA 驱动版本过旧，训练只能使用 CPU
- **目标机器**：需要在安装了所有 MotrixLab 依赖的机器上运行训练

### 16.3 下一步扩展路线

#### Phase 1: 基础训练验证（1-2 周）

- 在目标机器上运行训练
- 调整奖励权重（可能需要多次实验找到最佳权重）
- 观察模型是否学会踢球

#### Phase 2: 加入守门员（2-3 周）

```
场景变为:
  射门机器人 (RL 控制, 12 维动作)
  +
  守门员机器人 (规则控制, 在球门线上横向移动)
```

- 守门员用简单规则控制（例如：根据球的位置左右移动）
- 射门机器人需要学会观察守门员位置，选择射门角度
- 观察空间需要增加：守门员相对位置 [3 维]

#### Phase 3: 课程学习（2-3 周）

```
Level 1: 球固定在罚球点正中央
Level 2: 球在罚球点 ±0.1m 范围内随机
Level 3: 球在罚球点 ±0.2m 范围内随机，角度随机
Level 4: 球在罚球点 ±0.3m 范围内随机，加入守门员
```

课程学习让 AI 从简单任务开始，逐步增加难度，避免一开始就面对太难的任务导致学不会。

#### Phase 4: 自对弈（4-6 周）

- 射门机器人和守门员都使用 RL 训练
- 双方对抗训练：你变强，我也变强
- 类似 AlphaGo 的自对弈机制

#### Phase 5: 部署到 Sim Manager

- 训练完成 → 导出 ONNX 模型
  ```python
  torch.onnx.export(policy_net, obs_tensor, "penalty_policy.onnx")
  ```
- 在 Sim Manager 中选择 RL 策略 → 机器人用 RL 模型射门
- 可与规则控制的 decider 机器人同场竞技

### 16.4 关键文件速查

| 想看什么 | 文件 |
|---------|------|
| 物理场景 | `simulation/MotrixLab-main/motrix_envs/src/motrix_envs/locomotion/k1_penalty/xmls/scene_penalty_shootout.xml` |
| 环境配置 | `simulation/MotrixLab-main/motrix_envs/src/motrix_envs/locomotion/k1_penalty/cfg.py` |
| 环境逻辑 | `simulation/MotrixLab-main/motrix_envs/src/motrix_envs/locomotion/k1_penalty/k1_penalty_np.py` |
| 训练超参 | `simulation/MotrixLab-main/motrix_rl/src/motrix_rl/tasks/k1_penalty.py` |
| 观察/奖励/终止 | 本文档第 6-11 章 |
| 项目总览 | `PROJECT_OVERVIEW.md` |
