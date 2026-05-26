# PPO 强化学习训练详解 — K1 点球射门

---

## 目录

1. [PPO 算法原理](#1-ppo-算法原理)
2. [网络架构：Actor 与 Critic](#2-网络架构actor-与-critic)
3. [输入：57 维观测向量](#3-输入57-维观测向量)
4. [输出：12 维动作与 PD 控制](#4-输出12-维动作与-pd-控制)
5. [奖励函数设计](#5-奖励函数设计)
6. [训练循环：一帧的完整流程](#6-训练循环一帧的完整流程)
7. [PPO 超参数详解](#7-ppo-超参数详解)
8. [终止条件与回合管理](#8-终止条件与回合管理)
9. [并行训练：2048 个环境同时跑](#9-并行训练2048-个环境同时跑)
10. [训练监控：如何判断学得好不好](#10-训练监控如何判断学得好不好)
11. [常见问题与调参建议](#11-常见问题与调参建议)

---

## 1. PPO 算法原理

### 1.1 什么是 PPO

PPO（Proximal Policy Optimization，近端策略优化）是 OpenAI 在 2017 年提出的强化学习算法。它目前是机器人学习领域最主流的算法，原因很简单：

- **稳定**：不会因为一次更新太大而导致策略崩溃
- **高效**：数据利用率高，可以大规模并行训练
- **简单**：相比 TRPO、DDPG 等算法，实现更简洁

### 1.2 核心概念：策略梯度

PPO 属于**策略梯度**（Policy Gradient）方法。核心思想：

```
策略梯度 = 好的动作出现的更多 × 每个动作的"好坏程度"
```

具体来说：

1. **采样**：用当前策略 π_old 在环境中运行，收集一批 (状态, 动作, 奖励) 数据
2. **评估**：估算每个动作的"优势"（advantage）——比平均好多少
3. **更新**：增加高优势动作的概率，降低低优势动作的概率

### 1.3 PPO 的核心创新：Clip 机制

普通的策略梯度方法可能一次更新太大，导致策略突然变差。PPO 用了一个**裁剪**（clip）技巧：

```
PPO Loss = min(
    ratio * advantage,                          # 策略比率 × 优势
    clip(ratio, 1-ε, 1+ε) * advantage          # 裁剪后的版本
)
```

其中 `ratio = π_new(action|state) / π_old(action|state)` 是新旧策略的概率比。

**直觉**：
- 如果 advantage > 0（这个动作比平均好），ratio 被限制在 1+ε = 1.2 以内 → 最多增加 20% 的概率
- 如果 advantage < 0（这个动作比平均差），ratio 被限制在 1-ε = 0.8 以上 → 最多减少 20% 的概率
- 这样就保证了**每次更新都是"小步快跑"**

### 1.4 通用优势估计（GAE）

PPO 使用 GAE（Generalized Advantage Estimation）来估算每个动作的"优势"：

```
A_t = δ_t + γλ·δ_{t+1} + (γλ)²·δ_{t+2} + ...

其中:
  δ_t = r_t + γ·V(s_{t+1}) - V(s_t)    (TD 误差)
  γ = 0.99  (折扣因子，未来奖励的打折率)
  λ = 0.95  (GAE 平滑参数)
```

**直觉**：一个动作的"优势" = 这个动作之后获得的总奖励 - Critic 预测的"平均情况"。正优势 → 比预期好，负优势 → 比预期差。

### 1.5 PPO 与其他算法的对比

| 算法 | 稳定性 | 数据效率 | 实现复杂度 | 适用场景 |
|------|--------|----------|-----------|---------|
| PPO | 高 | 中 | 中 | 机器人、游戏、通用 |
| SAC | 高 | 中 | 中 | 连续控制（偏探索） |
| TD3 | 中 | 低 | 低 | 简单连续控制 |
| DQN | 中 | 低 | 低 | 离散动作空间 |
| TRPO | 最高 | 中 | 高 | 对稳定性要求极高 |

---

## 2. 网络架构：Actor 与 Critic

### 2.1 双网络结构

PPO 使用两个神经网络（Actor-Critic 架构）：

```
                    ┌─────────────────────┐
    57 维观测 ──────┤                     ├────── 12 维动作 (Actor)
                    │   共享输入，独立权重  │
    57 维观测 ──────┤                     ├────── 1 个标量 (Critic)
                    └─────────────────────┘
```

两个网络接收相同的 57 维观测，但参数完全独立，各自优化。

### 2.2 Actor 网络（策略网络）

**职责**：给定状态 s，输出动作 a 的**概率分布**。

```
Actor (Policy Network)
═══════════════════════════════════════════════════
输入: obs (57,) = [重力3, 角速度3, 线速度3, 关节差12, 关节速12, 上帧动作12, 球相对位置3, 球速度3, 球门相对位置3, 球到球门方向3]

层级:
  Linear(57 → 512)  +  ELU 激活
  Linear(512 → 256) +  ELU 激活
  Linear(256 → 128) +  ELU 激活
  Linear(128 → 12)                    ← 输出动作均值 μ
  可学习参数: log_std (12,)             ← 输出动作标准差 σ

输出: 高斯分布 N(μ, σ²) 中采样得到 action (12,)
═══════════════════════════════════════════════════

总参数: ~150,000 (512×57 + 512 + 512×256 + 256 + 256×128 + 128 + 128×12 + 12 + 12)
```

**详细说明**：

```
Layer 0: Linear(57, 512)
  Weight: (512, 57) = 29,184 parameters
  Bias:   (512,)    = 512 parameters
  Output: (512,) after ELU activation

Layer 1: Linear(512, 256)
  Weight: (256, 512) = 131,072 parameters
  Bias:   (256,)     = 256 parameters
  Output: (256,) after ELU activation

Layer 2: Linear(256, 128)
  Weight: (128, 256) = 32,768 parameters
  Bias:   (128,)     = 128 parameters
  Output: (128,) after ELU activation

Layer 3: Linear(128, 12)
  Weight: (12, 128) = 1,536 parameters
  Bias:   (12,)     = 12 parameters
  Output: (12,) = action mean μ

log_std: (12,) = 12 learnable parameters
  → std = exp(log_std), initialized to exp(-1.5) ≈ 0.223

Total: 29,184 + 512 + 131,072 + 256 + 32,768 + 128 + 1,536 + 12 + 12
     = 195,480 parameters
```

**ELU 激活函数**（Exponential Linear Unit）：

```
        { x              if x > 0
elu(x) = {
        { α(e^x - 1)     if x ≤ 0,  α = 1.0
```

ELU 比 ReLU 的优势：负值区域有非零梯度，避免了"死神经元"问题。输出均值接近零，有助于训练稳定。

### 2.3 Critic 网络（价值网络）

**职责**：给定状态 s，预测从该状态开始的**未来总回报**。

```
Critic (Value Network)
═══════════════════════════════════════════════════
输入: obs (57,) = 与 Actor 完全相同

层级:
  Linear(57 → 512)  +  ELU 激活
  Linear(512 → 256) +  ELU 激活
  Linear(256 → 128) +  ELU 激活
  Linear(128 → 1)                     ← 输出单个标量 V(s)

输出: V(s) = 从状态 s 开始，按照当前策略执行能获得的期望总奖励
═══════════════════════════════════════════════════

总参数: ~195,000 (与 Actor 结构相同，仅最后一层维度不同)
```

**Critic 的作用**：

1. **计算 Advantage**：`A(s,a) = r + γ·V(s') - V(s)`。Critic 告诉 Actor "你做得比预期的好/差"
2. **降低方差**：直接用累积奖励（Monte Carlo）评估动作好坏，方差很大。减去 Critic 的预测（Baseline）可以大幅降低方差，加快学习
3. **提供学习信号**：即使没有稀疏奖励（如进球），Critic 仍然能提供"你看起来离目标更近了"的信号

### 2.4 EmpiricalNormalization（经验归一化）

Actor 和 Critic 的输入都经过一个**在线归一化层**：

```python
EmpiricalNormalization():
    # 维护观测的滑动平均值和标准差
    running_mean  ← 0.99 * running_mean  + 0.01 * batch_mean
    running_std   ← 0.99 * running_std   + 0.01 * batch_std

    # 归一化输出
    normalized_obs = (obs - running_mean) / clip(running_std, 0.01, ∞)
```

效果：自动将观测缩放到 ~N(0,1) 范围，让网络训练更稳定。不需要手动调观测缩放。

### 2.5 网络结构的选择

| 方案 | 效果 |
|------|------|
| `[256, 128, 64]` | 轻量级，适合简单任务如 CartPole |
| `[512, 256, 128]` | 中等大小，适合 K1 点球任务 |
| `[1024, 512, 256]` | 大型网络，适合更复杂的全身控制 |

我们选择 `[512, 256, 128]` 的原因：57 维观测 + 12 维动作，复杂度适中。比 CartPole 复杂但比全身控制简单。

---

## 3. 输入：57 维观测向量

### 3.1 观测的来源

观测向量由环境类的 `_get_obs()` 方法构建，数据来自三个来源：

1. **MotrixSim 传感器**：IMU 读数（姿态、角速度、线速度）
2. **关节状态**：12 个腿关节的当前位置和速度
3. **物体位置计算**：球和球门的相对位置

### 3.2 逐段详解

```
观测向量 obs[0:57] — 57 个 float32 值

╔══════╤══════╤════════════════════════════════════════════════════════╗
║ 索引  │ 维度  │ 内容                                                    ║
╠══════╪══════╪════════════════════════════════════════════════════════╣
║ 0:3  │  3   │ 局部重力方向 (体坐标系)                                   ║
║      │      │ = quaternion.rotate_inverse(base_quat, [0,0,-1])       ║
║      │      │ 站立时 ≈ [0, 0, -1]; 倾斜时 ≠ [0, 0, -1]               ║
║      │      │ 告诉网络"我有没有摔倒或倾斜"                               ║
╟──────┼──────┼────────────────────────────────────────────────────────╢
║ 3:6  │  3   │ 角速度 (gyro, rad/s)                                     ║
║      │      │ = model.get_sensor_value("angular-velocity", data)      ║
║      │      │ [绕x轴, 绕y轴, 绕z轴] = [翻滚速度, 俯仰速度, 偏航速度]      ║
║      │      │ 告诉网络"我的身体在向哪个方向旋转，旋转多快"                  ║
╟──────┼──────┼────────────────────────────────────────────────────────╢
║ 6:9  │  3   │ 线速度 (local_linvel, m/s)                               ║
║      │      │ = model.get_sensor_value("local_linvel", data)          ║
║      │      │ [前向, 左向, 上向] 在机器人自身坐标系中的移动速度            ║
║      │      │ 告诉网络"我正在向哪个方向移动"                              ║
╟──────┼──────┼────────────────────────────────────────────────────────╢
║ 9:21 │  12  │ 关节位置差 (rad)                                         ║
║      │      │ = dof_pos - default_angles                              ║
║      │      │ "当前关节角度偏离默认站立姿态了多少"                         ║
║      │      │ 顺序: L_HipPitch, L_HipRoll, L_HipYaw, L_Knee,          ║
║      │      │       L_AnklePitch, L_AnkleRoll,                        ║
║      │      │       R_HipPitch, R_HipRoll, R_HipYaw, R_Knee,          ║
║      │      │       R_AnklePitch, R_AnkleRoll                         ║
╟──────┼──────┼────────────────────────────────────────────────────────╢
║21:33 │  12  │ 关节速度 (rad/s)                                         ║
║      │      │ = dof_vel * obs_scale_dof_vel (scale = 0.1)             ║
║      │      │ "我的腿关节正在以多快速度转动"                              ║
║      │      │ 缩放 0.1 是因为关节速度通常在 ±10 rad/s 范围内              ║
╟──────┼──────┼────────────────────────────────────────────────────────╢
║33:45 │  12  │ 上一帧动作 (action, 无量纲, 范围 [-1,1])                   ║
║      │      │ = current_actions (上一帧的输出，首帧为全零)                ║
║      │      │ 告诉网络"我上一步做了什么"，帮助理解动作的时间连续性          ║
║      │      │ 对 PD 控制至关重要：网络可以感知动作变化速率                 ║
╟──────┼──────┼────────────────────────────────────────────────────────╢
║45:48 │  3   │ 球相对机器人位置 (体坐标系, m)                              ║
║      │      │ = quaternion.rotate_inverse(base_quat, ball_world -     ║
║      │      │                            robot_world)                 ║
║      │      │ [前方距离, 左侧距离, 上方距离]                              ║
║      │      │ "球在我身体的哪个方向" — 体坐标系让网络不管朝向都能理解       ║
╟──────┼──────┼────────────────────────────────────────────────────────╢
║48:51 │  3   │ 球速度 (世界坐标系, m/s)                                   ║
║      │      │ = data.dof_vel[:, ball_qvel_indices[:3]]                ║
║      │      │ [vx, vy, vz] — 球在世界坐标系中的移动速度                   ║
║      │      │ 告诉网络"球飞得多快、飞向哪里"                               ║
╟──────┼──────┼────────────────────────────────────────────────────────╢
║51:54 │  3   │ 球门中心相对机器人位置 (体坐标系, m)                         ║
║      │      │ = quaternion.rotate_inverse(base_quat, [4.5,0,0.9] -    ║
║      │      │                            robot_world)                 ║
║      │      │ "球门在我身体的哪个方向"                                    ║
╟──────┼──────┼────────────────────────────────────────────────────────╢
║54:57 │  3   │ 球到球门方向 (世界坐标系, 归一化单位向量)                     ║
║      │      │ = normalize([4.5,0,0.9] - ball_world)                   ║
║      │      │ "从球的位置指向球门中心的方向"                               ║
║      │      │ 这是一个稳定的方向指引，不依赖机器人朝向                      ║
╚══════╧══════╧════════════════════════════════════════════════════════╝
```

### 3.3 观测数值范围（经验值）

| 观测段 | 典型范围 | 异常值 |
|--------|---------|--------|
| 局部重力 | [-1, 1] | >1 表示极端姿态 |
| 角速度 | [-5, 5] rad/s | >20 表示摔倒翻滚 |
| 线速度 | [-3, 3] m/s | >10 可能异常 |
| 关节位置差 | [-1, 1] rad | >2 关节极限预警 |
| 关节速度 | [-20, 20] rad/s | >50 异常 |
| 上帧动作 | [-1, 1] | 不变 |
| 球相对位置 | [-6, 5] m | 场内范围 |
| 球速度 | [-20, 20] m/s | 射门瞬间可达 10+ |
| 球门相对位置 | [-6, 5] m | 场内范围 |
| 球到球门方向 | [-1, 1] | 单位向量 |

---

## 4. 输出：12 维动作与 PD 控制

### 4.1 动作输出

```
Actor 输出分布: N(μ, σ²)

μ (mean): 12 维向量, 由网络最后一层 Linear(128→12) 直接输出
σ (std):  12 维向量, 从可学习参数 log_std 计算: σ = exp(log_std)

训练时: action[i] = μ[i] + σ[i] * ε[i],  其中 ε[i] ~ N(0,1)
推理时: action[i] = μ[i]                    (不加噪声, 确定性策略)

action 被 clip 到 [-1, 1]
```

### 4.2 动作到关节目标的映射

```
Step 1: 动作缩放
  joint_offset[i] = action[i] * action_scale
                  = action[i] * 0.25
                  → 范围 [-0.25, +0.25] rad

Step 2: 目标角度计算
  target_angle[i] = joint_offset[i] + default_angle[i]

  例如:
    Knee_Pitch action = +1.0
    → offset = 0.25 rad
    → target = 0.25 + 0.4 = 0.65 rad  (膝盖弯曲加大)

    Knee_Pitch action = -1.0
    → offset = -0.25 rad
    → target = -0.25 + 0.4 = 0.15 rad  (膝盖近乎伸直)

Step 3: PD 控制 → 力矩
  torque[i] = kp[i] * (target_angle[i] - current_angle[i])
            - kd[i] * current_velocity[i]
  torque[i] = clip(torque[i], -40.0, +40.0) Nm

Step 4: 非腿关节
  头/臂共 10 个执行器 → 力矩固定为 0 → 由 armature 阻尼保持原位
```

### 4.3 12 个腿关节的 PD 参数

| 关节 | 动作索引 | kp (刚度) | kd (阻尼) | 力矩上限 |
|------|---------|----------|----------|---------|
| Left_Hip_Pitch | 0 | 80.0 | 2.0 | ±40 Nm |
| Left_Hip_Roll | 1 | 80.0 | 2.0 | ±40 Nm |
| Left_Hip_Yaw | 2 | 80.0 | 2.0 | ±40 Nm |
| Left_Knee_Pitch | 3 | 80.0 | 2.0 | ±40 Nm |
| Left_Ankle_Pitch | 4 | 50.0 | 1.0 | ±40 Nm |
| Left_Ankle_Roll | 5 | 50.0 | 1.0 | ±40 Nm |
| Right_Hip_Pitch | 6 | 80.0 | 2.0 | ±40 Nm |
| Right_Hip_Roll | 7 | 80.0 | 2.0 | ±40 Nm |
| Right_Hip_Yaw | 8 | 80.0 | 2.0 | ±40 Nm |
| Right_Knee_Pitch | 9 | 80.0 | 2.0 | ±40 Nm |
| Right_Ankle_Pitch | 10 | 50.0 | 1.0 | ±40 Nm |
| Right_Ankle_Roll | 11 | 50.0 | 1.0 | ±40 Nm |

**踝关节刚度较低（50 vs 80）的原因**：脚需要一定柔顺性来适应地面，过于僵硬会导致站立不稳。PD 参数从初始的 200/5 降低到 80/2，以保证在 0.002s 时间步长下的数值稳定性。

### 4.4 动作安全防护

PPO 采样得到的动作在进入 PD 控制器前经过多层安全防护：

```
第 1 层 — clip_actions=True:      采样后 clamp 到 [-1, 1] (GaussianMixin.act)
第 2 层 — NaN/Inf guard:          np.nan_to_num + clip (SkrlNpWrapper.step)
第 3 层 — PD target clamp:        目标角度不超过 joint limits (k1_penalty_np._compute_torques)
第 4 层 — torque NaN guard+clip:  力矩清洗 + 限幅 ±40 Nm
```

### 4.5 物理时间步长

```
物理时间步长:  0.002s (500Hz)  — XML中写0.005, 被 Python cfg 覆盖为 0.002
控制时间步长:  0.020s (50Hz)   — 策略网络输出频率

每个控制步内 = 0.020 / 0.002 = 10 个物理子步
每个子步执行: PD 控制器 → 更新力矩 → bad-state guard → mtx.step() 推进物理
```

---

## 5. 奖励函数设计

### 5.1 设计原则

奖励函数是 RL 中**最重要的设计**。网络只会优化它能"看到"的奖励。三条原则：

1. **稀疏奖励给终极目标**：进球得 +1000，让 AI 有明确的终极追求
2. **稠密奖励给中间过程**：球朝球门移动 +2.0，帮助 AI 在早期探索中找到方向
3. **惩罚给不良行为**：动作抖动 -0.01，关节超限 -0.1，塑造稳定自然的动作

### 5.2 六项奖励详解

#### R1: 进球奖励 (goal)

```python
R1 = 1000.0  if 球越过球门线且在门柱间且在横梁下 else 0.0
```

- **类型**：稀疏奖励（只在特殊事件时触发）
- **权重**：+1000.0
- **终止**：进球后回合立即结束
- **作用**：终极目标，告诉 AI "把球踢进球门就是你要做的事"

#### R2: 球朝球门运动 (ball_toward_goal)

```python
goal_center = [4.5, 0.0, 0.9]                          # 球门中心世界坐标
goal_dir = normalize(goal_center - ball_pos)            # 从球指向球门
vel_dir = normalize(ball_vel)                           # 球速度方向
R2 = 2.0 * clip(dot(vel_dir, goal_dir), 0.0, 1.0)     # 范围 [0, 2.0]
```

- **类型**：稠密奖励（每一步都计算）
- **权重**：+2.0
- **范围**：[0, 2.0]
- **作用**：鼓励 AI 把球踢向球门方向

**情景分析**：
| 球速度方向 | 点积值 | 奖励 |
|-----------|--------|------|
| 正对球门中心 | 1.0 | +2.00 |
| 偏离 30° | 0.87 | +1.73 |
| 偏离 60° | 0.5 | +1.00 |
| 横向 (90°) | 0.0 | 0.00 |
| 远离球门 (>90°) | 负值 → clip 为 0 | 0.00 |

#### R3: 球速奖励 (ball_speed)

```python
target_speed = 3.0          # m/s, 理想球速
sigma = 1.0                 # 容差
speed = norm(ball_vel)      # 当前实际球速
R3 = 1.0 * exp(-0.5 * ((target_speed - speed) / sigma)^2)
```

- **类型**：稠密奖励
- **权重**：+1.0
- **范围**：[0, 1.0]
- **作用**：鼓励有力射门，但不要过于暴力

**奖励曲线**：
```
speed=0     → R3 = exp(-4.5)  ≈ 0.011  (几乎没奖励)
speed=1.0   → R3 = exp(-2.0)  ≈ 0.135
speed=2.0   → R3 = exp(-0.5)  ≈ 0.607
speed=3.0   → R3 = exp(0)     = 1.000  (最佳！)
speed=4.0   → R3 = exp(-0.5)  ≈ 0.607
speed=6.0   → R3 = exp(-4.5)  ≈ 0.011
```

**为什么选 3.0 m/s？** 点球距离约 1.5m，3.0 m/s 的球速意味着球在 0.5 秒后到达球门，这个速度既有威胁又不会飞过横梁。

#### R4: 存活奖励 (alive)

```python
R4 = 0.5  # 每一步，只要机器人没摔倒
```

- **类型**：稠密奖励（常数）
- **权重**：+0.5
- **作用**：鼓励机器人维持站立并持续尝试，而不是过早放弃
- **每步总奖励占比**：约 15-30%

#### R5: 动作平滑惩罚 (action_rate)

```python
R5 = -0.01 * sum((action_current - action_previous)^2)  # 12 个关节的动作差平方和
```

- **类型**：稠密惩罚
- **权重**：-0.01
- **范围**：约 [−0.12, 0]（每个关节最大变化 2 = 2²×12×0.01 = 0.48 → 实际约 −0.05 到 0）
- **作用**：鼓励平滑连续的动作，避免剧烈抖动

#### R6: 关节极限惩罚 (joint_limit)

```python
for each leg joint i:
    if dof_pos[i] < joint_min[i]:
        violation += joint_min[i] - dof_pos[i]
    if dof_pos[i] > joint_max[i]:
        violation += dof_pos[i] - joint_max[i]
R6 = -0.1 * total_violation
```

- **类型**：稠密惩罚
- **权重**：-0.1
- **范围**：≤ 0（正常时 = 0）
- **作用**：阻止 AI 将关节推到物理极限之外

### 5.3 奖励总结

| 奖励项 | 类型 | 权重 | 范围 | 每一步贡献 |
|--------|------|------|------|-----------|
| goal | 稀疏 | +1000.0 | {0, 1000} | 极少 |
| ball_toward_goal | 稠密 | +2.0 | [0, 2.0] | ~0.5-1.5 |
| ball_speed | 稠密 | +1.0 | [0, 1.0] | ~0.1-0.5 |
| alive | 稠密 | +0.5 | {0.5} | 0.5 |
| action_rate | 稠密 | -0.01 | [-0.12, 0] | ~-0.03 |
| joint_limit | 稠密 | -0.1 | ≤ 0 | ~0 |
| **平均总计** | | | | **~1.0-2.5** |

### 5.4 奖励配置代码

```python
@dataclass
class K1RewardConfig:
    scales: dict[str, float] = field(default_factory=lambda: {
        "goal": 1000.0,
        "ball_toward_goal": 2.0,
        "ball_speed": 1.0,
        "alive": 0.5,
        "action_rate": -0.01,
        "joint_limit": -0.1,
    })
    ball_speed_target: float = 3.0    # m/s
    ball_speed_sigma: float = 1.0     # tolerance
```

---

## 6. 训练循环：一帧的完整流程

### 6.1 数据流全貌

```
PPO 训练循环 (每轮 iteration)

┌─────────────────────────────────────────────────────────────┐
│ Phase 1: 数据收集 (Collection)                               │
│                                                              │
│  for step in range(64):                  # rollouts (SKRL)   │
│                                                              │
│    1. Actor 推理:                                             │
│       obs (num_envs, 57) → Actor网络 → μ, σ                  │
│       action = sample(N(μ, σ²))   → (num_envs, 12)          │
│       clip_actions: clamp(action, -1, 1)                     │
│                                                              │
│    2. 环境执行:                                               │
│       action → NaN guard → clip[-1,1] → env.step()          │
│         → PD控制器 → target clamp → 力矩(22,)                │
│         → bad-state guard → mtx.step()×10                    │
│         → 新的 obs, reward, terminated, truncated, info     │
│                                                              │
│    3. 存入 Buffer:                                           │
│       transitions.append(obs, action, reward, done,          │
│                          values, log_prob)                   │
│                                                              │
│  结果: 64步 × num_envs 条经验, 共 64×N 条                    │
├─────────────────────────────────────────────────────────────┤
│ Phase 2: 优势估计 (GAE)                                      │
│                                                              │
│  for each transition (t from T down to 0):                   │
│    δ[t] = reward[t] + γ·value[t+1] · (1-done[t]) - value[t] │
│    A[t] = δ[t] + γ·λ·A[t+1] · (1-done[t])                  │
│    R[t] = A[t] + value[t]                   # returns       │
│                                                              │
│  结果: advantages (64, N), returns (64, N)                   │
├─────────────────────────────────────────────────────────────┤
│ Phase 3: 策略更新 (Learning, 重复 5 次)                       │
│                                                              │
│  for epoch in range(5):                 # learning_epochs    │
│    for mini_batch in shuffled(data, num_mini_batches=4):    │
│      # 64步 × N envs = 64N 条经验, 分4批, 每批 16N 条       │
│                                                              │
│      1. 前向传播:                                             │
│         new_log_prob = Actor(obs).log_prob(action)           │
│         new_value = Critic(obs)                              │
│                                                              │
│      2. 计算 Loss:                                            │
│         ratio = exp(new_log_prob - old_log_prob)             │
│         surr1 = ratio * advantages                           │
│         surr2 = clip(ratio, 0.8, 1.2) * advantages           │
│         actor_loss = -mean(min(surr1, surr2))                │
│         critic_loss = mean((returns - new_value)^2)          │
│         entropy_bonus = mean(entropy)                        │
│         total_loss = actor_loss                             │
│                    + 0.5 * critic_loss                       │
│                    - 0.01 * entropy_bonus                    │
│                                                              │
│      3. 反向传播:                                             │
│         optimizer.zero_grad()                                │
│         total_loss.backward()                                │
│         clip_grad_norm_(max_norm=1.0)                        │
│         optimizer.step()                                     │
│                                                              │
├─────────────────────────────────────────────────────────────┤
│ Phase 4: 日志与保存                                           │
│                                                              │
│  if iteration % save_interval == 0:                          │
│    save checkpoint → runs/.../model_{iter}.pt                │
│  log to TensorBoard: reward, episode_length, losses         │
└─────────────────────────────────────────────────────────────┘
```

### 6.2 关键公式汇总

| 公式 | 代码 | 含义 |
|------|------|------|
| `μ = Actor(obs)` | `actor(obs)` | 动作均值 |
| `a = μ + σ·ε` | `distribution.sample()` | 随机采样动作 |
| `τ = kp·(a·0.25+d - q) - kd·q̇` | `_compute_torques()` | PD 力矩 |
| `δ = r + γ·V(s') - V(s)` | TD error | 单步优势估计 |
| `A = Σ(γλ)^k·δ_k` | `compute_gae()` | GAE 优势 |
| `ratio = π_new/π_old` | `exp(new_logp - old_logp)` | 策略比率 |
| `L_clip = -min(ratio·A, clip(ratio)·A)` | PPO loss | 裁剪损失 |
| `L_value = (V(s) - R)^2` | MSE | 价值损失 |
| `L = L_clip + 0.5·L_value - 0.01·H` | total_loss | 总损失 |

---

## 7. PPO 超参数详解

### 7.1 超参数表 (SKRL, 默认框架)

```
═══════════════════════════════════════════════════════════════
                    PPO 训练超参数 (SKRL)
═══════════════════════════════════════════════════════════════
网络架构
  policy_hiddens:     [512, 256, 128]   三层 MLP + ELU
  value_hiddens:      [512, 256, 128]   与 Policy 共享躯干 (shared)
  clip_actions:       True              采样后裁剪到 [-1, 1]
  initial_log_std:    -1.5              初始探索噪声 (std ≈ 0.223)
  max_log_std:        2.0               log_std 上限 (std ≤ 7.39)
  min_log_std:        -20.0             log_std 下限

训练规模
  num_envs:           2048              并行仿真环境数
  rollouts:           64                每环境每轮收集的步数
  trainer.timesteps:  50000             总 batch 环境步数

  总步数 = 2048 × 50000 = 102,400,000 环境步 (含 learning_epochs 复用)
  每更新总步数 = 64 × 2048 = 131,072 步
  每 mini-batch = 131,072 / 4 = 32,768 步

PPO 算法
  learning_rate:      3e-4 (0.0003)     Adam 优化器学习率
  learning_epochs:    5                 每批数据学习的轮数
  mini_batches:       4                 每 epoch 分成几个小批
  discount_factor γ:  0.99              未来奖励折扣因子
  lam λ:              0.95              GAE 平滑参数
  ratio_clip ε:       0.2               PPO 裁剪范围
  entropy_loss_scale: 0.0               熵正则化系数 (禁用)
  grad_norm_clip:     1.0               梯度裁剪阈值

环境
  control_frequency:  50 Hz            策略推理频率 (ctrl_dt=0.02)
  physics_frequency:  500 Hz           物理仿真频率 (sim_dt=0.002)
  max_episode_seconds: 10.0            单回合最大时长
  max_episode_steps:  500              = 10.0 / 0.02
═══════════════════════════════════════════════════════════════
```

### 7.1b RSLRL 超参数 (备选框架)

```
═══════════════════════════════════════════════════════════════
                    PPO 训练超参数 (RSLRL)
═══════════════════════════════════════════════════════════════
训练规模
  num_envs:           2048              并行仿真环境数
  num_steps_per_env:  24                每环境每轮收集的步数
  max_iterations:     3000              总训练轮数

  总步数 = 2048 × 24 × 3000 = 147,456,000 环境步

PPO 算法
  entropy_coef:       0.01              熵正则化系数 (RSLRL 启用)
  (其他参数与 SKRL 相同)
═══════════════════════════════════════════════════════════════
```

### 7.2 超参数影响详解

#### learning_rate (3e-4)

学习率控制每次更新的步长。太小 → 学得太慢；太大 → 可能不收敛。

```
learning_rate = 1e-4: 太保守，3000 迭代可能不够
learning_rate = 3e-4: 平衡值，适合我们的任务 ✓
learning_rate = 1e-3: 可能不稳定，奖励波动大
```

#### gamma (0.99)

折扣因子决定 AI "看多远"。γ 越接近 1，越看重长期奖励。

```
γ = 0.95: 只看重 ~20 步内的奖励 → 短视
γ = 0.99: 看重大约 100 步内的奖励 → 适合点球 ✓
γ = 0.999: 看重 ~1000 步内的奖励 → 适合超长任务
```

#### rollouts (64, SKRL) / num_steps_per_env (24, RSLRL)

**SKRL**：每次每个环境连续收集 64 步数据。64 × 50Hz = 1.28 秒的连续轨迹。episode 最长 500 步，一次 rollout 覆盖约 13%。窗口足够大来捕捉踢球动作的完整时序，同时保持策略更新频率。

**RSLRL**：每次每个环境收集 24 步。24 × 50Hz = 0.48 秒。

#### entropy_coef (SKRL: 0.0, RSLRL: 0.01)

熵奖励鼓励**探索**。SKRL 通过 `clip_log_std` 和 `initial_log_std=-1.5` 来控制探索（通过可学习的 log_std 参数），不使用显式熵奖励。RSLRL 使用 0.01 的熵系数。

#### num_mini_batches (4)

把 64×2048 条经验 (SKRL) 分成 4 个小批，每批 64×2048/4 = 32,768 条。在 4 个小批上各更新一次 → 每 epoch 更新 4 次。

---

## 8. 终止条件与回合管理

### 8.1 终止条件（terminated）

| 条件 | 代码 | 含义 |
|------|------|------|
| 进球 | `ball_x >= 4.5 and abs(ball_y) <= 0.95 and ball_z <= 1.8` | 成功 |
| 出界 | `abs(ball_x) > 5.0 or abs(ball_y) > 3.5` | 失败 |
| 躯干触地 | `trunk_geom contacts ground_geom` | 摔倒 |
| 高度过低 | `base_z < 0.3` | 摔倒（备用检测） |

### 8.2 截断条件（truncated）

| 条件 | 值 |
|------|-----|
| 超时 | `steps >= 500`（= 10 秒） |

### 8.3 坏状态保护（Bad-State Guard）

除常规终止条件外，在每次 `physics_step()` 前有额外的**物理安全检测**：

```
_guard_bad_state() 检查:
  - qpos/qvel 是否 finite (NaN/Inf 检测)
  - qvel 最大值是否 < 100 rad/s (速度爆炸检测)

发现坏状态 → 立即 reset 该环境 → 不让坏状态进入 motrixsim 求解器
```

这防止了极端动作引起的物理求解器崩溃（LTL 矩阵分解失败）。默认启用，可通过 `--bad-state-reset=false` 关闭。

### 8.4 回合重置（Auto-Reset）

NpEnv 基类在每步结束后自动检查 `done = terminated | truncated`，对 done 的环境调用 `reset()`：

```python
# NpEnv._reset_done_envs() 的伪代码
done_envs = where(state.done == True)          # 找出所有已结束的环境
new_obs, new_info = self.reset(data[done_envs])# 只重置已结束的环境
state.obs[done_envs] = new_obs                 # 更新观察
state.info["steps"][done_envs] = 0             # 步数归零
```

这意味着每个环境是**独立**的——一个进球了，其他 2047 个还在继续。批量重置机制保证了高效的并行训练。

---

## 9. 并行训练：2048 个环境同时跑

### 9.1 为什么需要并行

```
1 个环境, 50000 steps (SKRL timesteps):
  总步数 = 50,000
  → 经验太少，学不好

2048 个环境, 50000 steps (SKRL timesteps):
  总 env steps = 2048 × 50000 = 102,400,000
  每更新 = 64 × 2048 = 131,072 步
  → 足够的学习信号
```

并行训练的关键是 MotrixSim 支持**批量物理仿真**：

```python
data = mtx.SceneData(model, batch=[2048])  # 2048 个独立的世界
```

所有环境共享相同的物理参数和场景结构，但状态各自独立演化。

### 9.2 并行训练的数据流

```
GPU/NPU 上的网络推理 (batch_size=2048):
  obs (2048, 57) → Actor (2048, 12) / Critic (2048, 1)

CPU 上的物理仿真 (每个环境独立):
  action[0] → env_0.physics_step() → obs[0], reward[0], done[0]
  action[1] → env_1.physics_step() → obs[1], reward[1], done[1]
  ...
  action[2047] → env_2047.physics_step() → obs[2047], reward[2047], done[2047]
```

### 9.3 不同规模的训练建议 (SKRL)

| num_envs | 适用场景 | 每更新步数 | 总 env steps | 预估时间 |
|----------|---------|-----------|-------------|---------|
| 4 | 调试/开发 | 256 | 50,000 | ~15 min |
| 64 | 快速实验 | 4,096 | 50,000 | ~15 min |
| 512 | 中等训练 | 32,768 | 50,000 | ~20 min |
| 2048 | 大规模训练 | 131,072 | 50,000 | ~90 min |

注意：env 数越多，每更新步数越大（rollouts=64 固定），GPU 利用率越高。但 wall-clock 时间受物理仿真瓶颈影响。

---

## 10. 训练监控：如何判断学得好不好

### 10.1 所有监控指标

训练日志中输出的监控指标（SKRL 格式，进度条以 timesteps 显示）：

```
  12%|█▏ | 6001/50000 [01:45<12:50, 57.13it/s]

  throughput: 57.1 env_steps/s          ← 每秒处理的环境步数

  Reward Instant / goal (max): 1000.0   ← 即刻奖励分量
  Reward Instant / alive (mean): 0.5
  Reward Total / goal (mean): 1000.0    ← 回合总奖励分量

  eval/success_rate: 0.42               ← ★ 进球率（最重要指标）
  eval/contact_rate: 0.78               ← 触球率
  eval/fall_rate: 0.03                  ← 摔倒率
  eval/mean_ball_speed: 3.21            ← 平均最大球速
  eval/mean_time_to_goal: 2.34          ← 平均进球时间 (秒)
```

注意：SKRL 使用 `trainer.timesteps=50000`，进度条显示 `N/50000`（batch env steps）。

### 10.2 好训练的预期特征

| 指标 | 初期 (step 0-5000) | 中期 (step 10000-25000) | 后期 (step 40000-50000) |
|------|-------------------|------------------------|------------------------|
| Mean reward | 20-30 | 60-300 | 500-1000+ |
| Mean episode length | 30-50 | 60-120 | 150-300 |
| Mean action noise std | 0.22 | 0.3-0.5 | 0.2-0.4 |
| Mean value loss | 波动大 | 稳定下降 | <10 |

注意：SKRL 使用 `initial_log_std=-1.5`（std ≈ 0.223），初始探索噪声比 RSLRL（std=1.0）低很多。

### 10.3 坏训练的警告信号

| 信号 | 含义 | 解决 |
|------|------|------|
| reward 在几百步后不增长 | 没有学到有用行为 | 增加熵 coef，降低学习率 |
| reward 突然崩到 0 | 策略崩溃 | 减小 clip_param，降低学习率 |
| episode length 总是 500（满） | AI 不知道要射门 | 增加 ball_toward_goal 权重 |
| episode length <5 | 立刻摔倒 | 降低 action_scale，增加 alive 权重 |
| value loss 爆炸 | Critic 不稳定 | 降低学习率，增加 batch size |
| entropy 降到 ~5 以下太快 | 过早收敛 | 增加 entropy_coef |

### 10.4 使用 TensorBoard

```bash
cd /opt/sim_soccer2/simulation/MotrixLab-main
uv run tensorboard --logdir runs/k1-penalty-shootout
# 浏览器打开 http://localhost:6006
```

TensorBoard 会显示（SKRL 格式）：
- `eval/success_rate`: 进球率（最重要的图）
- `eval/contact_rate`: 触球率
- `eval/fall_rate`: 摔倒率
- `eval/mean_ball_speed`: 平均最大球速
- `eval/mean_time_to_goal`: 平均进球时间
- `Reward Instant / goal (max)`: 即时进球奖励
- `Reward Total / goal (mean)`: 回合总进球奖励

---

## 11. 常见问题与调参建议

### 11.1 机器人不踢球，只是站着

**问题**：reward 停留在 ~25（~0.5×50步），机器人只是站着不动。

**原因**：`ball_toward_goal` 和 `ball_speed` 奖励不足以引导踢球行为。站着的奖励（alive=0.5）比试图踢球（可能摔倒、得到 0 奖励）更稳定。

**解决**：
- 降低 alive 权重：`"alive": 0.1`
- 增加球相关奖励：`"ball_toward_goal": 5.0`
- 增加探索：`entropy_coef = 0.02`

### 11.2 机器人总是摔倒

**问题**：episode length < 10，reward 很低。

**原因**：动作太大或不协调，导致失去平衡。

**解决**：
- 降低 action_scale：`0.25 → 0.1`
- 增加 alive 权重：`"alive": 1.0`
- 降低学习率：`3e-4 → 1e-4`

### 11.3 训练稳定但 reward 不增长

**问题**：reward 卡在 ~30-40，持续数百轮不再增长。

**原因**：陷入了局部最优——学会了站立，但没有动机去踢球。

**解决**：
- 增加 goal 奖励：`1000 → 2000`
- 增加 ball_toward_goal 奖励：`2.0 → 5.0`
- 增加探索：`entropy_coef = 0.02 → 0.03`

### 11.4 reward 突然崩到接近 0

**问题**：之前 reward 在增长，突然归零并保持。

**原因**：PPO clip 机制失效，策略在某次更新中改变了太多，学到的行为全丢了。

**解决**：
- 降低 clip_param：`0.2 → 0.1`
- 降低学习率：`3e-4 → 5e-5`
- 从最近的 checkpoint 恢复训练

### 11.5 球速不够快（射门无力）

**问题**：AI 学会了踢球，但球速不到 1 m/s，很容易被守门。

**原因**：`ball_speed` 奖励的目标速度太低或权重不足。

**解决**：
- 增加 target_speed：`3.0 → 5.0`
- 增加 ball_speed 权重：`1.0 → 3.0`

---

## 附录A：完整训练命令

```bash
cd /opt/sim_soccer2/simulation/MotrixLab-main

# ========== 调试（物理稳定性验证） ==========

# 零动作测试
uv run scripts/debug_penalty_physics.py --env k1-penalty-shootout \
  --num-envs 1 --num-steps 1000 --zero-actions

# 随机动作测试
uv run scripts/debug_penalty_physics.py --env k1-penalty-shootout \
  --num-envs 64 --num-steps 2000 --seed 42 --random-actions

# ========== 训练 ==========

# SKRL (默认, PyTorch) — 小规模快速测试
uv run scripts/train.py --env k1-penalty-shootout --num-envs 64 --seed 42

# SKRL — 标准训练
uv run scripts/train.py --env k1-penalty-shootout --num-envs 2048 --seed 42

# SKRL — 带安全参数覆盖
uv run scripts/train.py --env k1-penalty-shootout --num-envs 2048 \
  --action-scale 0.125 --bad-state-reset true

# RSLRL (备选框架)
uv run scripts/train.py --env k1-penalty-shootout --rllib rslrl --num-envs 2048

# ========== 评估 ==========

# 可视化测试（自动发现最新 best_agent.pt）
uv run scripts/play.py --env k1-penalty-shootout --num-envs 1

# 量化评估（批量统计）
uv run scripts/eval_penalty.py --env k1-penalty-shootout \
  --policy runs/k1-penalty-shootout/skrl/.../checkpoints/best_agent.pt \
  --num-envs 256 --num-episodes 2048 --deterministic \
  --json-out results.json --csv-out results.csv

# ========== 监控 ==========

# TensorBoard
uv run tensorboard --logdir runs/k1-penalty-shootout

# ========== 环境 smoke test ==========

uv run python -c "
import numpy as np
from motrix_envs import registry
env = registry.make('k1-penalty-shootout', num_envs=1)
state = env.init_state()
print(f'obs: {state.obs.shape}, action_space: {env.action_space}')
for i in range(100):
    state = env.step(np.random.uniform(-1,1,(1,12)).astype(np.float32))
print(f'100 steps: reward={state.reward[0]:.3f}')
"
```

## 附录B：关键文件索引

| 文件 | 内容 |
|------|------|
| `motrix_envs/.../k1_penalty/cfg.py` | 环境配置：PD 参数(80/2)、奖励权重、安全参数 |
| `motrix_envs/.../k1_penalty/k1_penalty_np.py` | 环境类(~960行)：obs、reward、termination、task metrics |
| `motrix_envs/.../k1_penalty/xmls/scene_penalty_shootout.xml` | MJCF 物理场景(258行, 22actuator, impratio=1) |
| `motrix_rl/.../tasks/k1_penalty.py` | PPO 超参数：[512,256,128]、clip_actions、initial_log_std |
| `motrix_rl/.../skrl/config.py` | SKRL 配置默认值(clip_actions=True, initial_log_std=-1.5) |
| `motrix_rl/.../skrl/torch/wrap_np.py` | Action NaN/Inf guard + clip |
| `motrix_envs/.../np/env.py` | Bad-state guard (_guard_bad_state) |
| `scripts/train.py` | 训练入口 (+ CLI 安全参数) |
| `scripts/debug_penalty_physics.py` | 物理调试脚本 |
| `scripts/eval_penalty.py` | 独立评估脚本 |
| `scripts/play.py` | 可视化评估入口 |
