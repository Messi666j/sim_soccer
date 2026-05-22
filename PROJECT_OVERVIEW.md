# sim-soccer2 项目全景介绍

这是一个**双足/多足机器人足球仿真与决策控制**项目，目标机器人是**Unitree K1**（22自由度人形）和 **Pi Plus**（20自由度人形）。

---

## 一、总体架构（两层）

多机器人足球系统的核心就两层：

```
┌─────────────────────────────────────────────────┐
│  decider/  决策层（规则状态机）                    │
│  - 追球、带球、射门、守门                           │
│  - 每个机器人一个独立进程                           │
└─────────────────────┬───────────────────────────┘
                      │ ZMQ (JSON: cmd ↔ state)
┌─────────────────────┴───────────────────────────┐
│  Sim Manager  Web UI (:8000)                     │
│  ┌─────────────────────────────────────────────┐ │
│  │ MotrixSim 物理引擎                            │ │
│  │ - 神经网络策略推理（机器人走路）                 │ │
│  │ - PD 控制器 → 关节力矩                         │ │
│  │ - 裁判系统（开球、出界、进球等）                 │ │
│  │ - WebView 3D 可视化                           │ │
│  └─────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────┘
```

---

## 二、核心模块详解

### 1. `decider/` — 机器人"大脑"

入口文件是 [user_entry.py](decider/user_entry.py)，由 [decider.py](decider/decider.py) 中的 `Agent`（ROS2模式）或 `SimAgent`（ZMQ仿真模式）调用。

**决策流程（`_mvp_game()`）：**

```
比赛状态判断 → 角色分配:
├── 守门员 (ID=0) → goalkeeper 状态机
│   └── 状态：goalkeep → charge_out → clearance → save
│   └── 使用最小二乘法预测球轨迹，判断是否需要扑救
│
└── 其他球员 → _playing_logic():
    ├── 看不到球 → find_ball（旋转搜索）
    ├── 球较远   → chase_ball（转向+前进逼近）
    └── 球够近   → AdvancedDribbler（矢量场导航带球射门）
```

**三层状态机（都使用 `transitions` 库）：**

| 层级 | 目录 | 职责 | 示例 |
|------|------|------|------|
| 子状态机 | `logic/sub_statemachines/` | 原子行为 | find_ball, chase_ball, dribble, kick, go_back_to_field |
| 策略状态机 | `logic/policy_statemachines/` | 角色策略 | goalkeeper |
| 战术状态机 | `logic/strategy_statemachines/` | 多机协作 | attack, defend, dribble_ball, shoot_ball |

**接口层（`interfaces/`）：**

| 文件 | 用途 |
|------|------|
| [vision.py](decider/interfaces/vision.py) | 视觉感知（球位置、自身位姿、坐标变换） |
| [action.py](decider/interfaces/action.py) | 动作执行（cmd_vel, kick, save_ball） |
| [gamecontroller.py](decider/interfaces/gamecontroller.py) | 比赛状态解析 |
| [sim_client.py](decider/interfaces/sim_client.py) | ZMQ 通信 |

**关键点：decider 全部是规则逻辑，没有学习成分。**

---

### 2. `simulation/motrixsim/` — 物理仿真引擎（即本项目所需的仿真环境）

本项目需要的仿真环境就是 **Sim Manager**（基于 MotrixSim 的机器人足球仿真服务）。

#### 2a. Sim Manager — 足球仿真入口

Sim Manager 是一个 FastAPI Web 服务，提供**一键启动/管理机器人足球仿真**的界面。

**启动：**

```bash
conda activate motrixsim0508
cd /opt/sim_soccer2/simulation/motrixsim
python sim_manager.py --host 0.0.0.0 --port 8000
```

然后浏览器打开 **`http://127.0.0.1:8000/`**，即可：

1. 填写 `team_size`（每队机器人数量，如 3），点击 Start
2. 系统自动分配 ZMQ 端口、启动 MotrixSim 仿真进程
3. 页面上自动生成每个机器人的 Decider 启动命令，复制到终端执行
4. 页面提供仿真进程管理（查看状态、停止、重启）
5. WebView 3D 渲染链接也可在页面上直接打开

**Sim Manager 做的事情：**
- 管理多个仿真进程（启动/停止/监控）
- 自动生成 Decider 连接命令
- 提供 Swagger API（`/docs`）用于程序化控制
- 管理端口分配，避免冲突

**底层实现：** `sim_manager.py` → uvicorn 启动 `simulation/labbridge/sim_manager2.py` 中的 FastAPI app。

#### 2b. MultiRobotMotrixSim — 仿真核心类

如果不用 Sim Manager，也可以直接命令行启动（不推荐日常使用）：

```bash
conda activate motrixsim0508
cd /opt/sim_soccer2/simulation/motrixsim
python sim2sim_runner.py --team-size 3 --real-time --use-referee
```

核心类：`MultiRobotMotrixSim`（在 [multi_robot_sim.py](simulation/motrixsim/app/multi_robot_sim.py)）

**仿真循环（`zmq_loop()`）：**

```
每帧循环:
1. 接收 ZMQ 速度指令 {"commands": [{"id": 0, "cmd": [vx, vy, w]}, ...]}
2. _step_once():
   a. 每 control_decimation 步：_compute_targets()
      - _obs_for_robot() → 构建 47维 观测（重力、陀螺仪、关节状态、速度指令等）
      - 神经网络推理（ONNX/TorchScript） → 12维 关节目标位置
   b. _apply_torque() → PD控制: τ = kp*(target - q) + kd*(0 - qd)
   c. model.step() → MotrixSim 物理步进（dt=0.005s）
   d. 更新裁判、检测摔倒恢复
3. 回复 ZMQ 状态（所有机器人位姿、球位置、比赛状态）
4. 渲染 Web 3D 画面
```

**支持的策略类型（在 [runtime_config.py](simulation/motrixsim/app/runtime_config.py) 中配置）：**

| 类型 | 观测维度 | 动作维度 | 控制关节 | 用途 |
|------|---------|---------|---------|------|
| K1 Legged-Gym | 47 | 12 | 腿部12关节 | 行走（默认） |
| K1 AMP | 375 (5帧堆叠) | 22 | 全身22关节 | 行走+姿态 |
| K1 Full-Body | 78 | 22 | 全身22关节 | 行走（旧版） |
| Pi Plus | 可变 | 20 | 全身20关节 | 行走 |

**裁判系统（[soccer_referee.py](simulation/motrixsim/app/soccer_referee.py)）：**

- 完整实现 RoboCup GameController 协议
- 状态：INITIAL → READY → SET → PLAYING → FINISHED
- 判罚：开球、界外球、角球、球门球、进球、犯规
- 支持自动超时推进

**机器人资产（`assets/robots/k1/`）：**

- `K1_22dof.xml` — 22自由度 MJCF 模型
- 关节：头(2) + 肩(2×3) + 肘(2×1) + 髋(2×3) + 膝(2×1) + 踝(2×2)

---

### 3. `legged_gym/` — 步态策略（机器人走路的神经网络）

包含各机器人的**行走策略推理环境**：

- `envs/base/legged_robot.py` — 基类，提供 `step()`, `reset()`, `compute_obs()`, PD 控制
- `scripts/K1_play.py` — 加载 ONNX 模型，循环推理
- 预训练模型位于 `policy/` 目录（`*.onnx`）
- 这些策略被 MotrixSim 加载，提供机器人行走能力

---

## 三、数据流全景

```
┌─────────────┐  ZMQ JSON    ┌──────────────────┐
│  Decider    │ ──────────>  │  Sim Manager     │
│  (规则AI)   │  cmd: vx,vy,w│  :8000 Web UI     │
│             │ <──────────  │  ┌──────────────┐ │
│  10-50 Hz   │  state: pose │  │ MotrixSim    │ │
└─────────────┘  +ball+game  │  │ 500Hz 物理    │ │
                             │  │ 125Hz 推理    │ │
                             │  │ PD → 力矩     │ │
                             │  └──────────────┘ │
                             │  ┌──────────────┐ │
                             │  │ WebView 3D   │ │
                             │  │ :8080 画面    │ │
                             │  └──────────────┘ │
                             └──────────────────┘
```

- **Sim Manager** (`:8000`) — Web UI 管理仿真进程的生命周期
- **Decider** 算出高层速度指令 (vx, vy, vtheta)
- **MotrixSim** 接收速度指令，转成 47维观测 → 神经网络推理 → 12个关节目标 → PD力矩 → 物理步进
- **裁判** 独立运行，追踪比赛状态和得分
- 机器人行走由预训练神经网络策略驱动，比赛决策由 decider 规则状态机驱动

---

## 四、关键文件速查表

| 用途 | 文件路径 |
|------|---------|
| Decider 入口 | [decider/user_entry.py](decider/user_entry.py) |
| Decider 主类 | [decider/decider.py](decider/decider.py) |
| Decider 配置 | [decider/config.yaml](decider/config.yaml) |
| 守门员逻辑 | [decider/logic/policy_statemachines/goalkeeper.py](decider/logic/policy_statemachines/goalkeeper.py) |
| 仿真核心类 | [simulation/motrixsim/app/multi_robot_sim.py](simulation/motrixsim/app/multi_robot_sim.py) |
| 仿真配置 | [simulation/motrixsim/app/runtime_config.py](simulation/motrixsim/app/runtime_config.py) |
| 裁判系统 | [simulation/motrixsim/app/soccer_referee.py](simulation/motrixsim/app/soccer_referee.py) |
| 比赛配置 | [simulation/motrixsim/assets/config/match_config.json](simulation/motrixsim/assets/config/match_config.json) |
| 步态策略部署 | [legged_gym/envs/base/legged_robot.py](legged_gym/envs/base/legged_robot.py) |

---

## 五、运行环境

### Conda 环境

| 环境名 | Python | 用途 | 关键依赖 |
|--------|--------|------|---------|
| `motrixsim0508` | 3.12 | 仿真引擎 (Sim Manager) | motrixsim, torch, zmq, flask, fastapi, uvicorn |
| `k2` | 3.8 | 决策器 (Decider) | numpy, PyYAML, transitions, pyzmq |

### 启动仿真（Sim Manager，推荐方式）

```bash
# 终端 1 —— 启动 Sim Manager
conda activate motrixsim0508
cd /opt/sim_soccer2/simulation/motrixsim
python sim_manager.py --host 0.0.0.0 --port 8000
# 浏览器打开 http://127.0.0.1:8000/
# 在页面上设置 team_size，点击 Start，复制生成的 Decider 命令
```

```bash
# 终端 2 —— 启动 Decider（复制 Sim Manager 页面生成的命令）
conda activate k2
cd /opt/sim_soccer2
python decider/decider.py --simulation --ip 127.0.0.1 --port 5555 --color red --id 0
# 每个机器人需要一个独立进程
```

### 直接启动仿真（不推荐日常使用，绕过 Sim Manager）

```bash
conda activate motrixsim0508
cd /opt/sim_soccer2/simulation/motrixsim
python sim2sim_runner.py --team-size 3 --real-time --use-referee
# WebView: http://127.0.0.1:5811
# ZMQ: tcp://*:5555
```

顶层 Docker 镜像基于 `nvcr.io/nvidia/isaac-lab:2.3.2`。

---

## 六、点球大战 RL 训练 —— 后续工作安排

### 目标

训练一个强化学习模型，控制 K1 机器人在点球场景中完成射门（最终可扩展到包含守门员的完整点球大战）。

### 现有基础（可直接复用）

| 组件 | 位置 | 说明 |
|------|------|------|
| 物理引擎 | MotrixSim (类 MuJoCo API) | 提供 `load_model()`, `SceneData`, `step()` |
| 球场场景 | `assets/world.xml` | 含球、球门、场地标记（罚球点、禁区线等） |
| K1 机器人模型 | `assets/robots/k1/K1_22dof.xml` | 22 自由度 MJCF 模型 |
| 预训练步态策略 | `model_20000_new.onnx` (47→12) | 腿部 12 关节行走 |
| 进球检测 | `soccer_referee.py` → `_is_ball_in_left_goal()` | 球门区域判定 |
| 仿真 API | `teleport_ball()`, `teleport_robot()` | 回合重置用 |
| 比赛配置 | `match_config.json` | 罚球点距离、球门尺寸等参数 |

### 需要新建/修改的文件

```
opt/sim_soccer2/
├── envs/                                    # ← 修改此目录
│   ├── soccer_env_motrix.py                 # 改：骨架已存在，填入真正的物理+奖励逻辑
│   └── penalty_shootout_config.py           # 新：奖励权重、训练超参
│
├── scripts/                                 # ← 新建此目录
│   └── train_penalty_shootout.py            # 新：PPO 训练入口脚本
│
└── simulation/motrixsim/app/                # ← 不改代码，只读其接口
    ├── multi_robot_sim.py                   # 参考：teleport_ball、进球检测逻辑
    ├── soccer_referee.py                    # 参考：_is_ball_in_left_goal()
    └── runtime_config.py                    # 参考：K1 关节参数、PD 增益
```

**具体文件职责：**

| 文件 | 状态 | 职责 |
|------|------|------|
| `envs/soccer_env_motrix.py` | **改造**（骨架已有 56 行） | `step()` 推物理+算奖励、`reset()` 摆球摆机器人、`_get_obs()` 拼观测向量 |
| `envs/penalty_shootout_config.py` | **新建** | dataclass：场地尺寸、奖励权重、训练超参 |
| `scripts/train_penalty_shootout.py` | **新建** | `stable_baselines3.PPO` 训练循环 + 模型保存 |

**不需要动的目录：**

| 目录 | 原因 |
|------|------|
| `simulation/motrixsim/` | 只作为 `import motrixsim` 库使用，不动源码 |
| `decider/` | 训练完成后，导出的 ONNX 模型交给 Sim Manager 加载即可 |
| `legged_gym/` | 预训练步态权重只读，可选用于 warm-start |

### RL 环境设计

**与 Sim Manager 的关系：** RL 环境不是替代 Sim Manager，而是在 MotrixSim 物理引擎外包一层标准 RL 接口。Sim Manager 面向人类操作（Web UI + ZMQ），RL 环境面向 PPO 算法（`env.step(action)` → `obs, reward, done`）。底层共用同一个物理引擎和球场场景。

```
┌─────────────────────────────────┐
│  PPO 训练脚本                     │
│  env.step(action) → obs, reward │
└──────────────┬──────────────────┘
               │ Python 直接调用（不走 ZMQ）
┌──────────────┴──────────────────┐
│  PenaltyShootoutEnv (gym.Env)   │  ← 需要新建
│  - reset(): 摆球、摆机器人        │
│  - step(): 推物理、算奖励、判终止  │
│  - _compute_reward(): 奖励函数   │
└──────────────┬──────────────────┘
               │ mtx.load_model / mtx.SceneData / mtx.step
┌──────────────┴──────────────────┐
│  MotrixSim 物理引擎（复用）        │
│  - 球场场景 + K1 机器人 + 球      │
│  - 500Hz 物理步进                │
│  - PD 控制器 → 关节力矩           │
└─────────────────────────────────┘
```

**观察空间（~80-100 维）**
```
球相对机器人位置 [3]       # 最关键：球在哪
球速度 [3]                 # 球是否在移动
球门相对机器人位置 [2]      # 射门目标方向
机器人本体感知 [47]         # 复用 K1 legged_gym 观察（重力、陀螺仪、关节状态）
球门尺寸 [2]               # 宽度、高度作为上下文
```

**动作空间**
```
方案 A（推荐，22 维）: 全身关节目标位置 → 训练完整的踢球动作
方案 B（12 维）:       腿部关节目标位置 → 配合预训练步态，从行走进化出踢球
```

**奖励函数**
```python
reward = (
    +2.0  * goal_scored           # 进球，最大正奖励
    +0.5  * ball_toward_goal      # 球向球门移动（每步递增）
    +0.3  * foot_contact_ball     # 脚触球
    -0.1  * distance_to_ball      # 离球太远
    -0.05 * step_penalty          # 每步微小惩罚，鼓励快速射门
    -1.0  * ball_out_of_bounds    # 球出界
)
```

**终止条件 & 重置**
```
终止:
  - 球进入球门 → 成功
  - 超时（如 500 步 = 2.5 秒）→ 失败
  - 球出界 → 失败
  - 球静止超过 1 秒（射门后）→ 判定是否进球

重置（reset）:
  - 球传送到罚球点（从 match_config.json 读取 penalty_spot_distance）
  - 机器人传送到罚球点后方 0.5m
  - 可选：随机扰动球的位置（±0.2m）增加泛化性
  - 可选：随机切换射门目标方向（球门左/中/右）
```

### 训练方案

**第一阶段：静态球射门（MVP）**

- 球静止在罚球点
- 无守门员
- 训练机器人走到球前完成射门动作
- 目标：学会用脚触球并将球踢向球门方向

**第二阶段：加入课程学习**

- 随机化球的位置（罚球点 ± 随机偏移）
- 加入守门员（先用规则控制：在门线横向移动封堵）
- 目标：学会根据守门员位置选择射门角度

**第三阶段：完整点球大战**

- 自对弈训练（shooter vs goalkeeper 同时训练）
- 多轮点球（5 轮制）
- 目标：训练出有竞争力的射门和守门策略

### 训练工具选择

有两种方案可供选择：

| 方案 | 优点 | 缺点 |
|------|------|------|
| **A. stable-baselines3** | 轻量，快速上手，文档丰富 | 不支持大规模并行训练 |
| **B. MotrixLab (已有)** | 支持 2048 并行环境，训练快 | 需要理解其注册机制，较重 |

建议先用方案 A 验证整个 pipeline（能在几天内看到初步结果），确认观察/动作/奖励设计合理后，再迁移到方案 B 做大规模训练。

### 工作量估算

| 任务 | 预计工作量 | 优先级 |
|------|-----------|--------|
| 编写 `PenaltyShootoutEnv` | 1-2 天 | P0 |
| 调试观察空间和奖励函数 | 2-3 天 | P0 |
| stable-baselines3 训练脚本 + 初步训练 | 1-2 天 | P0 |
| 导出 ONNX 模型 + 在 Sim Manager 中加载验证 | 1 天 | P1 |
| 加入守门员 + 课程学习 | 3-5 天 | P1 |
| 大规模并行训练（MotrixLab） | 3-5 天 | P2 |
| 自对弈训练 | 5-10 天 | P2 |

---

## 七、当前 Git 状态

- **分支**: `motrixsim2`
- **未暂存修改**: decider 源码 (`user_entry.py`, `config.yaml`), 仿真文件 (`multi_robot_sim.py`, `runtime_config.py`), 大量 `__pycache__`
- **未跟踪**: `legged_gym/` 目录, `model_20000_new.onnx`, `model_4700.pt`, WebRTC server 等
