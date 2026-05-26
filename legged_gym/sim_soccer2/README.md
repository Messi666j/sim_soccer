# 机器人足球仿真环境项目速览



## 目录

- [一、项目概述](#一项目概述)

- [二、模块组成](#二模块组成)

- [三、技术架构](#三技术架构)

- [四、策略自定义开发](#四策略自定义开发)

- [五、Decider的API接口 ](#五Decider的API接口)

- [六、Decider 配置](#六decider-配置)

  

## 一、项目概述

**sim-soccer** 是一个面向双足 / 多足机器人（如 **K1**、**Pi Plus**）的足球竞技仿真与决策控制项目。它在虚拟环境中（MotrixSim / Isaac Sim）还原足球比赛场景，并为研究者提供一套完整的感知—决策—控制框架，可用于多智能体策略的研究、比赛训练和算法验证。

项目将**物理仿真、进程管理、网络通信**等底层复杂性全部封装，上层开发者只需使用简洁的 Python API，即可专注于编写多机器人的足球竞技策略。



## 二、模块组成

项目主要分为2部分：**决策层（Decider）** 和 **仿真层（Simulation）**。

```
mos-sim/
├── decider/                   # 决策模块（机器人大脑）
│   ├── decider.py             # 决策端入口
│   ├── user_entry.py          # 用户自定义策略入口
│   ├── config.yaml            # 决策端配置
│   ├── interfaces/            # 感知 / 动作 / 比赛控制接口
│   ├── logic/                 # 状态机实现
│   │   ├── sub_statemachines/     # 基础动作状态机（找球、追球等）
│   │   ├── strategy_statemachines/# 战术级状态机（进攻、防守等）
│   │   └── policy_statemachines/  # 策略级状态机（守门员等）
│   └── scripts/               # 整队启动、部署脚本
└── simulation/                # 仿真模块
    ├── motrixsim/             # MotrixSim 仿真环境（主推）
    ├── isaac_sim/             # Isaac Sim 仿真环境（旧版）
    └── labbridge/             # WebView / Bridge / Sim Manager
```

### 2.1 Decider 决策模块（`decider/`）

机器人的“大脑”，负责逻辑与策略开发。以客户端形式连接仿真服务器，获取感知数据（球的位置、机器人自身位姿等），并输出动作指令（速度控制、踢球等）。

### 2.2 Simulation 仿真模块（`simulation/`）

- **`motrixsim/`**：基于 MotrixSim 的物理仿真环境，**当前主推的仿真引擎**，附带基于 FastAPI 的可视化管理器（Sim Manager），方便在网页端启停和管理多个仿真实例。
- **`isaac_sim/`**：基于 NVIDIA Isaac Sim 的仿真环境（旧版，主实现已迁移到 MotrixSim）。
- **`labbridge/`**：独立的 WebView、通信桥梁（Bridge）以及仿真进程管理模块，负责仿真环境与外部界面的通信。



## 三、技术架构

### 3.1 通信机制

仿真环境（服务端）与决策代码（客户端）之间通过 **ZMQ (ZeroMQ)** 进行通信。每个机器人对应一个 ZMQ 端口，支持多机器人并发控制。

**请求格式（单指令）：**

```json
{"cmd":[vx, vy, w], "id":0, "timestamp": 0, "source":"xxx"}
```

**响应格式：**

```json
{
  "state": {
    "robots": [{"id":0, "name":"robot_rp0", "x":0, "y":0, "theta":0, "team":"red"}],
    "ball": {"x":0, "y":0, "z":0}
  },
  "sim_timestamp": 0,
  "step_latency": 0,
  "ack_timestamp": 0
}
```

### 3.2 Web 可视化管理（Sim Manager）

通过 Sim Manager 可以在浏览器中（`http://127.0.0.1:8000/`）：

- 可视化启停仿真实例
- 设置队伍规模（Team Size）
- 查看 PID、ZMQ Port、WebView 链接
- **一键生成 Decider 启动命令**（红蓝队、按机器人编号自动生成）

### 3.3 坐标系设计

项目定义了两套坐标系，分别用于相对感知与全局定位：

**机器人坐标系（Robot Frame）：**

- `X`：前向（米）
- `Y`：左向（米）
- `Theta`：逆时针（弧度），`0` 为正前方

**地图坐标系（Map Frame）：**

- `Y`：朝向对方球门
- `X`：右侧
- **蓝队相对仿真全局坐标做了镜像处理**，使得两队可以**完全复用同一套策略代码**，开发者无需为不同半场写两套逻辑。

### 3.4 固定 Robot ID 映射

无论 `team_size` 设置为多少，ID 映射始终固定：

- `0..6 -> robot_rp0..robot_rp6`（红队）
- `7..13 -> robot_bp0..robot_bp6`（蓝队）

未启用的 ID 会被自动忽略。



## 四、策略自定义开发

### 4.1 主入口

开发者在 `decider/user_entry.py` 中编写自定义的机器人策略。系统采用**状态机**架构来管理机器人的行为。

### 4.2 `game(agent)` 主循环

```python
def game(agent):
    # 如果没看到球，执行"找球"状态机
    if not agent.get_if_ball():
        agent.state_machine_runners['find_ball']()
    # 如果看到了球，执行"追球"状态机
    else:
        agent.state_machine_runners['chase_ball']()
```

### 4.3 内置状态机

项目按照**三层状态机**架构组织策略代码：

| 层级 | 目录 | 作用 | 示例 |
|---|---|---|---|
| **基础动作层** | `decider/logic/sub_statemachines/` | 基础足球动作 | `find_ball`、`chase_ball`、`dribble`、`kick`、`go_back_to_field` |
| **战术层** | `decider/logic/strategy_statemachines/` | 战术组合 | `attack`、`defend_ball`、`dribble_ball`、`shoot_ball` |
| **策略层** | `decider/logic/policy_statemachines/` | 角色策略 | `goalkeeper` 等 |

**调用方式：**

```python
agent.state_machine_runners['state_machine_name']()
```



## 五、Decider的API接口 

接口文件：

- **Action**：`decider/interfaces/action.py`
- **Vision**：`decider/interfaces/vision.py`
- **GameController**：`decider/interfaces/gamecontroller.py`

### 5.1 感知 API

| 方法 | 说明 | 返回 |
|---|---|---|
| `agent.get_ball_pos()` | 球在机器人坐标系的位置 | `[x, y]` 或 `[None, None]` |
| `agent.get_ball_distance()` | 到球的距离 | `float` |
| `agent.get_ball_angle()` | 球相对机器人坐标系的角度 | `float` |
| `agent.get_if_ball()` | 是否看见球 | `bool` |
| `agent.get_self_pos()` | 机器人在地图坐标系的位置 | `[x, y]` |
| `agent.get_self_yaw()` | 机器人在地图坐标系的朝向 | `float` |

### 5.2 动作 API

| 方法 | 说明 |
|---|---|
| `agent.cmd_vel(vx, vy, vtheta)` | 机器人坐标系下的速度控制 |
| `agent.stop()` | 停止运动 |
| `agent.kick()` | 执行踢球 |
| `agent.head_control(pitch, yaw)` | 头部 / 云台角度控制 |



## 六、Decider 配置

编辑 `decider/config.yaml` 可调整默认参数：

- `color`：默认队伍颜色
- `id`：默认机器人 ID

### `cmd_vel` 处理

Decider 不再在 `config.yaml` 中提供专门的 `cmd_vel` 限幅项。速度命令的缩放与整形由控制逻辑本身完成。
