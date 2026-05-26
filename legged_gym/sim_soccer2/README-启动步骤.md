# 机器人足球仿真环境启动步骤

整个**sim-soccer**环境分为两端，需分别启动并通过 **ZMQ** 进行通信：

- **仿真端（Sim）**：基于 motrixsim 的物理仿真 + Sim Manager 网页管理器
- **决策端（Decider）**：机器人策略大脑，以客户端方式连接仿真



## 目录

- [一、环境准备](#一环境准备)
- [二、启动 Sim Manager（推荐）](#二启动-sim-manager推荐)
- [三、启动 Decider 决策端](#三启动-decider-决策端)
- [四、纯命令行启动单个仿真（不用 Sim Manager）](#四纯命令行启动单个仿真不用-sim-manager)
- [五、可能遇到的问题](#五可能遇到的问题)
- [六、启动流程总览](#六启动流程总览)



## 一、环境准备

项目使用 **Conda** 管理两个独立的 Python 环境，分别对应仿真端和决策端。

| 环境名 | Python 版本 | 用途 | 依赖文件 |
|---|---|---|---|
| `motrixsim0508` | 5.8 | 运行 motrixsim 仿真和 Sim Manager | `simulation/motrixsim/requirements.txt` |
| `k2` | 3.8 | 运行 Decider 决策端 | `decider/requirements.txt` |

### 1.1 安装 Conda

```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
~/miniconda3/bin/conda init bash
source ~/.bashrc
```

### 1.2 创建仿真环境 `motrixsim0508`

```bash
cd ./sim_soccer2/simulation/motrixsim
conda create -n motrixsim0508 python=3.12 -y
conda activate motrixsim0508
pip install -r requirements.txt
```

可以先检查依赖是否就绪（检不检测都行，检测的话把稳一些^ ^）：

```bash
python -c "import motrixsim, torch, zmq, flask, fastapi, uvicorn; print('ok')"
```

### 1.3 创建决策环境 `k2`

```bash
cd ./sim_soccer2
conda create -n k2 python=3.8 -y
conda activate k2
pip install -r decider/requirements.txt
```



## 二、启动 Sim Manager（推荐）

Sim Manager 是一个基于 FastAPI 的网页管理器，可以在浏览器中可视化地启动、停止和管理多个仿真实例，并自动生成 Decider 启动命令。


####################################2026-05-12####################
### 2.1 启动 Manager 服务
```
cd ./sim_soccer2
conda activate motrixsim0508
python simulation/motrixsim/sim_manager.py --host 0.0.0.0 --port 8000
```


### 2.2 访问页面

| 入口 | 地址 |
|---|---|
| Manager 前端页面（核心） | `http://127.0.0.1:8000/` |
| Swagger 文档 | `http://127.0.0.1:8000/docs` |
| API 说明页 | `http://127.0.0.1:8000/manager/docs` |

### 2.3 在网页上的操作流程

1. **启动仿真实例**
   - 在 `Start Simulation` 区域填写 `team_size`（如 `3`），以及可选的渲染参数。
   - `zmq_port`、`webview_port` 可留空，Manager 会自动分配可用端口。
   - 点击 **Start**。

2. **查看实例资源**
   - 在 `Scanned Sim Processes` 可以看到：
     - `PID`
     - `ZMQ Port`（Decider 用来控制的端口）
     - `Team Size`
     - `WebView` 链接（点击即可查看 3D 仿真画面）

3. **停止仿真进程**
   - 在 `Process Control` 下拉选择 PID，点击 `Stop PID`。
   - 点击 `Stop External` 可停止非 Manager 创建但被扫描到的仿真进程。

4. **生成 Decider 启动命令**
   - 选择 PID 后展开 `Decider Commands`，页面会按红/蓝队和机器人编号自动生成命令，**可直接复制到新终端使用**。

5. **查看日志**
   - `Start Log`：启动结果
   - `Action Log`：停止/清理结果



## 三、启动 Decider 决策端

### 方式 A：单机器人启动（调试推荐）

打开**新的终端**，激活 `k2` 环境，使用 Sim Manager 分配的 ZMQ 端口启动：

```bash
cd ./sim_soccer2
conda activate k2

# 红队 0 号（--port 替换为 Sim Manager 分配的 ZMQ 端口）
python decider/decider.py --simulation --ip 127.0.0.1 --port 5555 --color red --id 0
# 红队 1 号
python decider/decider.py --simulation --ip 127.0.0.1 --port 5555 --color red --id 1
# 红队 2 号
python decider/decider.py --simulation --ip 127.0.0.1 --port 5555 --color red --id 2

# 蓝队 0 号
python decider/decider.py --simulation --ip 127.0.0.1 --port 5555 --color blue --id 0
# 蓝队 1 号
python decider/decider.py --simulation --ip 127.0.0.1 --port 5555 --color blue --id 1
# 蓝队 2 号
python3 decider/decider.py --simulation --ip 127.0.0.1 --port 5555 --color blue --id 2
s
```

主要参数：

| 参数 | 说明 |
|---|---|
| `--simulation` | 使用仿真模式（否则走真机协议） |
| `--ip` | Sim Manager 所在主机 IP（同机即 `127.0.0.1`） |
| `--port` | ZMQ 端口，从 Sim Manager 页面 `Scanned Sim Processes` 中复制 |
| `--color` | `red` 或 `blue` |
| `--id` | 机器人编号（`0..6`） |

> 说明：蓝队会根据 `match_config` 自动做队伍 ID 偏移，红/蓝队可共用同一套策略代码。

### 方式 B：一键启动整支球队（Linux / macOS）

```bash
# 按 match_config 自动启动
./decider/scripts/start_team.sh

# 自定义数量
./decider/scripts/start_team.sh --red 2 --blue 1

# 重启全部 / 停止全部
./decider/scripts/start_team.sh --restart
./decider/scripts/start_team.sh --kill
```

脚本会通过 `screen` 为每个机器人单独开一个会话。查看某个机器人的日志：

```bash
screen -r decider_red_0     # 进入会话；按 Ctrl+A 然后 D 退出
```



## 四、纯命令行启动单个仿真（不用 Sim Manager）

如果只想本地跑一个仿真实例，不需要网页管理器，可以直接启动 `sim2sim_runner.py`：

```bash
cd ./sim_soccer2/simulation/motrixsim
conda run -n motrixsim0508 python sim2sim_runner.py --team-size 3 --real-time
```

默认端口：

- WebView：`http://127.0.0.1:5811`
- ZMQ REP：`tcp://*:5555`

常用参数：

| 参数 | 说明 |
|---|---|
| `--robot-type` | 机器人类型，`k2`（默认）或 `pi_plus` |
| `--team-size` | 每队机器人数量（红蓝相等），范围 `0..7`，默认 `1` |
| `--use-referee` | 启用内置裁判盒（开球 / 出界 / 角球 / 门球 / 进球 / 超时等） |

示例（启动 `pi_plus`）：

```bash
conda run -n motrixsim0508 python sim2sim_runner.py --robot-type pi_plus --team-size 3 --real-time
```

### 固定 Robot ID 映射

- `0..6 -> robot_rp0..robot_rp6`（红队）
- `7..13 -> robot_bp0..robot_bp6`（蓝队）

即使 `--team-size` 小于 7，映射仍然固定；未启用的 ID 会被忽略。



## 五、可能遇到的问题

### 5.1 清理残留的 Decider 进程

```bash
ps -ef | grep decider.py | grep -v grep
sudo pkill -f decider.py
```

### 5.2 清理残留的仿真进程

- 方法 1：在 Sim Manager 页面点击 `Stop External`
- 方法 2：按 PID 手动 `kill`

### 5.3 Windows 用户注意事项

- `start_team.sh` 依赖 `screen` 和 `bash`，在 Windows 上建议通过 **WSL2** 或直接在 Linux 服务器上运行。
- `decider.py` 单独启动可以直接在 Windows PowerShell 里跑，把 `python3` 改成 `python` 即可：

  ```powershell
  conda activate k2
  python decider/decider.py --simulation --ip 127.0.0.1 --port 5555 --color red --id 0
  ```

### 5.4 将 Sim Manager 持久化为 systemd 服务（可选）

适用于断开 SSH 后需要持续运行、或开机自动拉起 Manager 的场景，详细步骤见 [`simulation/motrixsim/README.md`](./simulation/motrixsim/README.md#将-sim-manager-持久化为-systemd-服务)。



## 六、启动流程总览

```
┌──────────────────────────────────────────────────────────────┐
│ 终端 1（仿真端 / motrixsim0508 环境）                               │
│   conda run -n motrixsim0508 \                                   │
│     uvicorn simulation.labbridge.sim_manager:app \           │
│     --host 0.0.0.0 --port 8000                               │
└──────────────────────────────────────────────────────────────┘
                          │
                          ▼
   浏览器打开 http://127.0.0.1:8000/
     → 点击 Start 启动仿真
     → 得到 ZMQ 端口（例如 5555）
     → WebView 实时查看 3D 画面
                          │
                          ▼
┌──────────────────────────────────────────────────────────────┐
│ 终端 2（决策端 / k2 环境）                                      │
│   conda activate k2                                          │
│   python3 decider/decider.py \                               │
│     --simulation --ip 127.0.0.1 --port 5555 \                │
│     --color red --id 0                                       │
└──────────────────────────────────────────────────────────────┘
                          │
                          ▼
   Decider 通过 ZMQ 连接仿真，按照 decider/user_entry.py 中
   game(agent) 的逻辑驱动机器人在球场上行动。
```

启动后在 WebView 页面就能看到机器人按照策略开始行动。多机器人只需重复打开多个终端，改变 `--color` 和 `--id` 即可。
