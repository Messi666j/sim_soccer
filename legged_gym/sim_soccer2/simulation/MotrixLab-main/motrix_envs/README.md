# Motrix Environments (Motrix_envs)

motrix Environments 是一个基于 MotrixSim 仿真后端的强化学习环境库，提供了与具体 RL 框架无关的仿真环境定义。该模块设计上支持多种仿真后端，目前主要实现了 MotrixSim 的 NumPy 后端。

## 📁 项目结构

```
motrix_envs/
├── base.py              # 基础抽象类和配置
├── registry.py          # 环境注册系统
├── np/                  # NumPy 仿真后端实现
│   ├── env.py          # NumPy 环境基类
│   ├── renderer.py     # 渲染器
│   └── reward.py       # 奖励函数
├── basic/               # 基础环境
│   ├── cartpole/       # 倒立摆环境
│   └── walker/         # 步行者环境
├── locomotion/         # 运动控制环境
│   └── go1/            # GO1 机器人
│       ├── xmls/       # 机器人模型文件
│       ├── walk_np.py  # GO1 行走实现
│       └── cfg.py      # GO1 配置
└── common/              # 公共组件
```

## 🎯 内置环境

| 环境名称         | 注册标识符              | 后端 | 类型       | 描述                          |
| ---------------- | ----------------------- | ---- | ---------- | ----------------------------- |
| **倒立摆**       | `cartpole`              | np   | Basic      | 经典控制任务，保持杆子平衡    |
| **步行者**       | `walker`                | np   | Basic      | 平面双足步行机器人控制        |
| **GO1 平地行走** | `go1-flat-terrain-walk` | np   | Locomotion | 四足机器人 GO1 的平地行走任务 |

### 详细说明

#### 1. CartPole (倒立摆) - `cartpole`

-   **观测空间 (4 维)**:
    -   `cart_pos`: 小车位置 [-0.8, 0.8]
    -   `pole_angle`: 杆子角度 [-0.2, 0.2]
    -   `cart_vel`: 小车速度
    -   `pole_vel`: 杆子角速度
-   **动作空间 (1 维)**: 推力 `[-3.0, 3.0]`
-   **奖励**: 每步 +1.0
-   **终止条件**: 杆子角度 > 0.2 弧度 或 小车位置超出边界
-   **配置参数**: `reset_noise_scale=0.01`

#### 2. GO1 平地行走 - `go1-flat-terrain-walk`

-   **观测空间 (48 维)**: 包含关节位置、速度、IMU 数据、命令等
-   **动作空间 (12 维)**: 12 个关节的目标角度控制
-   **奖励组件**:
    -   `tracking_lin_vel`: 线速度跟踪 (权重: 1.0)
    -   `tracking_ang_vel`: 角速度跟踪 (权重: 0.5)
    -   `feet_air_time`: 足部空中时间 (权重: 1.0)
    -   `collision`: 碰撞惩罚 (权重: -1.0)
    -   `action_rate`: 动作变化率惩罚 (权重: -0.001)
-   **控制参数**: 刚度 80.0 Nm/rad, 阻尼 1.0 Nms/rad
-   **噪声模型**: 关节角度、速度、陀螺仪等多种传感器噪声

## 🛠️ 自定义环境

### 开发步骤

#### 1. 定义环境配置类

```python
from dataclasses import dataclass
from motrix_envs import registry
from motrix_envs.base import EnvCfg

@registry.envcfg("my-custom-env")
@dataclass
class MyEnvCfg(EnvCfg):
    """自定义环境配置"""
    # 继承基础配置
    reset_noise_scale: float = 0.01
    max_episode_seconds: float = 10.0

    # 添加自定义参数
    custom_param_1: float = 1.0
    custom_param_2: str = "default_value"
    enable_feature_x: bool = True
```

#### 2. 实现环境类

```python
import gymnasium as gym
import numpy as np
from motrix_envs import registry
from motrix_envs.np.env import NpEnv, NpEnvState

@registry.env("my-custom-env", "np")
class MyCustomEnv(NpEnv):
    """自定义环境实现"""

    def __init__(self, cfg: MyEnvCfg, num_envs: int = 1):
        super().__init__(cfg, num_envs=num_envs)

    @property
    def observation_space(self):
        """返回观测空间"""
        raise NotImplementedError

    @property
    def action_space(self):
        """返回动作空间"""
        raise NotImplementedError

    def apply_action(self, actions: np.ndarray, state: NpEnvState):
        """将actions应用到环境状态"""
        raise NotImplementedError
        return state

    def update_state(self, state: NpEnvState):
        """更新环境状态，计算观测、奖励、终止条件"""
        # 提取仿真数据
        data = state.data

        # 计算观测
        obs = self._compute_observation(data)

        # 计算奖励
        reward = self._compute_reward(data, obs)

        # 检查终止条件
        terminated = self._check_termination(data, obs)

        # 更新状态
        state.obs = obs
        state.reward = reward
        state.terminated = terminated
        return state

    def reset(self, data: mtx.SceneData) -> tuple[np.ndarray, dict]:
        """
        重置环境
        参数:
            data: motrixsim 场景数据对象，仅包含需要重置的环境实例
        返回:
            obs: 重置后场景的初始观测
            info: 额外信息
        """
        raise NotImplementedError

```

#### 3. 注册和使用环境

```python
# 确保导入了环境模块，触发注册
import motrix_envs.basic.my_custom_env  # noqa: F401

from motrix_envs import registry

# 创建环境实例
env = registry.make(
    name="my-custom-env",
    sim_backend="np",
    num_envs=256,
    env_cfg_override={
        "custom_param_1": 2.0,
        "reset_noise_scale": 0.02
    }
)

# 使用环境
for step in range(1000):
    actions = sample_actions_somehow()
    state = env.step(actions)
```
