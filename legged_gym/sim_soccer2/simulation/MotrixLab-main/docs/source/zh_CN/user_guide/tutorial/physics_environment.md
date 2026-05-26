# 物理环境配置

物理环境配置定义了强化学习训练中的仿真参数和模型文件设置。
MotrixLab 使用了[MotrixSim](https://motrixsim.readthedocs.io/zh-cn/latest/user_guide/index.html)作为物理仿真后端。

## 支持的文件格式

-   [**MJCF**](https://mujoco.readthedocs.io/en/stable/XMLreference.html)(MuJoCo XML 格式) - 提供丰富的物理特性和仿真配置

## 模型文件配置

需要在环境配置类中指定模型文件路径：

```python
@registry.envcfg("my-task")
@dataclass
class MyTaskEnvCfg(EnvCfg):
    # 模型文件路径（必需）
    model_file: str = "my_model.xml"

    # 仿真时间参数
    sim_dt: float = 0.002      # 仿真时间步
    ctrl_dt: float = 0.02      # 控制更新频率
```

### 推荐目录结构

```
motrix_envs/my_task/
├── __init__.py          # 模块初始化
├── cfg.py               # 环境配置
├── my_model.xml         # 物理模型文件
└── my_env.py            # 环境实现
```

对于结构复杂，引用文件较多的模型，推荐使用文件夹管理。

## 常见配置问题

### 文件路径问题

-   使用相对路径时，确保路径相对于配置文件位置
-   避免使用硬编码的绝对路径
-   检查文件权限和可访问性
-   确保所有引用的子文件都存在

### 时间步设置

-   `ctrl_dt` 应该是 `sim_dt` 的整数倍
-   `sim_dt` 过小会影响仿真性能
-   `ctrl_dt` 过大会影响控制精度
-   推荐 `sim_dt` 在 0.001-0.02 秒之间

### 仿真稳定性

-   避免过大的时间步长
-   合理设置接触参数避免穿透
-   质量和惯性分布要合理
-   关节限制要符合实际情况

通过合理的物理环境配置，您可以为强化学习训练创建准确且高效的仿真环境。
