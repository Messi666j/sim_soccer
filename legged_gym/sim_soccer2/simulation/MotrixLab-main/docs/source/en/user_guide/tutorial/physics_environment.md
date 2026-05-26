# Physics Environment Configuration

Physics environment configuration defines simulation parameters and model file settings in reinforcement learning training.
MotrixLab uses [MotrixSim](https://motrixsim.readthedocs.io/en/latest/user_guide/index.html) as the physics simulation backend.

## Supported File Formats

-   [**MJCF**](https://mujoco.readthedocs.io/en/stable/XMLreference.html) (MuJoCo XML format) - Provides rich physics features and simulation configuration

## Model File Configuration

You need to specify model file paths in environment configuration classes:

```python

@registry.envcfg("my-task")
@dataclass
class MyTaskEnvCfg(EnvCfg):
    # Model file path (required)
    model_file: str = "my_model.xml"

    # Simulation time parameters
    sim_dt: float = 0.002      # Simulation time step
    ctrl_dt: float = 0.02      # Control update frequency

    # Episode parameters
    max_episode_seconds: float = 20.0
    reset_noise_scale: float = 0.01
```

### Recommended Directory Structure

```
motrix_envs/my_task/
├── __init__.py          # Module initialization
├── cfg.py               # Environment configuration
├── my_model.xml         # Physics model file
└── my_env.py            # Environment implementation
```

For complex models with many referenced files, it's recommended to use folder management.

## Common Configuration Issues

### File Path Issues

-   When using relative paths, ensure paths are relative to the configuration file location
-   Avoid using hardcoded absolute paths
-   Check file permissions and accessibility
-   Ensure all referenced sub-files exist

### Time Step Settings

-   `ctrl_dt` should be an integer multiple of `sim_dt`
-   `sim_dt` that is too small will affect simulation performance
-   `ctrl_dt` that is too large will affect control precision
-   Recommend `sim_dt` between 0.001-0.02 seconds

### Simulation Stability

-   Avoid excessively large time steps
-   Set contact parameters reasonably to avoid penetration
-   Mass and inertia distribution should be reasonable
-   Joint limits should match actual conditions

Through proper physics environment configuration, you can create accurate and efficient simulation environments for reinforcement learning training.
