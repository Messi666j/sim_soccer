# Unitree GO1 Complex Terrain Locomotion

The Unitree GO1 Complex Terrain Walking Environment is a quadruped robot reinforcement learning task designed to train robots to achieve stable walking on challenging terrain. This environment includes two main terrain types: rough terrain and stairs terrain.

```{video} /_static/videos/go1_rough_terrain_walk.mp4
:poster: _static/images/poster/go1_rough_terrain_walk.jpg
:nocontrols:
:autoplay:
:playsinline:
:muted:
:loop:
:width: 100%
```

---

```{video} /_static/videos/go1_stairs_terrain_walk.mp4
:poster: _static/images/poster/go1_stairs_terrain_walk.jpg
:nocontrols:
:autoplay:
:playsinline:
:muted:
:loop:
:width: 100%
```

## Task Description

Train the Unitree GO1 quadruped robot to achieve stable and efficient quadruped walking on complex terrain. This environment uses MotrixSim physics engine for simulation, providing high-fidelity dynamic simulation. The agent controls target positions of each joint to achieve velocity tracking and attitude stability while adapting to different terrain challenges.

### Task Objectives

-   **Velocity Tracking**: Accurately track given linear and angular velocity commands
-   **Attitude Stability**: Maintain body attitude stability under various terrain conditions
-   **Energy Efficiency**: Achieve walking tasks with minimal energy consumption
-   **Terrain Adaptability**: Adapt to different challenges of rough terrain and stairs terrain

---

## Action Space

| Item          | Details                          |
| ------------- | -------------------------------- |
| **Type**      | `Box(-1.0, 1.0, (12,), float32)` |
| **Dimension** | 12                               |

Actions correspond to position control commands for 12 joints, including hip joints, thigh joints, and calf joints for all four legs.

---

## Observation Space

### Rough Terrain Observation Space (48-dimensional)

| Item          | Details                          |
| ------------- | -------------------------------- |
| **Type**      | `Box(-inf, inf, (48,), float32)` |
| **Dimension** | 48                               |

| Part                  | Content Description                         | Dim | Notes              |
| --------------------- | ------------------------------------------- | --- | ------------------ |
| **noisy_linvel**      | Linear velocity in body coordinate system   | 3   | With noise         |
| **noisy_gyro**        | Angular velocity in body coordinate system  | 3   | With noise         |
| **local_gravity**     | Local gravity direction                     | 3   | Gravity vector     |
| **noisy_joint_angle** | Joint angle deviation from default angles   | 12  | 12 joints          |
| **noisy_joint_vel**   | Joint angular velocities                    | 12  | With noise         |
| **last_actions**      | Control actions from previous time step     | 12  | Historical actions |
| **command**           | Target linear velocity and angular velocity | 3   | [vx, vy, vyaw]     |

### Stairs Terrain Observation Space (60-dimensional)

| Item          | Details                          |
| ------------- | -------------------------------- |
| **Type**      | `Box(-inf, inf, (60,), float32)` |
| **Dimension** | 60                               |

In addition to the rough terrain observations, it includes:

| Part                   | Content Description                 | Dim | Notes                    |
| ---------------------- | ----------------------------------- | --- | ------------------------ |
| **feet_contact_force** | Contact force vectors for four feet | 12  | 3D per foot (Fx, Fy, Fz) |

---

## Reward Function Design

GO1 complex terrain reward function adopts multi-objective weighted design:

```python
# Core reward components
reward_config.scales = {
    "tracking_lin_vel": 1.0,      # Linear velocity tracking accuracy
    "tracking_ang_vel": 0.5,      # Angular velocity tracking accuracy
    "orientation": -0.0,          # Body attitude stability penalty
    "torques": -0.00001,          # Joint torque penalty (energy efficiency)
    "dof_acc": -2.5e-7,           # Joint acceleration penalty
    "action_rate": -0.001,        # Action smoothness penalty
    "feet_air_time": 1.0,         # Foot air time reward (encourage large strides)
    "stand_still": 0.0,           # Joint position maintenance for stationary commands
    "hip_pos": -1,                # Hip joint position preference
    "calf_pos": -0.3,             # Calf joint position preference
    "feet_stumble": -0.5,         # Penalty when foot laterally touches obstacle
}

# Total reward = weighted combination of all above terms
```

---

## Initial State

### Rough Terrain Initialization

-   **Terrain Generation**: Use height maps to generate random terrain
-   **Terrain Height Levels**: Three preset height levels: -2.5m, 0.5m, 2.0m
-   **Position Randomization**: At basic training level, robot position is fixed; at advanced level, randomly selects from 25 preset positions in cycles

### Stairs Terrain Initialization

-   **Terrain Type**: Various step-based small terrain blocks arranged on continuous terrain
-   **Position Randomization**: Similar position randomization strategy as rough terrain

### Robot Initialization

-   **Joint Angles**: Set to default standing posture with [-0.125, 0.125] radian noise
-   **Velocity Initialization**: All linear and angular velocities initialized to zero

---

## Episode Termination Conditions

-   **Body Contact**: Robot trunk makes unexpected contact with ground
-   **Velocity Anomaly**: Sum of squared linear velocities exceeds threshold (1e8)

---

## Usage Guide

### 1. Environment Preview

```bash
uv run scripts/view.py --env go1-rough-terrain-walk
uv run scripts/view.py --env go1-stairs-terrain-walk
```

### 2. Start Training

```bash
uv run scripts/train.py --env go1-rough-terrain-walk
uv run scripts/train.py --env go1-stairs-terrain-walk
```

### 3. View Training Progress

```bash
uv run tensorboard --logdir runs/go1-rough-terrain-walk
```

### 4. Test Training Results

Since the rough terrain scene generates both an infinite plane and a rugged terrain height field, when testing training results, the agent will first be spawned on the flat terrain following the training process, complete one round of walking, and then be spawned onto the rugged terrain. Users need to actively adjust camera perspective and position to observe the agent's status.

```bash
uv run scripts/play.py --env go1-rough-terrain-walk
uv run scripts/play.py --env go1-stairs-terrain-walk
```

---

## Expected Training Results

### Rough Terrain Task (go1-rough-terrain-walk)

1. Capable of adapting to different rough terrain heights
2. High velocity tracking accuracy and stable posture
3. Coordinated gait with minimal foot slipping

### Stairs Terrain Task (go1-stairs-terrain-walk)

1. Capable of stably going up and down stairs
2. Capable of adapting to stairs of different heights and widths
3. Smooth movements without obvious stuttering

---

## Training Performance Reference

### go1-rough-terrain-walk

| Operating System | Training Backend | CPU               | GPU         | Num Environments | Training Time (30000 steps) |
| ---------------- | ---------------- | ----------------- | ----------- | ---------------- | --------------------------- |
| Ubuntu 22.04     | JAX              | AMD Ryzen 7 9700X | RTX 5070 Ti | 2048             | 7m20s                       |
| Ubuntu 22.04     | PyTorch          | AMD Ryzen 7 9700X | RTX 5070 Ti | 2048             | 8m30s                       |
| Windows 11       | PyTorch          | AMD Ryzen 7 9700X | RTX 5070 Ti | 2048             | 10m42s                      |

### go1-stairs-terrain-walk

| Operating System | Training Backend | CPU               | GPU         | Num Environments | Training Time (30000 steps) |
| ---------------- | ---------------- | ----------------- | ----------- | ---------------- | --------------------------- |
| Ubuntu 22.04     | JAX              | AMD Ryzen 7 9700X | RTX 5070 Ti | 2048             | 7m18s                       |
| Ubuntu 22.04     | PyTorch          | AMD Ryzen 7 9700X | RTX 5070 Ti | 2048             | 8m41s                       |
| Windows 11       | PyTorch          | AMD Ryzen 7 9700X | RTX 5070 Ti | 2048             | 10m52s                      |
