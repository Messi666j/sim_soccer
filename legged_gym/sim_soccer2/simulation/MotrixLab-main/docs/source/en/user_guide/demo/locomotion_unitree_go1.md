# Unitree GO1 Locomotion

Unitree GO1 is a quadruped robot platform. This example demonstrates how to train GO1 to achieve stable gait walking on flat terrain.

```{video} /_static/videos/go1_walk.mp4
:poster: _static/images/poster/go1_walk.jpg
:nocontrols:
:autoplay:
:playsinline:
:muted:
:loop:
:width: 100%
```

## Task Description

The GO1 quadruped robot has 12 degrees of freedom (3 joints per leg) and needs to learn coordinated gait control through deep reinforcement learning:

-   **State Space**: 48-dimensional, including robot linear velocity, angular velocity, posture, joint angles, joint velocities, actions, and commands
-   **Action Space**: 12-dimensional, controlling target positions of each joint (converted to torques through PD controller)
-   **Reward Function**: Composite reward including speed tracking, posture stability, energy efficiency, and other components
-   **Termination Conditions**: Robot trunk contacts ground or other unstable states

---

## Usage Guide

### 1. Environment Preview

```bash
uv run scripts/view.py --env go1-flat-terrain-walk
```

### 2. Start Training

```bash
uv run scripts/train.py --env go1-flat-terrain-walk
```

### 3. View Training Progress

```bash
uv run tensorboard --logdir runs/go1-flat-terrain-walk
```

### 4. Test Training Results

```bash
uv run scripts/play.py --env go1-flat-terrain-walk
```

## Reward Function Design

GO1's reward function is a complex composite function containing multiple components:

### Main Reward Components

```python
reward_config.scales = {
    "tracking_lin_vel": 1.0,      # Linear velocity tracking reward
    "tracking_ang_vel": 0.5,      # Angular velocity tracking reward
    "feet_air_time": 1.0,         # Foot air time reward
    "lin_vel_z": -2.0,            # Z-axis linear velocity penalty
    "ang_vel_xy": -0.05,          # XY-axis angular velocity penalty
    "orientation": -0.0,          # Posture deviation penalty
    "torques": -0.00001,          # Torque consumption penalty
    "dof_acc": -2.5e-7,           # Joint acceleration penalty
    "action_rate": -0.001,        # Action change rate penalty
    "hip_pos": -1,                # Hip joint position penalty
    "calf_pos": -0.3,             # Calf joint position penalty
}
```

### Key Reward Functions

#### Velocity Tracking Reward

```python
# Track linear velocity commands (xy plane)
def _reward_tracking_lin_vel(self, data, commands):

# Track angular velocity commands (yaw)
def _reward_tracking_ang_vel(self, data, commands):
```

#### Foot Air Time Reward

```python
def _reward_feet_air_time(self, commands, info):
```

## Observation Space Composition

GO1's observation space is 48-dimensional, containing the following information:

```python
obs = np.hstack([
    noisy_linvel,        # 3D: Local coordinate system linear velocity
    noisy_gyro,          # 3D: Gyroscope data
    local_gravity,       # 3D: Local gravity direction
    noisy_joint_angle,   # 12D: Joint angles (relative to default values)
    noisy_joint_vel,     # 12D: Joint velocities
    last_actions,        # 12D: Previous frame actions
    command,             # 3D: Velocity commands [vx, vy, vyaw]
])
```

## Motion Velocity Command Generation

Random velocity commands are generated during training to ensure the agent can track different movement speeds:

```python
def resample_commands(self, num_envs: int):
```

## Expected Training Results

1. Stable quadruped gait
2. Good speed tracking
