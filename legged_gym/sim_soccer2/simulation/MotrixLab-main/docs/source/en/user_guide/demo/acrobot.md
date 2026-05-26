# Acrobot

Acrobot is a two-link swing-up and balance task. The goal is to swing both arms up and reach a target position using one motor torque.

```{video} /_static/videos/acrobot.mp4
:poster: _static/images/poster/acrobot.jpg
:nocontrols:
:autoplay:
:playsinline:
:muted:
:loop:
:width: 100%
```

---

## Task Description

A two-link acrobot with one hinge joint is driven by a single motor. The motor is installed at the elbow joint, which is the only actuated joint in the system. The motor's torque rotates the rods in a plane, enabling swing-up from arbitrary initial angles and reaching a target position. Torque is limited by the actuator ctrlrange; by modulating its magnitude and direction, the policy must accumulate energy to swing up and reach the target while maintaining stability.

## Action Space

| Item          | Details                         |
| ------------- | ------------------------------- |
| **Type**      | `Box(-1.0, 1.0, (1,), float32)` |
| **Dimension** | 1                               |

---

## Observation Space

| Item          | Details                         |
| ------------- | ------------------------------- |
| **Type**      | `Box(-inf, inf, (6,), float32)` |
| **Dimension** | 6                               |

Order: `upper_arm_horizontal, lower_arm_horizontal, upper_arm_vertical, lower_arm_vertical, shoulder_velocity, elbow_velocity`.

---

## Reward Function Design

-   Base sparse reward: encourages the tip to enter the target region (radius = 0.2)
-   Continuous reward: provides 0.1 reward per step for staying in the target region
-   Distance shaping: 0.3 \* (1.0 - clip(distance / 2.0, 0, 1.0)) to encourage movement towards target
-   Velocity penalty: 0.01 \* max(0, velocity_magnitude - 2.0) to penalize excessive velocities

---

## Initial State

-   Shoulder angle randomized in `[-pi, pi]`
-   Elbow angle randomized in `[-pi, pi]`
-   Angular velocities initialized to zero

## Episode Termination Conditions

-   Episode length limited by `max_episode_seconds`
-   NaN check for observation values

---

### 1. Environment Preview

```bash
uv run scripts/view.py --env acrobot
```

### 2. Start Training

```bash
# Train with default parameters
uv run scripts/train.py --env acrobot

# Customize parallel environments
uv run scripts/train.py --env acrobot --num-envs 1024

# Enable rendering during training
uv run scripts/train.py --env acrobot --render
```

### 3. View Training Progress

```bash
uv run tensorboard --logdir runs/acrobot
```

### 4. Test Training Results

```bash
# Auto-discover best policy (recommended)
uv run scripts/play.py --env acrobot

# Manually specify a policy file
uv run scripts/play.py --env acrobot --policy runs/acrobot/nn/best_policy.pickle
```

> **Tip**: Policies are auto-selected from `runs/acrobot/`. You can override with `--policy`.

---

## Configuration Parameters

### Environment Configuration

```python
@dataclass
class AcrobotEnvCfg(EnvCfg):
    model_file: str = ".../acrobot.xml"  # MJCF model
    max_episode_seconds: float = 10.0
    sim_dt: float = 0.01
    ctrl_dt: float = 0.02
    reset_noise_scale: float = 0.1
    render_spacing: float = 2.0
```

### Training Configuration (PPO example)

```python
@rlcfg("acrobot", backend="jax")
@dataclass
class AcrobotPPO(PPOCfg):
    max_env_steps: int = 60_000_000
    check_point_interval: int = 500

    # Override PPO configuration
    policy_hidden_layer_sizes: tuple[int, ...] = (32, 32)
    value_hidden_layer_sizes: tuple[int, ...] = (32, 32)
    rollouts: int = 64
    learning_epochs: int = 5
    mini_batches: int = 8
    learning_rate: float = 3e-4
    grad_norm_clip: float = 0.1
    clip_predicted_values: bool = False
    value_clip: float = 10.0
    entropy_loss_scale: float = 0.1
    learning_rate_scheduler_kl_threshold: float = 0.02
    discount_factor: float = 0.995
    lambda_param: float = 0.97
    ratio_clip: float = 0.2
    value_loss_scale: float = 0.5
    random_timesteps: int = 0
    learning_starts: int = 0
    kl_threshold: float = 0.03
```

---

## Expected Training Results

1. Acrobot can swing up both arms to reach the target position
2. The tip can stay within the target region with stability
3. Excessive oscillations are reduced by velocity penalty
4. The policy efficiently approaches the target with smooth movements
