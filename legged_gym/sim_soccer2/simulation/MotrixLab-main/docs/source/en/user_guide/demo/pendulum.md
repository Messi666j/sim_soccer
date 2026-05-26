# Pendulum

Pendulum is a single-joint swing-up and balance task. The goal is to swing the pole up and keep it inverted using one motor torque.

```{video} /_static/videos/pendulum.mp4
:poster: _static/images/poster/pendulum.jpg
:nocontrols:
:autoplay:
:playsinline:
:muted:
:loop:
:width: 100%
```

---

## Task Description

A single-link pendulum with one hinge joint is driven by a single motor (configurable gear). The motor’s torque rotates the rod in a plane, enabling swing-up from arbitrary initial angles, inverted balance, and maintenance. Torque is limited by the actuator ctrlrange; by modulating its magnitude and direction, the policy must accumulate energy to swing up and stabilize near the inverted position while damping angular-velocity-induced oscillations.

## Action Space

| Item          | Details                         |
| ------------- | ------------------------------- |
| **Type**      | `Box(-1.0, 1.0, (1,), float32)` |
| **Dimension** | 1                               |

---

## Observation Space

| Item          | Details                         |
| ------------- | ------------------------------- |
| **Type**      | `Box(-inf, inf, (3,), float32)` |
| **Dimension** | 3                               |

Order: `cos(theta), sin(theta), angular velocity`.

---

## Reward Function Design

-   Upright reward: encourages angle near π (inverted)
-   Energy shaping: target energy near inverted position
-   Penalties: `ang_vel^2`, `ctrl^2`, `(ctrl - prev_ctrl)^2` to reduce oscillation and aggressive actuation

---

## Initial State

-   Angle randomized in `[-pi, pi]`
-   Angular velocity small random noise (if configured)
-   Control history (`prev_ctrl`) reset to zero

## Episode Termination Conditions

-   No fall/angle termination; only NaN check
-   Episode length limited by `max_episode_seconds`

---

### 1. Environment Preview

```bash
uv run scripts/view.py --env pendulum
```

### 2. Start Training

```bash
# Train with default parameters
uv run scripts/train.py --env pendulum

# Customize parallel environments
uv run scripts/train.py --env pendulum --num-envs 1024

# Enable rendering during training
uv run scripts/train.py --env pendulum --render
```

### 3. View Training Progress

```bash
uv run tensorboard --logdir runs/pendulum
```

### 4. Test Training Results

```bash
# Auto-discover best policy (recommended)
uv run scripts/play.py --env pendulum

# Manually specify a policy file
uv run scripts/play.py --env pendulum --policy runs/pendulum/nn/best_policy.pickle
```

> **Tip**: Policies are auto-selected from `runs/pendulum/`. You can override with `--policy`.

---

## Configuration Parameters

### Environment Configuration

```python
@dataclass
class PendulumEnvCfg(EnvCfg):
    model_file: str = ".../pendulum.xml"  # MJCF model (gear=5)
    max_episode_seconds: float = 20.0
    sim_dt: float = 0.0125
    ctrl_dt: float = 0.025
```

### Training Configuration (PPO example)

```python
@rlcfg("pendulum")
@dataclass
class PendulumPPO(PPOCfg):
    seed: int = 42
    max_env_steps: int = 10_000_000
    num_envs: int = 1024
    learning_rate: float = 3e-4
    rollouts: int = 32
    learning_epochs: int = 5
    mini_batches: int = 4
    policy_hidden_layer_sizes: tuple[int, ...] = (64, 64)
    value_hidden_layer_sizes: tuple[int, ...] = (64, 64)
```

---

## Expected Training Results

1. Pendulum can swing up and stay near inverted
2. Oscillation around upright is reduced by angular-velocity and control-change penalties
