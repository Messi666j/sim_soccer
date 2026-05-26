# CartPole

CartPole is a classic control task in reinforcement learning. The goal is to keep the pole balanced by controlling the cart's left-right movement.
![cartpole](/_static/images/poster/cartpole.jpg)

## Task Description

-   **State Space**: Cart position, cart velocity, pole angle, pole angular velocity
-   **Action Space**: Apply force left or right
-   **Reward Function**: +1 reward for each step the pole stays upright
-   **Termination Conditions**: Pole angle exceeds ±15 degrees or episode length exceeds 10 seconds

## Quick Start

### 1. Environment Preview

```bash
uv run scripts/view.py --env cartpole
```

### 2. Start Training

```bash
uv run scripts/train.py --env cartpole
```

### 3. View Training Progress

```bash
uv run tensorboard --logdir runs/cartpole
```

### 4. Test Training Results

```bash
uv run scripts/play.py --env cartpole
```

> **Tip**: The system will automatically find the latest and best policy files in the `runs/cartpole/` directory for testing. You can also manually specify specific policy files using the `--policy` parameter.

## Expected Results

-   Pole angle stays within ±5 degrees most of the time
-   Cart displacement range is reasonable

## Troubleshooting

If training performance is poor, you can try:

1. Adjust learning rate (try 1e-4 to 1e-3)
2. Increase number of environments (more parallel training)
3. Adjust reward function weights
4. Check if physical parameters are reasonable
