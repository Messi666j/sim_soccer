# Quick Start: Hello MotrixLab

This tutorial demonstrates the MotrixLab workflow through a simple example - loading and training a cartpole environment:

## Environment Preview

We provide a simple script to visualize an environment without executing any training. This helps you verify that system dependencies are correctly configured:

```bash
uv run scripts/view.py --env cartpole
```

This will open a visualization window showing the cartpole physics simulation environment with random actions for demonstration.

## Train Model

Start training the cartpole balancing task:

```bash
uv run scripts/train.py --env cartpole
```

The training process will automatically:

1. Automatically select training backend (JAX or PyTorch) based on hardware environment
2. Create training environments
3. Start PPO algorithm training

Training results will be saved in the `runs/cartpole/` directory, including:

-   Training checkpoints
-   TensorBoard log files

## Visualize Training Process

If you want to observe the model's learning process during training, you can enable visualization rendering:

```bash
uv run scripts/train.py --env cartpole --render
```

### 🎮 Interactive Rendering Control

> **Important Note**: Visualization significantly reduces training speed and is recommended mainly for debugging and demonstration purposes.

During visualized training, you can use the **spacebar** to dynamically control rendering:

-   **Enable Rendering**: Press spacebar to enable visualization and observe robot behavior
-   **Disable Rendering**: Press spacebar again to disable rendering and improve training speed
-   **Switch Anytime**: No need to restart the program; you can switch at any time during training

This interactive control allows you to observe training effects when needed and enjoy fast training when not needed. This feature also works during inference.

## View Training Results

Use TensorBoard to view training progress:

```bash
uv run tensorboard --logdir runs/cartpole
```

## Test Trained Model

After training is complete, test the trained policy:

```bash
# Automatically find best policy for testing (recommended)
uv run scripts/play.py --env cartpole

# Manually specify policy file for testing (if you need a specific version)
uv run scripts/play.py --env cartpole --policy runs/cartpole/YOUR_RESULT_NUMBER/best_agent.pickle
```

> **Tip**: The system will automatically find the latest and best policy files in the `runs/cartpole/` directory. Usually, using the auto-discovery feature is sufficient.

## That Completes Our Example

Next, you can try modifying parameters to observe physical effects under different settings, or try other environments.

## Next Steps

-   Learn about the [Basic Framework](../tutorial/basic_frame.md)
-   Study [Physics Environment Configuration](../tutorial/physics_environment.md)
-   View more [Training Examples](../demo/cartpole.md)
