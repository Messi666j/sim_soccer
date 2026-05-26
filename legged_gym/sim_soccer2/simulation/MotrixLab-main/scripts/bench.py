# Copyright (C) 2020-2025 Motphys Technology Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


"""Benchmark environment step performance."""

import time

import numpy as np
from absl import app, flags

from motrix_envs import registry

FLAGS = flags.FLAGS

flags.DEFINE_string("env", "cartpole", "Environment name to benchmark")
flags.DEFINE_integer("num_steps", 1000, "Number of steps to benchmark")
flags.DEFINE_boolean("random_actions", True, "Use random actions (True) or zero actions (False)")
flags.DEFINE_integer("num_envs", 1, "Number of parallel environments")
flags.DEFINE_string("sim_backend", None, "Simulation backend (auto-select if None)")


def generate_action(env, random_actions: bool) -> np.ndarray:
    """Generate action for the environment.

    Args:
        env: The environment instance.
        random_actions: If True, sample random actions from action space.
                       If False, use zero actions.

    Returns:
        Action array with shape (num_envs, *action_space.shape).
    """
    action_space = env.action_space

    if random_actions:
        # Sample random actions within bounds
        low = action_space.low
        high = action_space.high

        # Handle infinite bounds
        low = np.where(np.isneginf(low), -1e6, low)
        high = np.where(np.isposinf(high), 1e6, high)

        size = (env.num_envs, *action_space.shape)
        return np.random.uniform(low=low, high=high, size=size).astype(action_space.dtype)
    else:
        # Use zero actions
        return np.zeros((env.num_envs, *action_space.shape), dtype=action_space.dtype)


def main(argv):
    """Main benchmark function."""
    del argv  # Unused

    env_name = FLAGS.env
    num_steps = FLAGS.num_steps
    random_actions = FLAGS.random_actions
    num_envs = FLAGS.num_envs
    sim_backend = FLAGS.sim_backend

    # Create environment
    print(f"Creating environment: {env_name}")
    env = registry.make(env_name, sim_backend=sim_backend, num_envs=num_envs)

    # Print environment info
    print("\nBenchmark configuration:")
    print(f"  Environment: {env_name}")
    print(f"  Number of parallel environments: {num_envs}")
    print(f"  Action space shape: {env.action_space.shape}")
    print(f"  Random actions: {random_actions}")
    print(f"  Number of Batch steps: {num_steps}")
    print("\nRunning benchmark...\n")

    # Generate action
    action = generate_action(env, random_actions)

    # Warmup run (to reduce cold start effects)
    # Note: step() automatically initializes state if needed
    for _ in range(10):
        env.step(action)

    # Benchmark loop
    start_time = time.perf_counter()
    for _ in range(num_steps):
        env.step(action)
    end_time = time.perf_counter()

    # Calculate metrics
    total_time = end_time - start_time
    steps_per_second = num_steps / total_time
    time_per_step_ms = (total_time / num_steps) * 1000

    # Print results
    print("Results:")
    print(f"  Total time: {total_time:.4f} seconds")
    print(f"  Batch Steps per second: {steps_per_second:.2f}")
    print(f"  Total Steps per seconds: {steps_per_second * num_envs:.2f}")
    print(f"  Time per batch step: {time_per_step_ms:.4f} ms")


if __name__ == "__main__":
    app.run(main)
