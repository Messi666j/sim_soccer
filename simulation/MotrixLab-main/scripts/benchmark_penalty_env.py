# Copyright (C) 2020-2025 Motphys Technology Co., Ltd. All Rights Reserved.
"""Lightweight env-only benchmark for K1 penalty shootout.

Measures raw environment step throughput without policy inference overhead.
Supports multiple num_envs values for scaling analysis.

Usage:
    uv run scripts/benchmark_penalty_env.py --num-envs 128 256 512 1024 2048 --num-steps 1000
"""

import time
import numpy as np
from absl import app, flags

from motrix_envs import registry as env_registry

_ENV = flags.DEFINE_string("env", "k1-penalty-shootout", "Environment name.")
_NUM_ENVS = flags.DEFINE_multi_integer("num-envs", [128, 256, 512, 1024, 2048],
                                       "List of num_envs to benchmark.")
_NUM_STEPS = flags.DEFINE_integer("num-steps", 1000, "Number of env steps per benchmark.")
_WARMUP = flags.DEFINE_integer("warmup", 20, "Warmup steps before timing.")
_METRICS_MODE = flags.DEFINE_string("metrics-mode", "minimal", "Metrics mode (minimal/full).")


def benchmark(env_name: str, num_envs: int, num_steps: int, warmup: int, metrics_mode: str) -> dict:
    """Run benchmark for a single num_envs configuration.

    Returns dict with timing statistics.
    """
    env = env_registry.make(env_name, num_envs=num_envs)
    env.cfg.metrics_mode = metrics_mode
    state = env.init_state()

    # Warmup
    for _ in range(warmup):
        actions = np.random.uniform(-1, 1, (num_envs, 12)).astype(np.float32)
        state = env.step(actions)

    # Timed steps
    step_times = []
    t_start = time.perf_counter()
    for _ in range(num_steps):
        actions = np.random.uniform(-1, 1, (num_envs, 12)).astype(np.float32)
        t0 = time.perf_counter()
        state = env.step(actions)
        step_times.append(time.perf_counter() - t0)
    t_total = time.perf_counter() - t_start

    # Statistics
    step_times = np.array(step_times)
    env_steps_per_sec = (num_envs * num_steps) / t_total
    step_time_mean = np.mean(step_times) * 1000  # ms
    step_time_p95 = np.percentile(step_times, 95) * 1000  # ms
    step_time_p99 = np.percentile(step_times, 99) * 1000  # ms
    step_per_sec = num_steps / t_total

    return {
        "num_envs": num_envs,
        "wall_time_s": t_total,
        "step_per_sec": step_per_sec,
        "env_steps_per_sec": env_steps_per_sec,
        "step_time_mean_ms": step_time_mean,
        "step_time_p95_ms": step_time_p95,
        "step_time_p99_ms": step_time_p99,
    }


def main(argv):
    env_name = _ENV.value
    num_envs_list = _NUM_ENVS.value
    num_steps = _NUM_STEPS.value
    warmup = _WARMUP.value
    metrics_mode = _METRICS_MODE.value

    print(f"\nBenchmark: {env_name} ({metrics_mode} metrics)")
    print(f"Steps per run: {num_steps}, warmup: {warmup}")
    print()

    results = []
    for ne in num_envs_list:
        print(f"  num_envs={ne} ... ", end="", flush=True)
        r = benchmark(env_name, ne, num_steps, warmup, metrics_mode)
        results.append(r)
        print(f"{r['env_steps_per_sec']:.0f} env-steps/s "
              f"({r['step_time_mean_ms']:.2f} ms/step mean, "
              f"{r['step_time_p95_ms']:.2f} ms p95)")

    # Summary table
    print(f"\n{'num_envs':>10}  {'step/s':>8}  {'env-step/s':>12}  {'mean_ms':>8}  {'p95_ms':>8}  {'p99_ms':>8}")
    print("-" * 70)
    for r in results:
        print(f"{r['num_envs']:>10}  {r['step_per_sec']:>8.1f}  {r['env_steps_per_sec']:>12.1f}  "
              f"{r['step_time_mean_ms']:>8.2f}  {r['step_time_p95_ms']:>8.2f}  {r['step_time_p99_ms']:>8.2f}")

    # Scaling efficiency
    print(f"\n{'num_envs':>10}  {'efficiency':>12}")
    print("-" * 30)
    base = results[0]["env_steps_per_sec"] / results[0]["num_envs"]
    for r in results:
        per_env = r["env_steps_per_sec"] / r["num_envs"]
        eff = per_env / base * 100
        print(f"{r['num_envs']:>10}  {eff:>11.1f}%")
    print(f"\nBaseline: {base:.1f} env-steps/s per env at num_envs={results[0]['num_envs']}")
    print()


if __name__ == "__main__":
    app.run(main)
