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

"""Independent evaluation script for K1 penalty shootout RL policies.

Runs a trained policy in the environment without training, collects per-episode
task metrics, and prints a summary table. Supports JSON and CSV output for
experiment comparison tables.

Usage:
    uv run scripts/eval_penalty.py \\
      --env k1-penalty-shootout \\
      --policy runs/k1-penalty-shootout/skrl/.../checkpoints/best_agent.pt \\
      --num-envs 256 \\
      --num-episodes 2048 \\
      --deterministic \\
      --json-out results.json \\
      --csv-out results.csv
"""

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
from absl import app, flags

from motrix_rl import utils

logger = logging.getLogger(__name__)

_ENV = flags.DEFINE_string("env", "k1-penalty-shootout", "Environment name.")
_POLICY = flags.DEFINE_string("policy", None, "Path to policy checkpoint.")
_NUM_ENVS = flags.DEFINE_integer("num_envs", 256, "Number of parallel environments.")
_NUM_EPISODES = flags.DEFINE_integer("num_episodes", 2048, "Total episodes to evaluate.")
_DETERMINISTIC = flags.DEFINE_bool("deterministic", False, "Use deterministic (mean) actions.")
_SEED = flags.DEFINE_integer("seed", 0, "Random seed.")
_RLLIB = flags.DEFINE_string("rllib", None, "RL framework (skrl/rslrl). Auto-detected if not set.")
_JSON_OUT = flags.DEFINE_string("json_out", None, "Save results as JSON to this path.")
_CSV_OUT = flags.DEFINE_string("csv_out", None, "Save results as CSV to this path.")
_TAG = flags.DEFINE_string("tag", None, "Experiment tag for CSV. Defaults to policy filename stem.")


def _detect_rllib(policy_path: str) -> str:
    """Detect RL framework from policy file extension."""
    suffix = Path(policy_path).suffix
    if suffix == ".pickle":
        return "skrl"  # JAX SKRL uses pickle
    elif suffix == ".pt":
        # Could be either SKRL torch or RSLRL; default to skrl
        return "skrl"
    else:
        raise ValueError(f"Cannot detect RL framework from policy extension: {suffix}. "
                         f"Please specify --rllib explicitly.")


def _make_trainer_and_agent(env_name: str, policy_path: str, num_envs: int, seed: int,
                            rllib: str, deterministic: bool):
    """Create environment, load policy, and return (env, agent, act_fn)."""
    from motrix_envs import registry as env_registry

    device_supports = utils.get_device_supports()
    logger.info("Device supports: %s", device_supports)

    rl_override = {"play_num_envs": num_envs, "runner.seed": seed}

    if rllib == "rslrl":
        assert device_supports.torch, "PyTorch is not available."
        from motrix_rl.rslrl.torch.train import ppo as rslrl_ppo

        trainer = rslrl_ppo.Trainer(env_name, cfg_override=rl_override, enable_render=False)
        env = env_registry.make(env_name, num_envs=num_envs)
        import torch
        if trainer._rlcfg.runner.seed is not None:
            torch.manual_seed(trainer._rlcfg.runner.seed)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        from motrix_rl.rslrl.torch.wrap_vec_env import RslrlNpEnvWrap
        vec_env = RslrlNpEnvWrap(env, device)
        rslrl_cfg = trainer._create_rslrl_config()
        from rsl_rl.runners import OnPolicyRunner
        runner = OnPolicyRunner(vec_env, rslrl_cfg, log_dir=None, device=device)
        runner.load(policy_path)

        policy = runner.get_inference_policy(device=device)

        if not deterministic:
            logger.warning(
                "RSLRL get_inference_policy returns a deterministic model; "
                "stochastic evaluation is not supported for RSLRL. "
                "Use --rllib skrl for stochastic evaluation."
            )

        def _act(obs):
            return policy(obs)

        # Wrap vec_env to normalize step return to 5-tuple (obs, reward, term, trunc, info)
        class _RslrlStepWrapper:
            def __init__(self, venv, act_fn):
                self._venv = venv
                self.act = act_fn

            def reset(self):
                obs, extras = self._venv.reset()
                return obs, extras

            def step(self, actions):
                obs, rewards, dones, extras = self._venv.step(actions)
                # RSLRL merges terminated+truncated into "dones"; split for uniformity
                terminated = dones.to(dtype=torch.bool)
                truncated = torch.zeros_like(terminated)
                return obs, rewards, terminated, truncated, extras

        wrapped_env = _RslrlStepWrapper(vec_env, _act)
        return wrapped_env, _act, None

    else:
        # SKRL (torch or jax)
        policy_ext = Path(policy_path).suffix
        if policy_ext == ".pickle":
            backend = "jax"
            assert device_supports.jax, "JAX is not available."
            from skrl import config as skrl_config
            skrl_config.jax.backend = "jax"
            from motrix_rl.skrl.jax.train import ppo as skrl_ppo
        else:
            backend = "torch"
            assert device_supports.torch, "PyTorch is not available."
            from skrl import config as skrl_config
            skrl_config.torch.backend = "torch"
            from motrix_rl.skrl.torch.train import ppo as skrl_ppo

        trainer = skrl_ppo.Trainer(env_name, cfg_override=rl_override, enable_render=False)
        env = env_registry.make(env_name, num_envs=num_envs)
        from skrl.utils import set_seed
        set_seed(trainer._rlcfg.runner.seed)
        from motrix_rl.skrl.torch import wrap_env
        skrl_env = wrap_env(env, enable_render=False)
        models = trainer._make_model(skrl_env, trainer._rlcfg)
        ppo_cfg = trainer._rlcfg.runner.agent.to_dict()
        from motrix_rl.skrl.torch.train.ppo import _add_runtime_config
        _add_runtime_config(ppo_cfg, skrl_env)
        agent = trainer._make_agent(models, skrl_env, ppo_cfg, trainer._rlcfg.runner.memory)
        agent.load(policy_path)

        import torch
        def _act(obs):
            with torch.no_grad():
                outputs = agent.act(obs, timestep=0, timesteps=0)
                if deterministic:
                    return outputs[-1].get("mean_actions", outputs[0])
                else:
                    return outputs[0]

        return skrl_env, _act, agent.device


def _run_evaluation(env, act_fn, num_episodes: int, device) -> dict:
    """Run rollout and collect episode metrics.

    Args:
        env: Wrapped environment.
        act_fn: Function that takes obs tensor and returns actions tensor.
        num_episodes: Target number of completed episodes.
        device: torch device (or None for JAX).

    Returns:
        Dict of aggregated metrics.
    """
    all_episodes: list[dict] = []

    obs, _ = env.reset()
    episodes_completed = 0

    while episodes_completed < num_episodes:
        actions = act_fn(obs)
        obs, rewards, terminated, truncated, info = env.step(actions)

        # Collect episode summaries from _episode_log
        episode_log = info.pop("_episode_log", None) if isinstance(info, dict) else None
        if episode_log:
            all_episodes.extend(episode_log)
            episodes_completed += len(episode_log)

        if episodes_completed % 256 == 0 and episodes_completed > 0:
            logger.info("Completed %d / %d episodes", episodes_completed, num_episodes)

    # Trim to exact count
    all_episodes = all_episodes[:num_episodes]

    return _aggregate_metrics(all_episodes)


def _aggregate_metrics(episodes: list[dict]) -> dict:
    """Aggregate per-episode metrics into summary statistics."""
    n = len(episodes)
    if n == 0:
        return {"num_episodes": 0}

    bool_keys = ["goal", "contact_ball", "shot", "fall", "out_of_bounds"]
    float_keys = [
        "max_ball_speed", "speed_to_goal_at_end",
        "time_to_contact", "time_to_goal",
        "target_error_y", "target_error_z", "mean_action_rate",
        "mean_torque_square", "episode_length",
    ]

    result: dict = {"num_episodes": n}

    for k in bool_keys:
        values = [1.0 if ep.get(k, False) else 0.0 for ep in episodes]
        result[k] = sum(values) / n

    for k in float_keys:
        values = [ep[k] for ep in episodes if k in ep]
        if values:
            result[f"mean_{k}"] = sum(values) / len(values)

    # Derived metrics
    result["success_rate"] = result.get("goal", 0.0)
    result["contact_rate"] = result.get("contact_ball", 0.0)
    result["shot_rate"] = result.get("shot", 0.0)
    result["fall_rate"] = result.get("fall", 0.0)
    result["out_rate"] = result.get("out_of_bounds", 0.0)

    # Mean target error (combined)
    ty_vals = [ep.get("target_error_y", 0.0) for ep in episodes]
    tz_vals = [ep.get("target_error_z", 0.0) for ep in episodes]
    combined = [np.sqrt(y**2 + z**2) for y, z in zip(ty_vals, tz_vals)]
    result["mean_target_error"] = sum(combined) / n
    result["mean_target_error_y"] = sum(ty_vals) / n
    result["mean_target_error_z"] = sum(tz_vals) / n

    # Valid-time counts
    ttc_vals = [ep.get("time_to_contact", -1.0) for ep in episodes]
    valid_ttc = [v for v in ttc_vals if v > 0]
    result["mean_time_to_contact"] = sum(valid_ttc) / len(valid_ttc) if valid_ttc else float("nan")
    result["num_time_to_contact"] = len(valid_ttc)

    ttg_vals = [ep.get("time_to_goal", -1.0) for ep in episodes]
    valid_ttg = [v for v in ttg_vals if v > 0]
    result["mean_time_to_goal"] = sum(valid_ttg) / len(valid_ttg) if valid_ttg else float("nan")
    result["num_time_to_goal"] = len(valid_ttg)

    # Convert time_to steps into seconds (assuming 50 Hz ctrl_dt = 0.02s per step)
    STEP_TIME = 0.02
    result["mean_time_to_contact_s"] = result["mean_time_to_contact"] * STEP_TIME
    result["mean_time_to_goal_s"] = result["mean_time_to_goal"] * STEP_TIME

    return result


def _print_results(results: dict) -> None:
    """Print evaluation results to stdout."""
    print()
    print("=" * 52)
    print("  K1 Penalty Shootout — Evaluation Results")
    print("=" * 52)
    print(f"  num_episodes:          {results.get('num_episodes', 0)}")
    print(f"  success_rate:          {results.get('success_rate', 0):.4f}")
    print(f"  contact_rate:          {results.get('contact_rate', 0):.4f}")
    print(f"  shot_rate:             {results.get('shot_rate', 0):.4f}")
    print(f"  fall_rate:             {results.get('fall_rate', 0):.4f}")
    print(f"  out_rate:              {results.get('out_rate', 0):.4f}")
    print(f"  mean_ball_speed:       {results.get('mean_max_ball_speed', 0):.2f} m/s")
    print(f"  mean_speed_to_goal:    {results.get('mean_speed_to_goal_at_end', 0):.2f} m/s")
    print(f"  mean_target_error:     {results.get('mean_target_error', 0):.3f} m")
    print(f"  mean_target_error_y:   {results.get('mean_target_error_y', 0):.3f} m")
    print(f"  mean_target_error_z:   {results.get('mean_target_error_z', 0):.3f} m")
    ttc = results.get('mean_time_to_contact_s', float('nan'))
    ttg = results.get('mean_time_to_goal_s', float('nan'))
    print(f"  mean_time_to_contact:  {ttc:.2f} s  (n={results.get('num_time_to_contact', 0)})")
    print(f"  mean_time_to_goal:     {ttg:.2f} s  (n={results.get('num_time_to_goal', 0)})")
    print(f"  mean_action_rate:      {results.get('mean_mean_action_rate', 0):.4f}")
    print(f"  mean_torque_square:    {results.get('mean_mean_torque_square', 0):.4f}")
    print("=" * 52)
    print()


def _save_json(results: dict, path: str) -> None:
    """Save results as JSON."""
    out = {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in results.items()}
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    logger.info("Results saved to %s", path)


def _save_csv(results: dict, path: str, tag: str) -> None:
    """Save results as CSV (append if file exists, write header otherwise)."""
    columns = [
        "experiment", "success_rate", "contact_rate", "shot_rate",
        "fall_rate", "out_rate", "mean_ball_speed", "mean_speed_to_goal",
        "mean_target_error",
        "mean_target_error_y", "mean_target_error_z",
        "mean_time_to_contact_s", "mean_time_to_goal_s",
        "mean_action_rate", "mean_torque_square", "num_episodes",
    ]
    row = {
        "experiment": tag,
        "success_rate": results.get("success_rate", 0),
        "contact_rate": results.get("contact_rate", 0),
        "shot_rate": results.get("shot_rate", 0),
        "fall_rate": results.get("fall_rate", 0),
        "out_rate": results.get("out_rate", 0),
        "mean_ball_speed": results.get("mean_max_ball_speed", 0),
        "mean_speed_to_goal": results.get("mean_speed_to_goal_at_end", 0),
        "mean_target_error": results.get("mean_target_error", 0),
        "mean_target_error_y": results.get("mean_target_error_y", 0),
        "mean_target_error_z": results.get("mean_target_error_z", 0),
        "mean_time_to_contact_s": results.get("mean_time_to_contact_s", float("nan")),
        "mean_time_to_goal_s": results.get("mean_time_to_goal_s", float("nan")),
        "mean_action_rate": results.get("mean_mean_action_rate", 0),
        "mean_torque_square": results.get("mean_mean_torque_square", 0),
        "num_episodes": results.get("num_episodes", 0),
    }

    file_exists = Path(path).exists()
    with open(path, "a" if file_exists else "w") as f:
        if not file_exists:
            f.write(",".join(columns) + "\n")
        f.write(",".join(str(row.get(c, "")) for c in columns) + "\n")
    logger.info("CSV row appended to %s", path)


def main(argv):
    env_name = _ENV.value
    policy_path = _POLICY.value
    num_envs = _NUM_ENVS.value
    num_episodes = _NUM_EPISODES.value
    deterministic = _DETERMINISTIC.value
    seed = _SEED.value
    rllib = _RLLIB.value
    json_out = _JSON_OUT.value
    csv_out = _CSV_OUT.value
    tag = _TAG.value

    # Validate policy path
    if not policy_path:
        logger.error("Error: --policy is required. Please specify a checkpoint path.")
        logger.error("Example: --policy runs/k1-penalty-shootout/skrl/.../checkpoints/best_agent.pt")
        sys.exit(1)

    if not Path(policy_path).exists():
        logger.error("Error: Policy file not found: %s", policy_path)
        logger.error("Please check the path or train a model first.")
        sys.exit(1)

    # Auto-detect RL framework
    if not rllib:
        rllib = _detect_rllib(policy_path)
        logger.info("Auto-detected rllib: %s", rllib)

    if not tag:
        tag = Path(policy_path).stem

    logger.info("Environment: %s", env_name)
    logger.info("Policy: %s", policy_path)
    logger.info("RL framework: %s", rllib)
    logger.info("Num envs: %d", num_envs)
    logger.info("Num episodes: %d", num_episodes)
    logger.info("Deterministic: %s", deterministic)
    logger.info("Seed: %d", seed)

    # Setup
    env, act_fn, device = _make_trainer_and_agent(
        env_name, policy_path, num_envs, seed, rllib, deterministic)

    # Force full metrics collection for evaluation
    if hasattr(env, '_env'):
        env._env.cfg.metrics_mode = "full"

    # Run evaluation
    t_start = time.time()
    results = _run_evaluation(env, act_fn, num_episodes, device)
    elapsed = time.time() - t_start
    logger.info("Evaluation completed in %.1f seconds (%.1f episodes/s)",
                elapsed, num_episodes / elapsed)

    # Print results
    _print_results(results)

    # Save outputs
    if json_out:
        _save_json(results, json_out)
    if csv_out:
        _save_csv(results, csv_out, tag)


if __name__ == "__main__":
    app.run(main)
