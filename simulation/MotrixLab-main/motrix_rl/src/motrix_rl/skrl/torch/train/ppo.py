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

from typing import Any

import numpy as np
import torch
import torch.nn as nn
from skrl.agents.torch.ppo import PPO as BasePPO
from skrl.envs.torch import Wrapper
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveRL
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed

from motrix_envs import registry as env_registry
from motrix_rl import registry, utils
from motrix_rl.skrl import get_log_dir
from motrix_rl.skrl.config import SkrlCfg, SkrlMemoryCfg
from motrix_rl.skrl.torch import wrap_env


def _instantiate_memory(memory_cfg: SkrlMemoryCfg, memory_size: int, num_envs: int, device) -> Any:
    """Instantiate a SKRL Memory class based on configuration.

    Args:
        memory_cfg: Memory configuration with class_name and settings
        memory_size: Size of the memory buffer
        num_envs: Number of parallel environments
        device: Device to place memory on

    Returns:
        Instantiated SKRL Memory object

    Raises:
        ValueError: If class_name is not supported
    """
    from skrl.memories.torch import RandomMemory

    # Map class_name to actual Memory class
    memory_classes = {
        "RandomMemory": RandomMemory,
    }

    class_name = memory_cfg.class_name
    if class_name not in memory_classes:
        raise ValueError(f"Unsupported memory class_name: {class_name}. Supported: {list(memory_classes.keys())}")

    MemoryClass = memory_classes[class_name]
    return MemoryClass(memory_size=memory_size, num_envs=num_envs, device=device)


def _add_runtime_config(
    cfg: dict,
    env: Wrapper,
    log_dir: str = None,
) -> dict:
    """Add runtime-specific configuration to the base agent config.

    Args:
        cfg: Base configuration from agent.to_dict() (will be modified in-place)
        env: SKRL environment wrapper
        log_dir: Optional logging directory path

    Returns:
        The same cfg dict with runtime values added (modified in-place for convenience)
    """
    # Convert learning_rate_scheduler from string to actual class (if configured)
    if cfg.get("learning_rate_scheduler") == "KLAdaptiveLR":
        cfg["learning_rate_scheduler"] = KLAdaptiveRL
    elif cfg.get("learning_rate_scheduler") == "":
        cfg["learning_rate_scheduler"] = None  # Disabled

    # Add rewards shaper (conditional based on rewards_shaper_scale in cfg)
    if cfg.get("rewards_shaper_scale", 1.0) != 1.0:
        cfg["rewards_shaper"] = lambda reward, timestep, timesteps: reward * cfg["rewards_shaper_scale"]
    else:
        cfg["rewards_shaper"] = None

    # Add preprocessors (require runtime env values)
    cfg["state_preprocessor"] = RunningStandardScaler
    cfg["state_preprocessor_kwargs"] = {
        "size": env.observation_space,
        "device": env.device,
    }
    cfg["value_preprocessor"] = RunningStandardScaler
    cfg["value_preprocessor_kwargs"] = {"size": 1, "device": env.device}

    # Add experiment configuration (handle -1 -> "auto" conversion)
    if log_dir:
        cfg["experiment"]["write_interval"] = (
            "auto" if cfg["experiment"]["write_interval"] == -1 else cfg["experiment"]["write_interval"]
        )
        cfg["experiment"]["checkpoint_interval"] = (
            "auto" if cfg["experiment"]["checkpoint_interval"] == -1 else cfg["experiment"]["checkpoint_interval"]
        )
        cfg["experiment"]["directory"] = log_dir
    else:
        cfg["experiment"]["write_interval"] = 0
        cfg["experiment"]["checkpoint_interval"] = 0

    return cfg


class PPO(BasePPO):
    _total_custom_rewards: dict[str, torch.Tensor] = {}
    _step_count: int = 0
    _last_log_time: float = 0.0

    @staticmethod
    def _init_throughput_timer():
        import time
        PPO._step_count = 0
        PPO._last_log_time = time.perf_counter()

    def record_transition(
        self,
        states,
        actions,
        rewards,
        next_states,
        terminated,
        truncated,
        infos,
        timestep,
        timesteps,
    ) -> None:
        super().record_transition(
            states,
            actions,
            rewards,
            next_states,
            terminated,
            truncated,
            infos,
            timestep,
            timesteps,
        )
        if "Reward" in infos:
            reward_dict = infos["Reward"]
            keys = list(reward_dict.keys())
            # Stack all reward components into one (N_keys, num_envs) tensor to
            # reduce GPU sync points: 1 CPUp->GPU copy + 3 GPU->CPU copies instead
            # of 6*tensor() + 18*item() per step.
            stacked = torch.from_numpy(
                np.stack([reward_dict[k] for k in keys], axis=0)
            ).to(device=self.device)
            # Per-key instant tracking (vectorized max/min/mean)
            max_vals = torch.max(stacked, dim=1).values.cpu().numpy()
            min_vals = torch.min(stacked, dim=1).values.cpu().numpy()
            mean_vals = torch.mean(stacked, dim=1).cpu().numpy()
            for i, key in enumerate(keys):
                self.tracking_data[f"Reward Instant / {key} (max)"].append(float(max_vals[i]))
                self.tracking_data[f"Reward Instant / {key} (min)"].append(float(min_vals[i]))
                self.tracking_data[f"Reward Instant / {key} (mean)"].append(float(mean_vals[i]))
                if key not in self._total_custom_rewards:
                    self._total_custom_rewards[key] = torch.zeros_like(stacked[i])
                self._total_custom_rewards[key] += stacked[i]
            done = terminated | truncated
            done = done.reshape(-1)
            if done.any():
                for key in self._total_custom_rewards:
                    self.tracking_data[f"Reward Total/ {key} (mean)"].append(
                        torch.mean(self._total_custom_rewards[key][done]).item()
                    )
                    self.tracking_data[f"Reward Total/ {key} (min)"].append(
                        torch.min(self._total_custom_rewards[key][done]).item()
                    )
                    self.tracking_data[f"Reward Total/ {key} (max)"].append(
                        torch.max(self._total_custom_rewards[key][done]).item()
                    )

                    self._total_custom_rewards[key] = self._total_custom_rewards[key] * (~done)

        if "metrics" in infos:
            for key, value in infos["metrics"].items():
                tracked_value = torch.tensor(value, device=self.device)
                self.tracking_data[f"metrics / {key} (max)"].append(torch.max(tracked_value).item())
                self.tracking_data[f"metrics / {key} (min)"].append(torch.min(tracked_value).item())
                self.tracking_data[f"metrics / {key} (mean)"].append(torch.mean(tracked_value).item())

        # Episode-level task metrics from _episode_log
        episode_log = infos.pop("_episode_log", None)
        if episode_log:
            self._log_episode_metrics(episode_log)

        # Throughput logging every ~100 steps
        import time
        PPO._step_count += 1
        if PPO._step_count % 100 == 0:
            now = time.perf_counter()
            if PPO._last_log_time > 0:
                elapsed = now - PPO._last_log_time
                steps_per_sec = 100.0 / elapsed
                import logging
                _logger = logging.getLogger(__name__)
                _logger.info("throughput: %.1f env_steps/s (%.2f ms/step)",
                             steps_per_sec, elapsed / 100.0 * 1000.0)
            PPO._last_log_time = now

    def _log_episode_metrics(self, episode_log) -> None:
        """Aggregate and log episode-level task metrics to tracking_data.

        Supports two formats:
        - dict of numpy arrays (minimal mode): {"goal": np.array([1.,0.,...]), ...}
        - list of dicts (full mode): [{"goal": True, ...}, ...]
        """
        import numpy as np

        if isinstance(episode_log, dict):
            # Minimal mode: dict of numpy arrays, shape (num_done,)
            num_eps = len(next(iter(episode_log.values())))
            if num_eps == 0:
                return

            def _mean(key):
                arr = episode_log.get(key)
                return float(np.mean(arr)) if arr is not None and len(arr) > 0 else 0.0

            def _mean_valid(key):
                arr = episode_log.get(key)
                if arr is None or len(arr) == 0:
                    return 0.0
                valid = arr[arr > 0]
                return float(np.mean(valid)) if len(valid) > 0 else 0.0

            self.tracking_data["eval/success_rate"].append(_mean("goal"))
            self.tracking_data["eval/contact_rate"].append(_mean("contact_ball"))
            self.tracking_data["eval/shot_rate"].append(_mean("shot"))
            self.tracking_data["eval/fall_rate"].append(_mean("fall"))
            self.tracking_data["eval/out_rate"].append(_mean("out_of_bounds"))
            self.tracking_data["eval/mean_ball_speed"].append(_mean("max_ball_speed"))
            self.tracking_data["eval/mean_speed_to_goal"].append(_mean("speed_to_goal_at_end"))
            self.tracking_data["eval/mean_target_error_y"].append(_mean("target_error_y"))
            self.tracking_data["eval/mean_target_error_z"].append(_mean("target_error_z"))
            ty = episode_log.get("target_error_y")
            tz = episode_log.get("target_error_z")
            if ty is not None and tz is not None and len(ty) > 0:
                combined = np.sqrt(ty**2 + tz**2)
                self.tracking_data["eval/mean_target_error"].append(float(np.mean(combined)))
            self.tracking_data["eval/mean_time_to_contact"].append(_mean_valid("time_to_contact"))
            self.tracking_data["eval/mean_time_to_goal"].append(_mean_valid("time_to_goal"))
            self.tracking_data["eval/mean_action_rate"].append(_mean("mean_action_rate"))
            self.tracking_data["eval/mean_torque_square"].append(_mean("mean_torque_square"))
        else:
            # Full mode: list of dicts (original format for eval_penalty.py)
            metrics_accum: dict[str, list[float]] = {}
            bool_keys = ["goal", "contact_ball", "shot", "fall", "out_of_bounds"]
            float_keys = [
                "max_ball_speed", "speed_to_goal_at_end",
                "time_to_contact", "time_to_goal",
                "target_error_y", "target_error_z", "mean_action_rate",
                "mean_torque_square",
            ]

            for ep in episode_log:
                for k in bool_keys:
                    metrics_accum.setdefault(k, []).append(1.0 if ep.get(k, False) else 0.0)
                for k in float_keys:
                    if k in ep:
                        metrics_accum.setdefault(k, []).append(ep[k])

            if not metrics_accum:
                return

            num_eps = len(episode_log)
            self.tracking_data["eval/success_rate"].append(
                sum(metrics_accum.get("goal", [])) / max(num_eps, 1))
            self.tracking_data["eval/contact_rate"].append(
                sum(metrics_accum.get("contact_ball", [])) / max(num_eps, 1))
            self.tracking_data["eval/shot_rate"].append(
                sum(metrics_accum.get("shot", [])) / max(num_eps, 1))
            self.tracking_data["eval/fall_rate"].append(
                sum(metrics_accum.get("fall", [])) / max(num_eps, 1))
            self.tracking_data["eval/out_rate"].append(
                sum(metrics_accum.get("out_of_bounds", [])) / max(num_eps, 1))
            if "max_ball_speed" in metrics_accum:
                self.tracking_data["eval/mean_ball_speed"].append(
                    sum(metrics_accum["max_ball_speed"]) / len(metrics_accum["max_ball_speed"]))
            if "speed_to_goal_at_end" in metrics_accum:
                self.tracking_data["eval/mean_speed_to_goal"].append(
                    sum(metrics_accum["speed_to_goal_at_end"]) / len(metrics_accum["speed_to_goal_at_end"]))
            if "target_error_y" in metrics_accum and "target_error_z" in metrics_accum:
                errors_y = metrics_accum["target_error_y"]
                errors_z = metrics_accum["target_error_z"]
                self.tracking_data["eval/mean_target_error_y"].append(sum(errors_y) / len(errors_y))
                self.tracking_data["eval/mean_target_error_z"].append(sum(errors_z) / len(errors_z))
                combined = [np.sqrt(y**2 + z**2) for y, z in zip(errors_y, errors_z)]
                self.tracking_data["eval/mean_target_error"].append(sum(combined) / len(combined))
            if "time_to_contact" in metrics_accum:
                valid_ttc = [v for v in metrics_accum["time_to_contact"] if v > 0]
                if valid_ttc:
                    self.tracking_data["eval/mean_time_to_contact"].append(sum(valid_ttc) / len(valid_ttc))
            if "time_to_goal" in metrics_accum:
                valid_ttg = [v for v in metrics_accum["time_to_goal"] if v > 0]
                if valid_ttg:
                    self.tracking_data["eval/mean_time_to_goal"].append(sum(valid_ttg) / len(valid_ttg))
            if "mean_action_rate" in metrics_accum:
                self.tracking_data["eval/mean_action_rate"].append(
                    sum(metrics_accum["mean_action_rate"]) / len(metrics_accum["mean_action_rate"]))
            if "mean_torque_square" in metrics_accum:
                self.tracking_data["eval/mean_torque_square"].append(
                    sum(metrics_accum["mean_torque_square"]) / len(metrics_accum["mean_torque_square"]))


class Trainer:
    _trainer: SequentialTrainer
    _env_name: str
    _sim_backend: str
    _rlcfg: SkrlCfg
    _enable_render: bool
    _env_cfg_override: dict

    def __init__(
        self,
        env_name: str,
        sim_backend: str = None,
        enable_render: bool = False,
        cfg_override: dict = None,
        env_cfg_override: dict = None,
    ) -> None:
        rlcfg = registry.default_rl_cfg(env_name, "skrl", backend="torch")
        if cfg_override is not None:
            rlcfg = utils.cfg_override(rlcfg, cfg_override)
        self._rlcfg = rlcfg
        self._env_name = env_name
        self._sim_backend = sim_backend
        self._enable_render = enable_render
        self._env_cfg_override = env_cfg_override

    def train(self) -> None:
        """
        Start training the agent.
        """
        from datetime import datetime
        import os

        rlcfg = self._rlcfg
        env = env_registry.make(self._env_name, sim_backend=self._sim_backend, num_envs=rlcfg.num_envs, env_cfg_override=self._env_cfg_override)
        # Apply nested overrides (dot-notation keys like "control.action_scale")
        # that the registry's flat setattr can't handle
        if self._env_cfg_override:
            for key, value in self._env_cfg_override.items():
                if "." in key:
                    obj = env.cfg
                    parts = key.split(".")
                    for part in parts[:-1]:
                        obj = getattr(obj, part)
                    setattr(obj, parts[-1], value)
        set_seed(rlcfg.runner.seed)
        skrl_env = wrap_env(env, self._enable_render)
        models = self._make_model(skrl_env, rlcfg)
        # Get base configuration from config object
        ppo_cfg = rlcfg.runner.agent.to_dict()

        # Save full config with timestamped filenames
        base_log_dir = get_log_dir(self._env_name, rllib="skrl", agent_name="PPO")
        os.makedirs(base_log_dir, exist_ok=True)
        now = datetime.now()
        ts = now.strftime("%y-%m-%d_%H-%M-%S")
        full_cfg = utils.class_to_dict(rlcfg)
        # Save to base dir with timestamp so each run has its own config files;
        # SKRL will create its own subdirectory inside base_log_dir for checkpoints/events.
        utils.save_run_config(base_log_dir, full_cfg, extra_meta={"run_timestamp": ts})
        # Also copy to timestamped names for non-overlapping records
        import shutil
        shutil.copy2(os.path.join(base_log_dir, "config.yaml"),
                     os.path.join(base_log_dir, f"config_{ts}.yaml"))
        shutil.copy2(os.path.join(base_log_dir, "args.json"),
                     os.path.join(base_log_dir, f"args_{ts}.json"))

        # Add runtime-specific configuration; SKRL creates its own subdir under base_log_dir
        _add_runtime_config(ppo_cfg, skrl_env, log_dir=base_log_dir)
        # Init throughput timer
        PPO._init_throughput_timer()
        agent = self._make_agent(models, skrl_env, ppo_cfg, rlcfg.runner.memory)
        cfg_trainer = {
            "timesteps": rlcfg.runner.trainer.timesteps,
            "headless": not self._enable_render,
        }
        trainer = SequentialTrainer(cfg=cfg_trainer, env=skrl_env, agents=agent)
        trainer.train()

    def play(self, policy: str) -> None:
        import time

        rlcfg = self._rlcfg
        env = env_registry.make(self._env_name, sim_backend=self._sim_backend, num_envs=rlcfg.play_num_envs)
        set_seed(rlcfg.runner.seed)
        env = wrap_env(env, self._enable_render)
        models = self._make_model(env, rlcfg)
        # Get base configuration from config object
        ppo_cfg = rlcfg.runner.agent.to_dict()
        # Add runtime-specific configuration
        _add_runtime_config(ppo_cfg, env)
        agent = self._make_agent(models, env, ppo_cfg, rlcfg.runner.memory)
        agent.load(policy)
        with torch.no_grad():
            obs, _ = env.reset()
            fps = 60
            while True:
                t = time.time()
                outputs = agent.act(obs, timestep=0, timesteps=0)
                actions = outputs[-1].get("mean_actions", outputs[0])
                obs, _, _, _, _ = env.step(actions)
                env.render()
                delta_time = time.time() - t
                if delta_time < 1.0 / fps:
                    time.sleep(1.0 / fps - delta_time)

    def _make_model(self, env: Wrapper, rlcfg: SkrlCfg) -> dict[str, Model]:
        _activation_fn = {
            "elu": nn.ELU,
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "sigmoid": nn.Sigmoid,
            "leaky_relu": nn.LeakyReLU,
            "selu": nn.SELU,
        }

        policy_cfg = rlcfg.runner.models.policy
        value_cfg = rlcfg.runner.models.value
        separate = rlcfg.runner.models.separate

        def resolve_activations(activation_names: list[str], hiddens: list[int]) -> list:
            if len(activation_names) == 1:
                return [_activation_fn[activation_names[0]]] * len(hiddens)
            if len(activation_names) != len(hiddens):
                raise ValueError(
                    f"hidden_activation length ({len(activation_names)}) must be 1 or "
                    f"match hiddens length ({len(hiddens)})"
                )
            return [_activation_fn[name] for name in activation_names]

        policy_acts = resolve_activations(policy_cfg.hidden_activation, policy_cfg.hiddens)
        value_acts = resolve_activations(value_cfg.hidden_activation, value_cfg.hiddens)

        def build_mlp(input_size: int, hidden_sizes: list[int], activations: list) -> nn.Sequential:
            layers = []
            current_size = input_size
            for hidden_size, act in zip(hidden_sizes, activations):
                layers.append(nn.Linear(current_size, hidden_size))
                layers.append(act())
                current_size = hidden_size
            return nn.Sequential(*layers)

        models = {}

        if separate:

            class Policy(GaussianMixin, Model):
                def __init__(self, observation_space, action_space, device, **kwargs):
                    Model.__init__(self, observation_space, action_space, device, **kwargs)
                    GaussianMixin.__init__(
                        self,
                        policy_cfg.clip_actions,
                        policy_cfg.clip_log_std,
                        policy_cfg.min_log_std,
                        policy_cfg.max_log_std,
                        policy_cfg.reduction,
                    )
                    self.net = build_mlp(self.num_observations, policy_cfg.hiddens, policy_acts)
                    self.mean_layer = nn.Linear(policy_cfg.hiddens[-1], self.num_actions)
                    self.log_std_parameter = nn.Parameter(torch.full((self.num_actions,), policy_cfg.initial_log_std))

                def compute(self, inputs, role):
                    x = self.net(inputs["states"])
                    return self.mean_layer(x), self.log_std_parameter, {}

            class Value(DeterministicMixin, Model):
                def __init__(self, observation_space, action_space, device, **kwargs):
                    Model.__init__(self, observation_space, action_space, device, **kwargs)
                    DeterministicMixin.__init__(self, value_cfg.clip_actions)
                    self.net = build_mlp(self.num_observations, value_cfg.hiddens, value_acts)
                    self.value_layer = nn.Linear(value_cfg.hiddens[-1], 1)

                def compute(self, inputs, role):
                    x = self.net(inputs["states"])
                    return self.value_layer(x), {}

            models["policy"] = Policy(
                observation_space=env.observation_space,
                action_space=env.action_space,
                device=env.device,
            )
            models["value"] = Value(
                observation_space=env.observation_space,
                action_space=env.action_space,
                device=env.device,
            )
        else:

            class Shared(GaussianMixin, DeterministicMixin, Model):
                def __init__(self, observation_space, action_space, device, **kwargs):
                    Model.__init__(self, observation_space, action_space, device, **kwargs)
                    GaussianMixin.__init__(
                        self,
                        policy_cfg.clip_actions,
                        policy_cfg.clip_log_std,
                        policy_cfg.min_log_std,
                        policy_cfg.max_log_std,
                        policy_cfg.reduction,
                    )
                    DeterministicMixin.__init__(self, value_cfg.clip_actions)
                    self.net = build_mlp(self.num_observations, policy_cfg.hiddens, policy_acts)
                    self.mean_layer = nn.Linear(policy_cfg.hiddens[-1], self.num_actions)
                    self.log_std_parameter = nn.Parameter(torch.full((self.num_actions,), policy_cfg.initial_log_std))
                    self.value_layer = nn.Linear(policy_cfg.hiddens[-1], 1)
                    self._shared_output = None

                def act(self, inputs, role):
                    if role == "policy":
                        return GaussianMixin.act(self, inputs, role)
                    elif role == "value":
                        return DeterministicMixin.act(self, inputs, role)

                def compute(self, inputs, role):
                    if role == "policy":
                        self._shared_output = self.net(inputs["states"])
                        return self.mean_layer(self._shared_output), self.log_std_parameter, {}
                    elif role == "value":
                        shared = self._shared_output if self._shared_output is not None else self.net(inputs["states"])
                        self._shared_output = None
                        return self.value_layer(shared), {}

            models["policy"] = Shared(
                observation_space=env.observation_space,
                action_space=env.action_space,
                device=env.device,
            )
            models["value"] = models["policy"]

        return models

    def _make_agent(
        self, models: dict[str, Model], env: Wrapper, ppo_cfg: dict[str, Any], memory_cfg: SkrlMemoryCfg
    ) -> PPO:
        # Use memory_size from SkrlMemoryCfg, fall back to rollouts if -1
        memory_size = memory_cfg.memory_size
        if memory_size == -1:
            memory_size = ppo_cfg["rollouts"]

        memory = _instantiate_memory(memory_cfg, memory_size, env.num_envs, env.device)

        agent = PPO(
            models=models,
            memory=memory,
            cfg=ppo_cfg,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=env.device,
        )
        return agent
