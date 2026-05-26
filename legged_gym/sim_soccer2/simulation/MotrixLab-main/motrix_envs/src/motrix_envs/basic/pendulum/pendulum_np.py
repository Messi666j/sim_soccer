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

import gymnasium as gym
import motrixsim as mtx
import numpy as np

from motrix_envs import registry
from motrix_envs.np import reward as reward_utils
from motrix_envs.np.env import NpEnv, NpEnvState

from .cfg import PendulumEnvCfg


@registry.env("pendulum", "np")
class PendulumEnv(NpEnv):
    _cfg: PendulumEnvCfg

    def __init__(self, cfg: PendulumEnvCfg, num_envs: int = 1):
        super().__init__(cfg, num_envs=num_envs)
        ctrl_limits = self._model.actuator_ctrl_limits
        self._action_low = float(ctrl_limits[0, 0])
        self._action_high = float(ctrl_limits[1, 0])
        self._action_space = gym.spaces.Box(-1.0, 1.0, (1,), dtype=np.float32)
        self._observation_space = gym.spaces.Box(-np.inf, np.inf, (3,), dtype=np.float32)
        self._num_dof_pos = self._model.num_dof_pos
        self._num_dof_vel = self._model.num_dof_vel
        self._init_dof_pos = self._model.compute_init_dof_pos()
        self._init_dof_vel = np.zeros(
            (self._model.num_dof_vel,),
            dtype=np.float32,
        )

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    def apply_action(self, actions: np.ndarray, state: NpEnvState):
        actions = np.clip(actions, -1.0, 1.0)
        scaled = self._action_low + (actions + 1.0) * 0.5 * (self._action_high - self._action_low)
        state.data.actuator_ctrls = scaled
        return state

    def update_state(self, state: NpEnvState):
        # compute observation
        data = state.data
        dof_pos = data.dof_pos
        dof_vel = data.dof_vel
        angle = dof_pos[:, 0]
        ang_vel = dof_vel[:, 0]
        obs = np.stack([np.cos(angle), np.sin(angle), ang_vel], axis=-1)
        assert obs.shape == (self._num_envs, 3)

        # compute reward
        angle_wrapped = (angle + np.pi) % (2 * np.pi) - np.pi
        ctrl = data.actuator_ctrls[:, 0]
        # In this model, zero angle corresponds to the hanging-down position.
        # Shift the target by pi to encourage the upright (inverted) posture.
        upright = (1.0 + np.cos(angle_wrapped)) * 0.5
        prev_ctrl = state.info.get("prev_ctrl", np.zeros_like(ctrl))
        ctrl_delta = ctrl - prev_ctrl
        vel_penalty = 0.2 * (ang_vel**2)
        energy = 0.5 * ang_vel**2 + (1.0 - np.cos(angle_wrapped))
        energy_target = 2.0
        energy_reward = reward_utils.tolerance(
            energy,
            bounds=(energy_target, energy_target),
            margin=2.0,
            value_at_margin=0.1,
            sigmoid="gaussian",
        )
        reward = (3.0 * upright + energy_reward - vel_penalty - 0.001 * ctrl**2 - 0.001 * ctrl_delta**2).astype(
            np.float32
        )

        # compute terminated
        terminated = np.isnan(obs).any(axis=-1)

        state.obs = obs
        state.reward = reward
        state.terminated = terminated
        state.info["prev_ctrl"] = ctrl
        return state

    def reset(self, data: mtx.SceneData):
        cfg: PendulumEnvCfg = self._cfg
        reset_noise_scale = getattr(cfg, "reset_noise_scale", 0.0)
        num_reset = data.shape[0]
        dof_pos = np.zeros((num_reset, self._num_dof_pos), dtype=np.float32)
        dof_vel = np.zeros((num_reset, self._num_dof_vel), dtype=np.float32)
        dof_pos[:, 0] = np.random.uniform(-np.pi, np.pi, size=(num_reset,))
        if reset_noise_scale > 0.0:
            dof_vel[:, 0] = np.random.uniform(-reset_noise_scale, reset_noise_scale, size=(num_reset,))

        data.reset(self._model)
        data.set_dof_vel(dof_vel)
        data.set_dof_pos(dof_pos, self._model)
        angle = dof_pos[:, 0]
        ang_vel = dof_vel[:, 0]
        obs = np.stack([np.cos(angle), np.sin(angle), ang_vel], axis=-1)
        return obs, {"prev_ctrl": np.zeros((num_reset,), dtype=np.float32)}
