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
from motrix_envs.np import reward
from motrix_envs.np.env import NpEnv, NpEnvState

from .cfg import AcrobotEnvCfg


@registry.env("acrobot", "np")
class AcrobotEnv(NpEnv):
    _cfg: AcrobotEnvCfg

    def __init__(self, cfg: AcrobotEnvCfg, num_envs: int = 1):
        super().__init__(cfg, num_envs=num_envs)
        self._action_space = gym.spaces.Box(-1.0, 1.0, (1,), dtype=np.float32)
        self._observation_space = gym.spaces.Box(-np.inf, np.inf, (6,), dtype=np.float32)
        self._num_dof_pos = self._model.num_dof_pos
        self._num_dof_vel = self._model.num_dof_vel

        self._tip = self._model.get_site("tip")
        self._target = self._model.get_site("target")
        self._upper_arm = self._model.get_link("upper_arm")
        self._lower_arm = self._model.get_link("lower_arm")

        self._target_radius = 0.2

        self._step_count = np.zeros(self._num_envs, dtype=np.int32)
        self._max_steps = int(cfg.max_episode_seconds / cfg.ctrl_dt)

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    def apply_action(self, actions: np.ndarray, state: NpEnvState):
        actions = np.clip(actions, -1.0, 1.0)
        state.data.actuator_ctrls = actions
        return state

    def _get_obs(self, data: mtx.SceneData) -> np.ndarray:
        dof_pos = data.dof_pos
        shoulder_angle = dof_pos[:, 0]
        elbow_angle = dof_pos[:, 1]

        upper_arm_horizontal = np.cos(shoulder_angle)
        upper_arm_vertical = np.sin(shoulder_angle)

        total_angle = shoulder_angle + elbow_angle
        lower_arm_horizontal = np.cos(total_angle)
        lower_arm_vertical = np.sin(total_angle)

        dof_vel = data.dof_vel

        obs = np.concatenate(
            [
                upper_arm_horizontal.reshape(-1, 1),
                lower_arm_horizontal.reshape(-1, 1),
                upper_arm_vertical.reshape(-1, 1),
                lower_arm_vertical.reshape(-1, 1),
                dof_vel,
            ],
            axis=-1,
        )

        return obs

    def update_state(self, state: NpEnvState) -> NpEnvState:
        data = state.data
        obs = self._get_obs(data)

        tip_pos = self._tip.get_pose(data)
        target_pos = self._target.get_pose(data)
        dist_to_target = np.linalg.norm(tip_pos[:, :3] - target_pos[:, :3], axis=-1)

        base_rwd = reward.tolerance(
            dist_to_target,
            bounds=(0, self._target_radius),
            margin=0,
            value_at_margin=0.0,
            sigmoid="linear",
        )

        in_target = dist_to_target < self._target_radius
        continuous_reward = 0.1 * in_target

        distance_reward = 0.3 * (1.0 - np.clip(dist_to_target / 2.0, 0, 1.0))

        dof_vel = data.dof_vel
        vel_magnitude = np.mean(np.abs(dof_vel), axis=-1)
        velocity_penalty = 0.01 * np.maximum(0, vel_magnitude - 2.0)

        rwd = base_rwd + continuous_reward + distance_reward - velocity_penalty

        self._step_count += 1

        terminated = np.zeros((self._num_envs,), dtype=bool)

        terminated = np.logical_or(self._step_count >= self._max_steps, terminated)

        terminated = np.logical_or(np.isnan(obs).any(axis=-1), terminated)

        state.obs = obs
        state.reward = rwd
        state.terminated = terminated
        return state

    def reset(self, data: mtx.SceneData) -> tuple[np.ndarray, dict]:
        data.reset(self._model)
        num_reset = data.shape[0]

        shoulder_angle = np.random.uniform(-np.pi, np.pi, size=num_reset).astype(np.float32)
        elbow_angle = np.random.uniform(-np.pi, np.pi, size=num_reset).astype(np.float32)

        dof_pos = np.stack([shoulder_angle, elbow_angle], axis=-1)
        dof_vel = np.zeros((*data.shape, self._num_dof_vel), dtype=np.float32)

        data.set_dof_vel(dof_vel)
        data.set_dof_pos(dof_pos, self._model)
        self._model.forward_kinematic(data)

        obs = self._get_obs(data)
        return obs, {}

    def _reset_done_envs(self):
        """
        Reset the environments that are done
        """
        super()._reset_done_envs()
        done = self._state.done
        if np.any(done):
            self._step_count[done] = 0
