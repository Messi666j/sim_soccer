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

from typing import Any, Tuple

import gymnasium
import numpy as np
import torch
from skrl.envs.torch import Wrapper as SkrlWrapper

from motrix_envs.np.env import NpEnv
from motrix_envs.np.renderer import NpRenderer


class SkrlNpWrapper(SkrlWrapper):
    """
    Wrap the numpy-based environment to be compatible with skrl (PyTorch)
    """

    _env: NpEnv
    _renderer: NpRenderer = None
    _action_clip_count: int = 0

    def __init__(self, env: NpEnv, enable_render: bool = False):
        super().__init__(env)
        if enable_render:
            self._renderer = NpRenderer(env)

    def reset(self) -> Tuple[torch.Tensor, Any]:
        state = self._env.init_state()
        return torch.tensor(state.obs, dtype=torch.float32, device=self.device), state.info

    def step(
        self, actions: torch.Tensor
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Any,
    ]:
        actions = actions.cpu().numpy()

        # ---- Action safety guard ----
        nan_count = int(np.isnan(actions).sum())
        inf_count = int(np.isinf(actions).sum())
        if nan_count > 0 or inf_count > 0:
            actions = np.nan_to_num(actions, nan=0.0, posinf=1.0, neginf=-1.0)
            SkrlNpWrapper._action_clip_count += 1

        actions = np.clip(actions, -1.0, 1.0)

        state = self._env.step(actions)

        return (
            torch.tensor(state.obs, dtype=torch.float32, device=self.device),
            torch.tensor(state.reward.reshape(-1, 1), dtype=torch.float32, device=self.device),
            torch.tensor(state.terminated.reshape(-1, 1), dtype=torch.bool, device=self.device),
            torch.tensor(state.truncated.reshape(-1, 1), dtype=torch.bool, device=self.device),
            state.info,
        )

    def render(self, *args, **kwargs) -> Any:
        if self._renderer:
            self._renderer.render()

    def close(self) -> None:
        pass

    @property
    def num_envs(self) -> int:
        return self._env.num_envs

    @property
    def observation_space(self) -> gymnasium.Space:
        return self._env.observation_space

    @property
    def action_space(self) -> gymnasium.Space:
        return self._env.action_space
