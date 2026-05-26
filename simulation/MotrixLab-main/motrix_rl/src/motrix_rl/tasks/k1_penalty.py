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

"""RL training configuration for K1 penalty shootout task."""

from dataclasses import dataclass

from motrix_rl.registry import rlcfg
from motrix_rl.rslrl.cfg import RslrlCfg
from motrix_rl.skrl.config import SkrlCfg


class skrl:
    """SKRL PPO configurations for K1 penalty shootout."""

    @rlcfg("k1-penalty-shootout")
    @dataclass
    class K1PenaltySkrlPpo(SkrlCfg):
        """K1 penalty shootout - SKRL PPO configuration."""

        def __post_init__(self):
            runner = self.runner
            models = runner.models
            agent = runner.agent
            trainer = runner.trainer

            models.policy.hiddens = [512, 256, 128]
            models.policy.clip_actions = True
            models.policy.initial_log_std = -1.5
            models.value.hiddens = [512, 256, 128]

            agent.rollouts = 64
            agent.learning_epochs = 5
            agent.mini_batches = 4
            agent.learning_rate = 3e-4
            agent.learning_rate_scheduler = ""  # Disable KLAdaptiveLR (incompatible with PyTorch 2.7)

            trainer.timesteps = 50000


class rslrl:
    """RSLRL PPO configurations for K1 penalty shootout."""

    @rlcfg("k1-penalty-shootout")
    @dataclass
    class K1PenaltyRslrlPpo(RslrlCfg):
        """K1 penalty shootout - RSLRL PPO configuration."""

        def __post_init__(self):
            runner = self.runner
            algo = runner.algorithm

            runner.seed = 42
            runner.max_iterations = 3000
            runner.num_steps_per_env = 24
            runner.experiment_name = "k1_penalty_shootout"
            runner.save_interval = 100

            runner.actor.hidden_dims = [512, 256, 128]
            runner.critic.hidden_dims = [512, 256, 128]

            algo.learning_rate = 3e-4
            algo.num_learning_epochs = 5
            algo.num_mini_batches = 4
            algo.entropy_coef = 0.01
