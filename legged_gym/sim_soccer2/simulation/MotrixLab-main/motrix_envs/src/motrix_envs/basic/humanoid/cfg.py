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

import os
from dataclasses import dataclass, field

from motrix_envs import registry
from motrix_envs.base import EnvCfg

model_file = os.path.dirname(__file__) + "/humanoid.xml"


@dataclass
class InitStateConfig:
    reset_height_factor: float = 0.95
    reset_qvel_range: float = 0.01
    reset_actuator_range: float = 0.02
    hip_yaw_range: tuple[float, float] = (-15.0, 15.0)
    hip_roll_range: tuple[float, float] = (-12.0, 12.0)
    hip_pitch_range: tuple[float, float] = (-12.0, 12.0)
    symmetric_leg_pairs: list[tuple[int, int, tuple[float, float]]] = field(
        default_factory=lambda: [
            (10, 16, (-18.0, 2.0)),
            (11, 17, (-25.0, 20.0)),
            (12, 18, (-70.0, 5.0)),
            (13, 19, (-45.0, -25.0)),
            (14, 20, (-40.0, 0.0)),
            (15, 21, (-25.0, 25.0)),
        ]
    )
    symmetric_arm_pairs: list[tuple[int, int]] = field(
        default_factory=lambda: [
            (22, 25),
            (23, 26),
            (24, 27),
        ]
    )
    arm_margin_factor: float = 0.1


@dataclass
class TerminationConfig:
    head_height_factor: float = 0.5
    torso_upright_threshold: float = 0.2
    extreme_vel_threshold: float = 200.0


@registry.envcfg("dm-humanoid-walk")
@dataclass
class HumanoidWalkCfg(EnvCfg):
    model_file: str = model_file
    max_episode_seconds: float = 25.0

    sim_dt: float = 0.01
    ctrl_dt: float = 0.01
    move_speed: float = 1.0
    stand_height: float = 1.4

    init_state: InitStateConfig = field(default_factory=InitStateConfig)
    termination_config: TerminationConfig = field(default_factory=TerminationConfig)


@registry.envcfg("dm-humanoid-stand")
@dataclass
class HumanoidStandCfg(HumanoidWalkCfg):
    move_speed: float = 0.0


@registry.envcfg("dm-humanoid-run")
@dataclass
class HumanoidRunCfg(HumanoidWalkCfg):
    move_speed: float = 10.0
