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

"""Configuration for Shadow Hand Cube Reorientation Environment"""

import os
from dataclasses import dataclass
from typing import List, Tuple

from motrix_envs import registry
from motrix_envs.base import EnvCfg

# Path to the repose_cube.xml model
model_file = os.path.join(os.path.dirname(__file__), "xmls", "repose_cube.xml")


@registry.envcfg("shadow-hand-repose")
@dataclass
class ShadowHandReposeEnvCfg(EnvCfg):
    """
    Configuration for Shadow Hand Cube Reorientation Environment

    This environment trains the Shadow Hand to reorient a cube to match
    randomly sampled target orientations.
    """

    # ====================
    # Model Configuration
    # ====================
    model_file: str = model_file

    # ====================
    # Simulation Parameters
    # ====================
    sim_dt: float = 0.01
    sim_substeps: int = 1  # Number of simulation steps per control step
    ctrl_dt: float = sim_dt * sim_substeps

    max_episode_seconds: float = 10.0
    max_episode_steps: int = int(max_episode_seconds / ctrl_dt)

    # ====================
    # Robot Configuration
    # ====================
    num_hand_dofs: int = 24  # Total DOFs in Shadow Hand
    num_actuators: int = 20  # Actuated joints

    # Fingertip link names for forward kinematics
    fingertip_link_names: List[str] = (
        "rh_ffdistal",  # First finger (index) distal
        "rh_mfdistal",  # Middle finger distal
        "rh_rfdistal",  # Ring finger distal
        "rh_lfdistal",  # Little finger distal
        "rh_thdistal",  # Thumb distal
    )

    # ====================
    # Object Configuration
    # ====================
    cube_initial_pos: Tuple[float, float, float] = (0.33, 0.00, 0.295)  # Initial cube position

    # ====================
    # Reward Parameters
    # ====================
    # Core reward components
    dist_reward_scale: float = -10.0  # Balanced for MotrixSim (推荐)
    rot_reward_scale: float = 1.0  # Moderate rotation reward
    rot_eps: float = 0.1  # Stable denominator

    action_penalty_scale: float = -0.0002

    # Success and failure criteria
    success_tolerance: float = 0.1  # ~8.6° (moderate challenge)
    reach_goal_bonus: float = 2.0  # Balanced incentive

    fall_dist: float = 0.24  # Reasonable manipulation space          # Distance threshold for dropping cube (meters)
    fall_penalty: float = 0.0  # Penalty for dropping the cube

    # In-hand distance threshold (only used for success check, not reward)
    in_hand_dist_threshold: float = 0.05  # Distance threshold for "in-hand" (5cm)

    # Success hold mechanism (uses max_consecutive_successes)
    max_consecutive_successes: int = 50  # Reset after holding success for this many steps

    # Averaging factor for consecutive successes tracking
    av_factor: float = 0.1

    # ====================
    # Reset Noise Parameters
    # ====================
    reset_position_noise: float = 0.01  # Increased robustness
    reset_dof_pos_noise: float = 0.2  # Higher generalization
    reset_dof_vel_noise: float = 0.0  # DOF velocity noise at reset

    # ====================
    # Observation Scaling
    # ====================
    vel_obs_scale: float = 0.2  # Scale factor for velocity observations

    # ====================
    # Action Processing
    # ====================
    act_moving_average: float = 1.0  # Action smoothing (1.0 = no smoothing)

    # ====================
    # Visualization
    # ====================
    # Offset for target visualization (relative to hand position)
    # Recommended: offset to upper-left to avoid occluding the real hand and cube
    viz_target_offset: Tuple[float, float, float] = (
        0.0,  # Left (negative X)
        0.0,  # Forward/Up (negative Y)
        0.2,  # Up (positive Z)
    )

    # ====================
    # Domain Randomization (Optional)
    # ====================
    # Enable domain randomization (recommended to start with False)
    enable_domain_randomization: bool = False

    # Randomization parameters (only used if enable_domain_randomization=True)
    randomize_friction: bool = False
    friction_range: Tuple[float, float] = (0.8, 12)

    randomize_mass: bool = False
    mass_range: Tuple[float, float] = (0.8, 1.2)

    randomize_com: bool = False
    com_displacement_range: float = 0.01
