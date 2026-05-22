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


@dataclass
class K1ControlConfig:
    """PD control configuration for K1 leg joints.

    Stiffness and damping arrays match the 12 leg actuator order:
    Left/Right: Hip_Pitch, Hip_Roll, Hip_Yaw, Knee_Pitch, Ankle_Pitch, Ankle_Roll.
    """

    stiffness: list[float] = field(default_factory=lambda: [
        200.0, 200.0, 200.0, 200.0, 50.0, 50.0,
        200.0, 200.0, 200.0, 200.0, 50.0, 50.0,
    ])
    damping: list[float] = field(default_factory=lambda: [
        5.0, 5.0, 5.0, 5.0, 1.0, 1.0,
        5.0, 5.0, 5.0, 5.0, 1.0, 1.0,
    ])
    # action_scale: action * scale = joint target offset from default angle (radians)
    action_scale: float = 0.25
    # torque_limits: maximum torque applied to each joint (Nm)
    torque_limits: float = 40.0


@dataclass
class K1InitState:
    """Initial state configuration for K1 robot in penalty shootout."""

    # initial position of the robot Trunk in world frame [x, y, z]
    pos: list[float] = field(default_factory=lambda: [2.5, 0.0, 0.55])

    # default joint angles for standing posture
    # matches legged_gym K1_config.py default angles
    default_joint_angles: dict[str, float] = field(default_factory=lambda: {
        "Left_Hip_Pitch": -0.2,
        "Left_Hip_Roll": 0.0,
        "Left_Hip_Yaw": 0.0,
        "Left_Knee_Pitch": 0.4,
        "Left_Ankle_Pitch": -0.25,
        "Left_Ankle_Roll": 0.0,
        "Right_Hip_Pitch": -0.2,
        "Right_Hip_Roll": 0.0,
        "Right_Hip_Yaw": 0.0,
        "Right_Knee_Pitch": 0.4,
        "Right_Ankle_Pitch": -0.25,
        "Right_Ankle_Roll": 0.0,
    })

    # noise added to joint angles during reset (radians, uniform)
    reset_joint_noise: float = 0.05
    # noise added to ball position during reset (meters, uniform)
    reset_ball_noise: float = 0.05


@dataclass
class K1RewardConfig:
    """Reward weights for penalty shootout."""

    scales: dict[str, float] = field(default_factory=lambda: {
        "goal": 1000.0,
        "ball_toward_goal": 2.0,
        "ball_speed": 1.0,
        "alive": 0.5,
        "action_rate": -0.01,
        "joint_limit": -0.1,
    })
    # Exponential decay for ball speed reward
    ball_speed_target: float = 3.0  # m/s target ball speed
    ball_speed_sigma: float = 1.0


@dataclass
class K1PenaltyShootoutCfg(EnvCfg):
    """Configuration for K1 penalty shootout RL environment."""

    # Scene XML path (relative to this file)
    model_file: str = os.path.dirname(__file__) + "/xmls/scene_penalty_shootout.xml"

    # Simulation timestep
    sim_dt: float = 0.005
    # Control (policy) timestep = sim_dt * sim_substeps
    ctrl_dt: float = 0.02  # 50 Hz policy rate

    # Episode ends after this many seconds
    max_episode_seconds: float = 10.0

    # Spacing between envs in vectorized rendering
    render_spacing: float = 2.0

    # Sub-configs
    control: K1ControlConfig = field(default_factory=K1ControlConfig)
    init_state: K1InitState = field(default_factory=K1InitState)
    reward: K1RewardConfig = field(default_factory=K1RewardConfig)

    # ---- Field geometry constants (meters) ----
    # Goal center x position (world frame)
    goal_x: float = 4.5
    # Goal width (full width, y-axis)
    goal_width: float = 1.9
    # Goal height (z-axis from ground)
    goal_height: float = 1.8
    # Penalty spot [x, y, z]
    penalty_spot: list[float] = field(default_factory=lambda: [3.0, 0.0, 0.11])
    # Field boundary limits (|x|, |y|)
    field_x_limit: float = 5.0
    field_y_limit: float = 3.5
    # Ball radius (used for contact distance check)
    ball_radius: float = 0.11

    # ---- Robot body/geom names ----
    robot_body_name: str = "Trunk"
    ball_body_name: str = "ball"
    ball_geom_name: str = "ball"
    ground_geom_name: str = "ground"
    foot_geom_names: list[str] = field(default_factory=lambda: [
        "Left_Foot", "Right_Foot",
    ])

    # ---- Sensor names ----
    local_linvel_sensor: str = "local_linvel"
    gyro_sensor: str = "angular-velocity"
    orientation_sensor: str = "orientation"

    # ---- Observation normalization scales ----
    obs_scale_gravity: float = 1.0
    obs_scale_gyro: float = 1.0
    obs_scale_linvel: float = 1.0
    obs_scale_dof_pos: float = 1.0
    obs_scale_dof_vel: float = 0.1
    obs_scale_ball_pos: float = 1.0
    obs_scale_ball_vel: float = 1.0
    obs_scale_goal_pos: float = 1.0

    # ---- Randomization (disabled by default for MVP) ----
    randomize_ball_position: bool = False
    randomize_robot_position: bool = False
    ball_randomization_range: float = 0.2  # meters
    robot_randomization_range: float = 0.1  # meters


@registry.envcfg("k1-penalty-shootout")
@dataclass
class K1PenaltyShootoutEnvCfg(K1PenaltyShootoutCfg):
    """Registered environment configuration for K1 penalty shootout."""
    pass
