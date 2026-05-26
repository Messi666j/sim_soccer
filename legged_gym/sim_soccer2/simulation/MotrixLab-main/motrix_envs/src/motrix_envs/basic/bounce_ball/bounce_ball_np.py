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
from motrix_envs.np.env import NpEnv, NpEnvState

from .cfg import BounceBallEnvCfg


@registry.env("bounce_ball", "np")
class BounceBallEnv(NpEnv):
    _cfg: BounceBallEnvCfg

    def __init__(self, cfg: BounceBallEnvCfg, num_envs: int = 1):
        super().__init__(cfg, num_envs=num_envs)

        # Action space: 6D joint position control
        self._action_space = gym.spaces.Box(-1.0, 1.0, (6,), dtype=np.float32)

        # Observation space: joint states + paddle position + target height (29D)
        self._observation_space = gym.spaces.Box(-np.inf, np.inf, (29,), dtype=np.float32)

        self._num_dof_pos = self._model.num_dof_pos
        self._num_dof_vel = self._model.num_dof_vel

        # Initial arm joint positions
        self._init_arm_qpos = np.array(self._cfg.arm_init_qpos, dtype=np.float32) * np.pi / 180.0
        self._init_dof_vel = np.zeros(self._model.num_dof_vel, dtype=np.float32)

        # Full DOF positions (6 arm joints + 7 ball free joint)
        self._init_dof_pos = np.zeros(self._model.num_dof_pos, dtype=np.float32)
        self._init_dof_pos[:6] = self._init_arm_qpos

        # Body and geom references
        self._paddle_geom = self._model.get_geom("blocker")
        self._ball_body_id = self._model.body_names.index("ball_link")

        # Mocap bodies for visual markers
        self._target_marker_body = self._model.get_body("target_height_marker")
        assert self._target_marker_body.is_mocap, "target_height_marker must be a mocap body"

        self._paddle_home_marker_body = self._model.get_body("paddle_home_marker")
        assert self._paddle_home_marker_body.is_mocap, "paddle_home_marker must be a mocap body"

        # Action scaling
        self._action_scale = np.array(self._cfg.action_scale, dtype=np.float32)
        self._action_bias = np.array(self._cfg.action_bias, dtype=np.float32)

        # Ball initial conditions
        self._ball_init_pos = np.array(self._cfg.ball_init_pos, dtype=np.float32)
        self._ball_init_vel = np.array(self._cfg.ball_init_vel, dtype=np.float32)

        # Constants for marker poses
        self._ball_radius = 0.019  # Ball radius in meters
        # Target marker base pose: [x, y, z_placeholder, qx, qy, qz, qw]
        self._target_marker_base_pose = np.array(
            [
                self._cfg.target_ball_x,
                self._cfg.target_ball_y,
                0.0,  # z will be set per environment
                0.0,
                0.0,
                0.0,
                1.0,  # identity quaternion
            ],
            dtype=np.float32,
        )
        # Paddle home marker pose: [x, y, z, qx, qy, qz, qw]
        self._paddle_home_marker_pose = np.array(
            [
                self._cfg.target_ball_x,
                self._cfg.target_ball_y,
                self._cfg.paddle_home_position_z,
                0.0,
                0.0,
                0.0,
                1.0,  # identity quaternion
            ],
            dtype=np.float32,
        )

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    def _denormalize_action(self, action: np.ndarray) -> np.ndarray:
        """Denormalize action from [-1, 1] to joint position changes"""
        return self._action_scale * action + self._action_bias

    def _compute_observation(self, data: mtx.SceneData, target_heights: np.ndarray) -> np.ndarray:
        """Compute observation: joint states + paddle position + target height (29D)"""
        dof_pos = data.dof_pos
        dof_vel = data.dof_vel

        # Get paddle position
        paddle_pose = self._paddle_geom.get_pose(data)
        paddle_xyz = paddle_pose[:, :3]

        # Concatenate: DOF pos (13) + DOF vel (12) + paddle xyz (3) + target height (1)
        obs = np.concatenate([dof_pos, dof_vel, paddle_xyz, target_heights[:, np.newaxis]], axis=-1)
        return obs.astype(np.float32)

    def _compute_reward(
        self,
        obs: np.ndarray,
        data: mtx.SceneData = None,
        consecutive_bounces: np.ndarray = None,
        bounce_detected: np.ndarray = None,
        target_heights: np.ndarray = None,
        current_actions: np.ndarray = None,
        last_actions: np.ndarray = None,
    ) -> tuple:
        """
        Compute reward based on ball position, velocity, and paddle alignment.

        The reward function uses a composite design with multiple reward and penalty terms
        to guide the robot to learn a stable ball bouncing strategy.

        Returns:
            tuple: (total_reward, reward_details) where reward_details contains
                   individual reward components for analysis.
        """
        # Extract ball state
        ball_x = obs[:, 6]
        ball_y = obs[:, 7]
        ball_z = obs[:, 8]
        ball_vz = obs[:, 13 + 8]

        # Extract paddle position
        paddle_xy = obs[:, 25:27]
        paddle_z = obs[:, 27]

        # Target positions
        target_ball_x = self._cfg.target_ball_x
        target_ball_y = self._cfg.target_ball_y
        target_height = target_heights

        # Physics constant
        g = self._cfg.gravity

        # ============================================================================
        # 1. Horizontal position reward (weighted by vertical distance)
        # Core reward ensuring ball stays directly above paddle
        # ============================================================================
        x_position_error = np.abs(ball_x - target_ball_x)
        y_position_error = np.abs(ball_y - target_ball_y)
        xy_position_error = np.sqrt(x_position_error**2 + y_position_error**2)

        vertical_dist = np.abs(ball_z - paddle_z)
        vertical_weight = np.exp(-vertical_dist / self._cfg.vertical_weight_scale)

        weighted_horizontal_scale = self._cfg.weighted_horizontal_base_scale * (
            1.0 + self._cfg.weighted_horizontal_weight_factor * vertical_weight
        )
        weighted_position_reward = np.exp(-(xy_position_error**2) / (2 * weighted_horizontal_scale**2))

        # ============================================================================
        # 2. Out-of-position penalty
        # Strong penalty for severe deviation to prevent ball flying out of control
        # ============================================================================
        out_of_position_penalty = -2.0 / (
            1.0
            + np.exp(-(xy_position_error - self._cfg.out_of_position_threshold) / self._cfg.out_of_position_sharpness)
        )

        # ============================================================================
        # 3. Velocity matching reward
        # Based on projectile motion physics, encourages ball trajectory to have
        # desired velocity at target height for stable control
        # ============================================================================
        height_diff = target_height - ball_z
        desired_velocity_at_target = self._cfg.desired_velocity_at_target

        # Calculate velocity at target height using physics
        upward_below_condition = (ball_vz > 0) & (ball_z < target_height)
        velocity_squared_at_target_upward = np.where(upward_below_condition, ball_vz**2 - 2 * g * height_diff, 0.0)
        velocity_at_target_upward = np.sqrt(np.maximum(0, velocity_squared_at_target_upward))

        downward_above_condition = (ball_vz < 0) & (ball_z > target_height)
        velocity_squared_at_target_downward = np.where(
            downward_above_condition, ball_vz**2 + 2 * g * np.abs(height_diff), 0.0
        )
        velocity_at_target_downward = -np.sqrt(np.maximum(0, velocity_squared_at_target_downward))

        near_target_condition = np.abs(height_diff) < 0.05
        velocity_at_target_near = np.where(near_target_condition, ball_vz, 0.0)

        # Smooth combination
        upward_motion = 1.0 / (1.0 + np.exp(-ball_vz / 0.2))
        below_target = 1.0 / (1.0 + np.exp(-height_diff / 0.02))
        downward_motion = 1.0 - upward_motion
        above_target = 1.0 - below_target
        at_target_weight = np.exp(-(height_diff**2) / (2 * 0.01**2))

        velocity_at_target = (
            velocity_at_target_upward * upward_motion * below_target
            + velocity_at_target_downward * downward_motion * above_target
            + velocity_at_target_near * at_target_weight
        )

        velocity_error = np.abs(velocity_at_target - desired_velocity_at_target)
        velocity_matching_reward = np.exp(-(velocity_error**2) / (2 * self._cfg.velocity_error_sigma**2))

        # ============================================================================
        # 4. Height reward
        # Directly encourages ball to approach target height, core task objective
        # ============================================================================
        height_error = np.abs(ball_z - target_height)
        height_reward = np.exp(-(height_error**2) / (2 * self._cfg.height_error_sigma**2))

        # Height progress bonus
        height_progress_bonus = (
            np.maximum(0, ball_z - self._cfg.height_progress_threshold) * self._cfg.height_progress_scale
        )

        # ============================================================================
        # 5. Controlled upward velocity reward
        # Only rewards upward velocity when ball position is good, avoiding random hitting
        # Guides strategy to learn precise hitting force
        # ============================================================================
        positioning_quality = np.exp(-(xy_position_error**2) / (2 * self._cfg.positioning_quality_sigma**2))

        ideal_launch_velocity = np.sqrt(2 * g * np.maximum(0, height_diff))
        ideal_launch_velocity = np.clip(
            ideal_launch_velocity, self._cfg.ideal_velocity_min, self._cfg.ideal_velocity_max
        )

        upward_velocity_quality = np.exp(
            -((ball_vz - ideal_launch_velocity) ** 2) / (2 * self._cfg.upward_velocity_sigma**2)
        )

        upward_mask = 1.0 / (1.0 + np.exp(-ball_vz / self._cfg.upward_mask_scale))

        controlled_upward_reward = (
            positioning_quality
            * upward_velocity_quality
            * upward_mask
            * np.clip(ball_vz * 1.0, 0.0, self._cfg.controlled_upward_clip_max)
        )

        # ============================================================================
        # 6. Velocity penalties
        # Prevents ball velocity from being too fast or falling freely
        # Ensures ball motion stays within controllable range
        # ============================================================================
        excessive_upward_penalty = -1.0 / (
            1.0 + np.exp(-(ball_vz - self._cfg.excessive_upward_threshold) / self._cfg.excessive_upward_sharpness)
        )

        downward_penalty_magnitude = -ball_vz * np.clip(
            -ball_vz * self._cfg.downward_velocity_scale, 0.0, self._cfg.downward_velocity_clip_max
        )
        downward_penalty_trigger = 1.0 / (1.0 + np.exp((ball_vz - self._cfg.downward_velocity_threshold) / 0.2))
        downward_velocity_penalty = downward_penalty_magnitude * downward_penalty_trigger

        # ============================================================================
        # 7. Consecutive bounces reward
        # Encourages multiple consecutive successful bounces for stable long-term control
        # Uses logarithmic function to avoid infinite reward growth
        # ============================================================================
        bounce_positioning_quality = np.exp(-(xy_position_error**2) / (2 * self._cfg.bounce_positioning_sigma**2))

        bounce_log_reward = np.log(consecutive_bounces.astype(np.float32) + 1.0) * self._cfg.bounce_log_scale

        bounce_activation = (consecutive_bounces > 0).astype(np.float32) * bounce_positioning_quality
        consecutive_bounces_reward = bounce_log_reward * bounce_activation

        # High bounce count bonus
        high_bounce_activation = 1.0 / (
            1.0
            + np.exp(
                -(consecutive_bounces.astype(np.float32) - self._cfg.high_bounce_threshold)
                / self._cfg.high_bounce_sharpness
            )
        )
        high_bounce_bonus = (
            consecutive_bounces.astype(np.float32)
            * self._cfg.high_bounce_scale
            * bounce_positioning_quality
            * high_bounce_activation
        )

        # ============================================================================
        # 8. Paddle-ball horizontal alignment
        # Encourages paddle to actively move directly below ball
        # Extra reward at bounce moment to reinforce correct hitting behavior
        # ============================================================================
        if bounce_detected is None:
            bounce_detected = np.zeros(ball_x.shape[0], dtype=bool)

        ball_xy = np.stack([ball_x, ball_y], axis=1)
        paddle_ball_xy_error = np.linalg.norm(ball_xy - paddle_xy, axis=1)

        vertical_proximity_weight = np.exp(-vertical_dist / self._cfg.vertical_proximity_scale)

        paddle_alignment_quality = np.exp(-(paddle_ball_xy_error**2) / (2 * self._cfg.paddle_alignment_sigma**2)) * (
            1.0 + self._cfg.paddle_alignment_weight_factor * vertical_proximity_weight
        )

        bounce_boost = bounce_detected.astype(np.float32) * self._cfg.bounce_boost_factor + 1.0
        paddle_center_reward = paddle_alignment_quality * bounce_boost * self._cfg.paddle_center_scale

        # ============================================================================
        # 9. Paddle home position reward
        # Encourages paddle to return to home position when ball is far away
        # Makes paddle motion more energy-efficient and natural
        # ============================================================================
        paddle_home_z = self._cfg.paddle_home_position_z
        paddle_height_deviation = np.abs(paddle_z - paddle_home_z)

        # Distance-based dynamic factor
        distance_factor = 1.0 + self._cfg.distance_factor_scale / (
            1.0 + np.exp(-(vertical_dist - self._cfg.distance_factor_threshold) / self._cfg.distance_factor_sharpness)
        )

        home_position_reward = (
            np.exp(-(paddle_height_deviation**2) / (2 * self._cfg.home_position_sigma**2)) * distance_factor
        )

        # Height violation penalty
        max_deviation = self._cfg.max_paddle_height_deviation
        height_violation = np.maximum(0, paddle_height_deviation - max_deviation)
        height_violation_penalty = -height_violation * self._cfg.height_violation_scale

        # ============================================================================
        # 10. Action and velocity penalties
        # Penalizes drastic action changes and excessive joint velocities
        # Encourages smooth and energy-efficient control
        # ============================================================================
        num_envs = obs.shape[0]
        if current_actions is None:
            current_actions = np.zeros((num_envs, 6), dtype=np.float32)
        if last_actions is None:
            last_actions = np.zeros((num_envs, 6), dtype=np.float32)

        action_diff = current_actions - last_actions
        action_penalty = np.sum(np.square(action_diff), axis=-1)

        joint_vel = data.dof_vel[:, :6]
        joint_vel_penalty = np.sum(np.square(joint_vel), axis=-1)

        # ============================================================================
        # Total reward
        # ============================================================================
        action_penalty_rate = self._cfg.action_penalty_rate
        joint_vel_penalty_rate = self._cfg.joint_vel_penalty_rate

        total_reward = (
            weighted_position_reward * self._cfg.weighted_position_weight
            + velocity_matching_reward * self._cfg.velocity_matching_weight
            + height_reward * self._cfg.height_reward_weight
            + height_progress_bonus * self._cfg.height_progress_weight
            + controlled_upward_reward * self._cfg.controlled_upward_weight
            + consecutive_bounces_reward * self._cfg.consecutive_bounces_weight
            + high_bounce_bonus * self._cfg.high_bounce_weight
            + paddle_center_reward * self._cfg.paddle_center_weight
            + home_position_reward * self._cfg.home_position_weight
            + out_of_position_penalty * self._cfg.out_of_position_weight
            + excessive_upward_penalty * self._cfg.excessive_upward_weight
            + downward_velocity_penalty * self._cfg.downward_velocity_weight
            + height_violation_penalty * self._cfg.height_violation_weight
            - action_penalty_rate * action_penalty
            - joint_vel_penalty_rate * joint_vel_penalty
        )

        # ============================================================================
        # Reward details for debugging
        # ============================================================================
        reward_details = {
            "x_position_error": x_position_error,
            "y_position_error": y_position_error,
            "xy_position_error": xy_position_error,
            "ball_z": ball_z,
            "paddle_z": paddle_z,
            "vertical_dist": vertical_dist,
            "vertical_weight": vertical_weight,
            "ball_vz": ball_vz,
            "height_diff": height_diff,
            "height_error": height_error,
            "velocity_at_target": velocity_at_target,
            "desired_velocity_at_target": np.full_like(ball_vz, desired_velocity_at_target),
            "velocity_error": velocity_error,
            "ideal_launch_velocity": ideal_launch_velocity,
            "positioning_quality": positioning_quality,
            "upward_velocity_quality": upward_velocity_quality,
            "bounce_positioning_quality": bounce_positioning_quality,
            "paddle_ball_xy_error": paddle_ball_xy_error,
            "vertical_proximity_weight": vertical_proximity_weight,
            "bounce_detected": bounce_detected.astype(np.float32),
            "consecutive_bounces": consecutive_bounces.astype(np.float32),
            "action_penalty": action_penalty,
            "joint_vel_penalty": joint_vel_penalty,
            "distance_factor": distance_factor,
            "weighted_position_reward": weighted_position_reward * 2.0,
            "velocity_matching_reward": velocity_matching_reward * 2.0,
            "height_reward": height_reward * 4.5,
            "height_progress_bonus": height_progress_bonus * 1.0,
            "controlled_upward_reward": controlled_upward_reward * 1.5,
            "consecutive_bounces_reward": consecutive_bounces_reward * 0.8,
            "high_bounce_bonus": high_bounce_bonus * 0.3,
            "paddle_center_reward": paddle_center_reward * 0.6,
            "home_position_reward": home_position_reward * 1.5,
            "out_of_position_penalty": out_of_position_penalty * 1.0,
            "excessive_upward_penalty": excessive_upward_penalty * 1.0,
            "downward_velocity_penalty": downward_velocity_penalty * 1.0,
            "height_violation_penalty": height_violation_penalty * 1.0,
            "action_penalty_weighted": -action_penalty_rate * action_penalty,
            "joint_vel_penalty_weighted": -joint_vel_penalty_rate * joint_vel_penalty,
            "paddle_height_deviation": paddle_height_deviation,
            "total_reward": total_reward,
        }

        return total_reward, reward_details

    def _compute_terminated(self, obs: np.ndarray, target_heights: np.ndarray) -> np.ndarray:
        """Check if episode should terminate based on DOF states"""
        # Extract ball position from DOF (indices 6-8 for x,y,z)
        ball_x = obs[:, 6]  # Ball x position
        ball_y = obs[:, 7]  # Ball y position
        ball_z = obs[:, 8]  # Ball z position

        # Terminate if ball falls below ground or goes significantly higher than target
        terminated = (ball_z < 0.05) | (ball_z > target_heights + 1.0)

        # Also terminate if ball goes too far horizontally
        terminated |= (np.abs(ball_x) > 1.5) | (np.abs(ball_y) > 1.5)

        # Terminate if joint velocity is too high
        # Limit: 360 degrees/second = 2*pi rad/s ≈ 6.28 rad/s
        joint_vel = obs[:, 13:19]  # Joint velocities (indices 13-18 for 6 arm joints)
        max_joint_vel = 2.0 * np.pi  # 360 degrees/second in radians
        terminated |= np.abs(joint_vel).max(axis=-1) > max_joint_vel

        return terminated

    def apply_action(self, actions: np.ndarray, state: NpEnvState) -> NpEnvState:
        """Apply action to control paddle position"""
        # Store last actions for penalty calculation
        state.info["last_actions"] = state.info.get("current_actions", np.zeros_like(actions))
        state.info["current_actions"] = actions

        # Get current joint positions
        current_joint_pos = state.data.dof_pos[:, :6]  # First 6 DOFs are arm joints

        # Denormalize actions to get actual position changes
        delta_positions = self._denormalize_action(actions)

        # Calculate target positions = current positions + position changes
        target_positions = current_joint_pos + delta_positions

        # Apply target positions as actuator controls (position control)
        state.data.actuator_ctrls = target_positions
        return state

    def update_state(self, state: NpEnvState) -> NpEnvState:
        """Update state with new observations, rewards, and termination flags"""
        data = state.data

        # Get bounce tracking and target heights from info
        consecutive_bounces = state.info.get("consecutive_bounces", np.zeros(data.shape[0], dtype=np.int32))
        ball_was_upward = state.info.get("ball_was_upward", np.zeros(data.shape[0], dtype=bool))
        # Use mean of target_height_range as fallback
        default_height = np.mean(self._cfg.target_height_range)
        target_heights = state.info.get("target_heights", np.full(data.shape[0], default_height, dtype=np.float32))

        # Compute observation with target heights
        obs = self._compute_observation(data, target_heights)

        # Detect bounces and update consecutive bounce count
        current_ball_z = obs[:, 8]  # Ball z position
        current_ball_vz = obs[:, 21]  # Ball z velocity

        # Detect bounces: ball moving upward after being near paddle height
        near_paddle = (current_ball_z < 0.4) & (current_ball_z > 0.15)
        moving_upward = current_ball_vz > 0.01

        # A bounce is detected when ball was going down and now goes up near paddle height
        bounce_detected = ~ball_was_upward & moving_upward & near_paddle

        # Update consecutive bounce count
        consecutive_bounces = np.where(bounce_detected, consecutive_bounces + 1, consecutive_bounces)

        # Reset count if ball is falling too much (not bouncing properly)
        falling = (current_ball_vz < -0.5) & (current_ball_z < 0.4)
        consecutive_bounces = np.where(falling, 0, consecutive_bounces)

        # Update tracking variables in info
        state.info["consecutive_bounces"] = consecutive_bounces
        state.info["ball_was_upward"] = moving_upward

        # Track maximum bounces achieved
        max_current = np.max(consecutive_bounces)
        if "max_consecutive_bounces" not in state.info:
            state.info["max_consecutive_bounces"] = 0
        if max_current > state.info["max_consecutive_bounces"]:
            state.info["max_consecutive_bounces"] = max_current

        # For simplicity, use raw observation without normalization for now
        # Could add proper normalization later
        normalized_obs = obs

        # Compute reward and termination
        reward, reward_details = self._compute_reward(
            obs,
            data,
            consecutive_bounces,
            bounce_detected=bounce_detected,
            target_heights=target_heights,
            current_actions=state.info.get("current_actions"),
            last_actions=state.info.get("last_actions"),
        )
        terminated = self._compute_terminated(obs, target_heights=target_heights)

        state.obs = normalized_obs
        state.reward = reward
        state.terminated = terminated

        # Store reward details for debugging
        if self._cfg.store_reward_details:
            state.info["Reward"] = reward_details
        state.info["target_heights"] = target_heights  # Ensure target_heights persists across steps

        return state

    def reset(self, data: mtx.SceneData) -> tuple:
        """Reset environment to initial state with randomized target heights"""
        cfg: BounceBallEnvCfg = self._cfg
        num_reset = data.shape[0]

        # Randomize target heights for the environments being reset
        if cfg.randomize_target_height:
            min_height, max_height = cfg.target_height_range
            new_target_heights = np.random.uniform(min_height, max_height, num_reset).astype(np.float32)
        else:
            # Use mean of target_height_range when not randomizing
            default_height = np.mean(cfg.target_height_range)
            new_target_heights = np.full(num_reset, default_height, dtype=np.float32)

        # Add noise to initial arm joint positions only (not ball)
        arm_noise_pos = np.random.uniform(
            -cfg.reset_noise_scale,
            cfg.reset_noise_scale,
            (num_reset, 6),  # Only 6 arm joints
        )
        noise_vel = np.random.uniform(
            -cfg.reset_noise_scale,
            cfg.reset_noise_scale,
            (num_reset, self._num_dof_vel),
        )

        # Reset simulation first to get proper DOF structure
        data.reset(self._model)

        # Get current DOF positions
        current_dof_pos = data.dof_pos

        # === Modify all DOF positions ===
        # Set arm joint positions (first 6 DOFs)
        current_dof_pos[:, :6] = np.tile(self._init_arm_qpos, (num_reset, 1)) + arm_noise_pos

        # Set ball position in DOF (indices 6-8 for x, y, z positions)
        ball_noise_pos = np.random.uniform(-0.01, 0.01, (num_reset, 3))
        ball_pos = self._ball_init_pos + ball_noise_pos
        current_dof_pos[:, 6:9] = ball_pos

        # Apply all DOF position  changes
        data.set_dof_pos(current_dof_pos, self._model)

        # Get current DOF velocities
        current_dof_vel = data.dof_vel
        # === Modify all DOF velocities ===

        # Set arm joint velocities (first 6 DOFs)
        current_dof_vel[:, :6] = noise_vel[:, :6]

        # Apply ball linear velocity in DOF
        data.set_dof_vel(current_dof_vel)

        # Update target height marker position (thin cylinder disc)
        # Marker center aligns with ball top: marker_z = target_height + ball_radius
        target_marker_poses = np.tile(self._target_marker_base_pose, (num_reset, 1))
        target_marker_poses[:, 2] = new_target_heights + self._ball_radius  # Set z position

        self._target_marker_body.mocap.set_pose(data, target_marker_poses)

        # Update paddle home marker position
        paddle_home_marker_poses = np.tile(self._paddle_home_marker_pose, (num_reset, 1))

        # Set paddle home marker mocap body pose
        self._paddle_home_marker_body.mocap.set_pose(data, paddle_home_marker_poses)

        # Initialize info dict with bounce tracking variables
        info = {
            "consecutive_bounces": np.zeros(num_reset, dtype=np.int32),
            "ball_was_upward": np.zeros(num_reset, dtype=bool),
            "max_consecutive_bounces": 0,
            "target_heights": new_target_heights.copy(),  # Return target heights for this reset batch
            "current_actions": np.zeros((num_reset, 6), dtype=np.float32),
            "last_actions": np.zeros((num_reset, 6), dtype=np.float32),
        }

        # Compute initial observation with target heights
        obs = self._compute_observation(data, new_target_heights)
        normalized_obs = obs  # No normalization for now

        return normalized_obs, info
