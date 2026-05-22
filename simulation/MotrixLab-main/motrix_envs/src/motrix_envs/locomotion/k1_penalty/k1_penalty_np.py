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

"""K1 Penalty Shootout RL environment using MotrixSim physics.

A single K1 robot must kick a static ball from the penalty spot into the goal.
The robot starts behind the ball and the episode ends when:
  - The ball enters the goal (success)
  - The ball goes out of bounds (failure)
  - The robot falls over (failure)
  - Timeout after max episode steps

Observation space (57-dim):
  0:3   gravity vector in body frame
  3:6   angular velocity (gyro)
  6:9   linear velocity in body frame (local_linvel)
  9:21  joint position differences from default (12 DOF)
  21:33 joint velocities (12 DOF)
  33:45 last action (12 DOF)
  45:48 ball position relative to robot in body frame
  48:51 ball velocity in world frame
  51:54 goal center relative to robot in body frame
  54:57 vector from ball to goal center (world frame, normalized)

Action space (12-dim):
  Leg joint position offsets from default angles. Scaled by action_scale.
  Order: Left/Right Hip_Pitch, Hip_Roll, Hip_Yaw, Knee_Pitch, Ankle_Pitch, Ankle_Roll
"""

import gymnasium as gym
import motrixsim as mtx
import numpy as np

from motrix_envs import registry
from motrix_envs.locomotion.k1_penalty.cfg import K1PenaltyShootoutEnvCfg
from motrix_envs.math import quaternion
from motrix_envs.np.env import NpEnv, NpEnvState


@registry.env("k1-penalty-shootout", sim_backend="np")
class PenaltyShootoutEnv(NpEnv):
    """K1 robot penalty shootout RL environment.

    The robot must kick a ball from the penalty spot into the goal.
    Uses PD control on 12 leg joints with MotrixSim physics backend.
    """

    _init_dof_pos: np.ndarray
    _init_dof_vel: np.ndarray

    def __init__(self, cfg: K1PenaltyShootoutEnvCfg, num_envs: int = 1):
        super().__init__(cfg, num_envs)
        self._init_action_space()
        self._init_obs_space()

        # Body references
        self._body = self._model.get_body(self.cfg.robot_body_name)
        self._ball_body = self._model.get_body(self.cfg.ball_body_name)

        # Dimension counts
        self._num_action: int = self._action_space.shape[0]
        self._num_obs: int = self._observation_space.shape[0]
        self._num_dof_pos: int = self._model.num_dof_pos
        self._num_dof_vel: int = self._model.num_dof_vel

        # Ball DOF indices for velocity queries
        ball_qpos_start = self._ball_body.get_dof_pos_indices(include_floatingbase=True)[0]
        ball_qvel_start = self._ball_body.get_dof_vel_indices(include_floatingbase=True)[0]
        self._ball_qpos_indices: np.ndarray = np.arange(ball_qpos_start, ball_qpos_start + 7, dtype=np.int32)
        self._ball_qvel_indices: np.ndarray = np.arange(ball_qvel_start, ball_qvel_start + 6, dtype=np.int32)

        # Initial DOF state
        self._init_dof_vel = np.zeros((self._num_dof_vel,), dtype=np.float32)
        self._init_dof_pos = self._model.compute_init_dof_pos()
        self._init_buffer()

    # ---- Space initialization ----

    def _init_obs_space(self) -> None:
        """Initialize observation space: 57 dimensions."""
        num_gravity: int = 3
        num_gyro: int = 3
        num_linvel: int = 3
        num_joint_angle: int = 12  # leg joints only
        num_joint_vel: int = 12
        num_actions: int = 12
        num_ball_pos: int = 3
        num_ball_vel: int = 3
        num_goal_pos: int = 3
        num_ball_to_goal: int = 3

        num_obs = (
            num_gravity + num_gyro + num_linvel
            + num_joint_angle + num_joint_vel + num_actions
            + num_ball_pos + num_ball_vel + num_goal_pos + num_ball_to_goal
        )
        assert num_obs == 57, f"Expected 57 obs dims, got {num_obs}"
        self._observation_space = gym.spaces.Box(-np.inf, np.inf, (num_obs,), dtype=np.float32)

    def _init_action_space(self) -> None:
        """Initialize action space: 12 dimensions in [-1, 1]."""
        self._action_space = gym.spaces.Box(-1.0, 1.0, (12,), dtype=np.float32)

    @property
    def action_space(self) -> gym.spaces.Box:
        return self._action_space

    @property
    def observation_space(self) -> gym.spaces.Box:
        return self._observation_space

    # ---- Buffer initialization ----

    def _is_joint_in_body_subtree(self, jname: str) -> bool:
        """Check if a joint is in the robot body subtree (not ball or world)."""
        return jname not in ("world_joint", "ball-root")

    def _init_buffer(self) -> None:
        """Initialize PD gains, default angles, actuator-joint mapping, and contact detection."""
        cfg = self.cfg

        # PD gains
        self.kps = np.array(cfg.control.stiffness, dtype=np.float32)
        self.kds = np.array(cfg.control.damping, dtype=np.float32)

        # Gravity direction in world frame
        self.gravity_vec = np.array([0.0, 0.0, -1.0], dtype=np.float32)

        # Default joint angles for all 12 actuators
        self.default_angles = np.zeros(self._num_action, dtype=np.float32)
        for i in range(self._model.num_actuators):
            name = self._model.actuator_names[i]
            for k, v in cfg.init_state.default_joint_angles.items():
                if k in name:
                    self.default_angles[i] = v

        # Build mapping: actuator index → index in body subtree joint DOF array
        # body.get_joint_dof_pos() returns ALL 22 joints in Trunk subtree,
        # but we have only 12 actuators. We need to find which subtree indices
        # match our actuator target joints.
        # Strategy: create a temporary SceneData to probe the joint order,
        # then match actuator target names against joint names in that order.
        probe_data = mtx.SceneData(self._model, batch=[1])
        probe_data.set_dof_pos(self._init_dof_pos.reshape(1, -1), self._model)
        self._model.forward_kinematic(probe_data)
        all_body_joint_pos = self._body.get_joint_dof_pos(probe_data)
        num_body_joints: int = all_body_joint_pos.shape[1]  # 22 for K1

        # Build list of joint names in body subtree order
        body_joint_names: list[str] = []
        # Get all joints that contribute to the body's get_joint_dof_pos
        # These are sorted by depth-first traversal of the body subtree
        # All revolute joints have 1 DOF; free joints have 7 but are excluded
        for ji in range(self._model.num_joints):
            jname = self._model.joint_names[ji]
            jpos_start = self._model.joint_dof_pos_indices[ji]
            # A joint has DOF entries if its pos_index doesn't equal the next joint's
            # (or is the last joint). Simplest: exclude world_joint and ball-root.
            if jpos_start >= 0 and self._is_joint_in_body_subtree(jname):
                body_joint_names.append(jname)

        # Pad body_joint_names to match num_body_joints (in case of multi-DOF joints)
        # Actually, each joint has size 1 (revolute) or 7 (free joint for Trunk)
        # Free joint world_joint is NOT included in body.get_joint_dof_pos
        # So body_joint_names should already have num_body_joints entries
        assert len(body_joint_names) == num_body_joints, \
            f"Mismatch: {len(body_joint_names)} joint names vs {num_body_joints} body DOFs"

        self._actuator_joint_indices = np.zeros(self._num_action, dtype=np.int32)
        for ai in range(self._model.num_actuators):
            target_joint: str = self._model.actuators[ai].target_name  # type: ignore[attr-defined]
            found = False
            for bi, bname in enumerate(body_joint_names):
                if bname == target_joint:
                    self._actuator_joint_indices[ai] = bi
                    found = True
                    break
            if not found:
                raise RuntimeError(
                    f"Actuator {ai} targets joint '{target_joint}' not found in body subtree. "
                    f"Body joints: {body_joint_names}"
                )

        # Update init DOF pos with default angles at the global DOF indices
        for ai in range(self._model.num_actuators):
            target_joint: str = self._model.actuators[ai].target_name  # type: ignore[attr-defined]
            ji = self._model.get_joint_index(target_joint)
            if ji >= 0:
                jpos_start = self._model.joint_dof_pos_indices[ji]
                self._init_dof_pos[jpos_start] = self.default_angles[ai]

        # Goal position constant
        self.goal_pos_world = np.array(
            [cfg.goal_x, 0.0, cfg.goal_height / 2.0], dtype=np.float32
        )

        # Contact detection: Trunk-geom vs ground
        self._ground_geom_idx = self._model.get_geom_index(cfg.ground_geom_name)
        # Find trunk contact geometries (the main Trunk geom has contype=0, so use collision geoms)
        self._trunk_geom_indices: list[int] = []
        for name in self._model.geom_names:
            if name is not None and cfg.robot_body_name in name:
                idx = self._model.get_geom_index(name)
                if idx is not None and idx != self._ground_geom_idx:
                    self._trunk_geom_indices.append(idx)

        # Build termination contact pairs (trunk geoms vs ground)
        self._termination_pairs: np.ndarray = np.array(
            [[g, self._ground_geom_idx] for g in self._trunk_geom_indices],
            dtype=np.uint32,
        )
        self._num_termination_pairs: int = self._termination_pairs.shape[0] if self._termination_pairs.size > 0 else 0

        # Ball contact pairs (ball geom vs foot geoms)
        self._ball_geom_idx = self._model.get_geom_index(cfg.ball_geom_name)
        foot_geom_indices: list[int] = []
        for fname in cfg.foot_geom_names:
            for gname in self._model.geom_names:
                if gname is not None and fname in gname:
                    idx = self._model.get_geom_index(gname)
                    if idx is not None:
                        foot_geom_indices.append(idx)
        self._ball_contact_pairs: np.ndarray = np.array(
            [[self._ball_geom_idx, f] for f in foot_geom_indices],
            dtype=np.uint32,
        )
        self._num_ball_contact_pairs: int = self._ball_contact_pairs.shape[0] if self._ball_contact_pairs.size > 0 else 0

    # ---- Joint state access ----

    def get_dof_pos(self, data: mtx.SceneData) -> np.ndarray:
        """Get joint positions for the 12 actuated leg joints.

        Maps from the full body subtree DOFs to only the actuator-controlled joints.
        """
        all_pos = self._body.get_joint_dof_pos(data)  # (num_envs, 22)
        return all_pos[:, self._actuator_joint_indices]  # (num_envs, 12)

    def get_dof_vel(self, data: mtx.SceneData) -> np.ndarray:
        """Get joint velocities for the 12 actuated leg joints."""
        all_vel = self._body.get_joint_dof_vel(data)  # (num_envs, 22)
        return all_vel[:, self._actuator_joint_indices]  # (num_envs, 12)

    def get_local_linvel(self, data: mtx.SceneData) -> np.ndarray:
        """Get linear velocity at IMU site in local frame."""
        return self._model.get_sensor_value(self.cfg.local_linvel_sensor, data)

    def get_gyro(self, data: mtx.SceneData) -> np.ndarray:
        """Get angular velocity from gyro sensor."""
        return self._model.get_sensor_value(self.cfg.gyro_sensor, data)

    def get_orientation(self, data: mtx.SceneData) -> np.ndarray:
        """Get orientation quaternion from IMU sensor."""
        return self._model.get_sensor_value(self.cfg.orientation_sensor, data)

    def get_ball_position(self, data: mtx.SceneData) -> np.ndarray:
        """Get ball position in world frame. Shape: (num_envs, 3)."""
        return self._ball_body.get_position(data)

    def get_ball_velocity(self, data: mtx.SceneData) -> np.ndarray:
        """Get ball linear velocity in world frame. Shape: (num_envs, 3)."""
        return data.dof_vel[:, self._ball_qvel_indices[:3]]

    # ---- Action application ----

    def apply_action(self, actions: np.ndarray, state: NpEnvState) -> NpEnvState:
        """Apply PD torque control based on actions.

        Actions are 12-dim joint position offsets in [-1, 1].
        Mapped to: target = action * action_scale + default_angle.
        Torque: tau = kp * (target - q) - kd * qd, clipped to torque_limits.
        """
        state.info["last_actions"] = state.info.get("current_actions",
                                                     np.zeros_like(actions))
        state.info["current_actions"] = actions.copy()
        torques = self._compute_torques(actions, state.data)
        state.data.actuator_ctrls = torques
        return state

    def _compute_torques(self, actions: np.ndarray, data: mtx.SceneData) -> np.ndarray:
        """Compute PD torques from actions."""
        cfg = self.cfg
        actions_scaled = actions * cfg.control.action_scale
        dof_pos = self.get_dof_pos(data)
        dof_vel = self.get_dof_vel(data)
        torques = self.kps * (actions_scaled + self.default_angles - dof_pos) - self.kds * dof_vel
        torques = np.clip(torques, -cfg.control.torque_limits, cfg.control.torque_limits)
        return torques.astype(np.float32)

    # ---- State update ----

    def update_state(self, state: NpEnvState) -> NpEnvState:
        """Compute observations, rewards, and termination after physics step."""
        state = self._update_observation(state)
        state = self._update_terminated(state)
        state = self._update_reward(state)
        return state

    # ---- Observation ----

    def _get_obs(self, data: mtx.SceneData, info: dict) -> np.ndarray:
        """Build the 57-dim observation vector."""
        cfg = self.cfg

        # Proprioception
        pose = self._body.get_pose(data)
        base_quat = pose[:, 3:7]  # [x, y, z, w] format
        local_gravity = quaternion.rotate_inverse(base_quat, self.gravity_vec)

        gyro = self.get_gyro(data)
        linvel = self.get_local_linvel(data)

        dof_pos = self.get_dof_pos(data)
        dof_vel = self.get_dof_vel(data)
        joint_diff = dof_pos - self.default_angles

        last_actions = info.get("current_actions", np.zeros((data.shape[0], self._num_action), dtype=np.float32))

        # Ball info
        ball_pos_world = self.get_ball_position(data)
        ball_vel_world = self.get_ball_velocity(data)
        robot_pos_world = pose[:, :3]

        # Ball position relative to robot in body frame
        ball_rel_world = ball_pos_world - robot_pos_world
        ball_rel_body = quaternion.rotate_inverse(base_quat, ball_rel_world)

        # Goal position relative to robot in body frame
        goal_rel_world = self.goal_pos_world - robot_pos_world
        goal_rel_body = quaternion.rotate_inverse(base_quat, goal_rel_world)

        # Ball to goal vector (world frame, normalized)
        ball_to_goal = self.goal_pos_world - ball_pos_world
        ball_to_goal_norm = np.linalg.norm(ball_to_goal, axis=-1, keepdims=True)
        ball_to_goal_normalized = ball_to_goal / np.maximum(ball_to_goal_norm, 1e-8)

        # Concatenate observation
        obs = np.concatenate([
            local_gravity * cfg.obs_scale_gravity,           # 3
            gyro * cfg.obs_scale_gyro,                       # 3
            linvel * cfg.obs_scale_linvel,                   # 3
            joint_diff * cfg.obs_scale_dof_pos,              # 12
            dof_vel * cfg.obs_scale_dof_vel,                 # 12
            last_actions,                                     # 12
            ball_rel_body * cfg.obs_scale_ball_pos,          # 3
            ball_vel_world * cfg.obs_scale_ball_vel,         # 3
            goal_rel_body * cfg.obs_scale_goal_pos,           # 3
            ball_to_goal_normalized,                          # 3
        ], axis=-1).astype(np.float32)

        # Safety: clip to avoid extreme values
        obs = np.clip(obs, -100.0, 100.0)
        # Replace NaN/Inf with 0
        obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

        return obs

    def _update_observation(self, state: NpEnvState) -> NpEnvState:
        """Compute and store observation."""
        obs = self._get_obs(state.data, state.info)
        return state.replace(obs=obs)

    # ---- Termination ----

    def _update_terminated(self, state: NpEnvState) -> NpEnvState:
        """Check termination conditions: goal, out-of-bounds, fall, timeout."""
        data = state.data
        num_envs = data.shape[0]

        ball_pos = self.get_ball_position(data)
        terminated = np.zeros(num_envs, dtype=bool)

        # Goal scored: ball crosses goal line between posts and below crossbar
        goal_scored = self._is_goal(ball_pos)
        terminated = np.logical_or(terminated, goal_scored)
        state.info["goal_scored"] = goal_scored

        # Out of bounds: ball outside field limits
        out_of_bounds = self._is_out_of_bounds(ball_pos)
        terminated = np.logical_or(terminated, out_of_bounds)
        state.info["out_of_bounds"] = out_of_bounds

        # Robot fallen: trunk contacts ground
        if self._num_termination_pairs > 0:
            cquery = self._model.get_contact_query(data)
            trunk_ground_contact = cquery.is_colliding(self._termination_pairs)
            robot_fallen = trunk_ground_contact.any(axis=1)
            terminated = np.logical_or(terminated, robot_fallen)
            state.info["robot_fallen"] = robot_fallen
        else:
            state.info["robot_fallen"] = np.zeros(num_envs, dtype=bool)

        # Additional fall check: base height below threshold
        pose = self._body.get_pose(data)
        base_z = pose[:, 2]
        base_too_low = base_z < 0.3  # trunk height below 0.3m = fallen
        terminated = np.logical_or(terminated, base_too_low)
        state.info["base_too_low"] = base_too_low

        return state.replace(terminated=terminated)

    def _is_goal(self, ball_pos: np.ndarray) -> np.ndarray:
        """Check if ball has entered the goal.

        Goal is at x=goal_x, spanning y in [-goal_width/2, goal_width/2],
        z in [0, goal_height].
        """
        cfg = self.cfg
        x_check = ball_pos[:, 0] >= cfg.goal_x
        y_check = np.abs(ball_pos[:, 1]) <= cfg.goal_width / 2.0
        z_check = ball_pos[:, 2] <= cfg.goal_height
        z_check_ground = ball_pos[:, 2] >= 0.0
        return x_check & y_check & z_check & z_check_ground

    def _is_out_of_bounds(self, ball_pos: np.ndarray) -> np.ndarray:
        """Check if ball is outside the field boundaries."""
        cfg = self.cfg
        x_out = np.abs(ball_pos[:, 0]) > cfg.field_x_limit
        y_out = np.abs(ball_pos[:, 1]) > cfg.field_y_limit
        return x_out | y_out

    # ---- Reward ----

    def _update_reward(self, state: NpEnvState) -> NpEnvState:
        """Compute total reward for current step."""
        data = state.data
        terminated = state.terminated

        rewards = self._compute_rewards(data, state.info)
        total_reward = np.sum(list(rewards.values()), axis=0)

        # Zero reward for terminated envs
        total_reward = np.where(terminated, np.array(0.0, dtype=np.float32), total_reward)
        total_reward = np.nan_to_num(total_reward, nan=0.0, posinf=0.0, neginf=0.0)

        state.info["reward_components"] = {k: v for k, v in rewards.items()}
        return state.replace(reward=total_reward.astype(np.float32))

    def _compute_rewards(self, data: mtx.SceneData, info: dict) -> dict[str, np.ndarray]:
        """Compute all reward components.

        Returns:
            Dict mapping reward name to per-env reward values.
        """
        cfg = self.cfg
        scales = cfg.reward.scales

        rewards: dict[str, np.ndarray] = {}

        # Goal reward (sparse, large positive)
        if "goal" in scales and scales["goal"] != 0:
            rewards["goal"] = scales["goal"] * info.get("goal_scored",
                                                        np.zeros(data.shape[0], dtype=np.float32)).astype(np.float32)

        # Ball toward goal reward
        if "ball_toward_goal" in scales and scales["ball_toward_goal"] != 0:
            rewards["ball_toward_goal"] = scales["ball_toward_goal"] * self._reward_ball_toward_goal(data)

        # Ball speed reward
        if "ball_speed" in scales and scales["ball_speed"] != 0:
            rewards["ball_speed"] = scales["ball_speed"] * self._reward_ball_speed(data)

        # Alive reward (small constant per step)
        if "alive" in scales and scales["alive"] != 0:
            rewards["alive"] = scales["alive"] * np.ones(data.shape[0], dtype=np.float32)

        # Action rate penalty (smoothness)
        if "action_rate" in scales and scales["action_rate"] != 0:
            rewards["action_rate"] = scales["action_rate"] * self._reward_action_rate(info)

        # Joint limit penalty
        if "joint_limit" in scales and scales["joint_limit"] != 0:
            rewards["joint_limit"] = scales["joint_limit"] * self._reward_joint_limit(data)

        return rewards

    def _reward_ball_toward_goal(self, data: mtx.SceneData) -> np.ndarray:
        """Reward ball velocity component toward the goal.

        Returns dot product of ball velocity direction and goal direction,
        clipped to [0, 1] (only reward movement toward goal).
        """
        ball_vel = self.get_ball_velocity(data)
        ball_pos = self.get_ball_position(data)
        goal_dir = self.goal_pos_world - ball_pos
        goal_dir_norm = np.linalg.norm(goal_dir, axis=-1, keepdims=True)
        goal_dir = goal_dir / np.maximum(goal_dir_norm, 1e-8)

        vel_norm = np.linalg.norm(ball_vel, axis=-1, keepdims=True)
        vel_dir = ball_vel / np.maximum(vel_norm, 1e-8)

        dot = np.sum(vel_dir * goal_dir, axis=-1)
        return np.clip(dot, 0.0, 1.0).astype(np.float32)

    def _reward_ball_speed(self, data: mtx.SceneData) -> np.ndarray:
        """Reward ball speed, encouraging a strong kick.

        Uses an exponential: exp(-0.5 * ((target - speed) / sigma)^2).
        """
        cfg = self.cfg
        ball_vel = self.get_ball_velocity(data)
        ball_speed = np.linalg.norm(ball_vel, axis=-1)
        diff = cfg.reward.ball_speed_target - ball_speed
        return np.exp(-0.5 * (diff / cfg.reward.ball_speed_sigma) ** 2).astype(np.float32)

    def _reward_action_rate(self, info: dict) -> np.ndarray:
        """Penalize large changes between consecutive actions.

        Returns negative of squared action difference.
        """
        curr = info.get("current_actions", np.zeros((self._num_envs, self._num_action), dtype=np.float32))
        prev = info.get("last_actions", np.zeros_like(curr))
        return np.sum(np.square(curr - prev), axis=-1).astype(np.float32)

    def _reward_joint_limit(self, data: mtx.SceneData) -> np.ndarray:
        """Penalize joints that exceed safe range.

        Uses joint_limits from the model. For each actuated joint outside
        its range, penalty is proportional to the violation magnitude.
        """
        dof_pos = self.get_dof_pos(data)  # (num_envs, 12)
        # Build per-actuator joint limits from model.joint_limits
        # joint_limits shape: (2, num_joints), indexed by joint index
        pos_low = np.zeros(self._num_action, dtype=np.float32)
        pos_high = np.zeros(self._num_action, dtype=np.float32)
        for ai in range(self._num_action):
            target_joint: str = self._model.actuators[ai].target_name  # type: ignore[attr-defined]
            ji = self._model.get_joint_index(target_joint)
            if ji >= 0:
                pos_low[ai] = self._model.joint_limits[0, ji]
                pos_high[ai] = self._model.joint_limits[1, ji]
        violation_low = np.maximum(pos_low - dof_pos, 0.0)
        violation_high = np.maximum(dof_pos - pos_high, 0.0)
        total_violation = np.sum(violation_low + violation_high, axis=-1)
        return total_violation.astype(np.float32)

    # ---- Reset ----

    def reset(self, data: mtx.SceneData) -> tuple[np.ndarray, dict]:
        """Reset the environment: place robot at start position, ball at penalty spot.

        Args:
            data: SceneData for the environments to reset.

        Returns:
            (obs, info): Initial observations and info dict for reset envs.
        """
        cfg = self.cfg
        num_reset = data.shape[0]

        # Reset physics state
        data.reset(self._model)
        data.set_dof_vel(self._init_dof_vel)
        data.set_dof_pos(self._init_dof_pos, self._model)

        # Set robot base pose
        robot_xy = np.array(cfg.init_state.pos[:2], dtype=np.float32)
        robot_z = cfg.init_state.pos[2]
        if cfg.randomize_robot_position:
            noise = (np.random.uniform(-1, 1, (num_reset, 2)).astype(np.float32)
                     * cfg.robot_randomization_range)
            robot_xy = robot_xy + noise

        robot_qpos = np.zeros((num_reset, 7), dtype=np.float32)
        robot_qpos[:, 0] = robot_xy[:, 0] if cfg.randomize_robot_position else robot_xy[0]
        robot_qpos[:, 1] = robot_xy[:, 1] if cfg.randomize_robot_position else robot_xy[1]
        robot_qpos[:, 2] = robot_z
        robot_qpos[:, 6] = 1.0  # w component of identity quaternion

        # Joint noise
        joint_noise = np.zeros((num_reset, self._num_action), dtype=np.float32)
        joint_noise += np.random.uniform(
            -cfg.init_state.reset_joint_noise,
            cfg.init_state.reset_joint_noise,
            (num_reset, self._num_action),
        ).astype(np.float32)

        # Joint noise applied at global DOF indices for actuator joints
        dof_pos = np.tile(self._init_dof_pos, (num_reset, 1)).copy()
        for ai in range(self._num_action):
            target_joint: str = self._model.actuators[ai].target_name  # type: ignore[attr-defined]
            ji = self._model.get_joint_index(target_joint)
            if ji >= 0:
                jpos_start = self._model.joint_dof_pos_indices[ji]
                dof_pos[:, jpos_start] += joint_noise[:, ai]

        data.set_dof_pos(dof_pos, self._model)

        # Set ball position
        ball_xy = np.array(cfg.penalty_spot[:2], dtype=np.float32)
        ball_z = cfg.penalty_spot[2]
        if cfg.randomize_ball_position:
            noise = (np.random.uniform(-1, 1, (num_reset, 2)).astype(np.float32)
                     * cfg.ball_randomization_range)
            ball_xy = ball_xy + noise

        ball_qpos = np.zeros((num_reset, 7), dtype=np.float32)
        ball_qpos[:, 0] = ball_xy[:, 0] if cfg.randomize_ball_position else ball_xy[0]
        ball_qpos[:, 1] = ball_xy[:, 1] if cfg.randomize_ball_position else ball_xy[1]
        ball_qpos[:, 2] = ball_z
        ball_qpos[:, 6] = 1.0  # w component

        self._ball_body.set_dof_pos(data, ball_qpos)
        ball_vel = np.zeros((num_reset, 6), dtype=np.float32)
        self._ball_body.set_dof_vel(data, ball_vel)

        self._model.forward_kinematic(data)

        info = {
            "current_actions": np.zeros((num_reset, self._num_action), dtype=np.float32),
            "last_actions": np.zeros((num_reset, self._num_action), dtype=np.float32),
            "goal_scored": np.zeros(num_reset, dtype=bool),
            "out_of_bounds": np.zeros(num_reset, dtype=bool),
            "robot_fallen": np.zeros(num_reset, dtype=bool),
            "base_too_low": np.zeros(num_reset, dtype=bool),
        }

        obs = self._get_obs(data, info)
        return obs.astype(np.float32), info
