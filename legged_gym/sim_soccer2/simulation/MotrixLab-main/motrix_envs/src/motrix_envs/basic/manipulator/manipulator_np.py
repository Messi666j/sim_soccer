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
from motrix_envs.basic.manipulator.cfg import BringBallCfg
from motrix_envs.math import quaternion
from motrix_envs.np import reward as reward_utils
from motrix_envs.np.env import NpEnv, NpEnvState

_ARM_JOINTS = (
    "arm_root",
    "arm_shoulder",
    "arm_elbow",
    "arm_wrist",
    "finger",
    "fingertip",
    "thumb",
    "thumbtip",
)
_TOUCH_SENSORS = ("palm_touch", "finger_touch", "thumb_touch", "fingertip_touch", "thumbtip_touch")

_HAND_GEOMS = (
    "hand",
    "palm1",
    "palm2",
    "thumb1",
    "thumb2",
    "thumbtip1",
    "thumbtip2",
    "finger1",
    "finger2",
    "fingertip1",
    "fingertip2",
)


def _sanitize_joint_limits(low: np.ndarray, high: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    low = low.copy()
    high = high.copy()
    low = np.where(np.isfinite(low), low, -np.pi)
    high = np.where(np.isfinite(high), high, np.pi)
    return low.astype(np.float32), high.astype(np.float32)


def _quat_from_y_angle(angle: np.ndarray) -> np.ndarray:
    zeros = np.zeros_like(angle)
    return quaternion.from_euler(zeros, angle, zeros)


def _quat_to_z_axis(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float32)
    return quaternion.rotate_vector(quat, np.array([0.0, 0.0, 1.0], dtype=np.float32)).astype(np.float32)


def _tolerance(
    x: np.ndarray,
    *,
    bounds: tuple[float, float] = (0.0, 0.0),
    margin: float = 0.0,
    sigmoid: str = "gaussian",
    value_at_margin: float = 0.1,
) -> np.ndarray:
    """Vectorized tolerance reward (ported from dm_control-style reward_utils)."""
    return reward_utils.tolerance(x, bounds=bounds, margin=margin, sigmoid=sigmoid, value_at_margin=value_at_margin)


class ManipulatorBase(NpEnv):
    _cfg: BringBallCfg
    _observation_space: gym.spaces.Box
    _action_space: gym.spaces.Box

    def __init__(self, cfg: BringBallCfg, num_envs: int = 1):
        super().__init__(cfg, num_envs=num_envs)
        self._cfg = cfg

        self._joint_limit_low, self._joint_limit_high = _sanitize_joint_limits(*self._model.joint_limits)

        self._arm_joint_pos_indices = np.array([self._joint_pos_index(n) for n in _ARM_JOINTS], dtype=np.int32)
        self._arm_joint_vel_indices = np.array([self._joint_vel_index(n) for n in _ARM_JOINTS], dtype=np.int32)

        self._thumb_qpos_i = self._joint_pos_index("thumb")
        self._finger_qpos_i = self._joint_pos_index("finger")
        self._thumbtip_qpos_i = self._joint_pos_index("thumbtip")
        self._fingertip_qpos_i = self._joint_pos_index("fingertip")

        self._grasp_site = self._model.get_site("grasp")
        # Ensure correct actuator index is retrieved from base model
        self._grasp_act_i = int(self._model.get_actuator_index("grasp"))

        self._object_site = self._model.get_site("ball")
        self._target_site = self._model.get_site("target_ball")
        target_body = self._model.get_body("target_ball")
        if target_body is None:
            raise ValueError("Target body 'target_ball' not found in model")
        self._target_mocap = target_body.mocap

        object_qpos_joints = ("ball_x", "ball_z", "ball_y")
        object_geom_names = ("ball",)

        self._object_qpos_indices = np.array([self._joint_pos_index(n) for n in object_qpos_joints], dtype=np.int32)
        self._object_qvel_indices = np.array([self._joint_vel_index(n) for n in object_qpos_joints], dtype=np.int32)
        self._object_x_qvel_i = int(self._object_qvel_indices[0])

        self._hand_geom_indices = np.array([self._model.get_geom_index(name) for name in _HAND_GEOMS], dtype=np.uint32)
        object_geom_indices = np.array(
            [self._model.get_geom_index(name) for name in object_geom_names], dtype=np.uint32
        )
        self._hand_object_pairs = np.stack(
            [
                np.repeat(self._hand_geom_indices, object_geom_indices.shape[0]),
                np.tile(object_geom_indices, self._hand_geom_indices.shape[0]),
            ],
            axis=-1,
        ).astype(np.uint32)
        self._num_hand_object_pairs = int(self._hand_object_pairs.shape[0])

        self._init_dof_pos = self._model.compute_init_dof_pos().astype(np.float32)
        self._init_dof_vel = np.zeros((self._model.num_dof_vel,), dtype=np.float32)

        self._init_action_space()
        self._init_obs_space()

    def _joint_pos_index(self, joint_name: str) -> int:
        joint_index = self._model.get_joint_index(joint_name)
        return int(self._model.joint_dof_pos_indices[joint_index])

    def _joint_vel_index(self, joint_name: str) -> int:
        joint_index = self._model.get_joint_index(joint_name)
        return int(self._model.joint_dof_vel_indices[joint_index])

    def _init_action_space(self):
        low, high = self._model.actuator_ctrl_limits
        self._action_space = gym.spaces.Box(low, high, (self._model.num_actuators,), dtype=np.float32)

    def _init_obs_space(self):
        # arm_pos(sin,cos)=16 + arm_vel=8 + touch=5 + hand=3 + object=3 + target=3 + rel=3
        self._observation_space = gym.spaces.Box(-np.inf, np.inf, (41,), dtype=np.float32)

    @property
    def observation_space(self) -> gym.spaces.Box:
        return self._observation_space

    @property
    def action_space(self) -> gym.spaces.Box:
        return self._action_space

    def apply_action(self, actions: np.ndarray, state: NpEnvState) -> NpEnvState:
        actions = np.asarray(actions, dtype=np.float32)
        # Enforce actuator control limits to avoid out-of-range impulses.
        actions = np.clip(actions, self._action_space.low, self._action_space.high).astype(np.float32)
        state.info["last_actions"] = state.info["actions"]
        state.info["actions"] = actions
        state.data.actuator_ctrls = actions
        return state

    def _touch_raw(self, data: mtx.SceneData) -> np.ndarray:
        values = []
        for name in _TOUCH_SENSORS:
            v = np.asarray(self._model.get_sensor_value(name, data)).reshape(data.shape[0], -1)[:, 0]
            values.append(v)
        return np.stack(values, axis=-1).astype(np.float32)

    def _touch_log(self, data: mtx.SceneData) -> np.ndarray:
        return np.log1p(self._touch_raw(data))

    def _hand_pos(self, data: mtx.SceneData) -> np.ndarray:
        return self._grasp_site.get_position(data).astype(np.float32)

    def _object_pos(self, data: mtx.SceneData) -> np.ndarray:
        return self._object_site.get_position(data).astype(np.float32)

    def _target_pos(self, data: mtx.SceneData) -> np.ndarray:
        return self._target_site.get_position(data).astype(np.float32)

    def _contact_with_object(self, data: mtx.SceneData) -> np.ndarray:
        cquery = self._model.get_contact_query(data)
        colliding = cquery.is_colliding(self._hand_object_pairs)
        colliding = np.asarray(colliding).reshape((data.shape[0], self._num_hand_object_pairs))
        return colliding.any(axis=-1)

    def _get_obs(self, data: mtx.SceneData) -> np.ndarray:
        qpos = data.dof_pos[:, self._arm_joint_pos_indices]
        arm_pos = np.stack([np.sin(qpos), np.cos(qpos)], axis=-1).reshape(data.shape[0], -1)
        arm_vel = data.dof_vel[:, self._arm_joint_vel_indices]
        touch = self._touch_log(data)

        hand_pos = self._hand_pos(data)
        object_pos = self._object_pos(data)
        target_pos = self._target_pos(data)
        rel = object_pos - target_pos

        obs = np.concatenate([arm_pos, arm_vel, touch, hand_pos, object_pos, target_pos, rel], axis=-1)
        assert obs.shape == (data.shape[0], self._observation_space.shape[0])
        return obs.astype(np.float32)

    def _sample_arm_joint_angles(self, num: int) -> np.ndarray:
        joint_indices = np.array([self._model.get_joint_index(n) for n in _ARM_JOINTS], dtype=np.int32)
        low = self._joint_limit_low[joint_indices]
        high = self._joint_limit_high[joint_indices]
        return np.random.uniform(low=low, high=high, size=(num, joint_indices.shape[0])).astype(np.float32)

    def _sample_target_pose(self, num: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        cfg = self._cfg
        target_x = np.random.uniform(cfg.target_x_range[0], cfg.target_x_range[1], size=(num,)).astype(np.float32)
        target_z = np.random.uniform(cfg.target_z_range[0], cfg.target_z_range[1], size=(num,)).astype(np.float32)
        target_angle = np.random.uniform(cfg.target_angle_range[0], cfg.target_angle_range[1], size=(num,)).astype(
            np.float32
        )
        return target_x, target_z, target_angle

    def _set_target_mocap(
        self,
        data: mtx.SceneData,
        target_x: np.ndarray,
        target_z: np.ndarray,
        target_angle: np.ndarray,
    ):
        pose = np.zeros((data.shape[0], 7), dtype=np.float32)
        pose[:, 0] = target_x
        pose[:, 1] = float(self._cfg.target_y)
        pose[:, 2] = target_z
        pose[:, 3:7] = _quat_from_y_angle(target_angle)
        self._target_mocap.set_pose(data, pose)

    def _set_object_state(
        self,
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
        target_x: np.ndarray,
        target_z: np.ndarray,
        target_angle: np.ndarray,
        grasp_pos: np.ndarray,
    ):
        cfg = self._cfg
        num = dof_pos.shape[0]

        # Default: uniform in workspace.
        object_x = np.random.uniform(cfg.object_x_range[0], cfg.object_x_range[1], size=(num,)).astype(np.float32)
        object_z = np.random.uniform(cfg.object_z_range[0], cfg.object_z_range[1], size=(num,)).astype(np.float32)
        object_angle = np.random.uniform(cfg.object_angle_range[0], cfg.object_angle_range[1], size=(num,)).astype(
            np.float32
        )

        # dm_control-style object init distribution.
        r = np.random.uniform(0.0, 1.0, size=(num,)).astype(np.float32)
        in_hand = r < float(cfg.p_in_hand)
        in_target = (r >= float(cfg.p_in_hand)) & (r < float(cfg.p_in_hand + cfg.p_in_target))
        uniform = ~(in_hand | in_target)

        # Avoid initializing the object too close to the hand to prevent interpenetration / impulse explosions.
        min_dist = float(getattr(cfg, "min_object_hand_dist", 0.0))
        if min_dist > 0.0 and uniform.any():
            min_dist_sq = np.float32(min_dist * min_dist)
            max_attempts = 50
            pending = uniform.copy()
            for _ in range(max_attempts):
                if not pending.any():
                    break
                dx = object_x - grasp_pos[:, 0]
                dz = object_z - grasp_pos[:, 2]
                too_close = (dx * dx + dz * dz) < min_dist_sq
                pending = pending & too_close
                if not pending.any():
                    break
                n = int(pending.sum())
                object_x[pending] = np.random.uniform(cfg.object_x_range[0], cfg.object_x_range[1], size=(n,)).astype(
                    np.float32
                )
                object_z[pending] = np.random.uniform(cfg.object_z_range[0], cfg.object_z_range[1], size=(n,)).astype(
                    np.float32
                )

        object_x[in_target] = target_x[in_target]
        object_z[in_target] = target_z[in_target]
        object_angle[in_target] = target_angle[in_target]

        object_x[in_hand] = grasp_pos[in_hand, 0]
        object_z[in_hand] = grasp_pos[in_hand, 2]
        object_angle[in_hand] = 0.0

        dof_pos[:, self._object_qpos_indices] = np.stack([object_x, object_z, object_angle], axis=-1)

        dof_vel[:, self._object_qvel_indices] = 0.0
        if uniform.any():
            dof_vel[uniform, self._object_x_qvel_i] = np.random.uniform(
                cfg.object_x_vel_range[0], cfg.object_x_vel_range[1], size=(int(uniform.sum()),)
            ).astype(np.float32)

    def _settle(self, data: mtx.SceneData):
        control_steps = int(self._cfg.settle_steps)
        if control_steps <= 0:
            return
        substeps = int(self._cfg.sim_substeps)
        physics_steps = control_steps * max(substeps, 1)
        data.actuator_ctrls = np.zeros((data.shape[0], self._model.num_actuators), dtype=np.float32)
        for _ in range(physics_steps):
            self._model.step(data)
        if self._cfg.settle_zero_vel:
            data.set_dof_vel(np.zeros((data.shape[0], self._model.num_dof_vel), dtype=np.float32))
            self._model.forward_kinematic(data)

    def initialize_episode(self, data: mtx.SceneData) -> None:
        """Episode initialization with optional physics settling (dm_control-style)."""
        num = int(data.shape[0])
        dof_pos = np.tile(self._init_dof_pos, (num, 1))
        dof_vel = np.tile(self._init_dof_vel, (num, 1))

        # Optionally randomize arm joint angles and symmetrize the hand.
        if getattr(self._cfg, "randomize_arm", True):
            arm_angles = self._sample_arm_joint_angles(num)
            dof_pos[:, self._arm_joint_pos_indices] = arm_angles
        dof_pos[:, self._finger_qpos_i] = dof_pos[:, self._thumb_qpos_i]
        dof_pos[:, self._fingertip_qpos_i] = dof_pos[:, self._thumbtip_qpos_i]

        data.reset(self._model)
        data.set_dof_vel(dof_vel)
        data.set_dof_pos(dof_pos, self._model)
        self._model.forward_kinematic(data)

        target_x, target_z, target_angle = self._sample_target_pose(num)
        self._set_target_mocap(data, target_x, target_z, target_angle)
        self._model.forward_kinematic(data)

        grasp_pos = self._grasp_site.get_position(data)
        self._set_object_state(dof_pos, dof_vel, target_x, target_z, target_angle, grasp_pos)
        data.set_dof_vel(dof_vel)
        data.set_dof_pos(dof_pos, self._model)
        self._model.forward_kinematic(data)

        arm_qpos = data.dof_pos[:, self._arm_joint_pos_indices].copy()
        self._settle(data)
        if not getattr(self._cfg, "randomize_arm", True):
            dof_pos_after = data.dof_pos.copy()
            dof_vel_after = data.dof_vel.copy()
            dof_pos_after[:, self._arm_joint_pos_indices] = arm_qpos
            dof_vel_after[:, self._arm_joint_vel_indices] = 0.0
            data.set_dof_pos(dof_pos_after, self._model)
            data.set_dof_vel(dof_vel_after)
            self._model.forward_kinematic(data)

    def reset(self, data: mtx.SceneData) -> tuple[np.ndarray, dict]:
        num = int(data.shape[0])
        self.initialize_episode(data)

        obs = self._get_obs(data)
        info = {
            "actions": np.zeros((num, self._model.num_actuators), dtype=np.float32),
            "last_actions": np.zeros((num, self._model.num_actuators), dtype=np.float32),
        }
        return obs, info


@registry.env("dm-manipulator-bring-ball", "np")
class BringBall(ManipulatorBase):
    _cfg: BringBallCfg

    def __init__(self, cfg: BringBallCfg, num_envs: int = 1):
        super().__init__(cfg, num_envs=num_envs)

        # 1. Sensors setup
        self._fingertip_site = self._model.get_site("fingertip_touch")
        self._thumbtip_site = self._model.get_site("thumbtip_touch")
        self._touch_idx_palm = _TOUCH_SENSORS.index("palm_touch")
        self._touch_idx_fingertip = _TOUCH_SENSORS.index("fingertip_touch")
        self._touch_idx_thumbtip = _TOUCH_SENSORS.index("thumbtip_touch")

    def _compute_hand_direction(self, data: mtx.SceneData) -> np.ndarray:
        """Calculates the Z-axis vector of the hand (grasp site)."""
        grasp_pose = self._grasp_site.get_pose(data)
        return _quat_to_z_axis(grasp_pose[:, 3:])

    def _get_tip_positions(self, data: mtx.SceneData) -> tuple[np.ndarray, np.ndarray]:
        fingertip_pos = self._fingertip_site.get_position(data).astype(np.float32)
        thumbtip_pos = self._thumbtip_site.get_position(data).astype(np.float32)
        return fingertip_pos, thumbtip_pos

    def _compute_aim_direction(self, object_pos: np.ndarray, grasp_pos: np.ndarray) -> np.ndarray:
        vec_to_aim = object_pos - grasp_pos
        dist_to_aim = np.linalg.norm(vec_to_aim, axis=-1, keepdims=True)
        return vec_to_aim / (dist_to_aim + 1e-6)

    def _strict_grasp_condition(self, data: mtx.SceneData, object_pos: np.ndarray) -> np.ndarray:
        cfg = self._cfg
        height_ok = object_pos[:, 2] > float(cfg.lift_height_threshold)
        all_touch = self._touch_raw(data)

        touch_threshold = float(cfg.touch_threshold)

        touch_ok = (
            (all_touch[..., self._touch_idx_palm] > touch_threshold)
            | (all_touch[..., self._touch_idx_fingertip] > touch_threshold)
            | (all_touch[..., self._touch_idx_thumbtip] > touch_threshold)
        )

        object_contact_ok = self._contact_with_object(data)
        return height_ok & touch_ok & object_contact_ok

    def update_state(self, state: NpEnvState) -> NpEnvState:
        data = state.data
        cfg = self._cfg

        # 1. Observation
        obs = self._get_obs(data)
        terminated = np.isnan(obs).any(axis=-1)

        # 2. Positions
        object_pos = self._object_pos(data)
        target_pos = self._target_pos(data)
        grasp_pos = self._hand_pos(data)

        # 3. Kinematics
        fingertip_pos, thumbtip_pos = self._get_tip_positions(data)
        dist_finger = np.linalg.norm(fingertip_pos - object_pos, axis=-1)
        dist_thumb = np.linalg.norm(thumbtip_pos - object_pos, axis=-1)
        avg_tip_dist = ((dist_finger + dist_thumb) / 2.0).astype(np.float32)
        move_dist = np.linalg.norm(object_pos - target_pos, axis=-1).astype(np.float32)

        # 4. Dynamics
        arm_vel = data.dof_vel[:, self._arm_joint_vel_indices[:4]].astype(np.float32)
        arm_speed = np.linalg.norm(arm_vel, axis=-1).astype(np.float32)
        arm_speed_step = (arm_speed * float(cfg.ctrl_dt)).astype(np.float32)

        # 5. Logic Checks
        is_grasped = self._strict_grasp_condition(data, object_pos)
        contact_with_obj = self._contact_with_object(data)
        hover_threshold = float(cfg.hover_close_threshold)
        is_close_to_ball = (avg_tip_dist < hover_threshold).astype(np.float32)
        grasp_mask = is_grasped.astype(np.float32)
        post_grasp_scale = 1.0 - grasp_mask * float(cfg.post_grasp_discount)

        # --- Rewards ---

        # R1: Reach
        r_reach = _tolerance(avg_tip_dist, bounds=(0.0, 0.02), margin=0.25, sigmoid="linear").astype(np.float32)
        r_reach = (r_reach * post_grasp_scale).astype(np.float32)

        # R2: Orient
        hand_dir = self._compute_hand_direction(data)
        unit_vec_to_aim = self._compute_aim_direction(object_pos, grasp_pos)
        pointing_dot = np.sum(hand_dir * unit_vec_to_aim, axis=-1)

        # Dynamic tolerance
        dist_from_base = np.linalg.norm(object_pos[:, :2], axis=-1)
        orient_bound_lower = 0.95 * np.clip(dist_from_base / 0.5, 0.0, 1.0)

        r_orient_raw = 1.0 - orient_bound_lower + pointing_dot
        r_orient = np.clip(r_orient_raw, 0.0, 1.0).astype(np.float32)
        r_orient = (r_orient * post_grasp_scale).astype(np.float32)

        # R3: Pause
        r_pause = (
            _tolerance(arm_speed_step, bounds=(0.0, 0.05), margin=0.3, sigmoid="linear").astype(np.float32)
            * is_close_to_ball
        )
        r_pause = (r_pause * post_grasp_scale).astype(np.float32)

        # R4: Close
        default_actions = np.zeros((data.shape[0], self._model.num_actuators), dtype=np.float32)
        grasp_action = state.info.get("actions", default_actions)[:, self._grasp_act_i].astype(np.float32)
        r_close_intent = _tolerance(
            grasp_action, bounds=(0.8, 1.0), margin=1.0, sigmoid="linear", value_at_margin=0.01
        ).astype(np.float32)

        r_approach_grasp = r_close_intent * is_close_to_ball * r_orient * r_pause * contact_with_obj.astype(np.float32)
        r_sustain_grasp = r_close_intent * grasp_mask
        r_close = (r_approach_grasp * (1.0 - grasp_mask) + r_sustain_grasp).astype(np.float32)

        # R5: Lift & Transport
        lift_h = float(cfg.lift_height_threshold)
        ball_z = object_pos[:, 2].astype(np.float32)
        r_lift_height = (
            _tolerance(ball_z, bounds=(lift_h, lift_h + 0.15), margin=0.02, sigmoid="linear", value_at_margin=0.01)
            * grasp_mask
        ).astype(np.float32)
        r_transport = (_tolerance(move_dist, bounds=(0.0, 0.01), margin=0.3, sigmoid="linear") * grasp_mask).astype(
            np.float32
        )
        r_precision = (
            _tolerance(
                move_dist,
                bounds=(0.0, 0.0),
                margin=float(cfg.precision_margin),
                sigmoid="gaussian",
                value_at_margin=float(cfg.precision_value_at_margin),
            )
            * grasp_mask
        ).astype(np.float32)
        lift_height_weight = float(cfg.lift_height_weight)
        transport_weight = float(cfg.transport_weight)
        lift_norm = max(lift_height_weight + transport_weight, 1e-6)
        r_lift = ((lift_height_weight * r_lift_height + transport_weight * r_transport) / lift_norm).astype(np.float32)

        prev_move_dist = state.info.get("prev_move_dist")
        if prev_move_dist is None:
            prev_move_dist = move_dist
        else:
            prev_move_dist = np.asarray(prev_move_dist, dtype=np.float32)
        if "steps" in state.info:
            first_step = state.info["steps"] == 0
            prev_move_dist = np.where(first_step, move_dist, prev_move_dist)
        progress_clip = float(cfg.transport_progress_clip)
        progress = (prev_move_dist - move_dist) / max(progress_clip, 1e-6)
        progress = np.clip(progress, -1.0, 1.0).astype(np.float32)
        r_progress = (progress * float(cfg.transport_progress_scale) * grasp_mask).astype(np.float32)
        state.info["prev_move_dist"] = move_dist.astype(np.float32)

        # --- Penalties ---
        all_touch = self._touch_raw(data)
        side_touch_sum = (all_touch[..., 1] + all_touch[..., 2]).astype(np.float32)
        penalty_side = (
            -float(cfg.side_penalty_scale) * np.tanh(side_touch_sum * float(cfg.side_penalty_tanh_scale))
        ).astype(np.float32)

        hover_phase = (is_close_to_ball > 0.5) & (~contact_with_obj)
        penalty_hover = (-float(cfg.hover_penalty_scale) * hover_phase.astype(np.float32)).astype(np.float32)

        # --- Total ---
        reach_w = float(cfg.reach_weight)
        orient_w = float(cfg.orient_weight)
        pause_w = float(cfg.pause_weight)
        close_w = float(cfg.close_weight)
        lift_w = float(cfg.lift_reward_weight)
        precision_w = float(cfg.precision_weight)
        weight_sum = max(reach_w + orient_w + pause_w + close_w + lift_w + precision_w, 1e-6)
        reward = (
            (
                reach_w * r_reach
                + orient_w * r_orient
                + pause_w * r_pause
                + close_w * r_close
                + lift_w * r_lift
                + precision_w * r_precision
            )
            / weight_sum
            + penalty_side
            + penalty_hover
            + r_progress
        ).astype(np.float32)

        reward = np.where(terminated, 0.0, reward)

        state.info["Reward"] = {
            "reach": r_reach,
            "orient": r_orient,
            "close": r_close,
            "lift": r_lift,
            "transport": r_transport,
            "precision": r_precision,
            "progress": r_progress,
            "total": reward,
        }

        state.info["metrics"] = {
            "pointing_dot": pointing_dot,
            "is_grasped": is_grasped.astype(np.float32),
            "avg_tip_dist": avg_tip_dist,
            "move_dist": move_dist,
            "transport_reward": r_transport,
            "precision_reward": r_precision,
            "progress_reward": r_progress,
        }

        return state.replace(obs=obs, reward=reward, terminated=terminated)
