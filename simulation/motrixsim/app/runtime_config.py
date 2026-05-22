# SPDX-FileCopyrightText: Copyright (c) MOS-Brain Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import argparse
import re
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import numpy as np

K1_ROBOT_TYPE = "k1"
PI_PLUS_ROBOT_TYPE = "pi_plus"

K1_JOINTS_POLICY_ORDER = [
    "AAHead_yaw",
    "ALeft_Shoulder_Pitch",
    "ARight_Shoulder_Pitch",
    "Left_Hip_Pitch",
    "Right_Hip_Pitch",
    "Head_pitch",
    "Left_Shoulder_Roll",
    "Right_Shoulder_Roll",
    "Left_Hip_Roll",
    "Right_Hip_Roll",
    "Left_Elbow_Pitch",
    "Right_Elbow_Pitch",
    "Left_Hip_Yaw",
    "Right_Hip_Yaw",
    "Left_Elbow_Yaw",
    "Right_Elbow_Yaw",
    "Left_Knee_Pitch",
    "Right_Knee_Pitch",
    "Left_Ankle_Pitch",
    "Right_Ankle_Pitch",
    "Left_Ankle_Roll",
    "Right_Ankle_Roll",
]
PI_PLUS_JOINTS_POLICY_ORDER = [
    "l_hip_pitch_joint",
    "l_shoulder_pitch_joint",
    "r_hip_pitch_joint",
    "r_shoulder_pitch_joint",
    "l_hip_roll_joint",
    "l_shoulder_roll_joint",
    "r_hip_roll_joint",
    "r_shoulder_roll_joint",
    "l_thigh_joint",
    "l_upper_arm_joint",
    "r_thigh_joint",
    "r_upper_arm_joint",
    "l_calf_joint",
    "l_elbow_joint",
    "r_calf_joint",
    "r_elbow_joint",
    "l_ankle_pitch_joint",
    "r_ankle_pitch_joint",
    "l_ankle_roll_joint",
    "r_ankle_roll_joint",
]

PI_PLUS_KP_POLICY_ORDER = [
    80.0, 80.0, 80.0, 80.0, 60.0, 60.0, 30.0, 30.0, 30.0, 30.0,
    80.0, 80.0, 80.0, 80.0, 60.0, 60.0, 30.0, 30.0, 30.0, 30.0,
]

PI_PLUS_KD_POLICY_ORDER = [
    1.1, 1.1, 1.1, 1.1, 1.2, 1.2, 0.6, 0.6, 0.6, 0.6,
    1.1, 1.1, 1.1, 1.1, 1.2, 1.2, 0.6, 0.6, 0.6, 0.6,
]

OBS_TERMS_ORDER = [
    "base_lin_vel",
    "base_ang_vel",
    "gravity_orientation",
    "cmd",
    "joint_pos",
    "joint_vel",
    "last_action",
]

OBS_SCALE = {
    "base_lin_vel": 1.0,
    "base_ang_vel": 0.2,
    "gravity_orientation": 1.0,
    "cmd": 1.0,
    "joint_pos": 1.0,
    "joint_vel": 0.05,
    "last_action": 1.0,
}

K1_ACTION_SCALE = {
    ".*Head.*": 0.375,
    ".*Shoulder_Pitch": 0.875,
    ".*Shoulder_Roll": 0.875,
    ".*Elbow_Pitch": 0.875,
    ".*Elbow_Yaw": 0.875,
    ".*Hip_Pitch": 0.09375,
    ".*Hip_Roll": 0.109375,
    ".*Hip_Yaw": 0.0625,
    ".*Knee_Pitch": 0.125,
    ".*Ankle_Pitch": 1.0 / 6.0,
    ".*Ankle_Roll": 1.0 / 6.0,
}

PI_PLUS_ACTION_SCALE = {
    ".*": 0.25,
}

K1_MOTOR_EFFORT_LIMIT = {
    ".*Head.*": 6.0,
    ".*Shoulder.*": 14.0,
    ".*Elbow.*": 14.0,
    ".*Hip_Pitch": 30.0,
    ".*Hip_Roll": 35.0,
    ".*Hip_Yaw": 20.0,
    ".*Knee_Pitch": 40.0,
    ".*Ankle_.*": 20.0,
}

PI_PLUS_MOTOR_EFFORT_LIMIT = 20.0

K1_MOTOR_STIFFNESS = {
    ".*Head.*": 4.0,
    ".*Shoulder.*": 4.0,
    ".*Elbow.*": 4.0,
    ".*Hip_Pitch": 80.0,
    ".*Hip_Roll": 80.0,
    ".*Hip_Yaw": 80.0,
    ".*Knee_Pitch": 80.0,
    ".*Ankle_.*": 30.0,
}

PI_PLUS_MOTOR_STIFFNESS = {
    ".*l_hip_pitch_joint$": 80.0,
    ".*l_shoulder_pitch_joint$": 80.0,
    ".*r_hip_pitch_joint$": 80.0,
    ".*r_shoulder_pitch_joint$": 80.0,
    ".*l_hip_roll_joint$": 60.0,
    ".*l_shoulder_roll_joint$": 60.0,
    ".*r_hip_roll_joint$": 60.0,
    ".*r_shoulder_roll_joint$": 60.0,
    ".*l_thigh_joint$": 30.0,
    ".*l_upper_arm_joint$": 30.0,
    ".*r_thigh_joint$": 30.0,
    ".*r_upper_arm_joint$": 30.0,
    ".*l_calf_joint$": 80.0,
    ".*l_elbow_joint$": 80.0,
    ".*r_calf_joint$": 80.0,
    ".*r_elbow_joint$": 80.0,
    ".*l_ankle_pitch_joint$": 60.0,
    ".*r_ankle_pitch_joint$": 60.0,
    ".*l_ankle_roll_joint$": 30.0,
    ".*r_ankle_roll_joint$": 30.0,
}

K1_MOTOR_DAMPING = {
    ".*Head.*": 1.0,
    ".*Shoulder.*": 1.0,
    ".*Elbow.*": 1.0,
    ".*Hip_Pitch": 2.0,
    ".*Hip_Roll": 2.0,
    ".*Hip_Yaw": 2.0,
    ".*Knee_Pitch": 2.0,
    ".*Ankle_.*": 2.0,
}

PI_PLUS_MOTOR_DAMPING = {
    ".*l_hip_pitch_joint$": 1.1,
    ".*l_shoulder_pitch_joint$": 1.1,
    ".*r_hip_pitch_joint$": 1.1,
    ".*r_shoulder_pitch_joint$": 1.1,
    ".*l_hip_roll_joint$": 1.2,
    ".*l_shoulder_roll_joint$": 1.2,
    ".*r_hip_roll_joint$": 1.2,
    ".*r_shoulder_roll_joint$": 1.2,
    ".*l_thigh_joint$": 0.6,
    ".*l_upper_arm_joint$": 0.6,
    ".*r_thigh_joint$": 0.6,
    ".*r_upper_arm_joint$": 0.6,
    ".*l_calf_joint$": 1.1,
    ".*l_elbow_joint$": 1.1,
    ".*r_calf_joint$": 1.1,
    ".*r_elbow_joint$": 1.1,
    ".*l_ankle_pitch_joint$": 1.2,
    ".*r_ankle_pitch_joint$": 1.2,
    ".*l_ankle_roll_joint$": 0.6,
    ".*r_ankle_roll_joint$": 0.6,
}

# Backward-compatible aliases used by build_sim2sim_cfg.
MOTOR_EFFORT_LIMIT = K1_MOTOR_EFFORT_LIMIT
MOTOR_STIFFNESS = K1_MOTOR_STIFFNESS
MOTOR_DAMPING = K1_MOTOR_DAMPING

PITCH_SCALE = 0.45
SIM_DT = 0.005
CONTROL_DECIMATION = 4
ACTION_CLIP = (-100.0, 100.0)
ACTION_SMOOTH_FILTER = False
DEFAULT_CMD = [0.0, 0.0, 0.0]
USE_BODY_VEL_OBS = True
RELOCATION_HOLD_SEC = 0.5
MAX_ROBOTS_PER_TEAM = 7
DEFAULT_POS = np.array([-3.5, 0.0, 0.52], dtype=np.float32)
SLOWDOWN_FACTOR = 1.0

K1_RESET_JOINT_POS = {
    "Left_Shoulder_Roll": -1.3,
    "Right_Shoulder_Roll": 1.3,
}

# --- legged_gym K1 / T1 locomotion (47 obs, 12 leg actions) — see legged_gym/envs/T1/T1.py ---
K1_LEGGED_GYM_NUM_OBS = 47
K1_LEGGED_GYM_NUM_ACT = 12
# ZMQ/Web command_buffer: [vx, vy, yaw_rate] in SI-ish units after set_command clip; K1 obs[6:9] matches T1 (no [0,1] remap).
CMD_VEL_NORM_MIN = -1.0
CMD_VEL_NORM_MAX = 1.0
# Legacy full-body K1 policy (k1_model_46000.pt): base_lin + ang + gravity + cmd + joints + vels + last_a
K1_FULL_BODY_NUM_OBS = 78
K1_FULL_BODY_NUM_ACT = len(K1_JOINTS_POLICY_ORDER)
K1_LEGGED_GYM_GAIT_FREQUENCY = 1.5
# K1 AMP (legged_gym/envs/K1_amp): stacked proprio obs, full-body actions — see legged_gym_1.zip
K1_AMP_FRAME_STACK = 5
K1_AMP_NUM_SINGLE_OBS = 75
K1_AMP_NUM_OBS = K1_AMP_NUM_SINGLE_OBS * K1_AMP_FRAME_STACK
K1_AMP_NUM_ACT = 22
K1_AMP_ACTION_SCALE = 0.25
# Must match K1_22dof.xml <motor> order (training uses enumerate(model.actuators)).
K1_AMP_ACTUATOR_JOINT_ORDER = [
    "AAHead_yaw",
    "Head_pitch",
    "ALeft_Shoulder_Pitch",
    "Left_Shoulder_Roll",
    "Left_Elbow_Pitch",
    "Left_Elbow_Yaw",
    "ARight_Shoulder_Pitch",
    "Right_Shoulder_Roll",
    "Right_Elbow_Pitch",
    "Right_Elbow_Yaw",
    "Left_Hip_Pitch",
    "Left_Hip_Roll",
    "Left_Hip_Yaw",
    "Left_Knee_Pitch",
    "Left_Ankle_Pitch",
    "Left_Ankle_Roll",
    "Right_Hip_Pitch",
    "Right_Hip_Roll",
    "Right_Hip_Yaw",
    "Right_Knee_Pitch",
    "Right_Ankle_Pitch",
    "Right_Ankle_Roll",
]
K1_AMP_DEFAULT_JOINT_ANGLES = {
    "AAHead_yaw": 0.0,
    "Head_pitch": 0.0,
    "ALeft_Shoulder_Pitch": 0.0,
    "Left_Shoulder_Roll": -1.3,
    "Left_Elbow_Pitch": 0.0,
    "Left_Elbow_Yaw": -0.5,
    "ARight_Shoulder_Pitch": 0.0,
    "Right_Shoulder_Roll": 1.3,
    "Right_Elbow_Pitch": 0.0,
    "Right_Elbow_Yaw": 0.5,
    "Left_Hip_Pitch": -0.15,
    "Left_Hip_Roll": 0.0,
    "Left_Hip_Yaw": 0.0,
    "Left_Knee_Pitch": 0.3,
    "Left_Ankle_Pitch": -0.15,
    "Left_Ankle_Roll": 0.0,
    "Right_Hip_Pitch": -0.15,
    "Right_Hip_Roll": 0.0,
    "Right_Hip_Yaw": 0.0,
    "Right_Knee_Pitch": 0.3,
    "Right_Ankle_Pitch": -0.15,
    "Right_Ankle_Roll": 0.0,
}
K1_AMP_CMD_MAX_LIN_X = 0.5
K1_AMP_CMD_MAX_LIN_Y = 0.4
K1_AMP_CMD_MAX_YAW = 1.0
# Joint order for obs[11:35] and policy actions: must match the order of the *12 leg actuators*
# in the training MJCF (MotrixSim: same index order as `enumerate(model.actuators)` on the loco model).
# Soccer `K1_22dof.xml` lists leg <motor> as: full left chain, then full right chain — not L/R interleaved.
# If your ONNX was trained on a different MJCF ordering, permute this list to that asset's leg actuator order.
K1_LEGGED_GYM_LEG_JOINTS_POLICY_ORDER = [
    "Left_Hip_Pitch",
    "Left_Hip_Roll",
    "Left_Hip_Yaw",
    "Left_Knee_Pitch",
    "Left_Ankle_Pitch",
    "Left_Ankle_Roll",
    "Right_Hip_Pitch",
    "Right_Hip_Roll",
    "Right_Hip_Yaw",
    "Right_Knee_Pitch",
    "Right_Ankle_Pitch",
    "Right_Ankle_Roll",
]
# legged_gym T1Cfg.init_state.default_joint_angles (K1Cfg inherits; unspecified legs -> "default").
K1_LEGGED_GYM_DEFAULT_JOINT_ANGLES = {
    "Left_Hip_Pitch": -0.2,
    "Right_Hip_Pitch": -0.2,
    "Left_Knee_Pitch": 0.4,
    "Right_Knee_Pitch": 0.4,
    "Left_Ankle_Pitch": -0.25,
    "Right_Ankle_Pitch": -0.25,
    "default": 0.0,
}
# Training observation scalars (legged_gym/envs/T1/T1_config.py + K1 dof_vel override).
K1_LEGGED_GYM_GRAVITY_SCALE = 1.0
K1_LEGGED_GYM_GYRO_SCALE_LIN_VEL = 1.0  # T1 uses obs_scales.lin_vel for gyro (same as legged_gym).
K1_LEGGED_GYM_DOF_POS_SCALE = 1.0
K1_LEGGED_GYM_DOF_VEL_SCALE = 0.1  # K1Cfg.normalization.obs_scales.dof_vel
# PD and limits — legged_gym/envs/T1/T1_config.py (K1Cfg uses same stiffness/damping dict).
K1_LEGGED_GYM_ACTION_SCALE = 1.0
K1_LEGGED_GYM_TORQUE_LIMIT = 40.0
K1_LEGGED_GYM_KP = {
    "Left_Hip_Pitch": 200.0,
    "Right_Hip_Pitch": 200.0,
    "Left_Hip_Roll": 200.0,
    "Right_Hip_Roll": 200.0,
    "Left_Hip_Yaw": 200.0,
    "Right_Hip_Yaw": 200.0,
    "Left_Knee_Pitch": 200.0,
    "Right_Knee_Pitch": 200.0,
    "Left_Ankle_Pitch": 50.0,
    "Right_Ankle_Pitch": 50.0,
    "Left_Ankle_Roll": 50.0,
    "Right_Ankle_Roll": 50.0,
}
K1_LEGGED_GYM_KD = {
    "Left_Hip_Pitch": 5.0,
    "Right_Hip_Pitch": 5.0,
    "Left_Hip_Roll": 5.0,
    "Right_Hip_Roll": 5.0,
    "Left_Hip_Yaw": 5.0,
    "Right_Hip_Yaw": 5.0,
    "Left_Knee_Pitch": 5.0,
    "Right_Knee_Pitch": 5.0,
    "Left_Ankle_Pitch": 1.0,
    "Right_Ankle_Pitch": 1.0,
    "Left_Ankle_Roll": 1.0,
    "Right_Ankle_Roll": 1.0,
}

PI_PLUS_RESET_JOINT_POS = {
    "l_hip_pitch_joint": -0.25,
    "l_shoulder_pitch_joint": 0.0,
    "r_hip_pitch_joint": -0.25,
    "r_shoulder_pitch_joint": 0.0,
    "l_hip_roll_joint": 0.0,
    "l_shoulder_roll_joint": 0.2,
    "r_hip_roll_joint": 0.0,
    "r_shoulder_roll_joint": -0.2,
    "l_thigh_joint": 0.0,
    "l_upper_arm_joint": 0.0,
    "r_thigh_joint": 0.0,
    "r_upper_arm_joint": 0.0,
    "l_calf_joint": 0.65,
    "l_elbow_joint": -1.2,
    "r_calf_joint": 0.65,
    "r_elbow_joint": -1.2,
    "l_ankle_pitch_joint": -0.4,
    "r_ankle_pitch_joint": -0.4,
    "l_ankle_roll_joint": 0.0,
    "r_ankle_roll_joint": 0.0,
}

FIXED_ROBOT_ID_TO_NAME = {
    **{i: f"robot_rp{i}" for i in range(MAX_ROBOTS_PER_TEAM)},
    **{MAX_ROBOTS_PER_TEAM + i: f"robot_bp{i}" for i in range(MAX_ROBOTS_PER_TEAM)},
}
FIXED_ROBOT_NAME_TO_ID = {name: rid for rid, name in FIXED_ROBOT_ID_TO_NAME.items()}


@dataclass(frozen=True)
class RobotRuntimeConfig:
    robot_type: str
    policy: Path
    robot_xml: Path
    policy_joint_names: list[str]
    action_scale_cfg: dict[str, float]
    motor_effort_limit: float | dict[str, float]
    motor_stiffness: float | dict[str, float]
    motor_damping: float | dict[str, float]
    reset_joint_pos: dict[str, float]
    include_base_lin_vel_obs: bool
    obs_history_length: int
    obs_clip: float
    obs_scale: dict[str, float]
    cmd_clip: tuple[float, float, float] | None
    base_joint_name: str
    sim_dt: float
    control_decimation: int
    use_k1_legged_gym_policy: bool = False
    k1_stand_policy: Path | None = None
    use_k1_amp_onnx: bool = False


@dataclass
class RuntimeArgs:
    robot_type: str
    robot_cfg: RobotRuntimeConfig
    policy: Path
    robot_xml: Path
    soccer_world_xml: Path
    match_config: Path
    webview: bool
    zmq: bool
    webview_port: int
    web_fps: int
    web_width: int
    web_height: int
    web_jpeg_quality: int
    web_jpeg_subsampling: int
    render_collision_meshes: bool
    allow_keyboard_control: bool
    port: int
    team_size: int
    max_red_robots: int
    max_blue_robots: int
    use_referee: bool
    policy_device: str
    real_time: bool


def _clamp_team_count(v: int) -> int:
    return max(0, min(MAX_ROBOTS_PER_TEAM, int(v)))


def _normalize_robot_type(v: str) -> str:
    k = str(v).strip().lower().replace("-", "_")
    if k in ("k1",):
        return K1_ROBOT_TYPE
    if k in ("pi_plus", "piplus"):
        return PI_PLUS_ROBOT_TYPE
    raise ValueError(f"Unsupported robot type: {v}")


def infer_k1_pt_policy_io(path: Path) -> tuple[int, int] | None:
    """Detect (obs_dim, act_dim) for K1 *.pt: TorchScript module or actor state-dict checkpoint."""
    import torch

    if not path.is_file() or path.suffix.lower() != ".pt":
        return None
    try:
        m = torch.jit.load(str(path), map_location="cpu")
        m.eval()
        with torch.inference_mode():
            for n_in in (47, 78):
                try:
                    y = m(torch.zeros(1, n_in))
                    act = int(y.shape[-1])
                    if act > 0:
                        return (n_in, act)
                except Exception:
                    continue
    except Exception:
        pass
    try:
        ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    except Exception:
        return None
    if not isinstance(ckpt, dict):
        return None
    sd = ckpt.get("model_state_dict", ckpt)
    if not isinstance(sd, dict):
        return None
    actor_keys = sorted(
        (k for k in sd if re.match(r"^actor\.\d+\.weight$", k)),
        key=lambda s: int(s.split(".")[1]),
    )
    if not actor_keys:
        return None
    w0 = sd[actor_keys[0]]
    wn = sd[actor_keys[-1]]
    return (int(w0.shape[1]), int(wn.shape[0]))


def infer_k1_onnx_io(path: Path) -> tuple[int, int] | None:
    """Return (obs_dim, act_dim) for a 1-input / 1-output K1 ONNX policy."""
    try:
        import onnxruntime as ort
    except ImportError:
        return None
    if not path.is_file() or path.suffix.lower() != ".onnx":
        return None
    try:
        s = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        ins = s.get_inputs()
        outs = s.get_outputs()
        if len(ins) != 1 or len(outs) != 1:
            return None
        sh = ins[0].shape
        osh = outs[0].shape
        if len(sh) < 2 or len(osh) < 2:
            return None
        d_in = sh[-1]
        d_out = osh[-1]
        if not isinstance(d_in, int) or not isinstance(d_out, int) or d_in <= 0 or d_out <= 0:
            return None
        return (d_in, d_out)
    except Exception:
        return None


def build_robot_runtime_config(
    mujoco_dir: Path,
    *,
    robot_type: str,
    policy_override: Path | None,
    robot_xml_override: Path | None,
    use_k1_legged_gym: bool = True,
) -> RobotRuntimeConfig:
    rt = _normalize_robot_type(robot_type)
    if rt == K1_ROBOT_TYPE:
        repo_root = mujoco_dir.parent.parent
        default_pt_policy = mujoco_dir / "assets" / "policies" / "k1_model_46000.pt"
        default_legged_onnx_preferred = repo_root / "model_20000_new.onnx"
        default_legged_torch = repo_root / "model_4700.pt"
        default_legged_onnx = repo_root / "legged_gym" / "policy" / "booster_k1" / "model_4700.onnx"
        want_legged = bool(use_k1_legged_gym)
        use_k1_amp_onnx = False
        policy_path = Path(policy_override) if policy_override is not None else None

        if policy_path is not None:
            suf = policy_path.suffix.lower()
            if suf == ".onnx":
                od = infer_k1_onnx_io(policy_path)
                if od == (K1_AMP_NUM_OBS, K1_AMP_NUM_ACT):
                    use_k1_amp_onnx = True
                    want_legged = False
                elif od == (K1_LEGGED_GYM_NUM_OBS, K1_LEGGED_GYM_NUM_ACT):
                    want_legged = True
                elif od is None:
                    raise ValueError(f"Cannot read ONNX I/O for {policy_path} (need onnxruntime).")
                else:
                    raise ValueError(
                        f"Unsupported K1 ONNX shape {od[0]}->{od[1]} for {policy_path.name}; "
                        f"expected {K1_LEGGED_GYM_NUM_OBS}->{K1_LEGGED_GYM_NUM_ACT} (locomotion) or "
                        f"{K1_AMP_NUM_OBS}->{K1_AMP_NUM_ACT} (K1_amp stack)."
                    )
            elif suf == ".pt":
                dims_pt = infer_k1_pt_policy_io(policy_path)
                if dims_pt == (K1_LEGGED_GYM_NUM_OBS, K1_LEGGED_GYM_NUM_ACT):
                    if not use_k1_legged_gym:
                        raise ValueError(
                            f"{policy_path.name} is a {K1_LEGGED_GYM_NUM_OBS}\u2192{K1_LEGGED_GYM_NUM_ACT} legged policy; "
                            "remove --no-k1-legged-gym or pass a full-body checkpoint instead."
                        )
                    want_legged = True
                else:
                    want_legged = False

        k1_stand_policy: Path | None = None
        policy_final: Path | None = None
        if use_k1_amp_onnx:
            policy_final = policy_path
            if policy_final is None or not policy_final.is_file():
                raise FileNotFoundError("K1 AMP ONNX requires a valid --policy path.")
        elif want_legged:
            if policy_path is not None:
                policy_final = policy_path
            elif default_legged_onnx_preferred.is_file():
                policy_final = default_legged_onnx_preferred
                od = infer_k1_onnx_io(policy_final)
                if od == (K1_AMP_NUM_OBS, K1_AMP_NUM_ACT):
                    use_k1_amp_onnx = True
                    want_legged = False
            if not use_k1_amp_onnx:
                if policy_final is None and default_legged_torch.is_file():
                    policy_final = default_legged_torch
                if policy_final is None and default_legged_onnx.is_file():
                    policy_final = default_legged_onnx
                if policy_final is None:
                    print(
                        "[RobotRuntimeConfig] K1 legged default policy not found: "
                        f"{default_legged_onnx_preferred} or {default_legged_torch} or {default_legged_onnx}; "
                        f"falling back to {default_pt_policy.name}"
                    )
                    want_legged = False
                    policy_final = default_pt_policy
            if use_k1_amp_onnx:
                k1_stand_policy = None
            elif want_legged and not policy_final.is_file():
                raise FileNotFoundError(f"K1 legged policy not found: {policy_final}")
            if want_legged and not use_k1_amp_onnx and default_pt_policy.is_file():
                d_stand = infer_k1_pt_policy_io(default_pt_policy)
                if d_stand == (K1_FULL_BODY_NUM_OBS, K1_FULL_BODY_NUM_ACT):
                    k1_stand_policy = default_pt_policy
                elif d_stand is not None:
                    print(
                        "[RobotRuntimeConfig] K1 stand policy skipped "
                        f"({default_pt_policy.name} is {d_stand[0]}->{d_stand[1]}, "
                        f"expected {K1_FULL_BODY_NUM_OBS}->{K1_FULL_BODY_NUM_ACT})."
                    )
        else:
            policy_final = policy_path if policy_path is not None else default_pt_policy

        assert policy_final is not None
        return RobotRuntimeConfig(
            robot_type=K1_ROBOT_TYPE,
            policy=policy_final,
            robot_xml=robot_xml_override or (mujoco_dir / "assets" / "robots" / "k1" / "K1_22dof.xml"),
            policy_joint_names=K1_JOINTS_POLICY_ORDER,
            action_scale_cfg=K1_ACTION_SCALE,
            motor_effort_limit=K1_MOTOR_EFFORT_LIMIT,
            motor_stiffness=K1_MOTOR_STIFFNESS,
            motor_damping=K1_MOTOR_DAMPING,
            reset_joint_pos=K1_RESET_JOINT_POS,
            include_base_lin_vel_obs=not want_legged and not use_k1_amp_onnx,
            obs_history_length=1,
            obs_clip=100.0,
            obs_scale=OBS_SCALE,
            cmd_clip=None,
            base_joint_name="world_joint",
            sim_dt=0.005,
            control_decimation=4,
            use_k1_legged_gym_policy=want_legged and not use_k1_amp_onnx,
            k1_stand_policy=None if use_k1_amp_onnx else k1_stand_policy,
            use_k1_amp_onnx=use_k1_amp_onnx,
        )
    return RobotRuntimeConfig(
        robot_type=PI_PLUS_ROBOT_TYPE,
        policy=policy_override or (mujoco_dir / "assets" / "policies" / "pi_plus_model_40000.pt"),
        robot_xml=robot_xml_override or (mujoco_dir / "assets" / "robots" / "pi_plus" / "pi_plus.xml"),
        policy_joint_names=PI_PLUS_JOINTS_POLICY_ORDER,
        action_scale_cfg=PI_PLUS_ACTION_SCALE,
        motor_effort_limit=PI_PLUS_MOTOR_EFFORT_LIMIT,
        motor_stiffness=PI_PLUS_MOTOR_STIFFNESS,
        motor_damping=PI_PLUS_MOTOR_DAMPING,
        reset_joint_pos=PI_PLUS_RESET_JOINT_POS,
        include_base_lin_vel_obs=False,
        obs_history_length=5,
        obs_clip=100.0,
        obs_scale={
            "base_lin_vel": 1.0,
            "base_ang_vel": 1.0,
            "gravity_orientation": 1.0,
            "cmd": 1.0,
            "joint_pos": 1.0,
            "joint_vel": 1.0,
            "last_action": 1.0,
        },
        cmd_clip=(1.5, 1.0, 3.0),
        base_joint_name="floating_base_joint",
        sim_dt=0.002,
        control_decimation=10,
        use_k1_legged_gym_policy=False,
        k1_stand_policy=None,
        use_k1_amp_onnx=False,
    )


def parse_runtime_args(mujoco_dir: Path) -> RuntimeArgs:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--robot-type",
        type=str,
        default=K1_ROBOT_TYPE,
        help="Robot type. Supported: k1, pi_plus",
    )
    parser.add_argument("--policy", type=Path, default=None)
    parser.add_argument(
        "--robot-xml",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--soccer-world-xml",
        type=Path,
        default=mujoco_dir / "assets" / "environments" / "soccer" / "world.xml",
    )
    parser.add_argument(
        "--match-config",
        type=Path,
        default=mujoco_dir / "assets" / "config" / "match_config.json",
    )
    parser.add_argument("--team-size", type=int, default=1, help="Robots per team (0-7). Red/Blue are equal.")
    parser.add_argument("--webview", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--zmq", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--webview-port", type=int, default=5811)
    parser.add_argument("--web-fps", type=int, default=20)
    parser.add_argument("--web-width", type=int, default=1280)
    parser.add_argument("--web-height", type=int, default=720)
    parser.add_argument(
        "--web-jpeg-quality",
        type=int,
        default=82,
        help="WebView JPEG quality (1-95). Lower is faster and smaller.",
    )
    parser.add_argument(
        "--web-jpeg-subsampling",
        type=int,
        choices=[0, 1, 2],
        default=2,
        help="WebView JPEG chroma subsampling: 0=best/slowest (4:4:4), 2=fastest (4:2:0).",
    )
    parser.add_argument(
        "--render-collision-meshes",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Render collision geoms instead of visual geoms in the MuJoCo web viewer.",
    )
    parser.add_argument("--allow-keyboard-control", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--port", type=int, default=5555, help="ZeroMQ REP port.")
    parser.add_argument("--use-referee", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--real-time",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run simulation in real-time pace. Default false: run as fast as possible.",
    )
    parser.add_argument(
        "--policy-device",
        type=str,
        choices=["cpu", "gpu"],
        default="gpu",
        help="Policy inference device. If set to gpu but CUDA is unavailable, falls back to CPU.",
    )
    parser.add_argument(
        "--k1-legged-gym",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="K1 only (default on): 47-dim obs + 12-dim leg policy for locomotion; "
        "when k1_model_46000.pt is present, standstill uses that full-body policy (78->22) with command hysteresis. "
        "Default walk policy order: <repo>/model_20000_new.onnx, else model_4700.pt, else legged_gym/.../model_4700.onnx. "
        "Use --no-k1-legged-gym for legacy k1_model_46000.pt only. "
        "--policy overrides walk policy; .pt files are auto-classified (47->12 legged vs 78->22 full body).",
    )
    ns = parser.parse_args()
    team_size = _clamp_team_count(ns.team_size)
    robot_cfg = build_robot_runtime_config(
        mujoco_dir,
        robot_type=ns.robot_type,
        policy_override=ns.policy,
        robot_xml_override=ns.robot_xml,
        use_k1_legged_gym=ns.k1_legged_gym,
    )
    return RuntimeArgs(
        robot_type=robot_cfg.robot_type,
        robot_cfg=robot_cfg,
        policy=robot_cfg.policy,
        robot_xml=robot_cfg.robot_xml,
        soccer_world_xml=ns.soccer_world_xml,
        match_config=ns.match_config,
        webview=ns.webview,
        zmq=ns.zmq,
        webview_port=ns.webview_port,
        web_fps=ns.web_fps,
        web_width=ns.web_width,
        web_height=ns.web_height,
        web_jpeg_quality=ns.web_jpeg_quality,
        web_jpeg_subsampling=ns.web_jpeg_subsampling,
        render_collision_meshes=ns.render_collision_meshes,
        allow_keyboard_control=ns.allow_keyboard_control,
        port=ns.port,
        team_size=team_size,
        max_red_robots=team_size,
        max_blue_robots=team_size,
        use_referee=ns.use_referee,
        policy_device=ns.policy_device,
        real_time=ns.real_time,
    )


def build_action_scale_array(policy_joint_names: list[str], scale_cfg: dict[str, float]) -> np.ndarray:
    scales = np.zeros(len(policy_joint_names), dtype=np.float32)
    for i, joint_name in enumerate(policy_joint_names):
        for pattern, val in scale_cfg.items():
            if re.match(pattern, joint_name):
                scales[i] = float(val)
                break
        if scales[i] == 0.0:
            raise ValueError(f"No action scale matched for joint: {joint_name}")
    return scales


def parse_param_for_joint_names(joint_names: list[str], param: float | dict[str, float]) -> np.ndarray:
    out = np.zeros(len(joint_names), dtype=np.float32)
    if isinstance(param, (float, int)):
        out.fill(float(param))
        return out
    if not isinstance(param, dict):
        raise ValueError(f"Unsupported parameter type: {type(param)}")
    for i, name in enumerate(joint_names):
        matched = False
        for pattern, value in param.items():
            if re.match(pattern, name):
                out[i] = float(value)
                matched = True
                break
        if not matched:
            out[i] = 1e-7
    return out


def _ensure_trackerlab_stub() -> None:
    if "trackerLab.managers.motion_manager.motion_manager_cfg" in sys.modules:
        return
    trackerlab_pkg = sys.modules.setdefault("trackerLab", types.ModuleType("trackerLab"))
    managers_pkg = sys.modules.setdefault("trackerLab.managers", types.ModuleType("trackerLab.managers"))
    motion_pkg = sys.modules.setdefault("trackerLab.managers.motion_manager", types.ModuleType("trackerLab.managers.motion_manager"))
    motion_cfg_mod = types.ModuleType("trackerLab.managers.motion_manager.motion_manager_cfg")

    class MotionManagerCfg:
        pass

    motion_cfg_mod.MotionManagerCfg = MotionManagerCfg
    sys.modules["trackerLab.managers.motion_manager.motion_manager_cfg"] = motion_cfg_mod
    trackerlab_pkg.managers = managers_pkg
    managers_pkg.motion_manager = motion_pkg


def build_sim2sim_cfg(scene_xml: Path, policy_path: Path):
    _ensure_trackerlab_stub()
    from sim2simlib.model.actuator_motor import PIDMotor
    from sim2simlib.model.config import Actions_Config, Motor_Config, Observations_Config, Sim2Sim_Config

    return Sim2Sim_Config(
        robot_name="k1",
        simulation_dt=SIM_DT,
        control_decimation=CONTROL_DECIMATION,
        slowdown_factor=SLOWDOWN_FACTOR,
        xml_path=str(scene_xml),
        policy_path=str(policy_path),
        policy_joint_names=K1_JOINTS_POLICY_ORDER,
        default_pos=DEFAULT_POS.copy(),
        # Force both arms down at startup.
        default_angles={
            r".*": 0.0,
            r"^Left_Shoulder_Roll$": -1.3,
            r"^Right_Shoulder_Roll$": 1.3,
        },
        observation_cfg=Observations_Config(
            base_observations_terms=OBS_TERMS_ORDER,
            scale=OBS_SCALE,
            using_base_obs_history=False,
            base_obs_flatten=True,
            base_obs_his_length=1,
        ),
        action_cfg=Actions_Config(
            scale=build_action_scale_array(K1_JOINTS_POLICY_ORDER, K1_ACTION_SCALE),
            action_clip=ACTION_CLIP,
            smooth_filter=ACTION_SMOOTH_FILTER,
        ),
        motor_cfg=Motor_Config(
            motor_type=PIDMotor,
            effort_limit=MOTOR_EFFORT_LIMIT,
            stiffness=MOTOR_STIFFNESS,
            damping=MOTOR_DAMPING,
            saturation_effort=MOTOR_EFFORT_LIMIT,
            velocity_limit=40.0,
            friction=0.0,
        ),
        cmd=DEFAULT_CMD.copy(),
    )
