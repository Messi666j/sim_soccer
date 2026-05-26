# SPDX-FileCopyrightText: Copyright (c) MOS-Brain Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import json
import math
import queue
import re
import socket
import sys
import tempfile
import threading
import time
import types
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import motrixsim as mtx
import numpy as np
import torch
import torch.nn as nn
import zmq

from .runtime_config import (
    ACTION_CLIP,
    ACTION_SMOOTH_FILTER,
    DEFAULT_CMD,
    FIXED_ROBOT_ID_TO_NAME,
    FIXED_ROBOT_NAME_TO_ID,
    K1_AMP_ACTUATOR_JOINT_ORDER,
    K1_AMP_CMD_MAX_LIN_X,
    K1_AMP_CMD_MAX_LIN_Y,
    K1_AMP_CMD_MAX_YAW,
    K1_AMP_FRAME_STACK,
    K1_AMP_NUM_ACT,
    K1_AMP_NUM_OBS,
    K1_AMP_NUM_SINGLE_OBS,
    K1_AMP_ACTION_SCALE,
    K1_FULL_BODY_NUM_ACT,
    K1_FULL_BODY_NUM_OBS,
    K1_LEGGED_GYM_ACTION_SCALE,
    K1_LEGGED_GYM_DEFAULT_JOINT_ANGLES,
    K1_LEGGED_GYM_DOF_POS_SCALE,
    K1_LEGGED_GYM_DOF_VEL_SCALE,
    K1_LEGGED_GYM_GAIT_FREQUENCY,
    K1_LEGGED_GYM_GYRO_SCALE_LIN_VEL,
    K1_LEGGED_GYM_GRAVITY_SCALE,
    K1_LEGGED_GYM_KP,
    K1_LEGGED_GYM_KD,
    K1_LEGGED_GYM_LEG_JOINTS_POLICY_ORDER,
    K1_LEGGED_GYM_NUM_ACT,
    CMD_VEL_NORM_MAX,
    CMD_VEL_NORM_MIN,
    K1_LEGGED_GYM_NUM_OBS,
    K1_LEGGED_GYM_TORQUE_LIMIT,
    K1_ROBOT_TYPE,
    MAX_ROBOTS_PER_TEAM,
    PI_PLUS_KD_POLICY_ORDER,
    PI_PLUS_KP_POLICY_ORDER,
    PI_PLUS_ROBOT_TYPE,
    PITCH_SCALE,
    RobotRuntimeConfig,
    RuntimeArgs,
    build_action_scale_array,
    parse_param_for_joint_names,
)
from .soccer_referee import MujocoSoccerReferee
from .webview_server import MujocoLabWebView


def _mtx_resolve_joint_index(model: mtx.SceneModel, joint_name: str) -> int:
    """Resolve MJCF-style joint name to MotrixSim joint index (handles naming differences)."""
    j = model.get_joint_index(joint_name)
    if j is not None:
        return int(j)
    if "__" in joint_name:
        robot, short = joint_name.split("__", 1)
        for alt in (f"{robot}/{short}", f"{robot}.{short}", f"{robot}_{short}"):
            j = model.get_joint_index(alt)
            if j is not None:
                return int(j)
    names = getattr(model, "joint_names", None) or ()
    for i, jn in enumerate(names):
        if jn is None:
            continue
        if jn == joint_name:
            return int(i)
        if "__" in joint_name:
            robot, short = joint_name.split("__", 1)
            if short in str(jn) and robot in str(jn):
                return int(i)
    for joint in getattr(model, "joints", ()) or ():
        jn = getattr(joint, "name", None)
        if jn == joint_name:
            return int(joint.index)
        if jn and "__" in joint_name and joint_name == jn:
            return int(joint.index)
    sample = [x for x in names[:40] if x is not None]
    raise RuntimeError(f"MotrixSim: joint not found: {joint_name!r}. Sample joint_names={sample!r}")


def _mtx_root_base_dof_addrs(model: mtx.SceneModel, robot_body_name: str) -> tuple[int, int] | None:
    """
    Floating-base qpos/qvel slice starts from Body.get_dof_*_indices (MotrixSim may not
    register MJCF free-joint names on get_joint_index for the root link).
    """
    bd = model.get_body(robot_body_name)
    if bd is None:
        return None
    pos_idx = np.asarray(bd.get_dof_pos_indices(include_floatingbase=True), dtype=np.int64).reshape(-1)
    vel_idx = np.asarray(bd.get_dof_vel_indices(include_floatingbase=True), dtype=np.int64).reshape(-1)
    if pos_idx.size < 6 or vel_idx.size < 6:
        return None
    return int(pos_idx[0]), int(vel_idx[0])


def _mtx_joint_qpos_start(model: mtx.SceneModel, joint_name: str) -> int:
    ji = _mtx_resolve_joint_index(model, joint_name)
    return int(model.joint_dof_pos_indices[ji])


def _mtx_joint_qvel_start(model: mtx.SceneModel, joint_name: str) -> int:
    ji = _mtx_resolve_joint_index(model, joint_name)
    return int(model.joint_dof_vel_indices[ji])


def _mtx_joint_limit_range(model: mtx.SceneModel, joint_name: str) -> tuple[float, float]:
    """
    Return Motrix joint limit (lower, upper). Falls back to no-limit if unavailable.
    """
    ji = _mtx_resolve_joint_index(model, joint_name)
    joints = getattr(model, "joints", ()) or ()
    if ji < 0 or ji >= len(joints):
        return -np.inf, np.inf
    rng = np.asarray(getattr(joints[ji], "range", []), dtype=np.float64).reshape(-1)
    if rng.size >= 2 and np.isfinite(rng[0]) and np.isfinite(rng[1]) and rng[0] < rng[1]:
        return float(rng[0]), float(rng[1])
    return -np.inf, np.inf


def _mtx_actuator_index(model: mtx.SceneModel, actuator_name: str) -> int:
    a = model.get_actuator_index(actuator_name)
    if a is not None:
        return int(a)
    if "__" in actuator_name:
        robot, short = actuator_name.split("__", 1)
        for alt in (f"{robot}/{short}", f"{robot}.{short}", f"{robot}_{short}"):
            a = model.get_actuator_index(alt)
            if a is not None:
                return int(a)
    raise RuntimeError(f"MotrixSim: actuator not found: {actuator_name!r}")


def _mtx_sensor_vec(model: mtx.SceneModel, data: mtx.SceneData, sensor_name: str) -> np.ndarray:
    v = np.asarray(model.get_sensor_value(sensor_name, data), dtype=np.float32)
    if v.ndim >= 2 and v.shape[0] == data.shape[0]:
        v = v[0]
    return v.reshape(-1).astype(np.float32)


def _mtx_ball_robot_contact_pairs(model: mtx.SceneModel) -> tuple[np.ndarray, np.ndarray]:
    """Geom index pairs (ball, robot_geom) and parallel robot id (-1 if unknown)."""
    try:
        ball = int(model.get_geom_index("ball"))
    except Exception:
        return np.zeros((0, 2), dtype=np.uint32), np.zeros((0,), dtype=np.int32)
    pairs: list[tuple[int, int]] = []
    rids: list[int] = []
    for gname in model.geom_names:
        if not gname or gname == "ball":
            continue
        gl = gname.lower()
        if "pitch" in gl or gname == "ground":
            continue
        if not (gname.startswith("robot_rp") or gname.startswith("robot_bp")):
            continue
        if "__" not in gname:
            continue
        robot_name = gname.split("__", 1)[0]
        rid = FIXED_ROBOT_NAME_TO_ID.get(robot_name, -1)
        try:
            gid = int(model.get_geom_index(gname))
        except Exception:
            continue
        pairs.append((ball, gid))
        rids.append(int(rid))
    if not pairs:
        return np.zeros((0, 2), dtype=np.uint32), np.zeros((0,), dtype=np.int32)
    return np.asarray(pairs, dtype=np.uint32).reshape(-1, 2), np.asarray(rids, dtype=np.int32)


class _BlackFrameRenderer:
    def __init__(self, height: int, width: int):
        self._shape = (max(1, int(height)), max(1, int(width)), 3)

    def update_scene(self, data, camera=None, scene_option=None):
        pass

    def render(self) -> np.ndarray:
        return np.zeros(self._shape, dtype=np.uint8)

    def set_capture_camera(self, camera_name: str | None) -> None:
        pass

    def close(self) -> None:
        pass


def _motrix_capture_task_done(task) -> bool:
    """True when Motrix reports capture finished (binding may use str or enum-like object)."""
    st = getattr(task, "state", None)
    if st == "done" or st == "Done" or st is True:
        return True
    nm = getattr(st, "name", None)
    if isinstance(nm, str) and nm.lower() == "done":
        return True
    return False


def _motrix_capture_task_closed(task) -> bool:
    st = getattr(task, "state", None)
    if st == "closed":
        return True
    return "closed" in str(st).lower()


class _MotrixHeadlessRenderer:
    """Best-effort headless RGB using motrixsim.render (see MotrixSim docs)."""

    def __init__(self, model: mtx.SceneModel, width: int, height: int):
        from motrixsim.render import RenderApp, RenderSettings

        self._h = max(1, int(height))
        self._w = max(1, int(width))
        self._capture_cam_index = 0
        self._camera_indices: dict[str, int] = {}
        # Motrix docs: camera render target must be configured BEFORE RenderApp.launch().
        # Without this, headless capture can fallback to Window(Primary) and return no frames.
        try:
            cam_mgr = model.cameras
            for i in range(len(cam_mgr)):
                cam = cam_mgr[i]
                try:
                    nm = str(getattr(cam, "name", ""))
                    if nm:
                        self._camera_indices[nm] = int(i)
                except Exception:
                    pass
                try:
                    cam.set_render_target("image", self._w, self._h)
                except Exception:
                    continue
            # Prefer our injected scene camera when available.
            try:
                idx = cam_mgr.get_index("lab_webview_diagonal")
                if idx is not None:
                    self._capture_cam_index = int(idx)
                else:
                    idx = cam_mgr.get_index("lab_webview_camera")
                    if idx is not None:
                        self._capture_cam_index = int(idx)
            except Exception:
                pass
        except Exception:
            pass
        self._app = RenderApp(headless=True)
        settings = RenderSettings.performance()
        self._app.launch(model, batch=1, render_settings=settings)
        self._cam = self._app.get_camera(self._capture_cam_index)
        if self._cam is None:
            for i in range(32):
                c = self._app.get_camera(i)
                if c is not None:
                    self._cam = c
                    break
        self._capture_disabled = self._cam is None
        if self._capture_disabled:
            print(
                "[MotrixWebView] No MJCF camera found for RenderApp; video frames will be black "
                "(add <camera name='...'/> to the scene or extend _ensure_default_scene_camera_for_motrix)."
            )
        self._last_task = None
        self._last_rgb: np.ndarray | None = None
        self._capture_done_warned = False
        self._pending_data = None

    def set_capture_camera(self, camera_name: str | None) -> None:
        if not camera_name:
            return
        idx = self._camera_indices.get(str(camera_name), None)
        if idx is None:
            return
        try:
            cam = self._app.get_camera(int(idx))
        except Exception:
            cam = None
        if cam is None:
            return
        self._capture_cam_index = int(idx)
        self._cam = cam

    def update_scene(self, data, camera=None, scene_option=None):
        # All RenderApp.sync happens in render() so we do not sync here and again in render().
        self._pending_data = data
        self._last_task = None

    def render(self) -> np.ndarray:
        data = self._pending_data
        if self._capture_disabled or self._cam is None:
            self._last_task = None
            try:
                self._app.sync(data, wait=False)
            except Exception:
                pass
            if self._last_rgb is not None:
                return self._last_rgb
            return np.zeros((self._h, self._w, 3), dtype=np.uint8)

        # Async capture: push state, enqueue capture, then one wait sync (see RenderApp.sync docs).
        # Do not call sync inside the take_image retry loop.
        try:
            self._app.sync(data, wait=False)
        except Exception:
            pass
        try:
            self._last_task = self._cam.capture()
        except Exception:
            self._last_task = None

        task = self._last_task
        if task is None:
            if self._last_rgb is not None:
                return self._last_rgb
            return np.zeros((self._h, self._w, 3), dtype=np.uint8)

        img = None
        try:
            self._app.sync(None, wait=True)
        except Exception:
            pass
        for _ in range(4):
            try:
                img = task.take_image()
            except Exception:
                img = None
            if img is not None or _motrix_capture_task_closed(task):
                break
        if img is None and not self._capture_done_warned:
            st = getattr(task, "state", None)
            print(
                f"[MotrixWebView] capture still not ready (state={st!r} type={type(st).__name__}); "
                "video may stay black until Motrix returns a completed capture."
            )
            self._capture_done_warned = True
        if img is not None:
            px = np.asarray(img.pixels, dtype=np.uint8)
            if px.ndim == 3 and px.shape[2] == 4:
                px = px[:, :, :3]
            if px.ndim == 3 and px.shape[2] == 3:
                self._last_rgb = px
                return px
        if self._last_rgb is not None:
            return self._last_rgb
        return np.zeros((self._h, self._w, 3), dtype=np.uint8)

    def close(self) -> None:
        try:
            self._app.close()
        except Exception:
            pass


class _FreeCameraView:
    """Minimal stand-in for mujoco.MjvCamera (lookat, distance, azimuth, elevation)."""

    def __init__(self):
        self.lookat = np.array([0.0, 0.0, 0.8], dtype=np.float64)
        self.distance = 18.0
        self.azimuth = 45.0
        self.elevation = -35.0
        self.type = 0


FIELD_PRESETS = {
    "S": (9.0, 6.0),
    "M": (14.0, 9.0),
    "L": (22.0, 14.0),
}

DRAG_RESET_PROTECT_SEC = 0.5
DRAG_CMD_ZERO_POLICY_FRAMES = 5
FALL_RESET_PROTECT_SEC = 1.5
FALL_UPRIGHT_DOT_MIN = 0.2
FALL_CONFIRM_FRAMES = 10

# Extra root height when instantiating robots in the merged scene (meters).
# Helps avoid initial penetrations with non-planar or thick floor collision (e.g. mesh stadium).
ROBOT_SPAWN_Z_LIFT_M = 0.04


def _load_checkpoint_compat(path: Path, map_location: torch.device):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except ModuleNotFoundError as e:
        # pi_plus checkpoints may include rsl_rl.utils.utils.Normalizer in pickle payload.
        if "rsl_rl" not in str(e):
            raise
        rsl_rl_mod = types.ModuleType("rsl_rl")
        utils_pkg = types.ModuleType("rsl_rl.utils")
        utils_mod = types.ModuleType("rsl_rl.utils.utils")

        class Normalizer:
            pass

        Normalizer.__module__ = "rsl_rl.utils.utils"
        utils_mod.Normalizer = Normalizer
        rsl_rl_mod.utils = utils_pkg
        utils_pkg.utils = utils_mod
        sys.modules.setdefault("rsl_rl", rsl_rl_mod)
        sys.modules.setdefault("rsl_rl.utils", utils_pkg)
        sys.modules.setdefault("rsl_rl.utils.utils", utils_mod)
        return torch.load(path, map_location=map_location, weights_only=False)


def _load_field_size_from_match_config(match_config_path: Path | None) -> tuple[float, float] | None:
    if match_config_path is None or not match_config_path.exists():
        return None
    try:
        data = json.loads(match_config_path.read_text(encoding="utf-8"))
        field_cfg = data.get("field", {})
        preset = str(field_cfg.get("preset", "M")).upper()
        if preset in FIELD_PRESETS:
            return FIELD_PRESETS[preset]
        length = field_cfg.get("length")
        width = field_cfg.get("width")
        if length is not None and width is not None:
            return float(length), float(width)
    except Exception:
        pass
    return None


def _load_goal_config_from_match_config(match_config_path: Path | None) -> dict[str, object]:
    cfg: dict[str, object] = {
        "depth": 0.6,
        "width": 2.6,
        "height": 1.8,
        "post_radius": 0.05,
        "procedural_goals": True,
    }
    if match_config_path is None or not match_config_path.exists():
        return cfg
    try:
        data = json.loads(match_config_path.read_text(encoding="utf-8"))
        field_cfg = data.get("field", {}) if isinstance(data, dict) else {}
        goal_cfg = field_cfg.get("goal", data.get("goal", {}))
        if not isinstance(goal_cfg, dict):
            return cfg
        if "depth" in goal_cfg:
            cfg["depth"] = float(goal_cfg["depth"])
        if "width" in goal_cfg:
            cfg["width"] = float(goal_cfg["width"])
        if "height" in goal_cfg:
            cfg["height"] = float(goal_cfg["height"])
        if "post_radius" in goal_cfg:
            cfg["post_radius"] = float(goal_cfg["post_radius"])
        if "procedural_goals" in goal_cfg:
            cfg["procedural_goals"] = bool(goal_cfg["procedural_goals"])
    except Exception:
        pass
    return cfg


def _load_outer_floor_config_from_match_config(match_config_path: Path | None) -> dict[str, object]:
    cfg: dict[str, object] = {
        "enabled": True,
        "margin_ratio": 0.05,
        "min_margin": 1.0,
        "color": [0.2, 0.5, 0.2, 1.0],
        "collision": False,
        "edge_walls_enabled": True,
        "edge_wall_height": 0.8,
        "edge_wall_thickness": 0.04,
        "edge_wall_color": [0.8, 0.9, 1.0, 0.12],
        "edge_wall_collision": True,
    }
    if match_config_path is None or not match_config_path.exists():
        return cfg
    try:
        data = json.loads(match_config_path.read_text(encoding="utf-8"))
        field_cfg = data.get("field", {}) if isinstance(data, dict) else {}
        outer = field_cfg.get("outer_floor", {})
        if not isinstance(outer, dict):
            return cfg
        if "enabled" in outer:
            cfg["enabled"] = bool(outer["enabled"])
        if "margin_ratio" in outer:
            cfg["margin_ratio"] = float(outer["margin_ratio"])
        if "min_margin" in outer:
            cfg["min_margin"] = float(outer["min_margin"])
        if "collision" in outer:
            cfg["collision"] = bool(outer["collision"])
        if "color" in outer and isinstance(outer["color"], (list, tuple)) and len(outer["color"]) == 4:
            cfg["color"] = [float(x) for x in outer["color"]]
        if "edge_walls_enabled" in outer:
            cfg["edge_walls_enabled"] = bool(outer["edge_walls_enabled"])
        if "edge_wall_height" in outer:
            cfg["edge_wall_height"] = float(outer["edge_wall_height"])
        if "edge_wall_thickness" in outer:
            cfg["edge_wall_thickness"] = float(outer["edge_wall_thickness"])
        if "edge_wall_collision" in outer:
            cfg["edge_wall_collision"] = bool(outer["edge_wall_collision"])
        if "edge_wall_color" in outer and isinstance(outer["edge_wall_color"], (list, tuple)) and len(outer["edge_wall_color"]) == 4:
            cfg["edge_wall_color"] = [float(x) for x in outer["edge_wall_color"]]
    except Exception:
        pass
    return cfg


def _load_field_markings_config_from_match_config(
    match_config_path: Path | None, field_size: tuple[float, float] | None
) -> dict[str, object]:
    field_len = float(field_size[0]) if field_size is not None else 14.0
    field_wid = float(field_size[1]) if field_size is not None else 9.0
    cfg: dict[str, object] = {
        "enabled": False,
        "line_width": 0.05,
        "line_height": 0.001,
        "color": [1.0, 1.0, 1.0, 1.0],
        "field_length": field_len,
        "field_width": field_wid,
        "goal_area_depth": 1.0,
        "goal_area_width": 3.0,
        "penalty_area_depth": 2.0,
        "penalty_area_width": 4.0,
        "penalty_spot_distance": 1.5,
        "center_circle_diameter": 1.5,
    }
    if match_config_path is None or not match_config_path.exists():
        return cfg
    try:
        data = json.loads(match_config_path.read_text(encoding="utf-8"))
        field_cfg = data.get("field", {}) if isinstance(data, dict) else {}
        mk = field_cfg.get("markings", {})
        if not isinstance(mk, dict):
            return cfg
        if "enabled" in mk:
            cfg["enabled"] = bool(mk["enabled"])
        if "line_width" in mk:
            cfg["line_width"] = float(mk["line_width"])
        if "line_height" in mk:
            cfg["line_height"] = float(mk["line_height"])
        if "color" in mk and isinstance(mk["color"], (list, tuple)) and len(mk["color"]) == 4:
            cfg["color"] = [float(v) for v in mk["color"]]
        if "field_length" in mk:
            cfg["field_length"] = float(mk["field_length"])
        if "field_width" in mk:
            cfg["field_width"] = float(mk["field_width"])
        if "goal_area_depth" in mk:
            cfg["goal_area_depth"] = float(mk["goal_area_depth"])
        if "goal_area_width" in mk:
            cfg["goal_area_width"] = float(mk["goal_area_width"])
        if "penalty_area_depth" in mk:
            cfg["penalty_area_depth"] = float(mk["penalty_area_depth"])
        if "penalty_area_width" in mk:
            cfg["penalty_area_width"] = float(mk["penalty_area_width"])
        if "penalty_spot_distance" in mk:
            cfg["penalty_spot_distance"] = float(mk["penalty_spot_distance"])
        if "center_circle_diameter" in mk:
            cfg["center_circle_diameter"] = float(mk["center_circle_diameter"])
    except Exception:
        pass
    return cfg


def _load_referee_area_config_from_match_config(match_config_path: Path | None) -> dict[str, float]:
    cfg = {"goalie_area_depth": 1.0, "goalie_area_width": 3.0}
    if match_config_path is None or not match_config_path.exists():
        return cfg
    try:
        data = json.loads(match_config_path.read_text(encoding="utf-8"))
        field_cfg = data.get("field", {}) if isinstance(data, dict) else {}
        ref_cfg = field_cfg.get("referee", {})
        if isinstance(ref_cfg, dict):
            if "goalie_area_depth" in ref_cfg:
                cfg["goalie_area_depth"] = float(ref_cfg["goalie_area_depth"])
            if "goalie_area_width" in ref_cfg:
                cfg["goalie_area_width"] = float(ref_cfg["goalie_area_width"])
        mk_cfg = field_cfg.get("markings", {})
        if isinstance(mk_cfg, dict):
            if "goal_area_depth" in mk_cfg:
                cfg["goalie_area_depth"] = float(mk_cfg["goal_area_depth"])
            if "goal_area_width" in mk_cfg:
                cfg["goalie_area_width"] = float(mk_cfg["goal_area_width"])
    except Exception:
        pass
    return cfg


def _load_team_meta_from_match_config(match_config_path: Path | None) -> dict[str, dict[str, object]]:
    default = {
        "red": {"team_number": 12, "team_name": "Home"},
        "blue": {"team_number": 32, "team_name": "Away"},
    }
    if match_config_path is None or not match_config_path.exists():
        return default
    try:
        data = json.loads(match_config_path.read_text(encoding="utf-8"))
        teams = data.get("teams", {}) if isinstance(data, dict) else {}
        if not isinstance(teams, dict):
            return default
        for side in ("red", "blue"):
            tcfg = teams.get(side, {})
            if not isinstance(tcfg, dict):
                continue
            if "team_number" in tcfg:
                default[side]["team_number"] = int(tcfg["team_number"])
            elif "team_id" in tcfg:
                default[side]["team_number"] = int(tcfg["team_id"])
            if "team_name" in tcfg:
                default[side]["team_name"] = str(tcfg["team_name"])
            elif "name" in tcfg:
                default[side]["team_name"] = str(tcfg["name"])
    except Exception:
        pass
    return default


def _load_spawn_positions_from_match_config(
    match_config_path: Path | None,
) -> dict[str, list[tuple[float, float, float]]]:
    out: dict[str, list[tuple[float, float, float]]] = {"red": [], "blue": []}
    if match_config_path is None or not match_config_path.exists():
        return out
    try:
        data = json.loads(match_config_path.read_text(encoding="utf-8"))
        teams = data.get("teams", {}) if isinstance(data, dict) else {}
        if not isinstance(teams, dict):
            return out
        for side in ("red", "blue"):
            tcfg = teams.get(side, {})
            if not isinstance(tcfg, dict):
                continue
            raw = tcfg.get("spawn_positions", [])
            if not isinstance(raw, list):
                continue
            parsed: list[tuple[float, float, float]] = []
            for p in raw:
                if not isinstance(p, (list, tuple)) or len(p) < 2:
                    continue
                x = float(p[0])
                y = float(p[1])
                theta = float(p[2]) if len(p) >= 3 and p[2] is not None else 0.0
                parsed.append((x, y, theta))
            out[side] = parsed
    except Exception:
        pass
    return out


def _write_temp_xml(xml_text: str) -> Path:
    fd = tempfile.NamedTemporaryFile(prefix="multi_k1_scene_", suffix=".xml", delete=False)
    fd.write(xml_text.encode("utf-8"))
    fd.flush()
    fd.close()
    return Path(fd.name)


def _ensure_offscreen_buffer(root: ET.Element, offwidth: int = 1920, offheight: int = 1080):
    visual = root.find("visual")
    if visual is None:
        visual = ET.SubElement(root, "visual")
    global_tag = visual.find("global")
    if global_tag is None:
        global_tag = ET.SubElement(visual, "global")
    cur_w = int(global_tag.get("offwidth", "0") or "0")
    cur_h = int(global_tag.get("offheight", "0") or "0")
    global_tag.set("offwidth", str(max(cur_w, offwidth)))
    global_tag.set("offheight", str(max(cur_h, offheight)))


def _remove_all_plane_geoms(root: ET.Element):
    # Remove every plane geom defined in robot XML so only soccer world pitch remains.
    for worldbody in list(root.findall("worldbody")):
        for geom in list(worldbody.iter("geom")):
            if geom.get("type") != "plane":
                continue
            parent = next((p for p in worldbody.iter() if geom in list(p)), None)
            if parent is not None:
                parent.remove(geom)


def _find_template_robot_body(worldbody: ET.Element, base_joint_name: str) -> ET.Element:
    for body in list(worldbody.findall("body")):
        if body.find(f"joint[@name='{base_joint_name}']") is not None or body.find(
            f"freejoint[@name='{base_joint_name}']"
        ) is not None:
            return body
    raise RuntimeError(f"Cannot find template robot body with base joint '{base_joint_name}' in robot XML")


def _prefix_body_tree_names(body: ET.Element, robot_name: str):
    for elem in body.iter():
        name = elem.get("name")
        if not name:
            continue
        if elem.tag == "body":
            if elem is body:
                elem.set("name", robot_name)
            else:
                elem.set("name", f"{robot_name}__{name}")
        elif elem.tag in ("joint", "freejoint", "site", "geom", "camera", "light"):
            elem.set("name", f"{robot_name}__{name}")


def _ensure_default_scene_camera_for_motrix(worldbody: ET.Element) -> None:
    """Ensure Motrix webview cameras exist for preset switching."""

    def _xyaxes_from_eye_look(eye: tuple[float, float, float], look: tuple[float, float, float]) -> str:
        eye_v = np.asarray(eye, dtype=np.float64)
        look_v = np.asarray(look, dtype=np.float64)
        fwd = look_v - eye_v
        fn = float(np.linalg.norm(fwd))
        if fn < 1e-8:
            fwd = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        else:
            fwd = fwd / fn
        up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        if abs(float(np.dot(fwd, up))) > 0.98:
            up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        x = np.cross(up, fwd)
        xn = float(np.linalg.norm(x))
        if xn < 1e-8:
            x = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        else:
            x = x / xn
        # MuJoCo camera looks along -Z where Z = X x Y.
        y = np.cross(x, fwd)
        y = y / max(1e-8, float(np.linalg.norm(y)))
        # Rotate camera image by 180deg around view axis (flip screen up/right).
        x = -x
        y = -y
        return f"{x[0]:.7g} {x[1]:.7g} {x[2]:.7g} {y[0]:.7g} {y[1]:.7g} {y[2]:.7g}"

    existing = {str(ch.get("name", "")) for ch in worldbody if ch.tag == "camera"}
    # WebView capture uses these MJCF cameras (see set_capture_camera / get_camera). Eye/look must match
    # _apply_camera_preset in MultiRobotMotrixSim or preset buttons will not move the rendered view.
    presets = {
        "lab_webview_diagonal": ((-8.0, -8.0, 8.0), (0.0, 0.0, 0.8)),
        "lab_webview_top": ((0.0, 0.0, 18.0), (0.0, 0.0, 0.8)),
        "lab_webview_side": ((0.0, 9.0, 6.0), (0.0, 0.0, 0.8)),
        "lab_webview_goal_left": ((-8.5, 0.0, 6.0), (0.0, 0.0, 0.9)),
        "lab_webview_goal_right": ((8.5, 0.0, 6.0), (0.0, 0.0, 0.9)),
    }
    for cam_name, (eye, look) in presets.items():
        if cam_name in existing:
            continue
        ET.SubElement(
            worldbody,
            "camera",
            name=cam_name,
            pos=f"{eye[0]:.7g} {eye[1]:.7g} {eye[2]:.7g}",
            mode="fixed",
            fovy="55",
            xyaxes=_xyaxes_from_eye_look(eye, look),
        )
    # Backward-compatible alias used by older code paths.
    if "lab_webview_camera" not in existing:
        ET.SubElement(
            worldbody,
            "camera",
            name="lab_webview_camera",
            pos="-10 -10 10",
            mode="fixed",
            fovy="55",
            xyaxes=_xyaxes_from_eye_look((-10.0, -10.0, 10.0), (0.0, 0.0, 0.8)),
        )


def _spawn_xy_theta(team: str, idx: int, count: int, field_size: tuple[float, float] | None) -> tuple[float, float, float]:
    field_len = float(field_size[0]) if field_size is not None else 14.0
    y_spacing = 1.0
    start_y = -((count - 1) * y_spacing) * 0.5
    y = start_y + idx * y_spacing
    if team == "red":
        return (-field_len * 0.25, y, 0.0)
    return (field_len * 0.25, y, np.pi)


def _quat_from_yaw(theta: float) -> np.ndarray:
    # MJCF quaternion order is wxyz.
    half = 0.5 * float(theta)
    return np.array([np.cos(half), 0.0, 0.0, np.sin(half)], dtype=np.float32)


def _quat_xyzw_from_yaw(theta: float) -> np.ndarray:
    # Motrix runtime dof_pos quaternion order is xyzw.
    half = 0.5 * float(theta)
    return np.array([0.0, 0.0, np.sin(half), np.cos(half)], dtype=np.float32)


def _add_procedural_goals(
    worldbody: ET.Element,
    field_length: float,
    goal_depth: float = 0.6,
    goal_width: float = 2.6,
    goal_height: float = 1.8,
    post_radius: float = 0.05,
):
    goal_half_y = 0.5 * float(goal_width)
    field_half_x = 0.5 * float(field_length)

    post_rgba = "0.8 0.8 0.8 1"
    net_rgba = "1 1 1 0.2"
    y_quat = "0 0 0.7071068 0.7071068"

    for side, x_sign in (("left", -1.0), ("right", 1.0)):
        goal_name = f"goal-{side}"
        goal_body = ET.SubElement(worldbody, "body", name=goal_name, pos=f"{x_sign * field_half_x:g} 0 0")
        depth = x_sign * float(goal_depth)
        x_half_depth = 0.5 * abs(depth)

        # Vertical posts
        ET.SubElement(
            goal_body,
            "geom",
            name=f"{goal_name}-front-left-post",
            type="cylinder",
            pos=f"0 {-goal_half_y:g} {0.5 * goal_height:g}",
            size=f"{post_radius:g} {0.5 * goal_height:g}",
            rgba=post_rgba,
        )
        ET.SubElement(
            goal_body,
            "geom",
            name=f"{goal_name}-front-right-post",
            type="cylinder",
            pos=f"0 {goal_half_y:g} {0.5 * goal_height:g}",
            size=f"{post_radius:g} {0.5 * goal_height:g}",
            rgba=post_rgba,
        )
        ET.SubElement(
            goal_body,
            "geom",
            name=f"{goal_name}-back-left-post",
            type="cylinder",
            pos=f"{depth:g} {-goal_half_y:g} {0.5 * goal_height:g}",
            size=f"{post_radius:g} {0.5 * goal_height:g}",
            rgba=post_rgba,
        )
        ET.SubElement(
            goal_body,
            "geom",
            name=f"{goal_name}-back-right-post",
            type="cylinder",
            pos=f"{depth:g} {goal_half_y:g} {0.5 * goal_height:g}",
            size=f"{post_radius:g} {0.5 * goal_height:g}",
            rgba=post_rgba,
        )

        # Crossbars
        ET.SubElement(
            goal_body,
            "geom",
            name=f"{goal_name}-front-crossbar",
            type="cylinder",
            pos=f"0 0 {goal_height:g}",
            size=f"{post_radius:g} {goal_half_y:g}",
            quat=y_quat,
            rgba=post_rgba,
        )
        ET.SubElement(
            goal_body,
            "geom",
            name=f"{goal_name}-back-crossbar",
            type="cylinder",
            pos=f"{depth:g} 0 {goal_height:g}",
            size=f"{post_radius:g} {goal_half_y:g}",
            quat=y_quat,
            rgba=post_rgba,
        )
        ET.SubElement(
            goal_body,
            "geom",
            name=f"{goal_name}-left-side-crossbar",
            type="cylinder",
            pos=f"{0.5 * depth:g} {-goal_half_y:g} {goal_height:g}",
            size=f"{post_radius:g} {x_half_depth:g}",
            quat=f"0 {0.7071068 * x_sign:g} 0 0.7071068",
            rgba=post_rgba,
        )
        ET.SubElement(
            goal_body,
            "geom",
            name=f"{goal_name}-right-side-crossbar",
            type="cylinder",
            pos=f"{0.5 * depth:g} {goal_half_y:g} {goal_height:g}",
            size=f"{post_radius:g} {x_half_depth:g}",
            quat=f"0 {-0.7071068 * x_sign:g} 0 0.7071068",
            rgba=post_rgba,
        )

        # Light-weight visual nets (non-colliding)
        ET.SubElement(
            goal_body,
            "geom",
            name=f"{goal_name}-top-net",
            type="box",
            pos=f"{0.5 * depth:g} 0 {goal_height:g}",
            size=f"{x_half_depth:g} {goal_half_y:g} {post_radius:g}",
            rgba=net_rgba,
            contype="0",
            conaffinity="0",
        )
        ET.SubElement(
            goal_body,
            "geom",
            name=f"{goal_name}-back-net",
            type="box",
            pos=f"{depth:g} 0 {0.5 * goal_height:g}",
            size=f"{post_radius:g} {goal_half_y:g} {0.5 * goal_height:g}",
            rgba=net_rgba,
            contype="0",
            conaffinity="0",
        )


def _add_outer_floor_planes(
    worldbody: ET.Element,
    field_length: float,
    field_width: float,
    cfg: dict[str, object] | None = None,
):
    c = cfg if isinstance(cfg, dict) else {}
    if not bool(c.get("enabled", True)):
        return
    ratio = float(c.get("margin_ratio", 0.05))
    min_margin = float(c.get("min_margin", 1.0))
    margin_x = max(min_margin, 0.5 * field_length * ratio)
    margin_y = max(min_margin, 0.5 * field_width * ratio)
    field_half_x = 0.5 * float(field_length)
    field_half_y = 0.5 * float(field_width)
    rgba = c.get("color", [0.2, 0.5, 0.2, 1.0])
    if not isinstance(rgba, (list, tuple)) or len(rgba) != 4:
        rgba = [0.2, 0.5, 0.2, 1.0]
    rgba_str = " ".join(f"{float(v):g}" for v in rgba)
    coll = bool(c.get("collision", False))
    contype = "1" if coll else "0"
    conaffinity = "1" if coll else "0"

    # Left / right
    ET.SubElement(
        worldbody,
        "geom",
        name="left-floor",
        type="plane",
        pos=f"{-field_half_x - margin_x:g} 0 0",
        size=f"{margin_x:g} {field_half_y:g} 1",
        rgba=rgba_str,
        contype=contype,
        conaffinity=conaffinity,
    )
    ET.SubElement(
        worldbody,
        "geom",
        name="right-floor",
        type="plane",
        pos=f"{field_half_x + margin_x:g} 0 0",
        size=f"{margin_x:g} {field_half_y:g} 1",
        rgba=rgba_str,
        contype=contype,
        conaffinity=conaffinity,
    )
    # Top / bottom
    ET.SubElement(
        worldbody,
        "geom",
        name="top-floor",
        type="plane",
        pos=f"0 {field_half_y + margin_y:g} 0",
        size=f"{field_half_x + 2.0 * margin_x:g} {margin_y:g} 1",
        rgba=rgba_str,
        contype=contype,
        conaffinity=conaffinity,
    )
    ET.SubElement(
        worldbody,
        "geom",
        name="bottom-floor",
        type="plane",
        pos=f"0 {-field_half_y - margin_y:g} 0",
        size=f"{field_half_x + 2.0 * margin_x:g} {margin_y:g} 1",
        rgba=rgba_str,
        contype=contype,
        conaffinity=conaffinity,
    )

    # Add transparent boundary walls on outer-floor edges to keep robots inside controllable area.
    if bool(c.get("edge_walls_enabled", True)):
        wall_h = max(0.05, float(c.get("edge_wall_height", 0.8)))
        wall_t = max(0.01, float(c.get("edge_wall_thickness", 0.04)))
        wall_rgba = c.get("edge_wall_color", [0.8, 0.9, 1.0, 0.12])
        if not isinstance(wall_rgba, (list, tuple)) or len(wall_rgba) != 4:
            wall_rgba = [0.8, 0.9, 1.0, 0.12]
        wall_rgba_str = " ".join(f"{float(v):g}" for v in wall_rgba)
        wall_coll = bool(c.get("edge_wall_collision", True))
        wall_contype = "1" if wall_coll else "0"
        wall_conaffinity = "1" if wall_coll else "0"

        outer_half_x = field_half_x + 2.0 * margin_x
        outer_half_y = field_half_y + 2.0 * margin_y
        half_h = 0.5 * wall_h
        half_t = 0.5 * wall_t

        # Left/right walls (normal along X)
        ET.SubElement(
            worldbody,
            "geom",
            name="outer-wall-left",
            type="box",
            pos=f"{-outer_half_x - half_t:g} 0 {half_h:g}",
            size=f"{half_t:g} {outer_half_y:g} {half_h:g}",
            rgba=wall_rgba_str,
            contype=wall_contype,
            conaffinity=wall_conaffinity,
        )
        ET.SubElement(
            worldbody,
            "geom",
            name="outer-wall-right",
            type="box",
            pos=f"{outer_half_x + half_t:g} 0 {half_h:g}",
            size=f"{half_t:g} {outer_half_y:g} {half_h:g}",
            rgba=wall_rgba_str,
            contype=wall_contype,
            conaffinity=wall_conaffinity,
        )

        # Top/bottom walls (normal along Y)
        ET.SubElement(
            worldbody,
            "geom",
            name="outer-wall-top",
            type="box",
            pos=f"0 {outer_half_y + half_t:g} {half_h:g}",
            size=f"{outer_half_x:g} {half_t:g} {half_h:g}",
            rgba=wall_rgba_str,
            contype=wall_contype,
            conaffinity=wall_conaffinity,
        )
        ET.SubElement(
            worldbody,
            "geom",
            name="outer-wall-bottom",
            type="box",
            pos=f"0 {-outer_half_y - half_t:g} {half_h:g}",
            size=f"{outer_half_x:g} {half_t:g} {half_h:g}",
            rgba=wall_rgba_str,
            contype=wall_contype,
            conaffinity=wall_conaffinity,
        )


def _add_field_markings(
    worldbody: ET.Element,
    field_length: float,
    field_width: float,
    cfg: dict[str, object] | None = None,
):
    c = cfg if isinstance(cfg, dict) else {}
    if not bool(c.get("enabled", False)):
        return

    line_w = max(0.005, float(c.get("line_width", 0.05)))
    line_h = max(0.0002, float(c.get("line_height", 0.001)))
    rgba = c.get("color", [1.0, 1.0, 1.0, 1.0])
    if not isinstance(rgba, (list, tuple)) or len(rgba) != 4:
        rgba = [1.0, 1.0, 1.0, 1.0]
    rgba_str = " ".join(f"{float(v):g}" for v in rgba)

    mark_len = max(0.1, float(c.get("field_length", field_length)))
    mark_wid = max(0.1, float(c.get("field_width", field_width)))
    half_len = 0.5 * mark_len
    half_wid = 0.5 * mark_wid
    half_lw = 0.5 * line_w
    z = line_h
    half_h = 0.5 * line_h

    def add_line_box(name: str, x: float, y: float, sx: float, sy: float):
        ET.SubElement(
            worldbody,
            "geom",
            name=name,
            type="box",
            pos=f"{x:g} {y:g} {z:g}",
            size=f"{max(half_lw, sx):g} {max(half_lw, sy):g} {half_h:g}",
            rgba=rgba_str,
            contype="0",
            conaffinity="0",
        )

    # Boundary and center line
    add_line_box("line-boundary-top", 0.0, half_wid, half_len, half_lw)
    add_line_box("line-boundary-bottom", 0.0, -half_wid, half_len, half_lw)
    add_line_box("line-boundary-left", -half_len, 0.0, half_lw, half_wid)
    add_line_box("line-boundary-right", half_len, 0.0, half_lw, half_wid)
    add_line_box("line-center", 0.0, 0.0, half_lw, half_wid)

    goal_area_depth = max(0.05, float(c.get("goal_area_depth", 1.0)))
    goal_area_width = max(line_w, float(c.get("goal_area_width", 3.0)))
    penalty_area_depth = max(0.05, float(c.get("penalty_area_depth", 2.0)))
    penalty_area_width = max(line_w, float(c.get("penalty_area_width", 4.0)))

    for side, sgn in (("left", -1.0), ("right", 1.0)):
        for prefix, depth, box_w in (
            ("goal-area", goal_area_depth, goal_area_width),
            ("penalty-area", penalty_area_depth, penalty_area_width),
        ):
            x_outer = sgn * half_len
            x_inner = sgn * (half_len - depth)
            y_half = 0.5 * box_w
            y_half = min(y_half, half_wid)

            add_line_box(f"line-{prefix}-{side}-outer", x_outer, 0.0, half_lw, y_half)
            add_line_box(f"line-{prefix}-{side}-inner", x_inner, 0.0, half_lw, y_half)
            add_line_box(f"line-{prefix}-{side}-top", 0.5 * (x_outer + x_inner), y_half, 0.5 * depth, half_lw)
            add_line_box(f"line-{prefix}-{side}-bottom", 0.5 * (x_outer + x_inner), -y_half, 0.5 * depth, half_lw)

    spot_dist = max(0.05, float(c.get("penalty_spot_distance", 1.5)))
    spot_r = max(0.02, 0.5 * line_w)
    for side, sgn in (("left", -1.0), ("right", 1.0)):
        x_spot = sgn * (half_len - spot_dist)
        ET.SubElement(
            worldbody,
            "geom",
            name=f"line-penalty-spot-{side}",
            type="cylinder",
            pos=f"{x_spot:g} 0 {half_h:g}",
            size=f"{spot_r:g} {line_h:g}",
            rgba=rgba_str,
            contype="0",
            conaffinity="0",
        )

    circle_d = max(0.1, float(c.get("center_circle_diameter", 1.5)))
    circle_r = 0.5 * circle_d
    seg_n = 48
    # Slight overlap between adjacent segments avoids tiny visual gaps.
    seg_len = (2.0 * math.pi * circle_r / seg_n) * 1.08
    for i in range(seg_n):
        theta = (2.0 * math.pi * i) / seg_n
        x = circle_r * math.cos(theta)
        y = circle_r * math.sin(theta)
        # Align each short box with circle tangent, not radius.
        tangent_theta = theta + 0.5 * math.pi
        ET.SubElement(
            worldbody,
            "geom",
            name=f"line-center-circle-{i}",
            type="box",
            pos=f"{x:g} {y:g} {z:g}",
            quat=f"{math.cos(0.5 * tangent_theta):g} 0 0 {math.sin(0.5 * tangent_theta):g}",
            size=f"{0.5 * seg_len:g} {half_lw:g} {half_h:g}",
            rgba=rgba_str,
            contype="0",
            conaffinity="0",
        )


def _build_multi_robot_soccer_scene_xml(
    robot_xml: Path,
    soccer_world_xml: Path,
    max_red_robots: int,
    max_blue_robots: int,
    base_joint_name: str,
    pitch_scale: float = PITCH_SCALE,
    target_field_size: tuple[float, float] | None = None,
    goal_cfg: dict[str, object] | None = None,
    outer_floor_cfg: dict[str, object] | None = None,
    field_markings_cfg: dict[str, object] | None = None,
    spawn_positions_cfg: dict[str, list[tuple[float, float, float]]] | None = None,
    keep_robot_sensors: bool = False,
) -> tuple[Path, list[int]]:
    meshdir = robot_xml.parent / "meshes"
    robot_root = ET.fromstring(robot_xml.read_text(encoding="utf-8"))
    world_root = ET.parse(soccer_world_xml).getroot()

    robot_compiler = robot_root.find("compiler")
    if robot_compiler is not None:
        robot_compiler.set("meshdir", meshdir.as_posix())

    _ensure_offscreen_buffer(robot_root)
    _remove_all_plane_geoms(robot_root)

    worldbody = robot_root.find("worldbody")
    if worldbody is None:
        raise RuntimeError("Robot XML missing worldbody")

    template_body = _find_template_robot_body(worldbody, base_joint_name=base_joint_name)
    template_actuator = robot_root.find("actuator")
    if template_actuator is None:
        raise RuntimeError("Robot XML missing actuator section")
    template_actuators = list(template_actuator)

    worldbody.remove(template_body)
    for child in list(template_actuator):
        template_actuator.remove(child)

    template_sensor = robot_root.find("sensor")
    template_sensors: list[ET.Element] = []
    if template_sensor is not None:
        if keep_robot_sensors:
            template_sensors = list(template_sensor)
            for child in list(template_sensor):
                template_sensor.remove(child)
        else:
            robot_root.remove(template_sensor)
            template_sensor = None

    active_robot_ids: list[int] = []
    template_pos_vals = [float(v) for v in (template_body.get("pos", "0 0 0").split())]
    template_spawn_z = template_pos_vals[2] if len(template_pos_vals) >= 3 else 0.0

    def add_team(team: str, count: int):
        base_id = 0 if team == "red" else MAX_ROBOTS_PER_TEAM
        team_spawns = []
        if isinstance(spawn_positions_cfg, dict):
            team_spawns = spawn_positions_cfg.get(team, []) or []
        selected_spawns: list[tuple[float, float, float]] = []
        if team_spawns:
            n = len(team_spawns)
            m = int(count)
            if m <= n:
                start = max(0, (n - m) // 2)
                selected_spawns = team_spawns[start : start + m]
            else:
                selected_spawns = list(team_spawns)
        for i in range(count):
            rid = base_id + i
            robot_name = FIXED_ROBOT_ID_TO_NAME[rid]
            body_copy = deepcopy(template_body)
            _prefix_body_tree_names(body_copy, robot_name)
            if selected_spawns and i < len(selected_spawns):
                x, y, theta = selected_spawns[i]
            else:
                x, y, theta = _spawn_xy_theta(team, i, count, target_field_size)
            spawn_z = float(template_spawn_z) + float(ROBOT_SPAWN_Z_LIFT_M)
            body_copy.set("pos", f"{x:.6f} {y:.6f} {spawn_z:.6f}")
            body_copy.set("quat", " ".join(f"{v:.9g}" for v in _quat_from_yaw(theta)))
            worldbody.append(body_copy)

            for act in template_actuators:
                act_copy = deepcopy(act)
                if act_copy.get("name"):
                    act_copy.set("name", f"{robot_name}__{act_copy.get('name')}")
                if act_copy.get("joint"):
                    act_copy.set("joint", f"{robot_name}__{act_copy.get('joint')}")
                template_actuator.append(act_copy)
            if template_sensor is not None:
                for sen in template_sensors:
                    sen_copy = deepcopy(sen)
                    if sen_copy.get("name"):
                        sen_copy.set("name", f"{robot_name}__{sen_copy.get('name')}")
                    for attr in ("joint", "actuator", "site", "objname", "body", "tendon"):
                        ref = sen_copy.get(attr)
                        if ref:
                            sen_copy.set(attr, f"{robot_name}__{ref}")
                    template_sensor.append(sen_copy)

            active_robot_ids.append(rid)

    add_team("red", max_red_robots)
    add_team("blue", max_blue_robots)

    world_compiler = world_root.find("compiler")
    world_asset_dir = soccer_world_xml.parent
    if world_compiler is not None and world_compiler.get("assetdir"):
        world_asset_dir = soccer_world_xml.parent / world_compiler.get("assetdir")

    robot_asset = robot_root.find("asset")
    if robot_asset is None:
        robot_asset = ET.SubElement(robot_root, "asset")
    world_asset = world_root.find("asset")
    if world_asset is not None:
        for child in list(world_asset):
            copied = deepcopy(child)
            for attr in ("file", "fileup", "filedown", "filefront", "fileback", "fileleft", "fileright"):
                v = copied.get(attr)
                if v and not Path(v).is_absolute():
                    copied.set(attr, (world_asset_dir / v).as_posix())
            robot_asset.append(copied)

    world_worldbody = world_root.find("worldbody")
    out_field_len = float(target_field_size[0]) if target_field_size is not None else 14.0
    out_field_wid = float(target_field_size[1]) if target_field_size is not None else 9.0
    if world_worldbody is not None:
        for child in list(world_worldbody):
            copied = deepcopy(child)
            if copied.tag == "geom" and copied.get("name") == "pitch":
                size_str = copied.get("size")
                if size_str:
                    vals = [float(x) for x in size_str.split()]
                    if len(vals) >= 2:
                        if target_field_size is not None:
                            vals[0] = float(target_field_size[0]) * 0.5
                            vals[1] = float(target_field_size[1]) * 0.5
                        else:
                            vals[0] *= pitch_scale
                            vals[1] *= pitch_scale
                        copied.set("size", " ".join(f"{v:g}" for v in vals))
                        out_field_len = float(vals[0]) * 2.0
                        out_field_wid = float(vals[1]) * 2.0
                # Preserve MuJoCo pitch friction from source XML (do not override here).
                # Preserve MuJoCo pitch contact parameters from source XML.
                # Use flat color instead of texture so field markings stay controllable and clear.
                copied.attrib.pop("material", None)
                copied.set("rgba", "0.18 0.45 0.18 1")
            worldbody.append(copied)

    _ensure_default_scene_camera_for_motrix(worldbody)

    _add_outer_floor_planes(worldbody, field_length=out_field_len, field_width=out_field_wid, cfg=outer_floor_cfg)
    _add_field_markings(worldbody, field_length=out_field_len, field_width=out_field_wid, cfg=field_markings_cfg)

    g = goal_cfg if isinstance(goal_cfg, dict) else {}
    if bool(g.get("procedural_goals", True)):
        _add_procedural_goals(
            worldbody,
            field_length=out_field_len,
            goal_depth=float(g.get("depth", 0.6)),
            goal_width=float(g.get("width", 2.6)),
            goal_height=float(g.get("height", 1.8)),
            post_radius=float(g.get("post_radius", 0.05)),
        )

    xml_path = _write_temp_xml(ET.tostring(robot_root, encoding="unicode"))
    return xml_path, active_robot_ids


class MLPActor(nn.Module):
    def __init__(self, layer_dims: list[int]):
        super().__init__()
        if len(layer_dims) < 2:
            raise ValueError("Actor layer_dims must contain at least input and output sizes")
        layers: list[nn.Module] = []
        for i in range(len(layer_dims) - 1):
            layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
            if i < len(layer_dims) - 2:
                layers.append(nn.ELU())
        self.actor = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.actor(x)


def _quat_to_rot_world_from_body(quat_xyzw: np.ndarray) -> np.ndarray:
    x, y, z, w = quat_xyzw
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
            [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
            [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def _quat_apply_inverse(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    q_w = q[-1]
    q_vec = q[:3]
    a = v * (2.0 * q_w**2 - 1.0)
    b = np.cross(q_vec, v) * q_w * 2.0
    c = q_vec * np.dot(q_vec, v) * 2.0
    return a - b + c


# Keep the pi_plus constants aligned with sim2sim_pi_plus.py.
PI_PLUS_JOINTS_MUJOCO_ORDER = [
    "l_hip_pitch_joint",
    "l_hip_roll_joint",
    "l_thigh_joint",
    "l_calf_joint",
    "l_ankle_pitch_joint",
    "l_ankle_roll_joint",
    "l_shoulder_pitch_joint",
    "l_shoulder_roll_joint",
    "l_upper_arm_joint",
    "l_elbow_joint",
    "r_hip_pitch_joint",
    "r_hip_roll_joint",
    "r_thigh_joint",
    "r_calf_joint",
    "r_ankle_pitch_joint",
    "r_ankle_roll_joint",
    "r_shoulder_pitch_joint",
    "r_shoulder_roll_joint",
    "r_upper_arm_joint",
    "r_elbow_joint",
]
PI_PLUS_ISAAC_TO_MUJOCO_IDX = np.asarray([0, 4, 8, 12, 16, 18, 1, 5, 9, 13, 2, 6, 10, 14, 17, 19, 3, 7, 11, 15], dtype=np.int32)
PI_PLUS_MUJOCO_TO_ISAAC_IDX = np.asarray([0, 6, 10, 16, 1, 7, 11, 17, 2, 8, 12, 18, 3, 9, 13, 19, 4, 14, 5, 15], dtype=np.int32)
PI_PLUS_DEFAULT_DOF_POS_MUJOCO = np.asarray(
    [-0.25, 0.0, 0.0, 0.65, -0.4, 0.0, 0.0, 0.2, 0.0, -1.2, -0.25, 0.0, 0.0, 0.65, -0.4, 0.0, 0.0, -0.2, 0.0, -1.2],
    dtype=np.float32,
)

# K1 walk/stand hybrid: body-frame command (vx, vy, yaw rate) gates legged vs full-body policy.
K1_HYBRID_WALK_ENTER_LIN = 0.07
K1_HYBRID_WALK_EXIT_LIN = 0.035
K1_HYBRID_WALK_ENTER_YAW = 0.18
K1_HYBRID_WALK_EXIT_YAW = 0.07


@dataclass
class RobotSpec:
    rid: int
    name: str
    team: str
    qpos_idx: np.ndarray
    qvel_idx: np.ndarray
    act_idx: np.ndarray
    act_qpos_idx: np.ndarray
    act_qvel_idx: np.ndarray
    base_qpos_adr: int
    base_qvel_adr: int
    init_joint_pos: np.ndarray
    init_angles: np.ndarray
    filtered_dof_target: np.ndarray
    target_joint_pos: np.ndarray
    last_action: np.ndarray
    action_scale: np.ndarray
    kp: np.ndarray
    kd: np.ndarray
    effort: np.ndarray
    obs_step_dim: int
    obs_history: np.ndarray
    pi_qpos_idx_mujoco: np.ndarray | None = None
    pi_qvel_idx_mujoco: np.ndarray | None = None
    pi_act_idx_mujoco: np.ndarray | None = None
    pi_default_dof_pos: np.ndarray | None = None
    pi_isaac_to_mujoco_idx: np.ndarray | None = None
    pi_mujoco_to_isaac_idx: np.ndarray | None = None
    pi_filtered_dof_target: np.ndarray | None = None
    pi_target_dof_pos: np.ndarray | None = None
    joint_lower: np.ndarray | None = None
    joint_upper: np.ndarray | None = None
    pi_joint_lower_mujoco: np.ndarray | None = None
    pi_joint_upper_mujoco: np.ndarray | None = None
    # When K1 hybrid stand policy is loaded: False = full-body stand (46000), True = legged walk (4700).
    k1_hybrid_walking: bool = False
    # legged_gym K1 locomotion (47-dim obs, 12 leg actions) — optional per-robot buffers.
    k1_legged_last_action: np.ndarray | None = None
    k1_legged_default_dof: np.ndarray | None = None
    k1_legged_joint_indices: np.ndarray | None = None
    k1_non_leg_mask: np.ndarray | None = None
    k1_gait_phase: float | None = None
    k1_amp_hist: np.ndarray | None = None
    k1_amp_last_mjcf: np.ndarray | None = None
    k1_amp_gather_mjcf: np.ndarray | None = None


class MultiRobotMotrixSim:
    def __init__(self, args: RuntimeArgs):
        self.args = args
        self.robot_cfg: RobotRuntimeConfig = args.robot_cfg
        self.max_red_robots = args.max_red_robots
        self.max_blue_robots = args.max_blue_robots
        self.active_robot_ids = self._active_ids_from_limits(self.max_red_robots, self.max_blue_robots)

        field_size = _load_field_size_from_match_config(args.match_config)
        self._field_length = float(field_size[0]) if field_size is not None else 14.0
        self._field_width = float(field_size[1]) if field_size is not None else 9.0
        goal_cfg = _load_goal_config_from_match_config(args.match_config)
        self._goal_width = float(goal_cfg.get("width", 2.6))
        self._goal_height = float(goal_cfg.get("height", 1.8))
        outer_floor_cfg = _load_outer_floor_config_from_match_config(args.match_config)
        field_markings_cfg = _load_field_markings_config_from_match_config(args.match_config, field_size)
        self._field_markings_cfg = dict(field_markings_cfg)
        self._outer_floor_cfg = dict(outer_floor_cfg)
        ratio = float(self._outer_floor_cfg.get("margin_ratio", 0.05))
        min_margin = float(self._outer_floor_cfg.get("min_margin", 1.0))
        margin_x = max(min_margin, 0.5 * self._field_length * ratio)
        margin_y = max(min_margin, 0.5 * self._field_width * ratio)
        self._world_length = float(self._field_length + 4.0 * margin_x)
        self._world_width = float(self._field_width + 4.0 * margin_y)
        referee_area_cfg = _load_referee_area_config_from_match_config(args.match_config)
        team_meta_cfg = _load_team_meta_from_match_config(args.match_config)
        spawn_positions_cfg = _load_spawn_positions_from_match_config(args.match_config)
        scene_xml, _ = _build_multi_robot_soccer_scene_xml(
            args.robot_xml,
            args.soccer_world_xml,
            max_red_robots=self.max_red_robots,
            max_blue_robots=self.max_blue_robots,
            base_joint_name=self.robot_cfg.base_joint_name,
            pitch_scale=PITCH_SCALE,
            target_field_size=field_size,
            goal_cfg=goal_cfg,
            outer_floor_cfg=outer_floor_cfg,
            field_markings_cfg=field_markings_cfg,
            spawn_positions_cfg=spawn_positions_cfg,
            keep_robot_sensors=(
                self.robot_cfg.robot_type == PI_PLUS_ROBOT_TYPE
                or self.robot_cfg.use_k1_legged_gym_policy
                or self.robot_cfg.use_k1_amp_onnx
            ),
        )

        self.model = mtx.load_model(str(scene_xml))
        self.data = mtx.SceneData(self.model, batch=[1])
        # Keep physics coefficients aligned with MuJoCo source scene (world.xml),
        # including timestep/options loaded in the model.
        self.sim_dt = float(self.model.options.timestep)
        self.control_decimation = int(self.robot_cfg.control_decimation)

        self._ball_body = None
        try:
            self._ball_body = self.model.get_body("ball")
        except Exception:
            self._ball_body = None
        self._ball_geom_contact_pairs, self._ball_geom_contact_rids = _mtx_ball_robot_contact_pairs(self.model)
        self._ball_qpos_adr: int | None = None
        self._ball_qvel_adr: int | None = None
        self._ball_qpos_idx: np.ndarray | None = None
        self._ball_qvel_idx: np.ndarray | None = None
        try:
            self._ball_qpos_adr = _mtx_joint_qpos_start(self.model, "ball-root")
            self._ball_qvel_adr = _mtx_joint_qvel_start(self.model, "ball-root")
        except Exception:
            pass
        if self._ball_body is not None:
            try:
                pos_idx = np.asarray(
                    self._ball_body.get_dof_pos_indices(include_floatingbase=True), dtype=np.int64
                ).reshape(-1)
                vel_idx = np.asarray(
                    self._ball_body.get_dof_vel_indices(include_floatingbase=True), dtype=np.int64
                ).reshape(-1)
                if pos_idx.size >= 7:
                    self._ball_qpos_idx = pos_idx[:7].astype(np.int64, copy=False)
                if vel_idx.size >= 6:
                    self._ball_qvel_idx = vel_idx[:6].astype(np.int64, copy=False)
            except Exception:
                self._ball_qpos_idx = None
                self._ball_qvel_idx = None

        self.policy_device = self._resolve_policy_device(args.policy_device)
        print(f"[MultiRobotMotrixSim] policy device: {self.policy_device}")
        self._policy_is_onnx = False
        self._policy_is_torchscript = False
        self._ts_policy = None
        self._onnx_session = None
        self._onnx_input_name = ""
        self._onnx_output_name = ""
        self.policy: nn.Module | None = None
        self._policy_obs_dim = 0
        self._policy_action_dim = 0
        self._init_policy(args.policy)
        self._k1_stand_policy: nn.Module | None = None
        if self.robot_cfg.k1_stand_policy is not None and not self.robot_cfg.use_k1_amp_onnx:
            self._init_k1_stand_policy(self.robot_cfg.k1_stand_policy)

        self.robot_specs: dict[int, RobotSpec] = self._build_robot_specs()
        self._apply_team_body_colors()
        if self.robot_specs:
            sample_spec = next(iter(self.robot_specs.values()))
            if self.robot_cfg.use_k1_amp_onnx:
                expected_obs_dim = K1_AMP_NUM_OBS
                expected_act_dim = K1_AMP_NUM_ACT
            elif self.robot_cfg.use_k1_legged_gym_policy:
                expected_obs_dim = K1_LEGGED_GYM_NUM_OBS
                expected_act_dim = K1_LEGGED_GYM_NUM_ACT
            else:
                expected_obs_dim = len(sample_spec.obs_history)
                expected_act_dim = len(sample_spec.last_action)
            if expected_obs_dim != self._policy_obs_dim:
                raise RuntimeError(
                    f"Policy obs dim mismatch: policy={self._policy_obs_dim} robot={expected_obs_dim} type={self.robot_cfg.robot_type}"
                )
            if expected_act_dim != self._policy_action_dim:
                raise RuntimeError(
                    f"Policy action dim mismatch: policy={self._policy_action_dim} robot={expected_act_dim} type={self.robot_cfg.robot_type}"
                )
        self.command_buffer: dict[int, np.ndarray] = {
            rid: np.array(DEFAULT_CMD, dtype=np.float32) for rid in FIXED_ROBOT_ID_TO_NAME
        }
        self.command_ts: dict[int, float] = {rid: float("-inf") for rid in FIXED_ROBOT_ID_TO_NAME}
        self.command_received: dict[int, bool] = {rid: False for rid in FIXED_ROBOT_ID_TO_NAME}
        for rid in self.robot_specs:
            self.command_received[rid] = True
        self.last_msg_info = {"timestamp": 0.0, "id": -1, "source": "unknown"}
        self._policy_step_count = 0
        self._policy_print_step = 0
        self._printed_target_policy_io = False

        self._startup_qpos = np.asarray(self.data.dof_pos, dtype=np.float64).reshape(-1).copy()
        self._startup_qvel = np.asarray(self.data.dof_vel, dtype=np.float64).reshape(-1).copy()
        self._startup_ctrl = np.asarray(self.data.actuator_ctrls, dtype=np.float64).reshape(-1).copy()
        self._startup_act = np.array([], dtype=np.float32)
        self._saved_spawn_points: dict[str, list[float]] = {}
        self._robot_protect_until: dict[int, float] = {}
        self._robot_protect_pose: dict[int, tuple[float, float, float]] = {}
        self._robot_cmd_zero_frames_left: dict[int, int] = {}
        self._fall_candidate_frames: dict[int, int] = {}
        self._enable_fall_recovery = True
        self._ball_last_touch_rid: int | None = None

        self.use_referee = bool(args.use_referee)
        self.referee: MujocoSoccerReferee | None = None
        if self.use_referee:
            goalie_area_depth = float(referee_area_cfg.get("goalie_area_depth", 1.0))
            goalie_area_width = float(referee_area_cfg.get("goalie_area_width", 3.0))
            goalie_area_extra_width = max(0.0, 0.5 * (goalie_area_width - self._goal_width))
            self.referee = MujocoSoccerReferee(
                field_length=self._field_length,
                field_width=self._field_width,
                goal_width=self._goal_width,
                goal_height=self._goal_height,
                goalie_area_depth=goalie_area_depth,
                goalie_area_extra_width=goalie_area_extra_width,
                red_count=self.max_red_robots,
                blue_count=self.max_blue_robots,
                left_team_number=int(team_meta_cfg["red"]["team_number"]),
                right_team_number=int(team_meta_cfg["blue"]["team_number"]),
                left_team_name=str(team_meta_cfg["red"]["team_name"]),
                right_team_name=str(team_meta_cfg["blue"]["team_name"]),
            )
            print("[MultiRobotMotrixSim] referee: enabled")
        else:
            print("[MultiRobotMotrixSim] referee: disabled")

        self._web_camera = None
        self._web_capture_camera_name = "lab_webview_diagonal"
        self._fallback_frame_logged = False
        self._render_scene_option = None
        self._step_fail_count = 0
        self._last_step_fail_log = 0.0
        self._last_web_cmd_log = 0.0
        if args.render_collision_meshes:
            print("[MultiRobotMotrixSim] render_collision_meshes ignored (MotrixSim web render path)")

    def _apply_team_body_colors(self) -> None:
        """Team tint was implemented via MuJoCo geom_rgba; MotrixSim path skips (visual-only)."""
        print("[MultiRobotMotrixSim] team color tint skipped under MotrixSim (MJCF materials unchanged)")

    @staticmethod
    def _active_ids_from_limits(max_red: int, max_blue: int) -> list[int]:
        ids = []
        ids.extend(range(0, max_red))
        ids.extend(range(MAX_ROBOTS_PER_TEAM, MAX_ROBOTS_PER_TEAM + max_blue))
        return ids

    @staticmethod
    def _resolve_policy_device(requested: str) -> torch.device:
        req = str(requested).strip().lower()
        if req == "cpu":
            return torch.device("cpu")
        if req == "gpu":
            if torch.cuda.is_available():
                return torch.device("cuda")
            print("[MultiRobotMotrixSim] policy device requested=gpu but CUDA is unavailable, fallback to cpu")
            return torch.device("cpu")
        raise ValueError(f"Unsupported policy device: {requested}")

    def _init_policy(self, policy_path: Path) -> None:
        path = Path(policy_path)
        if self.robot_cfg.use_k1_amp_onnx:
            try:
                import onnxruntime as ort
            except ImportError as e:
                raise RuntimeError("Install onnxruntime to run the K1 AMP ONNX policy (pip install onnxruntime).") from e
            if not path.is_file():
                raise FileNotFoundError(f"K1 AMP ONNX policy not found: {path}")
            self._onnx_session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
            ins = self._onnx_session.get_inputs()
            outs = self._onnx_session.get_outputs()
            if len(ins) != 1 or len(outs) != 1:
                raise RuntimeError(f"Expected 1 ONNX input and 1 output; got {len(ins)} / {len(outs)}")
            self._onnx_input_name = ins[0].name
            self._onnx_output_name = outs[0].name
            self._policy_obs_dim = K1_AMP_NUM_OBS
            self._policy_action_dim = K1_AMP_NUM_ACT
            self._policy_is_onnx = True
            print(
                f"[MultiRobotMotrixSim] K1 AMP ONNX (stacked {K1_AMP_FRAME_STACK}x{K1_AMP_NUM_SINGLE_OBS}): {path} "
                f"(obs_dim={self._policy_obs_dim}, act_dim={self._policy_action_dim})"
            )
            return
        if self.robot_cfg.use_k1_legged_gym_policy:
            suf = path.suffix.lower()
            if suf == ".onnx":
                try:
                    import onnxruntime as ort
                except ImportError as e:
                    raise RuntimeError(
                        "Install onnxruntime to run the K1 legged_gym ONNX policy (pip install onnxruntime)."
                    ) from e
                if not path.is_file():
                    raise FileNotFoundError(
                        f"K1 legged_gym ONNX policy not found: {path}\n"
                        "Place model_4700.onnx under legged_gym/policy/booster_k1/ or pass --policy."
                    )
                self._onnx_session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
                ins = self._onnx_session.get_inputs()
                outs = self._onnx_session.get_outputs()
                if len(ins) != 1 or len(outs) != 1:
                    raise RuntimeError(f"Expected 1 ONNX input and 1 output; got {len(ins)} / {len(outs)}")
                self._onnx_input_name = ins[0].name
                self._onnx_output_name = outs[0].name
                self._policy_obs_dim = K1_LEGGED_GYM_NUM_OBS
                self._policy_action_dim = K1_LEGGED_GYM_NUM_ACT
                self._policy_is_onnx = True
                print(
                    f"[MultiRobotMotrixSim] K1 legged_gym ONNX: {path} (obs_dim={self._policy_obs_dim}, act_dim={self._policy_action_dim})"
                )
                return

            if suf == ".pt":
                ts = None
                try:
                    ts = torch.jit.load(str(path), map_location=self.policy_device)
                    ts.eval()
                except Exception:
                    ts = None
                if ts is not None:
                    with torch.inference_mode():
                        probe = ts(torch.zeros(1, K1_LEGGED_GYM_NUM_OBS, device=self.policy_device))
                    if int(probe.shape[-1]) != K1_LEGGED_GYM_NUM_ACT:
                        raise RuntimeError(
                            f"K1 legged TorchScript expected act_dim={K1_LEGGED_GYM_NUM_ACT}, "
                            f"got {probe.shape} from {path}"
                        )
                    self._ts_policy = ts
                    self._policy_is_torchscript = True
                    self._policy_is_onnx = False
                    self.policy = None
                    self._policy_obs_dim = K1_LEGGED_GYM_NUM_OBS
                    self._policy_action_dim = K1_LEGGED_GYM_NUM_ACT
                    print(
                        f"[MultiRobotMotrixSim] K1 legged_gym TorchScript: {path} "
                        f"(obs_dim={self._policy_obs_dim}, act_dim={self._policy_action_dim})"
                    )
                    return

                self.policy = self._load_torch_policy(path)
                if self._policy_obs_dim != K1_LEGGED_GYM_NUM_OBS or self._policy_action_dim != K1_LEGGED_GYM_NUM_ACT:
                    raise RuntimeError(
                        f"K1 legged_gym expects obs_dim={K1_LEGGED_GYM_NUM_OBS} act_dim={K1_LEGGED_GYM_NUM_ACT}, "
                        f"got {self._policy_obs_dim} x {self._policy_action_dim} from {path}"
                    )
                self._policy_is_onnx = False
                self._policy_is_torchscript = False
                print(
                    f"[MultiRobotMotrixSim] K1 legged_gym Torch actor: {path} "
                    f"(obs_dim={self._policy_obs_dim}, act_dim={self._policy_action_dim})"
                )
                return

            raise RuntimeError(f"K1 legged_gym policy must be .onnx or .pt, got {path}")

        self.policy = self._load_torch_policy(path)

    def _load_mlp_actor_from_checkpoint(self, policy_path: Path) -> tuple[nn.Module, int, int]:
        ckpt = _load_checkpoint_compat(policy_path, map_location=self.policy_device)
        state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        if not isinstance(state_dict, dict):
            raise RuntimeError(f"Unsupported policy checkpoint format: {type(state_dict)}")
        actor_state = {k: v for k, v in state_dict.items() if k.startswith("actor.")}
        if not actor_state:
            raise RuntimeError("Checkpoint does not contain actor.* weights")

        actor_layer_dims: list[int] = []
        actor_weight_keys = sorted(
            (k for k in actor_state if re.match(r"^actor\.\d+\.weight$", k)),
            key=lambda s: int(s.split(".")[1]),
        )
        for i, wk in enumerate(actor_weight_keys):
            w = actor_state[wk]
            out_dim, in_dim = int(w.shape[0]), int(w.shape[1])
            if i == 0:
                actor_layer_dims.append(in_dim)
            actor_layer_dims.append(out_dim)

        in_dim = int(actor_layer_dims[0])
        out_dim = int(actor_layer_dims[-1])
        policy = MLPActor(layer_dims=actor_layer_dims).to(self.policy_device)
        policy.load_state_dict(actor_state, strict=True)
        policy.eval()
        return policy, in_dim, out_dim

    def _load_torch_policy(self, policy_path: Path) -> nn.Module:
        policy, in_d, out_d = self._load_mlp_actor_from_checkpoint(policy_path)
        self._policy_obs_dim = in_d
        self._policy_action_dim = out_d
        return policy

    def _init_k1_stand_policy(self, policy_path: Path) -> None:
        path = Path(policy_path)
        if not path.is_file():
            raise FileNotFoundError(f"K1 stand policy not found: {path}")
        policy, in_d, out_d = self._load_mlp_actor_from_checkpoint(path)
        if in_d != K1_FULL_BODY_NUM_OBS or out_d != K1_FULL_BODY_NUM_ACT:
            raise RuntimeError(
                f"K1 stand policy expects obs dim {K1_FULL_BODY_NUM_OBS} and act dim {K1_FULL_BODY_NUM_ACT}, "
                f"got {in_d} x {out_d} from {path}"
            )
        traced = None
        try:
            policy.eval()
            with torch.inference_mode():
                example = torch.zeros((1, in_d), device=self.policy_device, dtype=torch.float32)
                traced = torch.jit.trace(policy, example)
            traced.eval()
        except Exception as e:
            print(
                f"[MultiRobotMotrixSim] K1 stand policy TorchScript trace failed ({type(e).__name__}: {e}); "
                "using eager nn.Module (often slower than legged TorchScript)."
            )
        self._k1_stand_policy = traced if traced is not None else policy
        mode = "TorchScript traced" if traced is not None else "eager MLP"
        print(f"[MultiRobotMotrixSim] K1 hybrid stand (full-body) policy: {path} ({in_d}->{out_d}, {mode})")

    def _infer_policy_actions(self, obs_batch: np.ndarray) -> np.ndarray:
        if self._policy_is_onnx:
            ob = obs_batch.astype(np.float32, copy=False)
            b = int(ob.shape[0])
            in_meta = self._onnx_session.get_inputs()[0]
            shape0 = in_meta.shape[0] if in_meta.shape else None
            fixed_b = int(shape0) if isinstance(shape0, int) and shape0 > 0 else None
            # Many exported AMP checkpoints freeze batch=1; run one row at a time when needed.
            if fixed_b == 1 and b != 1:
                outs: list[np.ndarray] = []
                for i in range(b):
                    outs.append(
                        self._onnx_session.run(
                            [self._onnx_output_name],
                            {self._onnx_input_name: ob[i : i + 1]},
                        )[0]
                    )
                return np.concatenate(outs, axis=0).astype(np.float32, copy=False)
            return np.asarray(
                self._onnx_session.run([self._onnx_output_name], {self._onnx_input_name: ob})[0],
                dtype=np.float32,
            )
        if self._policy_is_torchscript:
            if self._ts_policy is None:
                raise RuntimeError("TorchScript policy not loaded")
            with torch.inference_mode():
                obs_tensor = torch.from_numpy(obs_batch.astype(np.float32, copy=False)).to(self.policy_device)
                return self._ts_policy(obs_tensor).detach().cpu().numpy().astype(np.float32, copy=False)
        if self.policy is None:
            raise RuntimeError("Torch policy not loaded")
        with torch.inference_mode():
            obs_tensor = torch.from_numpy(obs_batch).to(self.policy_device)
            return self.policy(obs_tensor).detach().cpu().numpy().astype(np.float32, copy=False)

    def _infer_k1_stand_actions(self, obs_batch: np.ndarray) -> np.ndarray:
        if self._k1_stand_policy is None:
            raise RuntimeError("K1 stand policy not loaded")
        with torch.inference_mode():
            obs_tensor = torch.from_numpy(obs_batch.astype(np.float32, copy=False)).to(self.policy_device)
            return self._k1_stand_policy(obs_tensor).detach().cpu().numpy().astype(np.float32, copy=False)

    def _k1_update_hybrid_walk_state(self, spec: RobotSpec, cmd: np.ndarray) -> None:
        vxy = float(np.hypot(float(cmd[0]), float(cmd[1])))
        wz = abs(float(cmd[2]))
        prev = spec.k1_hybrid_walking
        if prev:
            if vxy < K1_HYBRID_WALK_EXIT_LIN and wz < K1_HYBRID_WALK_EXIT_YAW:
                spec.k1_hybrid_walking = False
        else:
            if vxy >= K1_HYBRID_WALK_ENTER_LIN or wz >= K1_HYBRID_WALK_ENTER_YAW:
                spec.k1_hybrid_walking = True
        if prev != spec.k1_hybrid_walking:
            spec.last_action[:] = 0.0
            if spec.k1_legged_last_action is not None:
                spec.k1_legged_last_action[:] = 0.0

    def _obs_k1_fullbody_for_hybrid(self, spec: RobotSpec, cmd_override: np.ndarray | None = None) -> np.ndarray:
        obs_scale = self.robot_cfg.obs_scale
        qpos = self.data.dof_pos[0]
        qvel = self.data.dof_vel[0]
        base_lin_world = qvel[spec.base_qvel_adr : spec.base_qvel_adr + 3]
        base_ang_world = qvel[spec.base_qvel_adr + 3 : spec.base_qvel_adr + 6]
        quat = qpos[spec.base_qpos_adr + 3 : spec.base_qpos_adr + 7]
        rot_wb = _quat_to_rot_world_from_body(quat)
        base_lin = (rot_wb.T @ base_lin_world).astype(np.float32) * obs_scale["base_lin_vel"]
        base_ang = (rot_wb.T @ base_ang_world).astype(np.float32) * obs_scale["base_ang_vel"]
        gravity = (rot_wb.T @ np.array([0.0, 0.0, -1.0], dtype=np.float32)) * obs_scale["gravity_orientation"]
        cmd_src = self.command_buffer[spec.rid] if cmd_override is None else cmd_override
        cmd = cmd_src * obs_scale["cmd"]
        joint_pos = (qpos[spec.qpos_idx] - spec.init_joint_pos).astype(np.float32) * obs_scale["joint_pos"]
        joint_vel = qvel[spec.qvel_idx].astype(np.float32) * obs_scale["joint_vel"]
        last_action = spec.last_action * obs_scale["last_action"]
        obs_step = np.concatenate([base_lin, base_ang, gravity, cmd, joint_pos, joint_vel, last_action], axis=-1).astype(
            np.float32
        )
        obs_step = np.nan_to_num(obs_step, nan=0.0, posinf=0.0, neginf=0.0)
        c = float(self.robot_cfg.obs_clip)
        return np.clip(obs_step, -c, c)

    def _build_robot_specs(self) -> dict[int, RobotSpec]:
        specs: dict[int, RobotSpec] = {}
        joint_names = self.robot_cfg.policy_joint_names
        action_scale = build_action_scale_array(joint_names, self.robot_cfg.action_scale_cfg)
        if self.robot_cfg.use_k1_amp_onnx:
            obs_step_dim = K1_AMP_NUM_OBS
            obs_history_len = 1
            obs_dim = K1_AMP_NUM_OBS
        elif self.robot_cfg.use_k1_legged_gym_policy:
            obs_step_dim = (9 if self.robot_cfg.include_base_lin_vel_obs else 6) + 3 + 3 * len(joint_names)
            obs_history_len = max(1, int(self.robot_cfg.obs_history_length))
            obs_dim = obs_step_dim * obs_history_len
        else:
            obs_step_dim = (9 if self.robot_cfg.include_base_lin_vel_obs else 6) + 3 + 3 * len(joint_names)
            obs_history_len = max(1, int(self.robot_cfg.obs_history_length))
            obs_dim = obs_step_dim * obs_history_len

        for rid in self.active_robot_ids:
            name = FIXED_ROBOT_ID_TO_NAME[rid]
            team = "red" if rid < MAX_ROBOTS_PER_TEAM else "blue"
            pref_joints = [f"{name}__{j}" for j in joint_names]
            qpos_idx = []
            qvel_idx = []
            joint_lower = []
            joint_upper = []
            for jn in pref_joints:
                try:
                    qpos_idx.append(_mtx_joint_qpos_start(self.model, jn))
                    qvel_idx.append(_mtx_joint_qvel_start(self.model, jn))
                    lo, hi = _mtx_joint_limit_range(self.model, jn)
                    joint_lower.append(lo)
                    joint_upper.append(hi)
                except Exception as e:
                    raise RuntimeError(f"Missing policy joint: {jn}") from e

            act_idx: list[int] = []
            for jn in pref_joints:
                an = f"{name}__{jn.split('__', 1)[1]}"
                try:
                    act_idx.append(_mtx_actuator_index(self.model, an))
                except Exception as e:
                    raise RuntimeError(f"Missing actuator for policy joint: {jn} ({an})") from e

            base_jn = f"{name}__{self.robot_cfg.base_joint_name}"
            try:
                root_addrs = _mtx_root_base_dof_addrs(self.model, name)
                if root_addrs is not None:
                    base_qpos_adr, base_qvel_adr = root_addrs
                else:
                    base_qpos_adr = _mtx_joint_qpos_start(self.model, base_jn)
                    base_qvel_adr = _mtx_joint_qvel_start(self.model, base_jn)
            except Exception as e:
                raise RuntimeError(f"Missing base freejoint for {name}") from e

            init_joint_pos = self.data.dof_pos[0, np.asarray(qpos_idx, dtype=np.int32)].astype(np.float32).copy()
            # Enforce requested startup/reset joint pose for every robot.
            for jname, val in self.robot_cfg.reset_joint_pos.items():
                try:
                    local_idx = joint_names.index(jname)
                except ValueError:
                    continue
                init_joint_pos[local_idx] = float(val)
                self.data.dof_pos[0, qpos_idx[local_idx]] = float(val)
            init_angles = init_joint_pos.copy()
            pi_qpos_idx_mujoco = None
            pi_qvel_idx_mujoco = None
            pi_act_idx_mujoco = None
            pi_default_dof_pos = None
            pi_isaac_to_mujoco_idx = None
            pi_mujoco_to_isaac_idx = None
            pi_filtered_dof_target = None
            pi_target_dof_pos = None
            pi_joint_lower_mujoco = None
            pi_joint_upper_mujoco = None

            if self.robot_cfg.robot_type == PI_PLUS_ROBOT_TYPE:
                if len(joint_names) != len(PI_PLUS_KP_POLICY_ORDER) or len(joint_names) != len(PI_PLUS_KD_POLICY_ORDER):
                    raise RuntimeError("pi_plus kp/kd config length mismatch")
                kp = np.asarray(PI_PLUS_KP_POLICY_ORDER, dtype=np.float32)
                kd = np.asarray(PI_PLUS_KD_POLICY_ORDER, dtype=np.float32)
                pi_pref_joints = [f"{name}__{j}" for j in PI_PLUS_JOINTS_MUJOCO_ORDER]
                pi_qpos = []
                pi_qvel = []
                pi_act = []
                pi_lower = []
                pi_upper = []
                for jn in pi_pref_joints:
                    try:
                        pi_qpos.append(_mtx_joint_qpos_start(self.model, jn))
                        pi_qvel.append(_mtx_joint_qvel_start(self.model, jn))
                        pi_act.append(_mtx_actuator_index(self.model, jn))
                        lo, hi = _mtx_joint_limit_range(self.model, jn)
                        pi_lower.append(lo)
                        pi_upper.append(hi)
                    except Exception as e:
                        raise RuntimeError(f"Missing pi_plus mujoco-order joint/actuator: {jn}") from e
                pi_qpos_idx_mujoco = np.asarray(pi_qpos, dtype=np.int32)
                pi_qvel_idx_mujoco = np.asarray(pi_qvel, dtype=np.int32)
                pi_act_idx_mujoco = np.asarray(pi_act, dtype=np.int32)
                pi_joint_lower_mujoco = np.asarray(pi_lower, dtype=np.float32)
                pi_joint_upper_mujoco = np.asarray(pi_upper, dtype=np.float32)
                pi_default_dof_pos = PI_PLUS_DEFAULT_DOF_POS_MUJOCO.copy()
                pi_isaac_to_mujoco_idx = PI_PLUS_ISAAC_TO_MUJOCO_IDX.copy()
                pi_mujoco_to_isaac_idx = PI_PLUS_MUJOCO_TO_ISAAC_IDX.copy()
                # Keep startup pose exactly aligned with sim2sim_pi_plus.py default_dof_pos.
                self.data.dof_pos[0, pi_qpos_idx_mujoco] = pi_default_dof_pos
                pi_filtered_dof_target = pi_default_dof_pos.copy()
                pi_target_dof_pos = pi_default_dof_pos.copy()
            else:
                kp = parse_param_for_joint_names(pref_joints, self.robot_cfg.motor_stiffness)
                kd = parse_param_for_joint_names(pref_joints, self.robot_cfg.motor_damping)
            effort = parse_param_for_joint_names(pref_joints, self.robot_cfg.motor_effort_limit)

            k1_legged_last_action = None
            k1_legged_default_dof = None
            k1_legged_joint_indices = None
            k1_non_leg_mask = None
            k1_gait_phase = None
            if self.robot_cfg.use_k1_legged_gym_policy:
                leg_idx_list = [joint_names.index(jn) for jn in K1_LEGGED_GYM_LEG_JOINTS_POLICY_ORDER]
                k1_legged_joint_indices = np.asarray(leg_idx_list, dtype=np.int32)
                k1_non_leg_mask = np.ones((len(joint_names),), dtype=bool)
                k1_non_leg_mask[k1_legged_joint_indices] = False
                k1_legged_default_dof = np.zeros(K1_LEGGED_GYM_NUM_ACT, dtype=np.float32)
                for i, jn in enumerate(K1_LEGGED_GYM_LEG_JOINTS_POLICY_ORDER):
                    k1_legged_default_dof[i] = float(
                        K1_LEGGED_GYM_DEFAULT_JOINT_ANGLES.get(jn, K1_LEGGED_GYM_DEFAULT_JOINT_ANGLES["default"])
                    )
                k1_legged_last_action = np.zeros(K1_LEGGED_GYM_NUM_ACT, dtype=np.float32)
                k1_gait_phase = 0.0
                for i, jn in enumerate(K1_LEGGED_GYM_LEG_JOINTS_POLICY_ORDER):
                    li = int(leg_idx_list[i])
                    kp[li] = float(K1_LEGGED_GYM_KP[jn])
                    kd[li] = float(K1_LEGGED_GYM_KD[jn])
                    effort[li] = float(K1_LEGGED_GYM_TORQUE_LIMIT)
            k1_amp_hist = None
            k1_amp_last_mjcf = None
            k1_amp_gather_mjcf = None
            if self.robot_cfg.use_k1_amp_onnx:
                k1_amp_gather_mjcf = np.asarray(
                    [joint_names.index(n) for n in K1_AMP_ACTUATOR_JOINT_ORDER],
                    dtype=np.int32,
                )
                k1_amp_hist = np.zeros((K1_AMP_FRAME_STACK, K1_AMP_NUM_SINGLE_OBS), dtype=np.float32)
                k1_amp_last_mjcf = np.zeros((K1_AMP_NUM_ACT,), dtype=np.float32)
            specs[rid] = RobotSpec(
                rid=rid,
                name=name,
                team=team,
                qpos_idx=np.asarray(qpos_idx, dtype=np.int32),
                qvel_idx=np.asarray(qvel_idx, dtype=np.int32),
                act_idx=np.asarray(act_idx, dtype=np.int32),
                act_qpos_idx=np.asarray(qpos_idx, dtype=np.int32),
                act_qvel_idx=np.asarray(qvel_idx, dtype=np.int32),
                base_qpos_adr=base_qpos_adr,
                base_qvel_adr=base_qvel_adr,
                init_joint_pos=init_joint_pos,
                init_angles=init_angles,
                filtered_dof_target=init_angles.copy(),
                target_joint_pos=init_angles.copy(),
                last_action=np.zeros(len(joint_names), dtype=np.float32),
                action_scale=action_scale.copy(),
                kp=kp,
                kd=kd,
                effort=effort,
                obs_step_dim=obs_step_dim,
                obs_history=np.zeros((obs_dim,), dtype=np.float32),
                pi_qpos_idx_mujoco=pi_qpos_idx_mujoco,
                pi_qvel_idx_mujoco=pi_qvel_idx_mujoco,
                pi_act_idx_mujoco=pi_act_idx_mujoco,
                pi_default_dof_pos=pi_default_dof_pos,
                pi_isaac_to_mujoco_idx=pi_isaac_to_mujoco_idx,
                pi_mujoco_to_isaac_idx=pi_mujoco_to_isaac_idx,
                pi_filtered_dof_target=pi_filtered_dof_target,
                pi_target_dof_pos=pi_target_dof_pos,
                joint_lower=np.asarray(joint_lower, dtype=np.float32),
                joint_upper=np.asarray(joint_upper, dtype=np.float32),
                pi_joint_lower_mujoco=pi_joint_lower_mujoco,
                pi_joint_upper_mujoco=pi_joint_upper_mujoco,
                k1_legged_last_action=k1_legged_last_action,
                k1_legged_default_dof=k1_legged_default_dof,
                k1_legged_joint_indices=k1_legged_joint_indices,
                k1_non_leg_mask=k1_non_leg_mask,
                k1_gait_phase=k1_gait_phase,
                k1_amp_hist=k1_amp_hist,
                k1_amp_last_mjcf=k1_amp_last_mjcf,
                k1_amp_gather_mjcf=k1_amp_gather_mjcf,
            )
        return specs

    def set_command(self, vx, vy, w, robot_id=0, timestamp=0, source="unknown"):
        if robot_id not in FIXED_ROBOT_ID_TO_NAME:
            return
        if robot_id not in self.robot_specs:
            return
        if not self._is_command_allowed(robot_id):
            return
        if self._is_robot_protected(robot_id):
            return
        vx = float(vx)
        vy = float(vy)
        w = float(w)
        if self.robot_cfg.cmd_clip is not None:
            vx_lim, vy_lim, w_lim = self.robot_cfg.cmd_clip
            vx = float(np.clip(vx, -float(vx_lim), float(vx_lim)))
            vy = float(np.clip(vy, -float(vy_lim), float(vy_lim)))
            w = float(np.clip(w, -float(w_lim), float(w_lim)))
        vx = float(np.clip(vx, CMD_VEL_NORM_MIN, CMD_VEL_NORM_MAX))
        vy = float(np.clip(vy, CMD_VEL_NORM_MIN, CMD_VEL_NORM_MAX))
        w = float(np.clip(w, CMD_VEL_NORM_MIN, CMD_VEL_NORM_MAX))
        ts = float(timestamp) if timestamp else time.time()
        if ts < self.command_ts[robot_id]:
            return
        self.command_ts[robot_id] = ts
        self.command_buffer[robot_id] = np.array([vx, vy, w], dtype=np.float32)
        self.command_received[robot_id] = True
        self.last_msg_info = {"timestamp": ts, "id": int(robot_id), "source": str(source)}

    def _is_command_allowed(self, robot_id: int) -> bool:
        # User policy: never block either team commands by play mode.
        # Referee remains active for game state/ball placement, but command gating is disabled.
        _ = robot_id
        return True

    def _k1_legged_cmd_obs_from_norm_cmd(self, cmd: np.ndarray) -> np.ndarray:
        """T1_env obs[6:9] is raw command × scale (~m/s, rad/s), not [0,1]. Zeros must stay 0.

        Do **not** use (u+1)/2: that maps a single-axis command to biased values on the other axes
        (e.g. [0,0,ω] → [0.5,0.5,…]) and breaks axis decoupling in the policy.
        """
        return np.clip(np.asarray(cmd, dtype=np.float32).reshape(3), CMD_VEL_NORM_MIN, CMD_VEL_NORM_MAX)

    def _obs_k1_legged_gym(self, spec: RobotSpec, cmd_override: np.ndarray | None = None) -> np.ndarray:
        """Observation matching legged_gym/envs/T1/T1.py compute_obs (47-dim K1 locomotion)."""
        if (
            spec.k1_legged_default_dof is None
            or spec.k1_legged_joint_indices is None
            or spec.k1_legged_last_action is None
            or spec.k1_gait_phase is None
        ):
            raise RuntimeError("K1 legged_gym buffers are not initialized on RobotSpec")
        qpos = self.data.dof_pos[0]
        qvel = self.data.dof_vel[0]
        quat = qpos[spec.base_qpos_adr + 3 : spec.base_qpos_adr + 7]
        rot_wb = _quat_to_rot_world_from_body(quat)
        gravity = (rot_wb.T @ np.array([0.0, 0.0, -1.0], dtype=np.float32)) * float(K1_LEGGED_GYM_GRAVITY_SCALE)

        # Prefer MJCF gyro on imu (matches legged_gym `get_sensor_value("gyro", ...)` on their scene).
        # Soccer K1 asset names the gyro `angular-velocity`; training scene may use `gyro`.
        gyro = None
        for sname in (f"{spec.name}__angular-velocity", f"{spec.name}__gyro"):
            try:
                raw = np.asarray(self.model.get_sensor_value(sname, self.data), dtype=np.float32).reshape(-1)
                if raw.size >= 3:
                    gyro = raw[:3].copy()
                    break
            except BaseException:
                continue
        if gyro is None:
            base_ang_world = qvel[spec.base_qvel_adr + 3 : spec.base_qvel_adr + 6]
            gyro = (rot_wb.T @ base_ang_world.astype(np.float32)).astype(np.float32)
        gyro_scaled = gyro.astype(np.float32) * float(K1_LEGGED_GYM_GYRO_SCALE_LIN_VEL)

        cmd_src = self.command_buffer[spec.rid] if cmd_override is None else cmd_override
        command_scale = np.asarray([1.0, 1.0, 1.0], dtype=np.float32)
        cmd_scaled = self._k1_legged_cmd_obs_from_norm_cmd(cmd_src) * command_scale

        gait = float(spec.k1_gait_phase)
        gf = float(K1_LEGGED_GYM_GAIT_FREQUENCY)
        gait_mask = 1.0 if gf > 1.0e-8 else 0.0
        c_g = float(np.cos(2 * np.pi * gait) * gait_mask)
        s_g = float(np.sin(2 * np.pi * gait) * gait_mask)

        q_j = qpos[spec.qpos_idx].astype(np.float32)
        dq_j = qvel[spec.qvel_idx].astype(np.float32)
        leg_i = spec.k1_legged_joint_indices
        dof_pos_leg = q_j[leg_i]
        dof_vel_leg = dq_j[leg_i]
        diff = (dof_pos_leg - spec.k1_legged_default_dof) * float(K1_LEGGED_GYM_DOF_POS_SCALE)
        dvel = dof_vel_leg * float(K1_LEGGED_GYM_DOF_VEL_SCALE)
        last_a = spec.k1_legged_last_action.astype(np.float32)

        obs = np.zeros((K1_LEGGED_GYM_NUM_OBS,), dtype=np.float32)
        obs[0:3] = gravity
        obs[3:6] = gyro_scaled
        obs[6:9] = cmd_scaled
        obs[9] = c_g
        obs[10] = s_g
        obs[11:23] = diff
        obs[23:35] = dvel
        obs[35:47] = last_a
        obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        c = float(self.robot_cfg.obs_clip)
        return np.clip(obs, -c, c)

    def _obs_k1_amp(self, spec: RobotSpec, cmd_override: np.ndarray | None = None) -> np.ndarray:
        """Observation matching legged_gym/envs/K1_amp/K1_amp.py (75 * frame_stack = 375)."""
        if spec.k1_amp_hist is None or spec.k1_amp_last_mjcf is None or spec.k1_amp_gather_mjcf is None:
            raise RuntimeError("K1 AMP buffers are not initialized on RobotSpec")
        qpos = self.data.dof_pos[0]
        qvel = self.data.dof_vel[0]
        cmd_src = self.command_buffer[spec.rid] if cmd_override is None else cmd_override
        cmd_n = np.clip(np.asarray(cmd_src, dtype=np.float32).reshape(3), CMD_VEL_NORM_MIN, CMD_VEL_NORM_MAX)
        phys = np.array(
            [
                float(cmd_n[0]) * float(K1_AMP_CMD_MAX_LIN_X),
                float(cmd_n[1]) * float(K1_AMP_CMD_MAX_LIN_Y),
                float(cmd_n[2]) * float(K1_AMP_CMD_MAX_YAW),
            ],
            dtype=np.float32,
        )
        cmd_obs = phys * np.array([1.0, 1.0, 0.25], dtype=np.float32)

        quat = qpos[spec.base_qpos_adr + 3 : spec.base_qpos_adr + 7]
        rot_wb = _quat_to_rot_world_from_body(quat)
        gravity = (rot_wb.T @ np.array([0.0, 0.0, -1.0], dtype=np.float32)).astype(np.float32)

        gyro = None
        for sname in (f"{spec.name}__angular-velocity", f"{spec.name}__gyro"):
            try:
                raw = np.asarray(self.model.get_sensor_value(sname, self.data), dtype=np.float32).reshape(-1)
                if raw.size >= 3:
                    gyro = raw[:3].copy()
                    break
            except BaseException:
                continue
        if gyro is None:
            base_ang_world = qvel[spec.base_qvel_adr + 3 : spec.base_qvel_adr + 6]
            gyro = (rot_wb.T @ base_ang_world.astype(np.float32)).astype(np.float32)
        gyro_scaled = gyro.astype(np.float32) * 0.25

        q_j = qpos[spec.qpos_idx].astype(np.float32)
        dq_j = qvel[spec.qvel_idx].astype(np.float32)
        gather = spec.k1_amp_gather_mjcf
        q_mjcf = q_j[gather]
        dq_mjcf = dq_j[gather]
        default_mjcf = spec.init_angles[gather]
        diff = (q_mjcf - default_mjcf) * 1.0
        dvel = dq_mjcf * 0.05

        single = np.zeros((K1_AMP_NUM_SINGLE_OBS,), dtype=np.float32)
        single[0:3] = cmd_obs
        single[3:6] = gravity
        single[6:9] = gyro_scaled
        single[9:31] = diff
        single[31:53] = dvel
        single[53:75] = spec.k1_amp_last_mjcf

        hist = spec.k1_amp_hist
        hist[:-1] = hist[1:].copy()
        hist[-1] = single
        flat = hist.reshape(-1)
        c = float(self.robot_cfg.obs_clip)
        return np.clip(np.nan_to_num(flat, nan=0.0, posinf=0.0, neginf=0.0), -c, c)

    def _obs_for_robot(self, spec: RobotSpec, cmd_override: np.ndarray | None = None) -> np.ndarray:
        if self.robot_cfg.use_k1_legged_gym_policy:
            return self._obs_k1_legged_gym(spec, cmd_override=cmd_override)
        obs_scale = self.robot_cfg.obs_scale
        qpos = self.data.dof_pos[0]
        qvel = self.data.dof_vel[0]

        base_lin_world = qvel[spec.base_qvel_adr : spec.base_qvel_adr + 3]
        base_ang_world = qvel[spec.base_qvel_adr + 3 : spec.base_qvel_adr + 6]
        quat = qpos[spec.base_qpos_adr + 3 : spec.base_qpos_adr + 7]
        rot_wb = _quat_to_rot_world_from_body(quat)

        base_lin = (rot_wb.T @ base_lin_world).astype(np.float32) * obs_scale["base_lin_vel"]
        if self.robot_cfg.robot_type == PI_PLUS_ROBOT_TYPE:
            # Match sim2sim_pi_plus.py: angular velocity term is consumed directly.
            base_ang = base_ang_world.astype(np.float32) * obs_scale["base_ang_vel"]
        else:
            base_ang = (rot_wb.T @ base_ang_world).astype(np.float32) * obs_scale["base_ang_vel"]
        gravity = (rot_wb.T @ np.array([0.0, 0.0, -1.0], dtype=np.float32)) * obs_scale["gravity_orientation"]
        cmd_src = self.command_buffer[spec.rid] if cmd_override is None else cmd_override

        cmd = cmd_src * obs_scale["cmd"]

        if (
            self.robot_cfg.robot_type == PI_PLUS_ROBOT_TYPE
            and spec.pi_qpos_idx_mujoco is not None
            and spec.pi_qvel_idx_mujoco is not None
            and spec.pi_default_dof_pos is not None
            and spec.pi_mujoco_to_isaac_idx is not None
        ):
            # Keep pi_plus observation assembly aligned with sim2sim_pi_plus.py.
            dof_pos = qpos[spec.pi_qpos_idx_mujoco].astype(np.float32)
            dof_vel = qvel[spec.pi_qvel_idx_mujoco].astype(np.float32)
            sensor_ang_name = f"{spec.name}__angular-velocity"
            sensor_ori_name = f"{spec.name}__orientation"
            try:
                base_ang_pi = _mtx_sensor_vec(self.model, self.data, sensor_ang_name)
            except Exception:
                base_ang_pi = base_ang.astype(np.float32)
            try:
                ori_wxyz = _mtx_sensor_vec(self.model, self.data, sensor_ori_name)
                quat_xyzw = ori_wxyz[[1, 2, 3, 0]]
            except Exception:
                quat_xyzw = np.asarray([quat[1], quat[2], quat[3], quat[0]], dtype=np.float32)
            gravity_pi = _quat_apply_inverse(quat_xyzw, np.array([0.0, 0.0, -1.0], dtype=np.float32)).astype(np.float32)
            obs_step = np.zeros((spec.obs_step_dim,), dtype=np.float32)
            obs_step[0:3] = base_ang_pi * obs_scale["base_ang_vel"]
            obs_step[3:6] = gravity_pi * obs_scale["gravity_orientation"]
            obs_step[6:9] = cmd.astype(np.float32)
            obs_step[9:29] = (
                (dof_pos - spec.pi_default_dof_pos)[spec.pi_mujoco_to_isaac_idx].astype(np.float32) * obs_scale["joint_pos"]
            )
            obs_step[29:49] = dof_vel[spec.pi_mujoco_to_isaac_idx].astype(np.float32) * obs_scale["joint_vel"]
            obs_step[49:69] = (
                np.clip(spec.last_action, ACTION_CLIP[0], ACTION_CLIP[1]).astype(np.float32) * obs_scale["last_action"]
            )
            obs_step = np.nan_to_num(obs_step, nan=0.0, posinf=0.0, neginf=0.0)
            if self.robot_cfg.obs_history_length <= 1:
                return np.clip(obs_step, -self.robot_cfg.obs_clip, self.robot_cfg.obs_clip)
            spec.obs_history = np.roll(spec.obs_history, shift=-spec.obs_step_dim)
            spec.obs_history[-spec.obs_step_dim :] = obs_step
            return np.clip(spec.obs_history, -self.robot_cfg.obs_clip, self.robot_cfg.obs_clip)

        joint_pos = (qpos[spec.qpos_idx] - spec.init_joint_pos).astype(np.float32) * obs_scale["joint_pos"]
        joint_vel = qvel[spec.qvel_idx].astype(np.float32) * obs_scale["joint_vel"]
        last_action = spec.last_action * obs_scale["last_action"]

        obs_terms = [base_ang, gravity, cmd, joint_pos, joint_vel, last_action]
        if self.robot_cfg.include_base_lin_vel_obs:
            obs_terms.insert(0, base_lin)
        obs_step = np.concatenate(obs_terms, axis=-1).astype(np.float32)
        obs_step = np.nan_to_num(obs_step, nan=0.0, posinf=0.0, neginf=0.0)

        if self.robot_cfg.obs_history_length <= 1:
            return np.clip(obs_step, -self.robot_cfg.obs_clip, self.robot_cfg.obs_clip)
        spec.obs_history = np.roll(spec.obs_history, shift=-spec.obs_step_dim)
        spec.obs_history[-spec.obs_step_dim :] = obs_step
        return np.clip(spec.obs_history, -self.robot_cfg.obs_clip, self.robot_cfg.obs_clip)

    def _compute_targets(self):
        self._policy_step_count += 1
        debug_rid = FIXED_ROBOT_NAME_TO_ID.get("robot_rp0")
        if debug_rid not in self.robot_specs:
            debug_rid = next(iter(self.robot_specs.keys()), None)
        debug_obs = None
        debug_act = None
        infer_specs: list[RobotSpec] = []
        infer_cmd_override: list[np.ndarray | None] = []
        default_cmd = np.asarray(DEFAULT_CMD, dtype=np.float32)
        hybrid = self._k1_stand_policy is not None

        for spec in self.robot_specs.values():
            if self._is_robot_protected(spec.rid):
                spec.last_action[:] = 0.0
                if spec.k1_legged_last_action is not None:
                    spec.k1_legged_last_action[:] = 0.0
                if spec.k1_amp_last_mjcf is not None:
                    spec.k1_amp_last_mjcf[:] = 0.0
                if spec.k1_amp_hist is not None:
                    spec.k1_amp_hist.fill(0.0)
                spec.k1_hybrid_walking = False
                spec.filtered_dof_target[:] = spec.init_angles
                spec.target_joint_pos[:] = spec.init_angles
                if spec.pi_default_dof_pos is not None and spec.pi_filtered_dof_target is not None and spec.pi_target_dof_pos is not None:
                    spec.pi_filtered_dof_target[:] = spec.pi_default_dof_pos
                    spec.pi_target_dof_pos[:] = spec.pi_default_dof_pos
                continue

            zero_left = self._robot_cmd_zero_frames_left.get(spec.rid, 0)
            cmd_ov = default_cmd if zero_left > 0 else None
            if zero_left > 0:
                self._robot_cmd_zero_frames_left[spec.rid] = zero_left - 1
            infer_specs.append(spec)
            infer_cmd_override.append(cmd_ov)

        walk_obs: list[np.ndarray] = []
        stand_obs: list[np.ndarray] = []
        debug_obs_list: list[np.ndarray] = []

        if len(infer_specs) != len(infer_cmd_override):
            raise RuntimeError("internal: infer_specs / infer_cmd_override length mismatch")
        for spec, cmd_ov in zip(infer_specs, infer_cmd_override):
            cmd_gate = default_cmd if cmd_ov is not None else self.command_buffer[spec.rid]
            if hybrid:
                self._k1_update_hybrid_walk_state(
                    spec, np.clip(np.asarray(cmd_gate, dtype=np.float32), CMD_VEL_NORM_MIN, CMD_VEL_NORM_MAX)
                )
                if spec.k1_hybrid_walking:
                    o = self._obs_k1_legged_gym(spec, cmd_override=cmd_ov)
                    walk_obs.append(o)
                else:
                    o = self._obs_k1_fullbody_for_hybrid(spec, cmd_override=cmd_ov)
                    stand_obs.append(o)
            elif self.robot_cfg.use_k1_amp_onnx:
                o = self._obs_k1_amp(spec, cmd_override=cmd_ov)
                walk_obs.append(o)
            elif self.robot_cfg.use_k1_legged_gym_policy:
                o = self._obs_k1_legged_gym(spec, cmd_override=cmd_ov)
                walk_obs.append(o)
            else:
                o = self._obs_for_robot(spec, cmd_override=cmd_ov)
                walk_obs.append(o)
            debug_obs_list.append(o)

        act_walk_batch: np.ndarray | None = None
        if walk_obs:
            obs_batch_w = np.stack(walk_obs, axis=0).astype(np.float32, copy=False)
            act_walk_batch = self._infer_policy_actions(obs_batch_w)
            if act_walk_batch.ndim == 1:
                act_walk_batch = act_walk_batch.reshape(1, -1)

        act_stand_batch: np.ndarray | None = None
        if stand_obs:
            obs_batch_s = np.stack(stand_obs, axis=0).astype(np.float32, copy=False)
            act_stand_batch = self._infer_k1_stand_actions(obs_batch_s)
            if act_stand_batch.ndim == 1:
                act_stand_batch = act_stand_batch.reshape(1, -1)

        wi = 0
        si = 0
        if infer_specs:
            for i, spec in enumerate(infer_specs):
                hybrid_stand = bool(hybrid and not spec.k1_hybrid_walking)
                if hybrid_stand:
                    if act_stand_batch is None:
                        raise RuntimeError("K1 hybrid stand policy produced no actions")
                    act = np.nan_to_num(act_stand_batch[si], nan=0.0, posinf=0.0, neginf=0.0)
                    si += 1
                else:
                    if act_walk_batch is None:
                        raise RuntimeError("K1 walk policy produced no actions")
                    act = np.nan_to_num(act_walk_batch[wi], nan=0.0, posinf=0.0, neginf=0.0)
                    wi += 1
                if spec.rid == debug_rid:
                    debug_obs = debug_obs_list[i].copy()
                    debug_act = act.copy()
                if self.robot_cfg.use_k1_amp_onnx:
                    act_mjcf = act.astype(np.float32, copy=False).reshape(K1_AMP_NUM_ACT)
                    if spec.k1_amp_last_mjcf is not None:
                        spec.k1_amp_last_mjcf[:] = act_mjcf
                    pol_to_mjcf = np.asarray(
                        [K1_AMP_ACTUATOR_JOINT_ORDER.index(n) for n in self.robot_cfg.policy_joint_names],
                        dtype=np.int64,
                    )
                    act_pol = act_mjcf[pol_to_mjcf]
                    spec.last_action[:] = act_pol
                    delta = np.clip(
                        act_pol * float(K1_AMP_ACTION_SCALE),
                        ACTION_CLIP[0],
                        ACTION_CLIP[1],
                    )
                    joint_pos_action = spec.init_angles + delta
                    if ACTION_SMOOTH_FILTER:
                        spec.filtered_dof_target[:] = spec.filtered_dof_target * 0.2 + joint_pos_action * 0.8
                    else:
                        spec.filtered_dof_target[:] = joint_pos_action
                    if spec.joint_lower is not None and spec.joint_upper is not None:
                        np.clip(
                            spec.filtered_dof_target,
                            spec.joint_lower,
                            spec.joint_upper,
                            out=spec.filtered_dof_target,
                        )
                    spec.target_joint_pos[:] = spec.filtered_dof_target
                    continue
                if self.robot_cfg.use_k1_legged_gym_policy:
                    if hybrid_stand:
                        spec.last_action[:] = act
                        act_scaled = np.clip(act * spec.action_scale, ACTION_CLIP[0], ACTION_CLIP[1])
                        joint_pos_action = spec.init_angles + act_scaled
                        if ACTION_SMOOTH_FILTER:
                            spec.filtered_dof_target[:] = spec.filtered_dof_target * 0.2 + joint_pos_action * 0.8
                        else:
                            spec.filtered_dof_target[:] = joint_pos_action
                        if spec.joint_lower is not None and spec.joint_upper is not None:
                            np.clip(
                                spec.filtered_dof_target,
                                spec.joint_lower,
                                spec.joint_upper,
                                out=spec.filtered_dof_target,
                            )
                        spec.target_joint_pos[:] = spec.filtered_dof_target
                        continue
                    if (
                        spec.k1_legged_joint_indices is None
                        or spec.k1_legged_default_dof is None
                        or spec.k1_legged_last_action is None
                    ):
                        continue
                    act = act.astype(np.float32, copy=False)
                    spec.k1_legged_last_action[:] = act
                    spec.last_action[:] = 0.0
                    for kk in range(K1_LEGGED_GYM_NUM_ACT):
                        jidx = int(spec.k1_legged_joint_indices[kk])
                        spec.last_action[jidx] = act[kk]
                    spec.target_joint_pos[:] = spec.init_angles
                    asc = float(K1_LEGGED_GYM_ACTION_SCALE)
                    for kk in range(K1_LEGGED_GYM_NUM_ACT):
                        jidx = int(spec.k1_legged_joint_indices[kk])
                        spec.target_joint_pos[jidx] = float(spec.k1_legged_default_dof[kk] + act[kk] * asc)
                    if spec.joint_lower is not None and spec.joint_upper is not None:
                        np.clip(spec.target_joint_pos, spec.joint_lower, spec.joint_upper, out=spec.target_joint_pos)
                    if spec.k1_non_leg_mask is not None:
                        spec.target_joint_pos[spec.k1_non_leg_mask] = spec.init_joint_pos[spec.k1_non_leg_mask]
                    spec.filtered_dof_target[:] = spec.target_joint_pos
                elif (
                    self.robot_cfg.robot_type == PI_PLUS_ROBOT_TYPE
                    and spec.pi_default_dof_pos is not None
                    and spec.pi_isaac_to_mujoco_idx is not None
                    and spec.pi_mujoco_to_isaac_idx is not None
                    and spec.pi_filtered_dof_target is not None
                    and spec.pi_target_dof_pos is not None
                ):
                    act = np.clip(act, ACTION_CLIP[0], ACTION_CLIP[1]).astype(np.float32, copy=False)
                    spec.last_action[:] = act
                    actions_scaled = act * spec.action_scale
                    target_dof_pos = actions_scaled[spec.pi_isaac_to_mujoco_idx] + spec.pi_default_dof_pos
                    if ACTION_SMOOTH_FILTER:
                        spec.pi_filtered_dof_target[:] = spec.pi_filtered_dof_target * 0.2 + target_dof_pos * 0.8
                    else:
                        spec.pi_filtered_dof_target[:] = target_dof_pos
                    if spec.pi_joint_lower_mujoco is not None and spec.pi_joint_upper_mujoco is not None:
                        np.clip(
                            spec.pi_filtered_dof_target,
                            spec.pi_joint_lower_mujoco,
                            spec.pi_joint_upper_mujoco,
                            out=spec.pi_filtered_dof_target,
                        )
                    spec.pi_target_dof_pos[:] = spec.pi_filtered_dof_target
                    # Keep legacy buffers coherent (policy order) for debug/reset consistency.
                    spec.target_joint_pos[:] = spec.pi_target_dof_pos[spec.pi_mujoco_to_isaac_idx]
                    spec.filtered_dof_target[:] = spec.target_joint_pos
                else:
                    spec.last_action[:] = act
                    act_scaled = np.clip(act * spec.action_scale, ACTION_CLIP[0], ACTION_CLIP[1])
                    joint_pos_action = spec.init_angles + act_scaled
                    if ACTION_SMOOTH_FILTER:
                        spec.filtered_dof_target[:] = spec.filtered_dof_target * 0.2 + joint_pos_action * 0.8
                    else:
                        spec.filtered_dof_target[:] = joint_pos_action
                    if spec.joint_lower is not None and spec.joint_upper is not None:
                        np.clip(spec.filtered_dof_target, spec.joint_lower, spec.joint_upper, out=spec.filtered_dof_target)
                    spec.target_joint_pos[:] = spec.filtered_dof_target

        if (
            not self._printed_target_policy_io
            and self._policy_step_count == self._policy_print_step
            and debug_rid is not None
            and debug_obs is not None
            and debug_act is not None
        ):
            debug_name = FIXED_ROBOT_ID_TO_NAME.get(debug_rid, f"id{debug_rid}")
            np.set_printoptions(precision=6, suppress=True)
            print(f"[Policy Frame {self._policy_step_count}] robot={debug_name} input(obs): {debug_obs}")
            print(f"[Policy Frame {self._policy_step_count}] robot={debug_name} output(action): {debug_act}")
            self._printed_target_policy_io = True

    def _set_actuator_ctrls(self, act_idx: np.ndarray, values: np.ndarray | float) -> None:
        """
        Prefer actuator.set_ctrl(): in some Motrix bindings, direct writes to
        data.actuator_ctrls are not applied to simulator state.
        """
        idx = np.asarray(act_idx, dtype=np.int64).reshape(-1)
        if idx.size == 0:
            return
        val = np.asarray(values, dtype=np.float32).reshape(-1)
        if val.size == 1 and idx.size > 1:
            val = np.full((idx.size,), float(val[0]), dtype=np.float32)
        n = min(idx.size, val.size)
        for k in range(n):
            ai = int(idx[k])
            ctrl = float(val[k])
            applied = False
            try:
                actuator = self.model.get_actuator(ai)
                if actuator is not None:
                    actuator.set_ctrl(self.data, np.array([ctrl], dtype=np.float32))
                    applied = True
            except Exception:
                applied = False
            if not applied:
                try:
                    self.data.actuator_ctrls[0, ai] = ctrl
                except Exception:
                    pass

    def _apply_torque(self):
        for spec in self.robot_specs.values():
            if self._is_robot_protected(spec.rid):
                self._set_actuator_ctrls(spec.act_idx, self._startup_ctrl[spec.act_idx])
                if spec.pi_act_idx_mujoco is not None:
                    self._set_actuator_ctrls(spec.pi_act_idx_mujoco, self._startup_ctrl[spec.pi_act_idx_mujoco])
                continue
            use_pi_pd = (
                self.robot_cfg.robot_type == PI_PLUS_ROBOT_TYPE
                and spec.pi_qpos_idx_mujoco is not None
                and spec.pi_qvel_idx_mujoco is not None
                and spec.pi_act_idx_mujoco is not None
                and spec.pi_target_dof_pos is not None
            )
            if use_pi_pd:
                q = self.data.dof_pos[0, spec.pi_qpos_idx_mujoco]
                qd = self.data.dof_vel[0, spec.pi_qvel_idx_mujoco]
            else:
                q = self.data.dof_pos[0, spec.qpos_idx]
                qd = self.data.dof_vel[0, spec.qvel_idx]
            q = np.nan_to_num(q, nan=0.0, posinf=0.0, neginf=0.0)
            qd = np.nan_to_num(qd, nan=0.0, posinf=0.0, neginf=0.0)
            if use_pi_pd:
                target = np.nan_to_num(spec.pi_target_dof_pos, nan=0.0, posinf=0.0, neginf=0.0)
            else:
                target = np.nan_to_num(spec.target_joint_pos, nan=0.0, posinf=0.0, neginf=0.0)
            tau = spec.kp * (target - q) + spec.kd * (0.0 - qd)
            tau = np.clip(tau, -spec.effort, spec.effort)
            tau = np.nan_to_num(tau, nan=0.0, posinf=0.0, neginf=0.0)
            if use_pi_pd:
                self._set_actuator_ctrls(spec.pi_act_idx_mujoco, tau)
            else:
                self._set_actuator_ctrls(spec.act_idx, tau)

    def _step_once(self, counter: int) -> int:
        pre_hold_changed = self._apply_robot_protection_holds()
        if pre_hold_changed:
            self.model.forward_kinematic(self.data)
        if counter % self.control_decimation == 0:
            self._compute_targets()
        self._apply_torque()
        try:
            self.model.step(self.data)
        except BaseException as e:
            if isinstance(e, (KeyboardInterrupt, SystemExit)):
                raise
            self._step_fail_count += 1
            now = time.monotonic()
            if now - self._last_step_fail_log >= 1.0:
                self._last_step_fail_log = now
                print(
                    "[MotrixSim] model.step failed; skip this step to keep web controls responsive "
                    f"(count={self._step_fail_count}, err={type(e).__name__}: {e})"
                )
            # Keep process alive and preserve current state/teleport commands.
            # Full reset here makes interactive controls appear ineffective.
            return counter + 1
        if self.robot_cfg.use_k1_legged_gym_policy:
            dt_g = float(self.sim_dt) * float(K1_LEGGED_GYM_GAIT_FREQUENCY)
            for sp in self.robot_specs.values():
                if sp.k1_gait_phase is None:
                    continue
                if self._k1_stand_policy is not None and not sp.k1_hybrid_walking:
                    continue
                sp.k1_gait_phase = float(np.fmod(sp.k1_gait_phase + dt_g, 1.0))
        if self._enforce_joint_state_limits():
            self.model.forward_kinematic(self.data)
        self._update_referee(self.sim_dt)
        post_hold_changed = self._apply_robot_protection_holds()
        if post_hold_changed:
            self.model.forward_kinematic(self.data)
        fall_recovered = self._recover_fallen_robots() if self._enable_fall_recovery else False
        if fall_recovered:
            self.model.forward_kinematic(self.data)
        if (
            not np.isfinite(self.data.dof_pos).all()
            or not np.isfinite(self.data.dof_vel).all()
            or not np.isfinite(self.data.actuator_ctrls).all()
        ):
            self.reset(preserve_ball=True)
            return 0
        return counter + 1

    def _enforce_joint_state_limits(self) -> bool:
        changed = False
        for spec in self.robot_specs.values():
            use_pi = (
                self.robot_cfg.robot_type == PI_PLUS_ROBOT_TYPE
                and spec.pi_qpos_idx_mujoco is not None
                and spec.pi_qvel_idx_mujoco is not None
                and spec.pi_act_idx_mujoco is not None
                and spec.pi_joint_lower_mujoco is not None
                and spec.pi_joint_upper_mujoco is not None
            )
            if use_pi:
                q_idx = spec.pi_qpos_idx_mujoco
                v_idx = spec.pi_qvel_idx_mujoco
                a_idx = spec.pi_act_idx_mujoco
                lower = spec.pi_joint_lower_mujoco
                upper = spec.pi_joint_upper_mujoco
            else:
                if spec.joint_lower is None or spec.joint_upper is None:
                    continue
                q_idx = spec.qpos_idx
                v_idx = spec.qvel_idx
                a_idx = spec.act_idx
                lower = spec.joint_lower
                upper = spec.joint_upper

            q = np.asarray(self.data.dof_pos[0, q_idx], dtype=np.float32)
            q_clip = np.clip(q, lower, upper)
            hit = np.abs(q - q_clip) > 1e-5
            if not np.any(hit):
                continue
            applied = False
            try:
                rb = self.model.get_body(spec.name)
                if rb is not None:
                    pos_idx = np.asarray(rb.get_dof_pos_indices(include_floatingbase=True), dtype=np.int64).reshape(-1)
                    vel_idx = np.asarray(rb.get_dof_vel_indices(include_floatingbase=True), dtype=np.int64).reshape(-1)
                    if pos_idx.size > 0 and vel_idx.size > 0:
                        q_all = np.asarray(self.data.dof_pos[0, pos_idx], dtype=np.float32).copy()
                        v_all = np.asarray(self.data.dof_vel[0, vel_idx], dtype=np.float32).copy()
                        mapped = 0
                        for j, gq in enumerate(np.asarray(q_idx, dtype=np.int64)):
                            q_loc = np.where(pos_idx == gq)[0]
                            if q_loc.size > 0:
                                q_all[int(q_loc[0])] = float(q_clip[j])
                                mapped += 1
                            if hit[j]:
                                gv = int(np.asarray(v_idx, dtype=np.int64)[j])
                                v_loc = np.where(vel_idx == gv)[0]
                                if v_loc.size > 0:
                                    v_all[int(v_loc[0])] = 0.0
                        if mapped > 0:
                            rb.set_dof_pos(self.data, q_all)
                            rb.set_dof_vel(self.data, v_all)
                            applied = True
            except Exception:
                applied = False

            if not applied:
                # Fallback path for environments where body-level set_* is unavailable.
                self.data.dof_pos[0, q_idx] = q_clip
                self.data.dof_vel[0, v_idx[hit]] = 0.0
            self._set_actuator_ctrls(a_idx[hit], 0.0)
            changed = True
        return changed

    def _detect_ball_contact_rid(self) -> int | None:
        if self._ball_geom_contact_pairs.size == 0:
            return None
        cquery = self.model.get_contact_query(self.data)
        hit = np.asarray(cquery.is_colliding(self._ball_geom_contact_pairs)).reshape(-1)
        if hit.size == 0 or not hit.any():
            return None
        active: set[int] = set()
        hit_idx = np.flatnonzero(hit)
        for k in hit_idx:
            rid = int(self._ball_geom_contact_rids[int(k)])
            if rid >= 0 and rid in self.robot_specs:
                active.add(rid)
        if not active:
            return None
        return min(active)

    def _recover_fallen_robots(self) -> bool:
        changed = False
        now = time.monotonic()
        for rid, spec in self.robot_specs.items():
            if self._is_robot_protected(rid):
                self._fall_candidate_frames[rid] = 0
                continue
            base_z = float(self.data.dof_pos[0, spec.base_qpos_adr + 2])
            startup_z = float(self._startup_qpos[spec.base_qpos_adr + 2])
            # Only trigger auto-recovery when the robot is both tilted and clearly low.
            # This avoids repeated false recoveries for walking/leaning states.
            low_z_gate = max(0.45, startup_z - 0.25)
            if not np.isfinite(base_z) or base_z > low_z_gate:
                self._fall_candidate_frames[rid] = 0
                continue
            quat = self.data.dof_pos[0][spec.base_qpos_adr + 3 : spec.base_qpos_adr + 7]
            rot_wb = _quat_to_rot_world_from_body(quat)
            upright_dot = float(rot_wb[2, 2])
            if not np.isfinite(upright_dot) or upright_dot >= FALL_UPRIGHT_DOT_MIN:
                self._fall_candidate_frames[rid] = 0
                continue
            fallen_frames = int(self._fall_candidate_frames.get(rid, 0)) + 1
            self._fall_candidate_frames[rid] = fallen_frames
            if fallen_frames < FALL_CONFIRM_FRAMES:
                continue
            self._fall_candidate_frames[rid] = 0
            x = float(self.data.dof_pos[0][spec.base_qpos_adr + 0])
            y = float(self.data.dof_pos[0][spec.base_qpos_adr + 1])
            theta = float(self._yaw_from_quat(quat))
            self._robot_protect_pose[rid] = (x, y, theta)
            self._robot_protect_until[rid] = now + FALL_RESET_PROTECT_SEC
            self._robot_cmd_zero_frames_left[rid] = max(
                int(self._robot_cmd_zero_frames_left.get(rid, 0)),
                DRAG_CMD_ZERO_POLICY_FRAMES,
            )
            self._hold_robot_at_reset_pose(spec, x, y, theta)
            changed = True
        return changed

    def _update_referee(self, dt: float):
        if self.referee is None:
            return
        ball_x, ball_y, ball_z = 0.0, 0.0, 0.075
        if self._ball_body is not None:
            p = np.asarray(self._ball_body.get_position(self.data), dtype=np.float64).reshape(-1)
            if p.size >= 3:
                ball_x, ball_y, ball_z = float(p[0]), float(p[1]), float(p[2])
        active_touch = self._detect_ball_contact_rid()
        if active_touch is not None:
            self._ball_last_touch_rid = active_touch
        self.referee.update(dt, ball_x, ball_y, ball_z, active_touch)
        place = self.referee.consume_ball_place()
        if place is not None:
            self.teleport_ball(float(place[0]), float(place[1]), None)

    def _get_ball_state(self):
        if self._ball_qpos_adr is None or self._ball_qvel_adr is None:
            return None
        qpos_adr = int(self._ball_qpos_adr)
        qvel_adr = int(self._ball_qvel_adr)
        return {
            "qpos": self.data.dof_pos[0][qpos_adr : qpos_adr + 7].copy(),
            "qvel": self.data.dof_vel[0][qvel_adr : qvel_adr + 6].copy(),
        }

    def _get_all_robot_states(self):
        out = {}
        for rid, spec in self.robot_specs.items():
            out[rid] = {
                "base_qpos": self.data.dof_pos[0][spec.base_qpos_adr : spec.base_qpos_adr + 7].copy(),
                "base_qvel": self.data.dof_vel[0][spec.base_qvel_adr : spec.base_qvel_adr + 6].copy(),
                "joint_qpos": self.data.dof_pos[0][spec.qpos_idx].copy(),
                "joint_qvel": self.data.dof_vel[0][spec.qvel_idx].copy(),
            }
        return out

    def _restore_ball_state(self, state):
        if state is None or self._ball_qpos_adr is None or self._ball_qvel_adr is None:
            return
        qpos_adr = int(self._ball_qpos_adr)
        qvel_adr = int(self._ball_qvel_adr)
        self.data.dof_pos[0, qpos_adr : qpos_adr + 7] = state["qpos"]
        self.data.dof_vel[0, qvel_adr : qvel_adr + 6] = state["qvel"]

    def _restore_all_robot_states(self, states):
        if not states:
            return
        for rid, state in states.items():
            spec = self.robot_specs.get(rid)
            if spec is None:
                continue
            self.data.dof_pos[0, spec.base_qpos_adr : spec.base_qpos_adr + 7] = state["base_qpos"]
            self.data.dof_vel[0, spec.base_qvel_adr : spec.base_qvel_adr + 6] = state["base_qvel"]
            self.data.dof_pos[0, spec.qpos_idx] = state["joint_qpos"]
            self.data.dof_vel[0, spec.qvel_idx] = state["joint_qvel"]

    def _reset_one_robot(self, spec: RobotSpec):
        self.data.dof_pos[0, spec.base_qpos_adr : spec.base_qpos_adr + 7] = self._startup_qpos[spec.base_qpos_adr : spec.base_qpos_adr + 7]
        self.data.dof_vel[0, spec.base_qvel_adr : spec.base_qvel_adr + 6] = self._startup_qvel[spec.base_qvel_adr : spec.base_qvel_adr + 6]
        self.data.dof_pos[0, spec.qpos_idx] = self._startup_qpos[spec.qpos_idx]
        self.data.dof_vel[0, spec.qvel_idx] = self._startup_qvel[spec.qvel_idx]
        if spec.pi_qpos_idx_mujoco is not None and spec.pi_qvel_idx_mujoco is not None:
            self.data.dof_pos[0, spec.pi_qpos_idx_mujoco] = self._startup_qpos[spec.pi_qpos_idx_mujoco]
            self.data.dof_vel[0, spec.pi_qvel_idx_mujoco] = self._startup_qvel[spec.pi_qvel_idx_mujoco]
        self._set_actuator_ctrls(spec.act_idx, self._startup_ctrl[spec.act_idx])
        if spec.pi_act_idx_mujoco is not None:
            self._set_actuator_ctrls(spec.pi_act_idx_mujoco, self._startup_ctrl[spec.pi_act_idx_mujoco])
        spec.last_action[:] = 0.0
        if spec.k1_legged_last_action is not None:
            spec.k1_legged_last_action[:] = 0.0
        if spec.k1_gait_phase is not None:
            spec.k1_gait_phase = 0.0
        spec.k1_hybrid_walking = False
        spec.filtered_dof_target[:] = spec.init_angles
        spec.target_joint_pos[:] = spec.init_angles
        if spec.pi_default_dof_pos is not None and spec.pi_filtered_dof_target is not None and spec.pi_target_dof_pos is not None:
            spec.pi_filtered_dof_target[:] = spec.pi_default_dof_pos
            spec.pi_target_dof_pos[:] = spec.pi_default_dof_pos

    def _hold_robot_at_reset_pose(self, spec: RobotSpec, x: float, y: float, theta: float):
        # Use a fixed reset height for k1 so post-fall recovery pose is stable and predictable.
        if self.robot_cfg.robot_type == K1_ROBOT_TYPE:
            cur_z = 0.6
        else:
            # Keep current base height for other robots to avoid lift/drop energy injection.
            cur_z = float(self.data.dof_pos[0, spec.base_qpos_adr + 2])
            if not np.isfinite(cur_z):
                cur_z = float(self._startup_qpos[spec.base_qpos_adr + 2])
        applied = False
        try:
            rb = self.model.get_body(spec.name)
            if rb is not None:
                pos_idx = np.asarray(rb.get_dof_pos_indices(include_floatingbase=True), dtype=np.int64).reshape(-1)
                vel_idx = np.asarray(rb.get_dof_vel_indices(include_floatingbase=True), dtype=np.int64).reshape(-1)
                if pos_idx.size >= 7 and vel_idx.size >= 6:
                    q = self._startup_qpos[pos_idx].copy()
                    v = self._startup_qvel[vel_idx].copy()
                    q[0] = float(x)
                    q[1] = float(y)
                    q[2] = cur_z
                    q[3:7] = _quat_xyzw_from_yaw(float(theta))
                    v[:6] = 0.0
                    rb.set_dof_pos(self.data, q)
                    rb.set_dof_vel(self.data, v)
                    applied = True
        except Exception:
            applied = False
        if not applied:
            self._reset_one_robot(spec)
            self.data.dof_pos[0, spec.base_qpos_adr + 0] = float(x)
            self.data.dof_pos[0, spec.base_qpos_adr + 1] = float(y)
            self.data.dof_pos[0, spec.base_qpos_adr + 2] = cur_z
            self.data.dof_pos[0, spec.base_qpos_adr + 3 : spec.base_qpos_adr + 7] = _quat_xyzw_from_yaw(float(theta))
            self.data.dof_vel[0, spec.base_qvel_adr : spec.base_qvel_adr + 6] = 0.0
        self.command_buffer[spec.rid] = np.array(DEFAULT_CMD, dtype=np.float32)
        self.command_ts[spec.rid] = float("-inf")
        self.command_received[spec.rid] = True

    def _is_robot_protected(self, rid: int) -> bool:
        until = self._robot_protect_until.get(rid)
        if until is None:
            return False
        if time.monotonic() < until:
            return True
        self._robot_protect_until.pop(rid, None)
        self._robot_protect_pose.pop(rid, None)
        return False

    def _apply_robot_protection_holds(self) -> bool:
        changed = False
        now = time.monotonic()
        for rid, spec in self.robot_specs.items():
            until = self._robot_protect_until.get(rid)
            if until is None:
                continue
            if now >= until:
                self._robot_protect_until.pop(rid, None)
                self._robot_protect_pose.pop(rid, None)
                continue
            pose = self._robot_protect_pose.get(rid)
            if pose is None:
                continue
            x, y, theta = pose
            self._hold_robot_at_reset_pose(spec, x, y, theta)
            changed = True
        return changed

    def reset(self, preserve_ball: bool = True, reset_referee: bool = True):
        ball_state = self._get_ball_state() if preserve_ball else None
        self.data = mtx.SceneData(self.model, batch=[1])
        self.data.dof_pos[0, :] = self._startup_qpos
        self.data.dof_vel[0, :] = self._startup_qvel
        self._set_actuator_ctrls(np.arange(self._startup_ctrl.shape[0], dtype=np.int64), self._startup_ctrl)
        self._apply_saved_spawn_points()
        self._restore_ball_state(ball_state)
        for spec in self.robot_specs.values():
            spec.last_action[:] = 0.0
            if spec.k1_legged_last_action is not None:
                spec.k1_legged_last_action[:] = 0.0
            if spec.k1_gait_phase is not None:
                spec.k1_gait_phase = 0.0
            if spec.k1_amp_last_mjcf is not None:
                spec.k1_amp_last_mjcf[:] = 0.0
            if spec.k1_amp_hist is not None:
                spec.k1_amp_hist.fill(0.0)
            spec.k1_hybrid_walking = False
            spec.filtered_dof_target[:] = spec.init_angles
            spec.target_joint_pos[:] = spec.init_angles
            if spec.pi_default_dof_pos is not None and spec.pi_filtered_dof_target is not None and spec.pi_target_dof_pos is not None:
                spec.pi_filtered_dof_target[:] = spec.pi_default_dof_pos
                spec.pi_target_dof_pos[:] = spec.pi_default_dof_pos
        self.command_buffer = {rid: np.array(DEFAULT_CMD, dtype=np.float32) for rid in FIXED_ROBOT_ID_TO_NAME}
        self.command_ts = {rid: float("-inf") for rid in FIXED_ROBOT_ID_TO_NAME}
        self.command_received = {rid: False for rid in FIXED_ROBOT_ID_TO_NAME}
        for rid in self.robot_specs:
            self.command_received[rid] = True
        self._robot_protect_until.clear()
        self._robot_protect_pose.clear()
        self._robot_cmd_zero_frames_left.clear()
        self._fall_candidate_frames.clear()
        self._ball_last_touch_rid = None
        if reset_referee and self.referee is not None:
            self.referee.reset()
        self.last_msg_info = {"timestamp": 0.0, "id": -1, "source": "unknown"}
        self._policy_step_count = 0
        self._printed_target_policy_io = False
        self.model.forward_kinematic(self.data)

    def set_spawn_points(self, spawn_points: dict[str, list[float]]):
        cleaned: dict[str, list[float]] = {}
        for name, arr in spawn_points.items():
            if not isinstance(name, str) or not isinstance(arr, (list, tuple)) or len(arr) < 2:
                continue
            if name != "ball" and name not in FIXED_ROBOT_NAME_TO_ID:
                continue
            if name != "ball":
                rid = FIXED_ROBOT_NAME_TO_ID[name]
                if rid not in self.robot_specs:
                    continue
            x = float(arr[0])
            y = float(arr[1])
            theta = float(arr[2]) if len(arr) >= 3 and arr[2] is not None else 0.0
            cleaned[name] = [x, y, theta]
        self._saved_spawn_points = cleaned

    def _apply_saved_spawn_points(self):
        if not self._saved_spawn_points:
            return
        for name, arr in self._saved_spawn_points.items():
            x, y = float(arr[0]), float(arr[1])
            theta = float(arr[2]) if len(arr) >= 3 else 0.0
            if name == "ball":
                if self._ball_qpos_adr is None or self._ball_qvel_adr is None:
                    continue
                qpos_adr = int(self._ball_qpos_adr)
                qvel_adr = int(self._ball_qvel_adr)
                z = float(self._startup_qpos[qpos_adr + 2])
                self.data.dof_pos[0, qpos_adr + 0] = x
                self.data.dof_pos[0, qpos_adr + 1] = y
                self.data.dof_pos[0, qpos_adr + 2] = z
                self.data.dof_pos[0, qpos_adr + 3 : qpos_adr + 7] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
                self.data.dof_vel[0, qvel_adr : qvel_adr + 6] = 0.0
                continue
            rid = FIXED_ROBOT_NAME_TO_ID.get(name)
            if rid is None or rid not in self.robot_specs:
                continue
            spec = self.robot_specs[rid]
            self.data.dof_pos[0, spec.base_qpos_adr + 0] = x
            self.data.dof_pos[0, spec.base_qpos_adr + 1] = y
            self.data.dof_pos[0, spec.base_qpos_adr + 3 : spec.base_qpos_adr + 7] = _quat_xyzw_from_yaw(theta)
            self.data.dof_vel[0, spec.base_qvel_adr : spec.base_qvel_adr + 6] = 0.0

    def teleport_robot(self, robot_name: str, x: float, y: float, theta: float | None):
        rid = FIXED_ROBOT_NAME_TO_ID.get(robot_name, None)
        if rid is None or rid not in self.robot_specs:
            return
        spec = self.robot_specs[rid]
        cur_theta = self._yaw_from_quat(self.data.dof_pos[0][spec.base_qpos_adr + 3 : spec.base_qpos_adr + 7])
        target_theta = cur_theta if theta is None else float(theta)
        # Dragging a robot on minimap should only reset/reposition this robot.
        self._robot_protect_pose[rid] = (float(x), float(y), target_theta)
        self._robot_protect_until[rid] = time.monotonic() + DRAG_RESET_PROTECT_SEC
        self._robot_cmd_zero_frames_left[rid] = DRAG_CMD_ZERO_POLICY_FRAMES
        self._hold_robot_at_reset_pose(spec, float(x), float(y), target_theta)
        self.model.forward_kinematic(self.data)

    def teleport_ball(self, x: float, y: float, z: float | None):
        if self._ball_body is not None:
            try:
                pos_idx = np.asarray(
                    self._ball_body.get_dof_pos_indices(include_floatingbase=True), dtype=np.int64
                ).reshape(-1)
                vel_idx = np.asarray(
                    self._ball_body.get_dof_vel_indices(include_floatingbase=True), dtype=np.int64
                ).reshape(-1)
                if pos_idx.size >= 7 and vel_idx.size >= 6:
                    q = self.data.dof_pos[0, pos_idx].copy()
                    v = self.data.dof_vel[0, vel_idx].copy()
                    if z is None:
                        z = float(q[2])
                    q[0] = float(x)
                    q[1] = float(y)
                    q[2] = float(z)
                    q[3:7] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
                    v[:6] = 0.0
                    self._ball_body.set_dof_pos(self.data, q)
                    self._ball_body.set_dof_vel(self.data, v)
                    self.model.forward_kinematic(self.data)
                    return
            except Exception:
                pass
        if self._ball_qpos_adr is not None and self._ball_qvel_adr is not None:
            qpos_adr = int(self._ball_qpos_adr)
            qvel_adr = int(self._ball_qvel_adr)
            if z is None:
                z = float(self.data.dof_pos[0, qpos_adr + 2])
            self.data.dof_pos[0, qpos_adr + 0] = float(x)
            self.data.dof_pos[0, qpos_adr + 1] = float(y)
            self.data.dof_pos[0, qpos_adr + 2] = float(z)
            self.data.dof_pos[0, qpos_adr + 3 : qpos_adr + 7] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
            self.data.dof_vel[0, qvel_adr : qvel_adr + 6] = 0.0
        elif self._ball_qpos_idx is not None and self._ball_qvel_idx is not None:
            if z is None:
                z = float(self.data.dof_pos[0, int(self._ball_qpos_idx[2])])
            self.data.dof_pos[0, self._ball_qpos_idx[0]] = float(x)
            self.data.dof_pos[0, self._ball_qpos_idx[1]] = float(y)
            self.data.dof_pos[0, self._ball_qpos_idx[2]] = float(z)
            self.data.dof_pos[0, self._ball_qpos_idx[3:7]] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
            self.data.dof_vel[0, self._ball_qvel_idx[:6]] = 0.0
        else:
            return
        self.model.forward_kinematic(self.data)

    def _yaw_from_quat(self, quat_xyzw: np.ndarray) -> float:
        qx, qy, qz, qw = quat_xyzw
        return float(np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz)))

    def state_for_web(self) -> dict:
        states = {}
        for rid, name in FIXED_ROBOT_ID_TO_NAME.items():
            if rid in self.robot_specs:
                spec = self.robot_specs[rid]
                x = float(self.data.dof_pos[0][spec.base_qpos_adr + 0])
                y = float(self.data.dof_pos[0][spec.base_qpos_adr + 1])
                z = float(self.data.dof_pos[0][spec.base_qpos_adr + 2])
                quat = self.data.dof_pos[0][spec.base_qpos_adr + 3 : spec.base_qpos_adr + 7]
                cmd = self.command_buffer.get(rid, np.array(DEFAULT_CMD, dtype=np.float32))
                states[name] = {
                    "x": x,
                    "y": y,
                    "z": z,
                    "yaw": self._yaw_from_quat(quat),
                    "active": True,
                    "team": spec.team,
                    "cmd_vel": [float(cmd[0]), float(cmd[1]), float(cmd[2])],
                }
            else:
                states[name] = {
                    "x": 100.0 + rid,
                    "y": 100.0,
                    "z": 0.0,
                    "yaw": 0.0,
                    "active": False,
                    "team": "red" if rid < MAX_ROBOTS_PER_TEAM else "blue",
                    "cmd_vel": [0.0, 0.0, 0.0],
                }

        ball_x, ball_y, ball_z = 0.0, 0.0, 0.075
        if self._ball_body is not None:
            p = np.asarray(self._ball_body.get_position(self.data), dtype=np.float64).reshape(-1)
            if p.size >= 3:
                ball_x, ball_y, ball_z = float(p[0]), float(p[1]), float(p[2])
        states["ball"] = {"x": ball_x, "y": ball_y, "z": ball_z, "yaw": 0.0, "active": True, "team": "none"}
        if self.referee is not None:
            states["_game"] = self.referee.game_state_dict()
        states["_last_msg"] = dict(self.last_msg_info)
        return states

    def _render_topdown_web_frame(self, width: int, height: int) -> np.ndarray:
        """Fallback frame when Motrix capture is unavailable: draw a simple top-down field view."""
        h = max(1, int(height))
        w = max(1, int(width))
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[:, :, :] = np.array([32, 110, 45], dtype=np.uint8)  # grass-like background

        world_l = max(1e-6, float(self._world_length))
        world_w = max(1e-6, float(self._world_width))
        sx = w / world_l
        sy = h / world_w
        cx = 0.5 * w
        cy = 0.5 * h

        def to_px(x: float, y: float) -> tuple[int, int]:
            px = int(round(cx + float(x) * sx))
            py = int(round(cy - float(y) * sy))
            return px, py

        # Draw field border.
        half_l = 0.5 * float(self._field_length)
        half_w = 0.5 * float(self._field_width)
        x0, y0 = to_px(-half_l, half_w)
        x1, y1 = to_px(half_l, -half_w)
        xa, xb = sorted((x0, x1))
        ya, yb = sorted((y0, y1))
        img[max(0, ya) : min(h, ya + 2), max(0, xa) : min(w, xb + 1)] = 230
        img[max(0, yb - 1) : min(h, yb + 1), max(0, xa) : min(w, xb + 1)] = 230
        img[max(0, ya) : min(h, yb + 1), max(0, xa) : min(w, xa + 2)] = 230
        img[max(0, ya) : min(h, yb + 1), max(0, xb - 1) : min(w, xb + 1)] = 230

        states = self.state_for_web()
        ball = states.get("ball", {})
        bx, by = to_px(float(ball.get("x", 0.0)), float(ball.get("y", 0.0)))
        rr = max(2, int(round(0.08 * min(sx, sy))))
        yy, xx = np.ogrid[:h, :w]
        mask_ball = (xx - bx) * (xx - bx) + (yy - by) * (yy - by) <= rr * rr
        img[mask_ball] = np.array([245, 245, 245], dtype=np.uint8)

        for rid, name in FIXED_ROBOT_ID_TO_NAME.items():
            st = states.get(name, None)
            if not isinstance(st, dict) or not bool(st.get("active", False)):
                continue
            rx, ry = to_px(float(st.get("x", 0.0)), float(st.get("y", 0.0)))
            team = str(st.get("team", "none"))
            color = np.array([220, 60, 60], dtype=np.uint8) if team == "red" else np.array([70, 120, 240], dtype=np.uint8)
            rr = max(3, int(round(0.11 * min(sx, sy))))
            mask = (xx - rx) * (xx - rx) + (yy - ry) * (yy - ry) <= rr * rr
            img[mask] = color
        return img

    def state_for_zmq(self) -> dict:
        robots = []
        for rid in self.active_robot_ids:
            spec = self.robot_specs[rid]
            x = float(self.data.dof_pos[0][spec.base_qpos_adr + 0])
            y = float(self.data.dof_pos[0][spec.base_qpos_adr + 1])
            quat = self.data.dof_pos[0][spec.base_qpos_adr + 3 : spec.base_qpos_adr + 7]
            robots.append(
                {
                    "id": rid,
                    "name": spec.name,
                    "x": x,
                    "y": y,
                    "theta": self._yaw_from_quat(quat),
                    "team": spec.team,
                }
            )

        ball_x, ball_y, ball_z = 0.0, 0.0, 0.075
        if self._ball_body is not None:
            p = np.asarray(self._ball_body.get_position(self.data), dtype=np.float64).reshape(-1)
            if p.size >= 3:
                ball_x, ball_y, ball_z = float(p[0]), float(p[1]), float(p[2])

        out = {"robots": robots, "ball": {"x": ball_x, "y": ball_y, "z": ball_z}}
        if self.referee is not None:
            out["gamecontroller"] = self.referee.game_state_dict()
        return out

    def _set_camera_eye_lookat(self, eye: tuple[float, float, float], lookat: tuple[float, float, float]):
        if self._web_camera is None:
            return
        eye_vec = np.asarray(eye, dtype=np.float32)
        look_vec = np.asarray(lookat, dtype=np.float32)
        d = eye_vec - look_vec
        dist = float(np.linalg.norm(d))
        if dist < 1e-6:
            return
        azimuth = np.degrees(np.arctan2(d[1], d[0]))
        elevation = -np.degrees(np.arcsin(np.clip(d[2] / dist, -1.0, 1.0)))
        self._web_camera.lookat[:] = look_vec
        self._web_camera.distance = dist
        self._web_camera.azimuth = float(azimuth)
        self._web_camera.elevation = float(elevation)

    def _apply_camera_preset(self, preset: str):
        # Eye/look must match lab_webview_* cameras in _ensure_default_scene_camera_for_motrix (MJCF poses
        # drive Motrix headless capture; _web_camera alone is not enough for Diagonal/Side/Top).
        presets = {
            "Top": ((0.0, 0.0, 18.0), (0.0, 0.0, 0.8)),
            # Mirror side camera so field orientation matches the mini-map (red left, blue right).
            "Side": ((0.0, 9.0, 6.0), (0.0, 0.0, 0.8)),
            "Diagonal": ((-8.0, -8.0, 8.0), (0.0, 0.0, 0.8)),
            "Goal_Left": ((-8.5, 0.0, 6.0), (0.0, 0.0, 0.9)),
            "Goal_Right": ((8.5, 0.0, 6.0), (0.0, 0.0, 0.9)),
        }
        preset_to_cam = {
            "Top": "lab_webview_top",
            "Side": "lab_webview_side",
            "Diagonal": "lab_webview_diagonal",
            "Goal_Left": "lab_webview_goal_left",
            "Goal_Right": "lab_webview_goal_right",
        }
        if preset in presets:
            eye, lookat = presets[preset]
            self._set_camera_eye_lookat(eye, lookat)
            self._web_capture_camera_name = preset_to_cam.get(preset, "lab_webview_diagonal")

    def _safe_create_renderer(self, width: int, height: int):
        candidates = [(width, height), (640, 480), (480, 360), (320, 240)]
        last_err = None
        for w, h in candidates:
            try:
                if w <= 0 or h <= 0:
                    continue
                return _MotrixHeadlessRenderer(self.model, w, h)
            except Exception as e:
                last_err = e
        print(f"[MotrixWebView] Headless renderer unavailable, using black frames: {last_err}")
        h0, w0 = candidates[-1][1], candidates[-1][0]
        return _BlackFrameRenderer(h0, w0)

    def _apply_web_commands(self, cmds, counter: int) -> tuple[int, bool]:
        reset_triggered = False
        now_log = time.monotonic()
        cmd_summary: list[str] = []
        if cmds.spawn_points is not None:
            self.set_spawn_points(cmds.spawn_points)
            cmd_summary.append("spawn_points")
        if cmds.velocity_cmds is not None:
            for name, vx, vy, wz in cmds.velocity_cmds:
                rid = FIXED_ROBOT_NAME_TO_ID.get(name, None)
                if rid is None:
                    continue
                self.set_command(float(vx), float(vy), float(wz), robot_id=rid, timestamp=time.time(), source="webview")
            cmd_summary.append(f"velocity:{len(cmds.velocity_cmds)}")
        if cmds.reset_env:
            # Reset robots/ball/runtime state but keep current referee state.
            self.reset(preserve_ball=False, reset_referee=False)
            counter = 0
            reset_triggered = True
            cmd_summary.append("reset_env")
        if cmds.restart_match:
            # Restart full match state: reset robots, ball, and referee.
            self.reset(preserve_ball=False, reset_referee=True)
            counter = 0
            reset_triggered = True
            cmd_summary.append("restart_match")
        if cmds.viewer_point is not None and self._web_camera is not None:
            look = tuple(float(x) for x in self._web_camera.lookat)
            eye = tuple(float(x) for x in cmds.viewer_point)
            self._set_camera_eye_lookat(eye, look)
            cmd_summary.append("viewer_point")
        if cmds.viewer_look_at is not None and self._web_camera is not None:
            d = float(self._web_camera.distance)
            az = np.radians(float(self._web_camera.azimuth))
            el = np.radians(float(self._web_camera.elevation))
            cur_look = np.array(self._web_camera.lookat, dtype=np.float32)
            eye = (
                cur_look[0] + d * np.cos(el) * np.cos(az),
                cur_look[1] + d * np.cos(el) * np.sin(az),
                cur_look[2] + d * np.sin(el),
            )
            look = tuple(float(x) for x in cmds.viewer_look_at)
            self._set_camera_eye_lookat(eye, look)
            cmd_summary.append("viewer_look_at")
        if cmds.camera_preset is not None:
            self._apply_camera_preset(cmds.camera_preset)
            cmd_summary.append(f"camera:{cmds.camera_preset}")
        if cmds.teleport_cmd is not None:
            name, x, y, z, theta = cmds.teleport_cmd
            if name in FIXED_ROBOT_NAME_TO_ID:
                self.teleport_robot(name, x, y, None if theta is None else float(theta))
                counter = 0
                reset_triggered = True
                cmd_summary.append(f"teleport_robot:{name}")
            elif name == "ball":
                # Teleport ball directly so robot pose/velocity/control state are unaffected.
                self.teleport_ball(x, y, None if z is None else float(z))
                cmd_summary.append("teleport_ball")
        if cmds.referee_command is not None and self.referee is not None:
            self._apply_referee_command(cmds.referee_command)
            cmd_summary.append(f"referee:{cmds.referee_command}")
        if cmd_summary and (now_log - self._last_web_cmd_log >= 0.8):
            self._last_web_cmd_log = now_log
            print(f"[MotrixWebView] apply_web_commands: {', '.join(cmd_summary)}")
        return counter, reset_triggered

    def _apply_referee_command(self, cmd: str):
        if self.referee is None:
            return
        from .soccer_referee import GCState, PlayMode
        cmd = cmd.lower().strip()
        if cmd == "ready":
            self.referee.state = GCState.READY
            self.referee.kick_off(self.referee.kicking_side)
        elif cmd == "set":
            self.referee.state = GCState.SET
            self.referee._set_play_mode(PlayMode.BEFORE_KICK_OFF)
        elif cmd == "play":
            self.referee.state = GCState.PLAYING
            self.referee._set_team_mode(self.referee.kicking_side, PlayMode.KICK_OFF_LEFT, PlayMode.KICK_OFF_RIGHT)
        elif cmd == "finish":
            self.referee.game_over()
        elif cmd == "stoptimer":
            self.referee.set_auto_state_enabled(False)

    @staticmethod
    def _render_thread(model, render_queue, stop_event, webview, web_fps, width, height):
        """Standalone render thread: owns its GPU context, never blocks physics/ZMQ."""
        try:
            r = _MotrixHeadlessRenderer(model, width, height)
        except Exception as e:
            print(f"[MotrixWebView] render thread init failed, using black frames: {e}")
            r = _BlackFrameRenderer(height, width)
        camera = _FreeCameraView()
        frame_interval = 1.0 / max(1, web_fps)
        next_frame = time.time()
        try:
            while not stop_event.is_set():
                try:
                    item = render_queue.get(timeout=0.05)
                except queue.Empty:
                    continue
                if item is None or item.get("data") is None:
                    continue
                data = item["data"]
                cam_state = item.get("camera_state")
                if cam_state is not None:
                    camera.lookat[:] = np.array(cam_state["lookat"], dtype=np.float64)
                    camera.distance = cam_state["distance"]
                    camera.azimuth = cam_state["azimuth"]
                    camera.elevation = cam_state["elevation"]
                capture_name = item.get("capture_camera")
                now = time.time()
                if now >= next_frame:
                    r.set_capture_camera(capture_name)
                    r.update_scene(data, camera=camera)
                    frame = r.render()
                    webview.emit_frame(frame)
                    next_frame = now + frame_interval
        finally:
            r.close()

    def _push_render_data(self, render_queue: queue.Queue) -> None:
        """Push latest scene + camera state to render thread (non-blocking)."""
        if self._web_camera is None:
            return
        item = {
            "data": self.data,
            "capture_camera": getattr(self, "_web_capture_camera_name", None),
            "camera_state": {
                "lookat": tuple(float(x) for x in self._web_camera.lookat),
                "distance": float(self._web_camera.distance),
                "azimuth": float(self._web_camera.azimuth),
                "elevation": float(self._web_camera.elevation),
            },
        }
        try:
            render_queue.put_nowait(item)
        except queue.Full:
            try:
                render_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                render_queue.put_nowait(item)
            except queue.Full:
                pass

    def zmq_loop(self, port: int, webview: MujocoLabWebView | None, web_fps: int, width: int, height: int):
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        socket.bind(f"tcp://*:{port}")
        print(f"[MotrixZMQ] Bound to tcp://*:{port}")
        use_real_time = bool(self.args.real_time or (webview is not None))
        if webview is not None and not self.args.real_time:
            print("[MotrixWebView] forcing real-time stepping for web responsiveness")

        # Render thread: GPU rendering in a separate thread so physics/ZMQ never wait on GPU.
        render_queue: queue.Queue | None = None
        render_stop: threading.Event | None = None
        render_thread: threading.Thread | None = None
        state_emit_interval = 1.0 / max(1, web_fps)
        next_state_emit_time = time.time()
        if webview is not None:
            render_queue = queue.Queue(maxsize=1)
            render_stop = threading.Event()
            self._web_camera = _FreeCameraView()
            self._apply_camera_preset("Diagonal")
            render_thread = threading.Thread(
                target=self._render_thread,
                args=(self.model, render_queue, render_stop, webview, web_fps, width, height),
                daemon=True,
            )
            render_thread.start()

        counter = 0
        try:
            while True:
                step_start = time.time()
                reset_triggered = False
                if webview is not None:
                    cmds = webview.poll_commands()
                    counter, reset_triggered = self._apply_web_commands(cmds, counter)

                # Always use NOBLOCK: ZMQ is a side channel; physics must not wait on clients.
                flags = zmq.NOBLOCK
                got_msg = False
                msg = None
                try:
                    msg = socket.recv_json(flags=flags)
                    got_msg = True
                except zmq.Again:
                    pass

                if got_msg and msg is not None:
                    client_ts = msg.get("timestamp", 0)
                    msg_source = msg.get("source", "unknown")
                    if self.referee is not None:
                        gc_cmd = msg.get("game_controller_cmd")
                        if isinstance(gc_cmd, (list, tuple)) and len(gc_cmd) == 5:
                            self.referee.apply_auto_ref_command(gc_cmd)
                    if "commands" in msg and isinstance(msg["commands"], list):
                        for item in msg["commands"]:
                            if not isinstance(item, dict):
                                continue
                            c = item.get("cmd", [0.0, 0.0, 0.0])
                            rid = int(item.get("id", 0))
                            ts = item.get("timestamp", client_ts)
                            src = item.get("source", msg_source)
                            if isinstance(c, (list, tuple)) and len(c) >= 3:
                                self.set_command(float(c[0]), float(c[1]), float(c[2]), robot_id=rid, timestamp=ts, source=src)
                    else:
                        c = msg.get("cmd", [0.0, 0.0, 0.0])
                        rid = int(msg.get("id", 0))
                        if isinstance(c, (list, tuple)) and len(c) >= 3:
                            self.set_command(float(c[0]), float(c[1]), float(c[2]), robot_id=rid, timestamp=client_ts, source=msg_source)

                    if not reset_triggered:
                        counter = self._step_once(counter)
                        step_latency = time.time() - step_start
                    else:
                        step_latency = 0.0
                    response = {
                        "state": self.state_for_zmq(),
                        "sim_timestamp": time.time(),
                        "step_latency": step_latency,
                        "ack_timestamp": client_ts,
                    }
                    socket.send_json(response)
                else:
                    # Physics always steps — never gated on ZMQ or rendering.
                    if not reset_triggered:
                        counter = self._step_once(counter)

                # Push latest scene to render thread (non-blocking).
                if render_queue is not None:
                    self._push_render_data(render_queue)

                if webview is not None:
                    now = time.time()
                    if now >= next_state_emit_time:
                        webview.emit_robot_states(self.state_for_web())
                        next_state_emit_time = now + state_emit_interval

                if use_real_time:
                    wait_time = float(self.model.options.timestep) - (time.time() - step_start)
                    if wait_time > 0:
                        time.sleep(wait_time)
                else:
                    time.sleep(0)
                if webview is not None:
                    slack = float(self.model.options.timestep) - (time.time() - step_start)
                    if slack > 0.002:
                        time.sleep(0.001)
        finally:
            socket.close()
            context.term()
            if render_thread is not None:
                render_stop.set()
                render_thread.join(timeout=2.0)


def run_sim(args: RuntimeArgs, template_dir: Path):
    sim = MultiRobotMotrixSim(args)
    webview = None
    if args.webview:
        # Avoid silently attaching to a stale viewer process on the same port.
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("0.0.0.0", int(args.webview_port)))
            except OSError as e:
                raise RuntimeError(
                    f"WebView port {args.webview_port} is already in use. "
                    "Please stop stale sim/webview processes or choose a different --webview-port."
                ) from e
        webview = MujocoLabWebView(
            template_dir=template_dir,
            allow_keyboard_control=args.allow_keyboard_control,
            web_jpeg_quality=args.web_jpeg_quality,
            web_jpeg_subsampling=args.web_jpeg_subsampling,
        )
        webview.start(port=args.webview_port)
        # Let the Flask/Socket.IO thread bind before we broadcast field_meta from the main thread.
        time.sleep(0.15)
        webview.set_field_meta(
            {
                "world_length": sim._world_length,
                "world_width": sim._world_width,
                "field_length": sim._field_length,
                "field_width": sim._field_width,
                "markings": {
                    "center_circle_diameter": float(sim._field_markings_cfg.get("center_circle_diameter", 1.5)),
                    "line_width": float(sim._field_markings_cfg.get("line_width", 0.05)),
                    "goal_area_depth": float(sim._field_markings_cfg.get("goal_area_depth", 1.0)),
                    "goal_area_width": float(sim._field_markings_cfg.get("goal_area_width", 3.0)),
                    "penalty_area_depth": float(sim._field_markings_cfg.get("penalty_area_depth", 2.0)),
                    "penalty_area_width": float(sim._field_markings_cfg.get("penalty_area_width", 4.0)),
                    "penalty_spot_distance": float(sim._field_markings_cfg.get("penalty_spot_distance", 1.5)),
                },
            }
        )
        print(f"[MotrixWebView] Started at http://localhost:{args.webview_port}")

    if not args.zmq:
        raise ValueError("ZMQ is required in simplified runner. Use default --zmq.")

    try:
        sim.zmq_loop(
            port=args.port,
            webview=webview,
            web_fps=args.web_fps,
            width=args.web_width,
            height=args.web_height,
        )
    finally:
        if webview is not None:
            try:
                webview.close()
            except Exception:
                pass
