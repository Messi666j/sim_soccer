# SPDX-FileCopyrightText: Copyright (c) MOS-Brain Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import base64
import threading
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path

import numpy as np
from flask import Flask, render_template
from flask_socketio import SocketIO
from PIL import Image


@dataclass
class WebMsgBuffer:
    reset_env: bool = False
    restart_match: bool = False
    viewer_point: list[float] | None = None
    viewer_look_at: list[float] | None = None
    camera_preset: str | None = None
    teleport_cmd: tuple[str, float, float, float | None, float | None] | None = None
    spawn_points: dict[str, list[float]] | None = None
    velocity_cmds: list[tuple[str, float, float, float]] | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)


class LabWebView:
    def __init__(self, template_dir: Path, allow_keyboard_control: bool = False):
        self.app = Flask(__name__, template_folder=str(template_dir))
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        self.msg = WebMsgBuffer()
        self.allow_keyboard_control = bool(allow_keyboard_control)
        self._field_meta: dict | None = None
        self._setup_routes_and_events()

    def _setup_routes_and_events(self):
        @self.app.route("/")
        def index():
            return render_template("index.html", allow_keyboard_control=self.allow_keyboard_control)

        @self.socketio.on("connect")
        def on_connect():
            print("[LabWebView] Client connected")
            if self._field_meta is not None:
                self.socketio.emit("field_meta", self._field_meta)

        @self.socketio.on("reset_env")
        def on_reset():
            with self.msg.lock:
                self.msg.reset_env = True

        @self.socketio.on("restart_match")
        def on_restart_match():
            with self.msg.lock:
                self.msg.restart_match = True

        @self.socketio.on("set_viewer_point")
        def on_view_point(data):
            with self.msg.lock:
                self.msg.viewer_point = data.get("point", [3.0, 3.0, 1.0])

        @self.socketio.on("set_viewer_look_at")
        def on_view_look(data):
            with self.msg.lock:
                self.msg.viewer_look_at = data.get("point", [0.0, 0.0, 1.0])

        @self.socketio.on("set_camera_preset")
        def on_camera_preset(data):
            with self.msg.lock:
                self.msg.camera_preset = data.get("preset", "Top")

        @self.socketio.on("teleport_entity")
        def on_teleport(data):
            with self.msg.lock:
                self.msg.teleport_cmd = (
                    data.get("name", ""),
                    float(data.get("x", 0.0)),
                    float(data.get("y", 0.0)),
                    data.get("z", None),
                    data.get("theta", None),
                )

        @self.socketio.on("set_initial_positions")
        def on_set_initial_positions(data):
            with self.msg.lock:
                self.msg.spawn_points = data if isinstance(data, dict) else {}

        @self.socketio.on("set_robot_velocity")
        def on_set_robot_velocity(data):
            name = str(data.get("name", ""))
            vx = float(data.get("vx", 0.0))
            vy = float(data.get("vy", 0.0))
            wz = float(data.get("wz", 0.0))
            with self.msg.lock:
                if self.msg.velocity_cmds is None:
                    self.msg.velocity_cmds = []
                self.msg.velocity_cmds.append((name, vx, vy, wz))

    def start(self, port: int = 5811):
        t = threading.Thread(
            target=lambda: self.socketio.run(
                self.app,
                host="0.0.0.0",
                port=port,
                use_reloader=False,
                debug=False,
                allow_unsafe_werkzeug=True,
            ),
            daemon=True,
        )
        t.start()

    def poll_commands(self) -> WebMsgBuffer:
        with self.msg.lock:
            out = WebMsgBuffer(
                reset_env=self.msg.reset_env,
                restart_match=self.msg.restart_match,
                viewer_point=self.msg.viewer_point,
                viewer_look_at=self.msg.viewer_look_at,
                camera_preset=self.msg.camera_preset,
                teleport_cmd=self.msg.teleport_cmd,
                spawn_points=self.msg.spawn_points,
                velocity_cmds=list(self.msg.velocity_cmds) if self.msg.velocity_cmds is not None else None,
            )
            self.msg.reset_env = False
            self.msg.restart_match = False
            self.msg.viewer_point = None
            self.msg.viewer_look_at = None
            self.msg.camera_preset = None
            self.msg.teleport_cmd = None
            self.msg.spawn_points = None
            self.msg.velocity_cmds = None
            return out

    def emit_frame(self, rgb: np.ndarray):
        image = Image.fromarray(rgb)
        bio = BytesIO()
        image.save(bio, format="JPEG", quality=92)
        payload = base64.b64encode(bio.getvalue()).decode("utf-8")
        self.socketio.emit("new_frame", {"image": payload})

    def emit_robot_states(self, states: dict):
        payload = dict(states) if isinstance(states, dict) else {}
        game = payload.get("_game")
        if isinstance(game, dict):
            payload["_game"] = self._normalize_gamecontroller_for_ui(game)
        self.socketio.emit("robot_states", payload)

    def set_field_meta(self, field_meta: dict):
        self._field_meta = dict(field_meta)

    @staticmethod
    def _normalize_gamecontroller_for_ui(game: dict) -> dict:
        out = dict(game)

        # Backfill state name from numeric state when needed.
        if "state_name" not in out and "state" in out:
            state_name_map = {
                0: "STATE_INITIAL",
                1: "STATE_READY",
                2: "STATE_SET",
                3: "STATE_PLAYING",
                4: "STATE_FINISHED",
                5: "STATE_STANDBY",
            }
            try:
                out["state_name"] = state_name_map.get(int(out.get("state", 0)), "STATE_INITIAL")
            except Exception:
                out["state_name"] = "STATE_INITIAL"

        # Backfill set play name from numeric value when needed.
        if "set_play_name" not in out and "set_play" in out:
            sp_name_map = {
                0: "SET_PLAY_NONE",
                1: "SET_PLAY_KICK_OFF",
                2: "SET_PLAY_KICK_IN",
                3: "SET_PLAY_GOAL_KICK",
                4: "SET_PLAY_CORNER_KICK",
                5: "SET_PLAY_DIRECT_FREE_KICK",
                6: "SET_PLAY_INDIRECT_FREE_KICK",
                7: "SET_PLAY_PENALTY_KICK",
            }
            try:
                out["set_play_name"] = sp_name_map.get(int(out.get("set_play", 0)), "SET_PLAY_NONE")
            except Exception:
                out["set_play_name"] = "SET_PLAY_NONE"

        # Backfill old scoreboard keys for legacy frontend snippets.
        if "teams" in out and isinstance(out["teams"], list) and len(out["teams"]) >= 2:
            try:
                out.setdefault("score_left", int(out["teams"][0].get("score", 0)))
                out.setdefault("score_right", int(out["teams"][1].get("score", 0)))
            except Exception:
                out.setdefault("score_left", 0)
                out.setdefault("score_right", 0)
        else:
            out.setdefault("score_left", 0)
            out.setdefault("score_right", 0)

        out.setdefault("scoring_allowed", "both")
        return out


# Backward-compatible alias for existing callers.
MujocoLabWebView = LabWebView
