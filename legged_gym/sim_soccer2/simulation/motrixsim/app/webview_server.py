# SPDX-FileCopyrightText: Copyright (c) MOS-Brain Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import base64
import json
import multiprocessing as mp
import os
import queue
import tempfile
import threading
import time
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from flask import Flask, Response, jsonify, render_template, request
from flask_socketio import SocketIO
from PIL import Image

_NS = "/"

_last_emit_error_at: dict[str, float] = {}


def _log_emit_throttled(key: str, msg: str, interval_sec: float = 2.0) -> None:
    now = time.monotonic()
    prev = _last_emit_error_at.get(key, 0.0)
    if now - prev >= interval_sec:
        _last_emit_error_at[key] = now
        print(msg)


def _jpeg_worker_main(
    in_q: queue.Queue,
    event_q: mp.Queue,
    quality: int,
    subsampling: int,
    stop_ev: threading.Event,
) -> None:
    """Encode RGB frames off the simulation thread so physics stepping is not blocked by PIL."""
    q = max(1, min(95, int(quality)))
    sub = max(0, min(2, int(subsampling)))
    while not stop_ev.is_set():
        try:
            arr = in_q.get(timeout=0.05)
        except queue.Empty:
            continue
        if arr is None:
            break
        try:
            if not isinstance(arr, np.ndarray) or arr.ndim != 3 or arr.shape[2] != 3:
                continue
            image = Image.fromarray(arr, mode="RGB")
            bio = BytesIO()
            image.save(bio, format="JPEG", quality=q, subsampling=sub)
            _queue_put_latest(event_q, {"type": "frame", "jpeg": bio.getvalue()}, max_drop=32)
        except Exception as e:
            _log_emit_throttled("jpeg_worker", f"[MujocoWebView] jpeg worker: {e}")


def _socket_json_safe(obj: Any) -> Any:
    """Ensure payloads are JSON-serializable for Engine.IO (no numpy scalars/arrays)."""
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, (int, float)):
        return obj
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): _socket_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_socket_json_safe(v) for v in obj]
    return str(obj)


def _queue_put_latest(q: mp.Queue, item: Any, *, max_drop: int = 16) -> bool:
    """Put latest item into bounded queue, dropping oldest items if full."""
    for _ in range(max_drop):
        try:
            q.put_nowait(item)
            return True
        except queue.Full:
            try:
                q.get_nowait()
            except queue.Empty:
                pass
    return False


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
    referee_command: str | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)


def _run_webview_process(
    template_dir: str,
    allow_keyboard_control: bool,
    port: int,
    command_q: mp.Queue,
    command_pipe,
    event_q: mp.Queue,
    command_file: str,
    parent_pid: int,
) -> None:
    app = Flask(__name__, template_folder=template_dir)
    socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

    latest_frame_jpeg: bytes | None = None
    latest_states: dict[str, Any] = {}
    field_meta: dict[str, Any] | None = None
    lock = threading.Lock()

    def push_cmd(payload: dict[str, Any]) -> None:
        sent = False
        if command_pipe is not None:
            try:
                command_pipe.send(payload)
                sent = True
            except Exception:
                sent = False
        if not sent:
            _queue_put_latest(command_q, payload, max_drop=64)
        # Durable fallback path: always mirror commands to a local file.
        try:
            with open(command_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=True) + "\n")
        except Exception:
            pass

    def handle_command_event(event: str, data: Any) -> bool:
        ev = str(event or "").strip()
        if ev == "reset_env":
            push_cmd({"type": "reset_env"})
            return True
        if ev == "restart_match":
            push_cmd({"type": "restart_match"})
            return True
        if ev == "set_viewer_point" and isinstance(data, dict):
            push_cmd({"type": "set_viewer_point", "point": data.get("point", [3.0, 3.0, 1.0])})
            return True
        if ev == "set_viewer_look_at" and isinstance(data, dict):
            push_cmd({"type": "set_viewer_look_at", "point": data.get("point", [0.0, 0.0, 1.0])})
            return True
        if ev == "set_camera_preset" and isinstance(data, dict):
            push_cmd({"type": "set_camera_preset", "preset": data.get("preset", "Top")})
            return True
        if ev == "teleport_entity" and isinstance(data, dict):
            push_cmd(
                {
                    "type": "teleport_entity",
                    "name": data.get("name", ""),
                    "x": float(data.get("x", 0.0)),
                    "y": float(data.get("y", 0.0)),
                    "z": data.get("z", None),
                    "theta": data.get("theta", None),
                }
            )
            return True
        if ev == "set_initial_positions" and isinstance(data, dict):
            push_cmd({"type": "set_initial_positions", "spawn_points": data})
            return True
        if ev == "set_robot_velocity" and isinstance(data, dict):
            push_cmd(
                {
                    "type": "set_robot_velocity",
                    "name": str(data.get("name", "")),
                    "vx": float(data.get("vx", 0.0)),
                    "vy": float(data.get("vy", 0.0)),
                    "wz": float(data.get("wz", 0.0)),
                }
            )
            return True
        if ev == "referee_command" and isinstance(data, dict):
            cmd = str(data.get("command", "")).strip()
            if cmd in ("ready", "set", "play", "finish", "stoptimer"):
                push_cmd({"type": "referee_command", "command": cmd})
                return True
        return False

    @app.route("/api/command", methods=["POST"])
    def api_command():
        try:
            payload = request.get_json(silent=True) or {}
            event = str(payload.get("event", "")).strip()
            data = payload.get("data", {})
            accepted = handle_command_event(event, data)
            print(f"[MujocoWebView] api_command event={event!r} accepted={accepted}", flush=True)
            return jsonify({"ok": True, "accepted": bool(accepted), "event": event})
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 400

    @app.route("/")
    def index():
        return render_template("index.html", allow_keyboard_control=bool(allow_keyboard_control))

    @app.route("/api/frame.jpg")
    def api_frame_jpg():
        nonlocal latest_frame_jpeg
        with lock:
            buf = latest_frame_jpeg
        if not buf:
            return Response(status=204)
        return Response(buf, mimetype="image/jpeg")

    @app.route("/api/states")
    def api_states():
        with lock:
            out = dict(latest_states)
        return jsonify(out)

    @socketio.on("connect")
    def on_connect():
        nonlocal field_meta
        print("[MujocoWebView] Client connected")
        if field_meta is not None:
            socketio.emit("field_meta", field_meta, namespace=_NS)

    @socketio.on("reset_env")
    def on_reset(data=None):
        handle_command_event("reset_env", {})

    @socketio.on("restart_match")
    def on_restart_match(data=None):
        handle_command_event("restart_match", {})

    @socketio.on("set_viewer_point")
    def on_view_point(data):
        handle_command_event("set_viewer_point", data)

    @socketio.on("set_viewer_look_at")
    def on_view_look(data):
        handle_command_event("set_viewer_look_at", data)

    @socketio.on("set_camera_preset")
    def on_camera_preset(data):
        handle_command_event("set_camera_preset", data)

    @socketio.on("teleport_entity")
    def on_teleport(data):
        handle_command_event("teleport_entity", data)

    @socketio.on("set_initial_positions")
    def on_set_initial_positions(data):
        handle_command_event("set_initial_positions", data)

    @socketio.on("set_robot_velocity")
    def on_set_robot_velocity(data):
        handle_command_event("set_robot_velocity", data)

    @socketio.on("referee_command")
    def on_referee_command(data):
        handle_command_event("referee_command", data)

    def event_pump() -> None:
        nonlocal latest_frame_jpeg, latest_states, field_meta
        while True:
            try:
                event = event_q.get(timeout=0.2)
            except queue.Empty:
                continue
            if event is None:
                break
            if not isinstance(event, dict):
                continue
            et = event.get("type", "")
            if et == "shutdown":
                break
            if et == "frame":
                jpeg = event.get("jpeg", None)
                if isinstance(jpeg, (bytes, bytearray)):
                    jb = bytes(jpeg)
                    with lock:
                        latest_frame_jpeg = jb
                    payload = base64.b64encode(jb).decode("utf-8")
                    socketio.emit("new_frame", {"image": payload}, namespace=_NS)
            elif et == "states":
                states = event.get("states", {})
                if isinstance(states, dict):
                    with lock:
                        latest_states = dict(states)
                    socketio.emit("robot_states", states, namespace=_NS)
            elif et == "field_meta":
                fm = event.get("field_meta", None)
                if isinstance(fm, dict):
                    with lock:
                        field_meta = dict(fm)
                    socketio.emit("field_meta", field_meta, namespace=_NS)

    t = threading.Thread(target=event_pump, daemon=True)
    t.start()

    def parent_watchdog() -> None:
        """Exit orphaned webview process when simulation parent is gone."""
        if parent_pid <= 1:
            return
        while True:
            try:
                os.kill(parent_pid, 0)
            except ProcessLookupError:
                # Parent process no longer exists; avoid serving stale UI with dead command path.
                os._exit(0)
            except Exception:
                # Permission/other transient errors should not kill webview process.
                pass
            time.sleep(1.0)

    threading.Thread(target=parent_watchdog, daemon=True).start()
    socketio.run(
        app,
        host="0.0.0.0",
        port=int(port),
        use_reloader=False,
        debug=False,
        allow_unsafe_werkzeug=True,
    )


class MujocoLabWebView:
    def __init__(
        self,
        template_dir: Path,
        allow_keyboard_control: bool = False,
        *,
        web_jpeg_quality: int = 82,
        web_jpeg_subsampling: int = 2,
    ):
        self.template_dir = Path(template_dir)
        self.msg = WebMsgBuffer()
        self.allow_keyboard_control = bool(allow_keyboard_control)
        self._web_jpeg_quality = int(web_jpeg_quality)
        self._web_jpeg_subsampling = int(web_jpeg_subsampling)
        self._field_meta: dict[str, Any] | None = None
        self._ctx = mp.get_context("spawn")
        self._command_q: mp.Queue | None = None
        self._command_recv = None
        self._command_send = None
        self._event_q: mp.Queue | None = None
        self._proc: mp.Process | None = None
        self._command_file: Path | None = None
        self._command_file_pos: int = 0
        self._jpeg_in_q: queue.Queue | None = None
        self._jpeg_thread: threading.Thread | None = None
        self._jpeg_stop: threading.Event | None = None

    def start(self, port: int = 5811):
        if self._proc is not None and self._proc.is_alive():
            return
        self._command_q = self._ctx.Queue(maxsize=512)
        self._command_recv, self._command_send = self._ctx.Pipe(duplex=False)
        self._event_q = self._ctx.Queue(maxsize=32)
        self._command_file = Path(tempfile.gettempdir()) / f"motrix_web_cmd_{int(port)}.jsonl"
        self._command_file_pos = 0
        try:
            self._command_file.write_text("", encoding="utf-8")
        except Exception:
            pass
        self._jpeg_stop = threading.Event()
        self._jpeg_in_q = queue.Queue(maxsize=2)
        self._jpeg_thread = threading.Thread(
            target=_jpeg_worker_main,
            args=(
                self._jpeg_in_q,
                self._event_q,
                self._web_jpeg_quality,
                self._web_jpeg_subsampling,
                self._jpeg_stop,
            ),
            name="motrix-webview-jpeg",
            daemon=True,
        )
        self._jpeg_thread.start()
        self._proc = self._ctx.Process(
            target=_run_webview_process,
            args=(
                str(self.template_dir),
                bool(self.allow_keyboard_control),
                int(port),
                self._command_q,
                self._command_send,
                self._event_q,
                str(self._command_file),
                int(os.getpid()),
            ),
            daemon=True,
        )
        self._proc.start()

    def poll_commands(self) -> WebMsgBuffer:
        out = WebMsgBuffer()
        def apply_one(cmd: Any) -> None:
            if not isinstance(cmd, dict):
                return
            ct = cmd.get("type", "")
            if ct == "reset_env":
                out.reset_env = True
            elif ct == "restart_match":
                out.restart_match = True
            elif ct == "set_viewer_point":
                out.viewer_point = cmd.get("point", [3.0, 3.0, 1.0])
            elif ct == "set_viewer_look_at":
                out.viewer_look_at = cmd.get("point", [0.0, 0.0, 1.0])
            elif ct == "set_camera_preset":
                out.camera_preset = str(cmd.get("preset", "Top"))
            elif ct == "teleport_entity":
                out.teleport_cmd = (
                    str(cmd.get("name", "")),
                    float(cmd.get("x", 0.0)),
                    float(cmd.get("y", 0.0)),
                    cmd.get("z", None),
                    cmd.get("theta", None),
                )
            elif ct == "set_initial_positions":
                sp = cmd.get("spawn_points", None)
                out.spawn_points = sp if isinstance(sp, dict) else {}
            elif ct == "set_robot_velocity":
                if out.velocity_cmds is None:
                    out.velocity_cmds = []
                out.velocity_cmds.append(
                    (
                        str(cmd.get("name", "")),
                        float(cmd.get("vx", 0.0)),
                        float(cmd.get("vy", 0.0)),
                        float(cmd.get("wz", 0.0)),
                    )
                )
            elif ct == "referee_command":
                out.referee_command = str(cmd.get("command", ""))

        if self._command_recv is not None:
            while True:
                try:
                    if not self._command_recv.poll():
                        break
                    cmd = self._command_recv.recv()
                    apply_one(cmd)
                except EOFError:
                    break
                except Exception:
                    break

        if self._command_q is not None:
            while True:
                try:
                    cmd = self._command_q.get_nowait()
                except queue.Empty:
                    break
                apply_one(cmd)

        if self._command_file is not None and self._command_file.exists():
            try:
                with self._command_file.open("r", encoding="utf-8") as f:
                    f.seek(self._command_file_pos)
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            cmd = json.loads(line)
                        except Exception:
                            continue
                        apply_one(cmd)
                    self._command_file_pos = f.tell()
            except Exception:
                pass
        return out

    def emit_frame(self, rgb: np.ndarray):
        try:
            if self._event_q is None or self._jpeg_in_q is None:
                return
            arr = np.ascontiguousarray(rgb)
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)
            if arr.ndim == 3 and arr.shape[2] == 4:
                arr = arr[:, :, :3]
            if arr.ndim != 3 or arr.shape[2] != 3:
                raise ValueError(f"expected HxWx3 RGB uint8, got shape={getattr(arr, 'shape', None)} dtype={arr.dtype}")
            frame = np.array(arr, copy=True, dtype=np.uint8, order="C")
            try:
                self._jpeg_in_q.put_nowait(frame)
            except queue.Full:
                try:
                    _ = self._jpeg_in_q.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._jpeg_in_q.put_nowait(frame)
                except queue.Full:
                    pass
        except Exception as e:
            _log_emit_throttled("emit_frame", f"[MujocoWebView] emit_frame failed: {e}")

    def emit_robot_states(self, states: dict):
        try:
            if self._event_q is None:
                return
            safe = _socket_json_safe(states)
            if isinstance(safe, dict):
                _queue_put_latest(self._event_q, {"type": "states", "states": safe}, max_drop=32)
        except Exception as e:
            _log_emit_throttled("emit_robot_states", f"[MujocoWebView] emit_robot_states failed: {e}")

    def set_field_meta(self, field_meta: dict):
        self._field_meta = _socket_json_safe(dict(field_meta))
        try:
            if self._event_q is None:
                return
            _queue_put_latest(self._event_q, {"type": "field_meta", "field_meta": dict(self._field_meta)}, max_drop=8)
        except Exception as e:
            _log_emit_throttled("field_meta", f"[MujocoWebView] field_meta broadcast failed: {e}")

    def close(self) -> None:
        try:
            if self._jpeg_in_q is not None and self._jpeg_stop is not None:
                self._jpeg_stop.set()
                try:
                    self._jpeg_in_q.put_nowait(None)
                except Exception:
                    pass
                if self._jpeg_thread is not None:
                    self._jpeg_thread.join(timeout=1.5)
                self._jpeg_in_q = None
                self._jpeg_thread = None
                self._jpeg_stop = None
            if self._event_q is not None:
                _queue_put_latest(self._event_q, {"type": "shutdown"}, max_drop=1)
                try:
                    self._event_q.put_nowait(None)
                except Exception:
                    pass
            if self._proc is not None and self._proc.is_alive():
                self._proc.join(timeout=1.5)
                if self._proc.is_alive():
                    self._proc.terminate()
                    self._proc.join(timeout=0.5)
            for conn in (self._command_send, self._command_recv):
                if conn is not None:
                    try:
                        conn.close()
                    except Exception:
                        pass
        except Exception:
            pass
