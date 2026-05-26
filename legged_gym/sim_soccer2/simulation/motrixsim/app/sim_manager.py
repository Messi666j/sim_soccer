# SPDX-FileCopyrightText: Copyright (c) MOS-Brain Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field


MUJOCO_DIR = Path(__file__).resolve().parents[1]
# Use current interpreter by default so spawned sims share the same conda/venv.
# Only honor $PYTHON when it is an explicit existing path (ignore bare "python").
_python_env = os.environ.get("PYTHON", "").strip()
if _python_env and Path(_python_env).exists():
    PYTHON_BIN = Path(_python_env)
else:
    PYTHON_BIN = Path(sys.executable)
REGISTRY_PATH = MUJOCO_DIR / ".sim_manager_registry.json"
MANAGER_WEB_DIR = MUJOCO_DIR / "web" / "manager"
MANAGER_INDEX_HTML = MANAGER_WEB_DIR / "index.html"
MANAGER_API_DOCS_HTML = MANAGER_WEB_DIR / "api_docs.html"


SIM_CMD_PATTERNS = [
    "motrixsim/sim2sim_runner.py",
    " app.runner",
    " app/runner.py",
]


def _now() -> float:
    return time.time()


def _safe_read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _safe_write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=True, indent=2), encoding="utf-8")


def _process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _cmd_is_sim(args: str) -> bool:
    return any(pat in args for pat in SIM_CMD_PATTERNS)


def _scan_sim_processes() -> list[dict[str, Any]]:
    out = subprocess.check_output(["ps", "-eo", "pid=,ppid=,args="], text=True)
    rows: list[dict[str, Any]] = []
    for raw in out.splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split(None, 2)
        if len(parts) < 3:
            continue
        pid = int(parts[0])
        ppid = int(parts[1])
        args = parts[2]
        if _cmd_is_sim(args):
            rows.append({"pid": pid, "ppid": ppid, "cmd": args})
    return rows


def _extract_int_arg(cmd: str, key: str) -> int | None:
    marker = f"{key} "
    idx = cmd.find(marker)
    if idx < 0:
        return None
    tail = cmd[idx + len(marker) :]
    token = tail.split(" ", 1)[0].strip()
    if not token:
        return None
    try:
        return int(token)
    except ValueError:
        return None


def _extract_str_arg(cmd: str, key: str) -> str | None:
    marker = f"{key} "
    idx = cmd.find(marker)
    if idx < 0:
        return None
    tail = cmd[idx + len(marker) :]
    token = tail.split(" ", 1)[0].strip()
    return token or None


def _terminate_pid(pid: int, graceful_timeout_sec: float = 5.0) -> bool:
    if not _process_exists(pid):
        return True
    try:
        os.killpg(pid, signal.SIGTERM)
    except Exception:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            return True
        except Exception:
            pass
    deadline = time.time() + graceful_timeout_sec
    while time.time() < deadline:
        if not _process_exists(pid):
            return True
        time.sleep(0.1)
    try:
        os.killpg(pid, signal.SIGKILL)
    except Exception:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            return True
        except Exception:
            pass
    time.sleep(0.1)
    return not _process_exists(pid)


@dataclass
class ManagedSim:
    sim_id: str
    pid: int
    created_at: float
    team_size: int
    robot_type: str
    zmq_port: int
    webview_port: int
    args: list[str] = field(default_factory=list)


class StartSimRequest(BaseModel):
    team_size: int = Field(default=1, ge=0, le=7)
    zmq_port: int | None = Field(default=None, ge=1, le=65535)
    webview_port: int | None = Field(default=None, ge=1, le=65535)
    webview: bool = True
    zmq: bool = True
    web_fps: int = Field(default=20, ge=1, le=120)
    web_width: int = Field(default=1280, ge=64, le=8192)
    web_height: int = Field(default=720, ge=64, le=8192)
    web_jpeg_quality: int = Field(default=82, ge=1, le=95)
    web_jpeg_subsampling: int = Field(default=2, ge=0, le=2)
    allow_keyboard_control: bool = False
    robot_type: str = Field(default="k1", pattern="^(k1|pi_plus)$")
    policy_device: str = Field(default="gpu", pattern="^(cpu|gpu)$")
    policy: str | None = None
    robot_xml: str | None = None
    soccer_world_xml: str | None = None
    match_config: str | None = None
    use_referee: bool = False


class StopRequest(BaseModel):
    pid: int


class SimManager:
    def __init__(self, registry_path: Path):
        self.registry_path = registry_path

    def _load(self) -> dict[str, Any]:
        raw = _safe_read_json(self.registry_path)
        if "managed" not in raw or not isinstance(raw["managed"], dict):
            raw["managed"] = {}
        return raw

    def _save(self, data: dict[str, Any]) -> None:
        _safe_write_json(self.registry_path, data)

    def _cleanup_dead(self) -> None:
        data = self._load()
        managed = data.get("managed", {})
        alive: dict[str, Any] = {}
        for sim_id, item in managed.items():
            pid = int(item.get("pid", -1))
            if pid > 0 and _process_exists(pid):
                alive[sim_id] = item
        data["managed"] = alive
        self._save(data)

    @staticmethod
    def _is_tcp_port_free(port: int) -> bool:
        if port <= 0 or port > 65535:
            return False
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("0.0.0.0", port))
            except OSError:
                return False
        return True

    def _pick_port(self, preferred: int, used: set[int]) -> int:
        for p in range(max(1, preferred), 65536):
            if p in used:
                continue
            if self._is_tcp_port_free(p):
                used.add(p)
                return p
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("0.0.0.0", 0))
            p = int(s.getsockname()[1])
            if p in used:
                raise HTTPException(status_code=500, detail="failed to allocate a free TCP port")
            used.add(p)
            return p

    def start(self, req: StartSimRequest) -> dict[str, Any]:
        self._cleanup_dead()
        used: set[int] = set()
        selected_zmq_port = self._pick_port(int(req.zmq_port) if req.zmq_port is not None else 5555, used)
        selected_webview_port = self._pick_port(
            int(req.webview_port) if req.webview_port is not None else 5811,
            used,
        )
        cmd = [
            str(PYTHON_BIN),
            str(MUJOCO_DIR / "sim2sim_runner.py"),
            "--robot-type",
            req.robot_type,
            "--team-size",
            str(req.team_size),
            "--port",
            str(selected_zmq_port),
            "--webview-port",
            str(selected_webview_port),
            "--web-fps",
            str(req.web_fps),
            "--web-width",
            str(req.web_width),
            "--web-height",
            str(req.web_height),
            "--web-jpeg-quality",
            str(req.web_jpeg_quality),
            "--web-jpeg-subsampling",
            str(req.web_jpeg_subsampling),
            "--policy-device",
            req.policy_device,
        ]
        cmd += ["--webview"] if req.webview else ["--no-webview"]
        cmd += ["--allow-keyboard-control"] if req.allow_keyboard_control else ["--no-allow-keyboard-control"]
        cmd += ["--zmq"] if req.zmq else ["--no-zmq"]
        cmd += ["--use-referee"] if req.use_referee else ["--no-use-referee"]
        if req.policy:
            cmd += ["--policy", req.policy]
        if req.robot_xml:
            cmd += ["--robot-xml", req.robot_xml]
        if req.soccer_world_xml:
            cmd += ["--soccer-world-xml", req.soccer_world_xml]
        if req.match_config:
            cmd += ["--match-config", req.match_config]

        proc = subprocess.Popen(
            cmd,
            cwd=str(MUJOCO_DIR),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            preexec_fn=os.setsid,
        )
        sim = ManagedSim(
            sim_id=str(uuid.uuid4()),
            pid=int(proc.pid),
            created_at=_now(),
            team_size=int(req.team_size),
            robot_type=str(req.robot_type),
            zmq_port=selected_zmq_port,
            webview_port=selected_webview_port,
            args=cmd,
        )
        data = self._load()
        data["managed"][sim.sim_id] = asdict(sim)
        self._save(data)
        out = asdict(sim)
        out["resolved"] = {"zmq_port": selected_zmq_port, "webview_port": selected_webview_port}
        return out

    def scan(self) -> dict[str, Any]:
        self._cleanup_dead()
        data = self._load()
        managed = data.get("managed", {})
        managed_pids = {int(item["pid"]) for item in managed.values()}
        scanned = _scan_sim_processes()
        for row in scanned:
            meta = next((item for item in managed.values() if int(item.get("pid", -1)) == row["pid"]), None)
            row["managed"] = row["pid"] in managed_pids
            if row["managed"]:
                row["sim_id"] = next(
                    (sid for sid, item in managed.items() if int(item.get("pid", -1)) == row["pid"]),
                    None,
                )
            row["team_size"] = (
                int(meta["team_size"])
                if meta is not None and "team_size" in meta
                else _extract_int_arg(row["cmd"], "--team-size")
            )
            row["robot_type"] = (
                str(meta["robot_type"])
                if meta is not None and "robot_type" in meta
                else (_extract_str_arg(row["cmd"], "--robot-type") or "k1")
            )
            row["zmq_port"] = (
                int(meta["zmq_port"])
                if meta is not None and "zmq_port" in meta
                else _extract_int_arg(row["cmd"], "--port")
            )
            row["webview_port"] = (
                int(meta["webview_port"])
                if meta is not None and "webview_port" in meta
                else _extract_int_arg(row["cmd"], "--webview-port")
            )
        return {"managed": list(managed.values()), "scanned": scanned}

    def stop_pid(self, pid: int) -> dict[str, Any]:
        scanned = _scan_sim_processes()
        scanned_pids = {int(row["pid"]) for row in scanned}
        if pid not in scanned_pids:
            raise HTTPException(status_code=404, detail=f"pid={pid} is not a scanned mujoco sim process")

        ok = _terminate_pid(pid)
        self._cleanup_dead()
        return {"pid": pid, "stopped": bool(ok)}

    def stop_external(self) -> dict[str, Any]:
        self._cleanup_dead()
        data = self._load()
        managed = data.get("managed", {})
        managed_pids = {int(item["pid"]) for item in managed.values()}

        scanned = _scan_sim_processes()
        targets = [int(row["pid"]) for row in scanned if int(row["pid"]) not in managed_pids]
        results = []
        for pid in targets:
            results.append({"pid": pid, "stopped": _terminate_pid(pid)})
        self._cleanup_dead()
        return {"targets": targets, "results": results}

    def stop_all_managed(self) -> dict[str, Any]:
        self._cleanup_dead()
        data = self._load()
        managed = data.get("managed", {})
        targets = [int(item["pid"]) for item in managed.values() if int(item.get("pid", -1)) > 0]
        results = []
        for pid in targets:
            results.append({"pid": pid, "stopped": _terminate_pid(pid)})
        data["managed"] = {}
        self._save(data)
        return {"targets": targets, "results": results}


manager = SimManager(REGISTRY_PATH)
app = FastAPI(title="Mujoco Sim Manager", version="1.0.0")


@app.on_event("shutdown")
def _shutdown_cleanup():
    try:
        result = manager.stop_all_managed()
        print(f"[SimManager] shutdown cleanup: {result}")
    except Exception as e:
        print(f"[SimManager] shutdown cleanup failed: {e}")


@app.get("/")
def manager_ui():
    if not MANAGER_INDEX_HTML.exists():
        raise HTTPException(status_code=500, detail=f"Missing UI file: {MANAGER_INDEX_HTML}")
    return FileResponse(str(MANAGER_INDEX_HTML))


@app.get("/manager/docs")
def manager_docs_page():
    if not MANAGER_API_DOCS_HTML.exists():
        raise HTTPException(status_code=500, detail=f"Missing docs file: {MANAGER_API_DOCS_HTML}")
    return FileResponse(str(MANAGER_API_DOCS_HTML))


@app.get("/healthz")
def healthz():
    return {"ok": True, "time": _now()}


@app.post("/sims/start")
def start_sim(req: StartSimRequest):
    return manager.start(req)


@app.get("/sims")
def list_sims():
    return manager.scan()


@app.post("/sims/stop")
def stop_sim(req: StopRequest):
    return manager.stop_pid(int(req.pid))


@app.post("/sims/stop-external")
def stop_external_sims():
    return manager.stop_external()
