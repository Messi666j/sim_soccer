# SPDX-FileCopyrightText: Copyright (c) MOS-Brain Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from .webview_server import LabWebView, WebMsgBuffer


class GenesisBridge:
    """Bridge Genesis simulation loop to the MuJoCo webview protocol."""

    def __init__(
        self,
        *,
        webview_port: int = 5811,
        web_fps: int = 20,
        enable_webview: bool = True,
        allow_keyboard_control: bool = False,
        template_dir: Path | None = None,
    ):
        self.enable_webview = bool(enable_webview)
        self.web_fps = max(1, int(web_fps))
        self._frame_interval = 1.0 / float(self.web_fps)
        self._state_interval = 1.0 / float(self.web_fps)
        now = time.time()
        self._next_frame_time = now
        self._next_state_time = now

        if template_dir is None:
            template_dir = (Path(__file__).resolve().parents[1] / "mujoco" / "web" / "templates")
        self.template_dir = Path(template_dir)

        self.webview: LabWebView | None = None
        if self.enable_webview:
            self.webview = LabWebView(
                template_dir=self.template_dir,
                allow_keyboard_control=allow_keyboard_control,
            )
            self.webview.start(port=int(webview_port))
            print(f"[GenesisBridge] WebView started at http://localhost:{int(webview_port)}")

    def poll_commands(self) -> WebMsgBuffer:
        """Poll one-shot commands from frontend (reset, teleport, velocity, camera)."""
        if self.webview is None:
            return WebMsgBuffer()
        return self.webview.poll_commands()

    def emit_frame(self, rgb: Any, *, force: bool = False) -> bool:
        """Emit encoded JPEG frame to frontend, throttled by web_fps."""
        if self.webview is None:
            return False
        now = time.time()
        if (not force) and now < self._next_frame_time:
            return False
        self.webview.emit_frame(rgb)
        self._next_frame_time = now + self._frame_interval
        return True

    def emit_states(self, states: dict[str, Any], *, force: bool = False) -> bool:
        """Emit robot/ball states to frontend, throttled by web_fps."""
        if self.webview is None:
            return False
        now = time.time()
        if (not force) and now < self._next_state_time:
            return False
        self.webview.emit_robot_states(states)
        self._next_state_time = now + self._state_interval
        return True

    def publish(self, *, rgb: Any | None = None, states: dict[str, Any] | None = None, force: bool = False):
        """Convenience helper to emit frame and/or states in one call."""
        if rgb is not None:
            self.emit_frame(rgb, force=force)
        if states is not None:
            self.emit_states(states, force=force)


def genesis_bridge(
    *,
    webview_port: int = 5811,
    web_fps: int = 20,
    enable_webview: bool = True,
    allow_keyboard_control: bool = False,
    template_dir: Path | None = None,
) -> GenesisBridge:
    """Factory function for Genesis webview bridge."""
    return GenesisBridge(
        webview_port=webview_port,
        web_fps=web_fps,
        enable_webview=enable_webview,
        allow_keyboard_control=allow_keyboard_control,
        template_dir=template_dir,
    )
