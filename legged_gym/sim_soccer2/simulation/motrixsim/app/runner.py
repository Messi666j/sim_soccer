#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) MOS-Brain Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import os
from pathlib import Path

from .multi_robot_sim import run_sim
from .runtime_config import parse_runtime_args

MUJOCO_DIR = Path(__file__).resolve().parents[1]


def main():
    # Keep render logs from flooding stdout/stderr (can starve webview responsiveness).
    os.environ.setdefault("RUST_LOG", "error,bevy_render::view::window::screenshot=off")
    args = parse_runtime_args(MUJOCO_DIR)
    template_dir = MUJOCO_DIR / "web" / "templates"
    run_sim(args=args, template_dir=template_dir)


if __name__ == "__main__":
    main()
