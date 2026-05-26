# SPDX-FileCopyrightText: Copyright (c) MOS-Brain Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Sim Manager HTTP API — compatibility module for historic imports and docs.

Many runbooks use::

    uvicorn simulation.labbridge.sim_manager:app

Implementation (defaults, logging, cwd) is maintained in :mod:`simulation.labbridge.sim_manager2`.
Import either module; they expose the same application object.
"""

from __future__ import annotations

from simulation.labbridge.sim_manager2 import (  # noqa: F401
    DEFAULT_RUNNER,
    DEFAULT_SIM_ROOT,
    MANAGER_API_DOCS_HTML,
    MANAGER_INDEX_HTML,
    MANAGER_WEB_DIR,
    MODULE_DIR,
    PROJECT_ROOT,
    PYTHON_BIN,
    REGISTRY_PATH,
    SIM_CMD_PATTERNS,
    ManagedSim,
    SimManager,
    StartSimRequest,
    StopRequest,
    app,
    create_app,
)

__all__ = [
    "DEFAULT_RUNNER",
    "DEFAULT_SIM_ROOT",
    "MANAGER_API_DOCS_HTML",
    "MANAGER_INDEX_HTML",
    "MANAGER_WEB_DIR",
    "MODULE_DIR",
    "PROJECT_ROOT",
    "PYTHON_BIN",
    "REGISTRY_PATH",
    "SIM_CMD_PATTERNS",
    "ManagedSim",
    "SimManager",
    "StartSimRequest",
    "StopRequest",
    "app",
    "create_app",
]
