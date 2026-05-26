# SPDX-FileCopyrightText: Copyright (c) MOS-Brain Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

import sys
import os
import time
import json
import logging

# Helper to find zmq_wrapper
# Priority:
# 1) Explicit env: MOS_BRAIN_MUJOCO_COMMON_PATH
# 2) Mujoco backend common path
# 3) Legacy Isaac simulation common path
COMMON_CANDIDATES = []
env_common = os.environ.get("MOS_BRAIN_MUJOCO_COMMON_PATH")
if env_common:
    COMMON_CANDIDATES.append(os.path.abspath(env_common))
COMMON_CANDIDATES.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../simulation/mujoco/src/common")))
COMMON_CANDIDATES.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../simulation/isaac_sim/src/common")))

COMMON_PATH = None
for candidate in COMMON_CANDIDATES:
    if os.path.isdir(candidate):
        COMMON_PATH = candidate
        if COMMON_PATH not in sys.path:
            sys.path.append(COMMON_PATH)
        break

try:
    from zmq_wrapper import ZMQWrapper
    import zmq
except ImportError:
    print(f"[SimClient] Error: Could not import zmq_wrapper or zmq. Ensure pyzmq is installed and path is correct: {COMMON_PATH}")
    # Fallback or exit
    ZMQWrapper = None

class SimClient:
    def __init__(self, ip="127.0.0.1", port="5555"):
        self.logger = logging.getLogger("SimClient")
        
        if ZMQWrapper is None:
            raise RuntimeError("ZMQ dependencies missing.")

        self.ip = ip
        self.port = port
        self.addr = f"tcp://{ip}:{port}"
        
        self.client = ZMQWrapper(socket_type=zmq.REQ, addr=self.addr)
        self.client.connect()
        self.logger.info(f"Connected to Sim Server at {self.addr}")
        
    def communicate(self, cmd_vel=[0.0, 0.0, 0.0], robot_id=0):
        """
        Send command and wait for state.
        cmd_vel: [vx, vy, w]
        robot_id: int (default 0)
        Returns: proper state dict or None on failure
        """
        req = {
            "cmd": cmd_vel,
            "id": robot_id,
            "timestamp": time.time()
        }
        
        try:
            self.client.send_json(req)
            resp = self.client.recv_json()
            # Basic validation
            if not resp or "state" not in resp:
                self.logger.warning("Received invalid response from Sim.")
                return None
            return resp
        except Exception as e:
            self.logger.error(f"Communication error: {e}")
            return None

    def force_reset(self):
        """Force close and recreate the ZMQ socket. Use this in shutdown/error recovery."""
        try:
             self.client.close()
        except:
             pass
        try:
             self.client = ZMQWrapper(socket_type=zmq.REQ, addr=self.addr)
             self.client.connect()
             self.logger.info("Socket force reset complete.")
        except Exception as e:
             self.logger.error(f"Failed to reset socket: {e}")

    def close(self):
        if self.client:
            self.client.close()
