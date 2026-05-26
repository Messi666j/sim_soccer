# SPDX-FileCopyrightText: Copyright (c) MOS-Brain Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

# interfaces/communication.py
#
#   @description:   Utilities to handle multi-robot communication data from ZMQ
#   @interfaces:
#       1. class Communication

import logging
import numpy as np

class Communication:
    def __init__(self, agent):
        self.agent = agent
        self.logger = agent.get_logger()
        # Initialize robots data storage
        # {
        #   "robot_id": {
        #       "status": "connected" / "disconnected",
        #       "data": { ... }
        #   }
        # }
        self.robots_data = {} 

    def update(self, comm_data: dict):
        """
        Update robots data from ZMQ dictionary.
        Expected structure: a dict of robot_id -> robot_data
        """
        if not comm_data:
            return

        try:
            # Assumes comm_data is already a dict of robot states
            # e.g. {'1': {'status': 'connected', 'data': {...}}, ...}
            # or just a list of robot states including self.
            
            # If it's a list, we might need to conform it to a dict keyed by ID.
            if isinstance(comm_data, list):
                 for robot in comm_data:
                     r_id = robot.get('id')
                     if r_id is not None:
                         self.robots_data[str(r_id)] = robot
            elif isinstance(comm_data, dict):
                 self.robots_data = comm_data
            
            # Filter out self if needed, or keep all. 
            # The agent logic usually filters self out when needed.

        except Exception as e:
            self.logger.error(f"Error updating Communication: {e}")

    def get_robots_data(self):
        return self.robots_data
