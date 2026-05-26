#!/usr/bin/env python3

import threading
import time
import numpy as np
import logging
import os
import sys
import math
from typing import Dict, List, Optional, Tuple

# Add path to find logic package if needed
DECIDER_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if DECIDER_PATH not in sys.path:
    sys.path.append(DECIDER_PATH)

from logic.strategy_statemachines.defend_ball_state_machine import DefendBallStateMachine
from logic.strategy_statemachines.dribble_ball_state_machine import DribbleBallStateMachine
from logic.strategy_statemachines.shoot_ball_state_machine import ShootBallStateMachine
from logic.strategy_statemachines.attack_state_machine import StateMachine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [TeamManager] - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("TeamManager")

class TeamManager:
    """
    Centralized Team Manager for Multi-Agent Strategy.
    Responsibilities:
    1. Collect world state from all robots (fusion).
    2. Assign dynamic roles (Striker, Defender, etc.).
    3. Dispatch high-level strategies/commands to robots.
    """
    def __init__(self, goalkeeper_id=4):
        self.robots_data: Dict[int, Dict] = {}
        self.goalkeeper_id = goalkeeper_id
        self.roles = {
            "striker": None,
            "defender": [],
            "goalkeeper": goalkeeper_id,
            "supporter": []
        }
        
        # World Model
        self.ball_pos_estimate = None
        self.ball_last_update = 0
        
        # Strategy State Machines
        # Note: These SMs generally expect an 'agent' with certain interfaces. 
        # We pass 'self' as the agent, so we must provide the expected interface methods.
        self._init_state_machines()
        
        # Command Buffer: robot_id -> command_string
        self.commands = {}

    def _init_state_machines(self):
        """Initialize functionality-specific state machines."""
        # These might need to be adapted if they were designed for single robots logic
        # For now, we assume they can run here to generate high-level decisions.
        self.state_machine = StateMachine(self)
        self.defend_sm = DefendBallStateMachine(self)
        self.dribble_sm = DribbleBallStateMachine(self)
        self.shoot_sm = ShootBallStateMachine(self)

    def update_robot_state(self, robot_id: int, data: Dict):
        """
        Update the internal state of a robot based on received data.
        data format expected: {'pose': [x, y, theta], 'ball': {'detected': bool, 'pos': [x, y]}, ...}
        """
        self.robots_data[robot_id] = {
            "timestamp": time.time(),
            "data": data
        }
        self._update_world_model()

    def _update_world_model(self):
        """Fuse data from multiple robots to estimate world state."""
        # 1. Fuse Ball Position (Simple Average of confident detections)
        ball_x_sum, ball_y_sum, count = 0, 0, 0
        
        current_time = time.time()
        for rid, info in self.robots_data.items():
            # Discard old data (> 1.0s)
            if current_time - info['timestamp'] > 1.0:
                continue
                
            data = info.get('data', {})
            ball_data = data.get('ball', {})
            
            if ball_data.get('detected'):
                # Transform local ball to global if needed, or assume data is global
                # Assuming data['ball']['pos'] is global or we interpret it as such for now
                # If robots send local, we need self_pose + local_ball -> global_ball
                
                # Check if data is already global map coordinates
                b_pos = ball_data.get('pos_global') or ball_data.get('pos')
                if b_pos:
                    ball_x_sum += b_pos[0]
                    ball_y_sum += b_pos[1]
                    count += 1
        
        if count > 0:
            self.ball_pos_estimate = (ball_x_sum / count, ball_y_sum / count)
            self.ball_last_update = current_time
        else:
            # Decay or reset? Keep last known for short time
            if current_time - self.ball_last_update > 5.0:
                self.ball_pos_estimate = None

    def step(self):
        """Main Strategy Loop Step"""
        self._assign_roles()
        self._execute_strategies()

    def _assign_roles(self):
        """
        Dynamic Role Assignment.
        Rule 1: Goalkeeper is fixed.
        Rule 2: Robot closest to ball is Striker.
        Rule 3: Others are Defenders/Supporters.
        """
        active_ids = [rid for rid, info in self.robots_data.items() 
                      if time.time() - info['timestamp'] < 2.0 and rid != self.goalkeeper_id]
        
        if not active_ids:
            return

        if self.ball_pos_estimate:
            # Sort by distance to ball
            def dist_to_ball(rid):
                r_pos = self.robots_data[rid]['data'].get('pose', [0,0])
                if not r_pos: return float('inf')
                return math.dist(r_pos[:2], self.ball_pos_estimate)
            
            sorted_robots = sorted(active_ids, key=dist_to_ball)
            
            self.roles['striker'] = sorted_robots[0]
            self.roles['defender'] = sorted_robots[1:] # Simple logic: rest are defenders
        else:
            # No ball? Stick to previous or default
            if not self.roles['striker'] and active_ids:
                self.roles['striker'] = active_ids[0]
                self.roles['defender'] = active_ids[1:]

    def _execute_strategies(self):
        """Decide what each role should do."""
        # Striker Logic
        striker_id = self.roles['striker']
        if striker_id:
            # By default, striker chases/dribbles/shoots
            # We can use the StateMachines here contextually
            # For simplicity, we just set a high-level command
            self.commands[striker_id] = "chase_and_kick"
        
        # Defender Logic
        for def_id in self.roles['defender']:
            self.commands[def_id] = "defend_zone"
            
        # Goalkeeper Logic
        if self.goalkeeper_id:
             self.commands[self.goalkeeper_id] = "goalkeeper_auto"

    def get_command(self, robot_id: int):
        return self.commands.get(robot_id, "stop")

    # --- Interface Methods for State Machines (Duck Typing) ---
    def get_ball_pos(self):
        return self.ball_pos_estimate or (0,0)
    
    def get_robot_pos(self, robot_id):
        if robot_id in self.robots_data:
            return self.robots_data[robot_id]['data'].get('pose', [0,0,0])
        return [0,0,0]

    # Add other methods required by DefendBallStateMachine etc. if they call agent methods


class StrategyServer:
    """Network Interface for TeamManager"""
    def __init__(self, ip="0.0.0.0", port="5555"):
        self.manager = TeamManager()
        self.running = False
        self.ip = ip
        self.port = port
        # TODO: Initialize ZMQ REP or SUB socket here
        
    def run(self):
        self.running = True
        logger.info(f"Strategy Server started on {self.ip}:{self.port}")
        while self.running:
            time.sleep(0.1)
            # Mock Loop:
            # 1. Receive data (mock)
            # self.manager.update_robot_state(...)
            
            # 2. Step Logic
            self.manager.step()
            
            # 3. Publish Commands
            # for rid in self.manager.robots_data:
            #     cmd = self.manager.get_command(rid)
            #     send_cmd(rid, cmd)
                
    def stop(self):
        self.running = False

if __name__ == "__main__":
    server = StrategyServer()
    try:
        server.run()
    except KeyboardInterrupt:
        server.stop()
