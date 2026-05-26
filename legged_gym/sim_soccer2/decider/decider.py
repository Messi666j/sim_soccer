# SPDX-FileCopyrightText: Copyright (c) MOS-Brain Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

# decider.py
#
#   @description : The entry py file for decision-making on clients
#

import sys, os, math, signal, time, traceback
import numpy as np
from pathlib    import Path
from datetime   import datetime
from typing     import Optional, Dict, Any, Callable, List

# Conditional ROS2 Import - allows running without ROS2 in simulation mode
ROS_AVAILABLE = False
try:
    import rclpy
    from rclpy.node import Node
    ROS_AVAILABLE = True
except ImportError:
    # Running without ROS2 - simulation mode only
    Node = object  # Fallback base class

# Interfaces with other components
import interfaces.action
import interfaces.vision
import configuration

# Import user_entry
import user_entry


class Agent(Node):
    """
    Main agent class responsible for decision-making and robot control.
    Manages state machines and handles communication with other components.
    """

    def __init__(self, args=None):
        """Initialize the Agent instance and set up ROS 2 node and components."""
        super().__init__('decider')
        self.logger = self.get_logger()
        self.logger.info("[Core] Initializing the core")
        self._config = configuration.load_config()
        self._action = interfaces.action.Action(self)
        self._vision = interfaces.vision.Vision(self)
        self.logger.info("[Core] Core initialized. Calling user's init()")

        user_entry.init(self)

        # 创建定时器，替代ROS 1中的while循环
        timer_period = 1.0 / 10  # 默认10Hz
        self.timer = self.create_timer(timer_period, self._timer_callback)
        self.logger.info("[Core] ALl initializing finished. ")

    def _timer_callback(self):
        """定时器回调函数，替代ROS 1中的主循环"""
        try:
            user_entry.loop(self)
        except Exception as e:
            self.logger.error(f"[Core] Error in timer callback: {e}")
            self.logger.error(f"[Core] Core exited !")
            self.stop()
            self.logger.error(f"[Core] The trackback infomation is shown below:")
            traceback.print_exc()
            exit(-1)

    def cmd_vel(self, vel_x: float, vel_y: float, vel_theta: float) -> None:
        """Set robot velocity with scaling from configuration."""
        vel_x *= self._config.get("max_walk_vel_x", 0.25)
        vel_y *= self._config.get("max_walk_vel_y", 0.1)
        vel_theta *= self._config.get("max_walk_vel_theta", 0.5)
        if vel_x > 0.001:
            vel_x += self._config.get("min_walk_vel_x", 0.2)
        elif vel_x < -0.001:
            vel_x -= self._config.get("min_walk_vel_x", 0.2)
        if vel_y > 0.001:
            vel_y += self._config.get("min_walk_vel_y", 0.2)
        elif vel_y < -0.001:
            vel_y -= self._config.get("min_walk_vel_y", 0.2)
        if vel_theta > 0.001:
            vel_theta += self._config.get("min_walk_vel_theta", 0.3)
        elif vel_theta < -0.001:
            vel_theta -= self._config.get("min_walk_vel_thetea", 0.3)
        self._action.cmd_vel(vel_x, vel_y, vel_theta)


    def move_head(self, pitch: float, yaw: float) -> None:
        """Set robot's head and neck angles."""
        self._action._move_head(pitch, yaw)

    def stop(self, sleep_time: float = 0) -> None:
        """Stop the robot's movement and optionally sleep."""
        self._action.cmd_vel(0.0, 0.0, 0.0)
        time.sleep(sleep_time)

    def kick(self, foot=0, death=0) -> None:
        """Execute the kicking action."""
        self._action.do_kick(foot, death)

    def save_ball(self, direction=1):
        """Execute the saving action."""
        self._action.save_ball(direction)

    # normalize(-pi,pi)
    def angle_normalize(self, angle: float) -> float:
        """Normalize angle to the range (-pi, pi)."""
        if angle is None:
            return None
        angle = angle % (2 * math.pi)
        if angle > math.pi:
            angle -= 2 * math.pi
        elif angle < -math.pi:
            angle += 2 * math.pi
        self.get_logger().debug(f"Normalized angle: {angle}")
        return angle

    # Data retrieval methods
    def relocate(self, x: float = 0, y : float = 0, theta : float = 0):
        """reset current location as reference point"""
        return self._vision.relocate(x, y, theta)
    
    def get_objects(self) -> list:
        # A list of dict:
        #    {
        #        'label': label,
        #        'relative_pos': relative_pos,
        #        'absolute_pos': absolute_coord,
        #        'distance': distance,
        #        'confidence': obj.confidence,
        #        'bounding_box_center': curr_coord / 1000,
        #        'bound_left_low': np.array(obj.bound_left_low[:2]) if obj.bound_left_low else None,
        #        'bound_right_low': np.array(obj.bound_right_low[:2]) if obj.bound_right_low else None,
        #        'timestamp': time.time()
        #    }
        return self._vision.get_objects()

    def get_self_pos(self) -> np.ndarray:
        """Return self position in map coordinates."""
        return self._vision.self_pos

    def get_self_yaw(self) -> float:
        """Return self orientation angle."""
        return self._vision.self_yaw

    def get_ball_pos(self) -> List[Optional[float]]:
        """Return ball position relative to robot or [None, None] if not found."""
        if self.get_if_ball():
            return self._vision.get_ball_pos()
        else:
            return [None, None]

    def get_ball_angle(self) -> Optional[float]:
        """Calculate and return ball angle relative to robot.
        NEW: X-Forward, Y-Left coordinate system
        Angle = atan2(Left, Forward) = atan2(ball_y, ball_x)
        Positive angle = ball is to the left
        Negative angle = ball is to the right
        """
        ball_pos = self.get_ball_pos()
        ball_x = ball_pos[0]  # Forward
        ball_y = ball_pos[1]  # Left

        if ball_pos is not None and ball_x is not None and ball_y is not None and self.get_if_ball():
            if ball_x == 0 and ball_y == 0:
                return None
            else:
                # atan2(Left, Forward) gives angle from forward direction
                angle_rad = math.atan2(ball_y, ball_x)
                return angle_rad
        else:
            return None

    def get_ball_pos_in_vis(self) -> np.ndarray:
        """Return ball position in vision coordinates."""
        return self._vision.get_ball_pos_in_vis()

    def get_ball_pos_in_map(self) -> np.ndarray:
        """Return ball position in global map coordinates."""
        if not self.get_if_ball():
            self.get_logger().warning("Ball not detected, returning None")
            return None
        return self._vision.get_ball_pos_in_map()

    def get_if_ball(self) -> bool:
        """Return whether ball is detected."""
        return self._vision.get_if_ball()

    def get_if_close_to_ball(self) -> bool:
        """Return whether robot is close to the ball."""
        if self.get_if_ball():
            return self.get_ball_distance() < self._config.get("close_to_ball_threshold", 0.5)
        else:
            return False

    def get_ball_distance(self) -> float:
        """Return distance to ball or a large value if not detected."""
        if self.get_if_ball():
            return self._vision.ball_distance
        else:
            return 1e6

    def get_neck(self) -> float:
        """Return neck angle."""
        return self._vision.neck

    def get_head(self) -> float:
        """Return head angle."""
        return self._vision.head


    def get_ball_history(self):
        """Return the history of ball positions."""
        return self._vision.get_ball_history()

    def get_config(self) -> Dict:
        """Return agent configuration."""
        return self._config



class SimAgent:
    """
    Agent for Simulation mode (ZeroMQ, No ROS).
    Mimics the interface of Agent.
    """
    
    class _ROS2CompatibleLogger:
        """Wrapper to provide ROS2-like logger interface using Python logging."""
        def __init__(self, name):
            self._logger = logging.getLogger(name)
        
        def get_child(self, suffix):
            """Mimic ROS2 get_child - returns a logger with hierarchical name."""
            return SimAgent._ROS2CompatibleLogger(f"{self._logger.name}.{suffix}")
        
        def info(self, msg): self._logger.info(msg)
        def debug(self, msg): self._logger.debug(msg)
        def warning(self, msg): self._logger.warning(msg)
        def error(self, msg): self._logger.error(msg)
        def fatal(self, msg): self._logger.critical(msg)
    
    def __init__(self, args=None, ip="127.0.0.1", port="5555", sim_hz=None):
        # Setup logging (std out and file)
        log_fmt = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')
        
        # Console Handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(log_fmt)
        
        # File Handler
        try:
            os.makedirs("logs", exist_ok=True)
            file_handler = logging.FileHandler(f"logs/sim_decider_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
            file_handler.setFormatter(log_fmt)
            handlers = [console_handler, file_handler]
        except Exception as e:
            print(f"Failed to setup file logging: {e}")
            handlers = [console_handler]

        logging.basicConfig(level=logging.INFO, handlers=handlers, force=True)
        
        self.logger = SimAgent._ROS2CompatibleLogger("SimDecider")
        self.logger.info("[SimCore] Initializing SimAgent in ROS-Free Mode (Logging to Console + File)")
        
        self.is_simulation = True
        self._config = configuration.load_config()
        self.id = self._config.get("id", 0)
        self.league = self._config.get("league", "S") # Fix: Explicitly set league (Default S)
        # Simulation control frequency (Hz). <= 0 means unlimited lockstep speed.
        self.sim_hz = self._config.get("sim_hz", 50.0) if sim_hz is None else sim_hz
        
        # Override with command line arg if provided
        if args:
            if hasattr(args, "id") and args.id is not None:
                self.id = args.id
                self._config["id"] = self.id # [FIX] Also update config so Vision module sees correct Local ID
            
            # Determine color: priority CLI > Config
            self.color = self._config.get("color", "red")
            if hasattr(args, "color") and args.color is not None:
                 self.color = args.color

            if hasattr(args, "sim_hz") and args.sim_hz is not None:
                self.sim_hz = args.sim_hz

            # Configure team offset based on resolved color
            if self.color:
                import json
                try:
                    # Resolve config path relative to this file.
                    # Priority: Mujoco backend config -> legacy simulation config.
                    cfg_candidates = [
                        os.path.abspath(
                            os.path.join(os.path.dirname(__file__), "../simulation/mujoco/assets/config/match_config.json")
                        ),
                        os.path.abspath(
                            os.path.join(os.path.dirname(__file__), "../simulation/isaac_sim/config/match_config.json")
                        ),
                    ]
                    cfg_path = None
                    for c in cfg_candidates:
                        if os.path.exists(c):
                            cfg_path = c
                            break
                    if cfg_path is None:
                        raise FileNotFoundError("No match_config.json found in simulation/mujoco or simulation/isaac_sim")

                    with open(cfg_path, 'r') as f:
                        match_config = json.load(f)
                    
                    red_count = match_config.get("teams", {}).get("red", {}).get("count", 1)
                    
                    if self.color == "blue":
                        self.logger.info(f"[SimCore] Team Blue specified. Adding offset {red_count} to ID {self.id}")
                        self.id += red_count
                    elif self.color == "red":
                         self.logger.info(f"[SimCore] Team Red specified. ID {self.id} unchanged")
                         
                except Exception as e:
                    self.logger.warning(f"[SimCore] Failed to load match config for team offset: {e}. Using raw ID.")

        try:
            self.sim_hz = float(self.sim_hz)
        except (TypeError, ValueError):
            self.logger.warning(f"[SimCore] Invalid sim_hz={self.sim_hz}, fallback to 50.0")
            self.sim_hz = 50.0

        if self.sim_hz > 0:
            self.logger.info(f"[SimCore] Control frequency limited to {self.sim_hz:.2f} Hz")
        else:
            self.logger.info("[SimCore] Control frequency unlimited (lockstep max speed)")

        self.logger.info(f"[SimCore] Final Robot ID: {self.id}")
        
        # Init ZMQ Client
        from interfaces.sim_client import SimClient
        self.client = SimClient(ip=ip, port=port)
        
        # Init Interfaces (mocking or adapting)
        # We need to tell them we are in simulation
        self._action = interfaces.action.Action(self) 
        self._vision = interfaces.vision.Vision(self)
        
        # [NEW] Integrate GameController and Communication
        from interfaces.gamecontroller import GameController
        from interfaces.communication import Communication
        self.gamecontroller = GameController(self)
        self.communication = Communication(self)
        
        self.logger.info("[SimCore] Core initialized. Calling user's init()")
        user_entry.init(self)
        
        # State
        self.current_cmd = [0.0, 0.0, 0.0]

    def get_logger(self):
        return self.logger

    def get_clock(self):
        # Mock clock for simple time access
        class MockClock:
            def now(self):
                class MockTime:
                    def to_msg(self):
                        return time.time()
                return MockTime()
        return MockClock()

    def run(self):
        self.logger.info("[SimCore] Starting Loop...")
        tick_period = (1.0 / self.sim_hz) if self.sim_hz > 0 else None
        while True:
            try:
                loop_start = time.perf_counter()
                # 1. User Loop (Think)
                user_entry.loop(self)
                
                # 2. Sync with Sim (Action -> State)
                # Send current_cmd, receive new state
                state = self.client.communicate(self.current_cmd, robot_id=self.id)
                
                if state:
                    # 3. Update Perception
                    sim_state = state.get("state", {})
                    self._vision.update_from_sim_state(sim_state)
                    # [NEW] Update GameController and Communication
                    # Actual ZMQ protocol structure: {"state": {"robots": {...}, "ball": {...}, "gamecontroller": {...}}, ...}
                    self.gamecontroller.update(sim_state.get("gamecontroller", {}))
                    self.communication.update(sim_state.get("communication", {}))
                
                if tick_period is not None:
                    elapsed = time.perf_counter() - loop_start
                    sleep_time = tick_period - elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                self.logger.error(f"[SimCore] Error in loop: {e}")
                traceback.print_exc()
                break
        
        self.stop()

    # --- Action Interface ---
    def cmd_vel(self, vel_x: float, vel_y: float, vel_theta: float) -> None:
        """Set robot velocity (Store for next sync step)."""
        # Apply scaling logic same as Agent.cmd_vel
        vel_x *= self._config.get("max_walk_vel_x", 0.25)
        vel_y *= self._config.get("max_walk_vel_y", 0.1)
        vel_theta *= self._config.get("max_walk_vel_theta", 0.5)
        # ... (Clipping logic omitted for brevity, user can copy if needed, or we refactor logic to utils)
        # Ideally we reuse the clipping logic. 
        # For compatibility, let's just copy the logic or call _action.cmd_vel
        
        # We delegate to _action which handles Sim check
        self._action.cmd_vel(vel_x, vel_y, vel_theta)

    def move_head(self, pitch: float, yaw: float) -> None:
        self._action._move_head(pitch, yaw)

    def stop(self, sleep_time: float = 0) -> None:
        self.current_cmd = [0.0, 0.0, 0.0]
        self._action.cmd_vel(0.0, 0.0, 0.0)
        
        # [NEW] Force send zero command to simulator to ensure stop
        try:
             if hasattr(self, 'client'):
                  # Force clean socket first (in case loop was interrupted in bad state)
                  self.client.force_reset()
                  self.client.communicate(self.current_cmd, robot_id=self.id)
        except Exception as e:
             # Ignore errors during shutdown
             pass
             
        time.sleep(sleep_time)

    def kick(self, foot=0, death=0) -> None:
        self._action.do_kick(foot, death)
    
    def save_ball(self, direction=1):
        self._action.save_ball(direction)

    def angle_normalize(self, angle: float) -> float:
        if angle is None: return None
        angle = angle % (2 * math.pi)
        if angle > math.pi: angle -= 2 * math.pi
        elif angle < -math.pi: angle += 2 * math.pi
        return angle

    # --- Vision Proxies ---
    # Redirect calls to self._vision same as Agent
    def relocate(self, x=0, y=0, theta=0): return self._vision.relocate(x, y, theta)
    def get_objects(self): return self._vision.get_objects()
    def get_self_pos(self): return self._vision.self_pos
    def get_self_yaw(self): return self._vision.self_yaw
    def get_ball_pos(self): return self._vision.get_ball_pos() if self.get_if_ball() else [None, None]
    
    def get_ball_angle(self):
        """Calculate ball angle. NEW: atan2(Left, Forward)"""
        ball_pos = self.get_ball_pos()
        if ball_pos[0] is None: return None
        if ball_pos[0] == 0 and ball_pos[1] == 0: return None
        # atan2(Left, Forward) for X-Forward, Y-Left system
        return math.atan2(ball_pos[1], ball_pos[0])

    def get_ball_pos_in_vis(self): return self._vision.get_ball_pos_in_vis()
    def get_ball_pos_in_map(self): return self._vision.get_ball_pos_in_map()
    def get_if_ball(self): return self._vision.get_if_ball()
    
    def get_if_close_to_ball(self):
        return (self.get_ball_distance() < self._config.get("close_to_ball_threshold", 0.5)) if self.get_if_ball() else False

    def get_ball_distance(self): return self._vision.ball_distance if self.get_if_ball() else 1e6
    def get_neck(self): return self._vision.neck
    def get_head(self): return self._vision.head
    def get_ball_history(self): return self._vision.get_ball_history()
    def get_config(self): return self._config
    
    # --- Command Interface (for state machine compatibility) ---
    def get_command(self):
        """
        Returns the current command dict for state machines.
        In simulation mode, we provide a default structure.
        """
        return {
            "command": getattr(self, '_command', 'stop'),
            "params": getattr(self, '_command_params', {})
        }
    
    def set_command(self, command: str, params: dict = None):
        """Set the current command (for external control)."""
        self._command = command
        self._command_params = params or {}

    def get_obstacle_avoidance_velocity(self):
        """
        Dummy implementation for obstacle avoidance velocity.
        In simulation, we assume no obstacles or handle them differently for now.
        Returns:
            tuple: (vx, vy, vtheta)
        """
        return None, None, None



if __name__ == "__main__":
    """Main entry point."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulation", action="store_true", help="Run in Sim Mode (ZMQ, No ROS)")
    parser.add_argument("--ip", default="127.0.0.1", help="Sim IP")
    parser.add_argument("--port", default="5555", help="Sim Port")
    parser.add_argument("--id", type=int, default=None, help="Robot ID (overrides config)")
    parser.add_argument("--color", type=str, default=None, choices=["red", "blue"], help="Team color (red/blue). If set, --id is interpreted as player number within team.")
    parser.add_argument("--sim-hz", dest="sim_hz", type=float, default=None, help="Simulation control frequency in Hz (<=0 for unlimited)")
    
    # We need to handle known vs unknown args because ROS args might be present if users mistake
    # But since we control the call:
    args, unknown = parser.parse_known_args()

    if args.simulation:
        # Simulation Mode (ROS-free)
        import logging
        agent = SimAgent(args, ip=args.ip, port=args.port, sim_hz=args.sim_hz)
        agent.run()
    else:
        # ROS Mode - check if ROS2 is available
        if not ROS_AVAILABLE:
            print("=" * 60)
            print("[ERROR] ROS2 is not available!")
            print("To run in simulation mode without ROS2, use: --simulation")
            print("To run in ROS2 mode, please install rclpy first.")
            print("=" * 60)
            sys.exit(1)
            
        rclpy.init(args=sys.argv)
        agent = Agent(sys.argv)
    
        def signal_handler(sig, frame):
            if agent:
                print("\n")
                agent.get_logger().warning(f"[Core] Interrupted")
                try:
                    agent.stop()
                    agent.destroy_node()
                    rclpy.shutdown()
                except Exception as e:
                    print(f"======== [Core] Error during stop command in signal handler: {e} ========")
                sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler) 
        rclpy.spin(agent)
