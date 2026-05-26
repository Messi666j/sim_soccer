import math
import numpy as np
import matplotlib.pyplot as plt
import os
from collections import deque

class DribbleStateMachine:
    """
    Continuous P-Control Dribbling Strategy
    Replaces discrete State Machine with continuous control loop.
    """

    def __init__(self, agent):
        """
        Initialize the P-Control Dribbler
        :param agent: Robot Agent Object
        """
        self.agent = agent
        self.logger = self.agent.get_logger().get_child("dribble_p_control")
        self._config = self.agent.get_config()
        self.read_params()
        self.logger.info("[DRIBBLE P-CONTROL] Initialized.")

    def read_params(self):
        """Read P-Control parameters from config"""
        dribble_cfg = self._config.get("dribble", {})
        
        # P Gains
        self.kp_x = dribble_cfg.get("kp_x", 1.5)
        self.kp_y = dribble_cfg.get("kp_y", 1.5)
        self.kp_theta = dribble_cfg.get("kp_theta", 1.0)
        
        # Distance Thresholds
        self.dist_setup = dribble_cfg.get("dist_setup", 0.45)        # Positioning distance
        self.dist_dribble = dribble_cfg.get("dist_dribble", -0.1)    # Pushing distance (negative = through ball)
        
        # Logic Thresholds
        self.align_threshold_deg = dribble_cfg.get("align_threshold_deg", 15.0)
        self.align_threshold_rad = math.radians(self.align_threshold_deg)
        
        # Limits
        self.max_vel_linear = dribble_cfg.get("max_vel_linear", 1.0)
        self.min_push_vel = dribble_cfg.get("min_push_vel", 0.5)  # Minimum forward velocity in PUSH mode
        self.orbit_efficiency = dribble_cfg.get("orbit_efficiency", 0.8) # Preference for lateral move when misaligned
        self.stop_threshold_dist = 0.05 # Deadband
        
        # Data Recording
        self.history_len = 500
        self.history = {
            "time": deque(maxlen=self.history_len),
            "e_x": deque(maxlen=self.history_len),
            "e_y": deque(maxlen=self.history_len),
            "e_th": deque(maxlen=self.history_len),
            "v_x": deque(maxlen=self.history_len),
            "v_y": deque(maxlen=self.history_len),
            "v_th": deque(maxlen=self.history_len),
            "mode": deque(maxlen=self.history_len)
        }
        self.plot_interval = 10  # Plot every 10 steps for faster debugging
        self.step_count = 0
        self.plot_save_path = os.path.join(os.path.dirname(__file__), "../../logs/dribble_debug.png")
        os.makedirs(os.path.dirname(self.plot_save_path), exist_ok=True)

    def run(self, aim_yaw=None, _kick_when_dribble=False):
        """
        Main Control Loop
        Calculates and sends velocity commands to position robot behind ball and push it.
        """
        # 1. Check Ball Visibility
        if not self.agent.get_if_ball():
            self.logger.warning("[DRIBBLE] No ball seen. Stopping.")
            self.agent.cmd_vel(0, 0, 0)
            return

        # 2. Get Inputs
        # - Ball position in Vision Frame (X=Forward, Y=Left) - NEW coordinate system
        ball_pos_vis = self.agent.get_ball_pos() 
        b_forward, b_left = ball_pos_vis[0], ball_pos_vis[1]
        
        # - Control Frame is same as Vision Frame (X=Forward, Y=Left)
        b_x = b_forward        # Forward
        b_y = b_left           # Left
        
        # - Global Yaws
        curr_yaw = self.agent.get_self_yaw()
        
        # - Determine Target Yaw
        target_yaw = self._get_target_yaw(aim_yaw)
        
        # 3. Calculate Orientation Error (Alpha)
        # alpha = target_yaw - curr_yaw (Normalized to [-pi, pi])
        alpha = self.agent.angle_normalize(target_yaw - curr_yaw)
        
        self.logger.debug(f"[DRIBBLE] Ball: ({b_x:.2f}, {b_y:.2f}), Alpha: {math.degrees(alpha):.1f} deg")

        # 4. Determine Desired Dribble Distance (Dynamic D)
        # If aligned, push the ball (aggressive). If not, stay back and orbit (safe).
        if abs(alpha) < self.align_threshold_rad:
            target_dist = self.dist_dribble
            mode = "PUSH"
        else:
            target_dist = self.dist_setup
            mode = "SETUP"
            
        # 5. Calculate Target Position in Robot Frame
        # We want to be at 'target_dist' behind the ball relative to vector pointing to target_yaw.
        # Vector pointing to target in Robot Frame:
        # v_target_local = [cos(alpha), sin(alpha)]
        #
        # Desired Robot Position (Logic: We want Ball to be at +target_dist along this vector)
        # P_des = Ball_Pos - target_dist * v_target_local
        
        vec_target_local_x = math.cos(alpha)
        vec_target_local_y = math.sin(alpha)
        
        p_des_x = b_x - target_dist * vec_target_local_x
        p_des_y = b_y - target_dist * vec_target_local_y
        
        # Error Vector (Target - Current(0,0))
        e_x = p_des_x
        e_y = p_des_y
        e_theta = alpha
        
        self.logger.debug(f"[DRIBBLE] Mode: {mode}, T_Dist: {target_dist}, Errors: x={e_x:.2f}, y={e_y:.2f}, th={e_theta:.2f}")

        v_x = self.kp_x * e_x
        v_y = self.kp_y * e_y
        v_theta = self.kp_theta * e_theta

        # 7.5. Minimum Velocity Clamp (Aggressive Push)
        # If in PUSH mode, ALWAYS maintain minimum forward velocity
        # This prevents oscillation when robot is close to target position
        if mode == "PUSH":
            # Always push forward at minimum velocity in PUSH mode
            if v_x < self.min_push_vel:
                v_x = self.min_push_vel

        # 7. Collision Avoidance / Orbit Logic (Refined)
        # If we are in SETUP mode (large angle error), we should prioritize rotating/orbiting 
        # over "closing the distance" (v_x) to avoid hitting the ball or spiraling.
        if mode == "SETUP":
            # Dampen forward velocity based on misalignment
            # If Alpha is 90 deg, we basically want 0 forward velocity relative to target frame logic,
            # but e_x/e_y are in Robot Frame.
            # Ideally: "Don't drive INTO the ball if not aligned".
            # The ball is at (b_x, b_y). Check if velocity vector points at ball?
            pass 
            
            # Simple Heuristic: Limit v_x (Forward) when misalignment is high
            # We want to maintain 'dist_setup'.
            # If we are further than dist_setup, we can approach.
            # If we are closer, we must back off.
            # But mostly we want to move laterally (v_y).
            
            # Dampen v_x if |alpha| is large to encourage orbiting
            damp_factor = max(0.0, 1.0 - (abs(alpha) / math.pi)) # 1.0 at 0 deg, 0.0 at 180 deg
            v_x *= damp_factor
            
            # Boost v_y (Orbit) if needed? P-control handles it if P_des is correct.
            # But P_des might be "inside" the ball radius if we just rotate frame.
            # Let's trust P_des for y, but dampen x.

        # 8. Velocity Clamping (Vector Normalization)
        v_linear_mag = math.hypot(v_x, v_y)
        if v_linear_mag > self.max_vel_linear:
            scale = self.max_vel_linear / v_linear_mag
            v_x *= scale
            v_y *= scale
            
        # Detailed Logging
        log_msg = (f"[DRIBBLE] Mode: {mode[0]} | "
                   f"Alph: {math.degrees(alpha):.0f} | "
                   f"T_Yaw: {math.degrees(target_yaw):.0f} | "
                   f"Bal: ({b_x:.2f},{b_y:.2f}) | "
                   f"Cmd: {v_x:.2f}, {v_y:.2f}, {v_theta:.2f}")
        self.logger.info(log_msg)

        # 9. Send Command
        self.agent.cmd_vel(v_x, v_y, v_theta)
        self.agent.move_head(math.inf, math.inf) # Look at ball/forward
        
        # 10. Record & Plot
        self._record_data(e_x, e_y, e_theta, v_x, v_y, v_theta, mode)

    def _record_data(self, ex, ey, eth, vx, vy, vth, mode):
        self.step_count += 1
        self.history["time"].append(self.step_count)
        self.history["e_x"].append(ex)
        self.history["e_y"].append(ey)
        self.history["e_th"].append(math.degrees(eth))
        self.history["v_x"].append(vx)
        self.history["v_y"].append(vy)
        self.history["v_th"].append(vth)
        self.history["mode"].append(1 if mode=="PUSH" else 0)
        
        if self.step_count % self.plot_interval == 0:
            self._save_plot()
            
    def _save_plot(self):
        try:
            plt.figure(figsize=(10, 8))
            
            # Subplot 1: Errors
            plt.subplot(3, 1, 1)
            plt.plot(self.history["time"], self.history["e_x"], label="Err X (m)")
            plt.plot(self.history["time"], self.history["e_y"], label="Err Y (m)")
            plt.title("Position Errors (Control Frame)")
            plt.grid(True)
            plt.legend()
            
            # Subplot 2: Angular Error
            plt.subplot(3, 1, 2)
            plt.plot(self.history["time"], self.history["e_th"], label="Err Theta (deg)", color='orange')
            plt.plot(self.history["time"], self.history["mode"], label="Mode (1=Push)", color='green', alpha=0.3)
            plt.title("Angular Error & Mode")
            plt.grid(True)
            plt.legend()
            
            # Subplot 3: Commands
            plt.subplot(3, 1, 3)
            plt.plot(self.history["time"], self.history["v_x"], label="Cmd Vx")
            plt.plot(self.history["time"], self.history["v_y"], label="Cmd Vy")
            plt.title("Velocity Commands")
            plt.grid(True)
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(self.plot_save_path)
            plt.close()
        except Exception as e:
            self.logger.warning(f"Plotting failed: {e}")

    def _get_target_yaw(self, override_yaw=None):
        """Determine target yaw from command or heuristic"""
        # 1. High priority: Override arg
        if override_yaw is not None:
            return override_yaw
            
        # 2. Medium priority: External Command
        cmd_yaw = self.agent.get_command().get('data', {}).get('aim_yaw', None)
        if cmd_yaw is not None:
            return cmd_yaw
            
        # 3. Low priority: Goal Center
        # Stabilization: Use last valid or default 0.0
        # If get_self_pos fails, it might fluctuate.
        
        try:
            # Reverted to using self.agent.league as SimAgent now exposes it
            field_len = self._config.get("field_size", {}).get(self.agent.league, [14.0, 9.0])[0]
            
            goal_x = field_len / 2.0
            my_pos = self.agent.get_self_pos() # [x, y] or None or [None, None]
            
            # Check validity
            if my_pos is not None and len(my_pos) == 2 and my_pos[0] is not None:
                # NEW: X-Forward coordinate system
                # Goal is at (L/2, 0), angle from Robot to Goal
                goal_angle = math.atan2(0 - my_pos[1], goal_x - my_pos[0])
                return goal_angle
        except Exception as e:
            self.logger.warning(f"Error calcing target yaw: {e}")
            pass
            
        return 0.0 # Default forward direction
