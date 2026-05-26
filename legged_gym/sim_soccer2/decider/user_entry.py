# user_entry.py
#
#   @description:   Logic entry point for mos-brain decider (Simplified)
#                   Game logic is separated into game() function.
#

import time
import traceback
import sys
import os
import math
import numpy as np

# Ensure we can find the logic package
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
if CUR_DIR not in sys.path:
    sys.path.append(CUR_DIR)

from logic.sub_statemachines import chase_ball, find_ball, go_back_to_field, dribble
from logic.policy_statemachines import goalkeeper

import csv
from datetime import datetime

class DataRecorder:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filepath = os.path.join(log_dir, f"dribble_debug_{timestamp}.csv")
        self.file = open(self.filepath, 'w', newline='')
        self.writer = csv.writer(self.file)
        self.header_written = False
        
    def log(self, data):
        if not self.header_written:
            self.writer.writerow(data.keys())
            self.header_written = True
        self.writer.writerow(data.values())
        self.file.flush()

    def close(self):
        self.file.close()

class AdvancedDribbler:
    def __init__(self, agent):
        self.agent = agent
        self.logger = agent.get_logger().get_child("AdvDribble")
        
        # Instrumentation
        # Use project-relative debug_logs directory (parent of `decider`)
        log_dir = os.path.abspath(os.path.join(CUR_DIR, '..', 'debug_logs'))
        print(f"[DEBUG] DataRecorder log_dir: {log_dir}")
        self.recorder = DataRecorder(log_dir)
        
        # Parameters
        self.bturn_p = 2.0
        self.side_correction_p = 2.5
        self.forward_p = 1.0
        
        self.setup_dist = 0.40
        self.dribble_dist = 0.20 # Ball should be slightly in front
        self.max_fw_vel = 0.8
        
        self.field_length = 14.0 # Default M
        self.field_width = 9.0
        
        # Anti-Oscillation
        self.spread_factor_max = 20.0 # degrees
        self.spread_factor_min = 5.0 # degrees

        # Hysteresis to avoid mode chattering near b_x threshold
        self.turn_to_ball_enter_bx = 0.03
        self.turn_to_ball_exit_bx = 0.08
        self.turn_to_ball_mode = False
        self.direction_consistency_bx = 0.12  # keep turn direction consistent near mode boundary
        
    def get_target_vector(self):
        """
        Calculate dribbling direction (Vector Field)
        COORDINATE SYSTEM (NEW):
        - Origin: Center of field
        - X+: Points to Opponent Goal (Front)
        - Y+: Points to Left side
        - Field Length: X-axis
        - Field Width: Y-axis
        """
        # 1. Goal Attraction
        # Goal is at (L/2, 0) - forward on X, center on Y
        goal_x = self.field_length / 2.0
        goal_y = 0.0
        
        my_pos = self.agent.get_self_pos()
        if my_pos is None:
            return np.array([1.0, 0.0]), 0 # Default forward (X+)
            
        # Global Vector to Goal
        g_dx = goal_x - my_pos[0]
        g_dy = goal_y - my_pos[1]
        
        # 2. Boundary Repulsion (Side Lines are at Y = +/- W/2)
        # If too close to side lines, push towards center (Y=0)
        dist_to_left = (self.field_width / 2.0) - my_pos[1]   # Y+ is left
        dist_to_right = my_pos[1] - (-self.field_width / 2.0)  # Y- is right
        
        repulsion_y = 0.0
        margin = 1.0 # Buffer
        
        # If close to Left Boundary (Y > 0), push Right (Y-)
        if dist_to_left < margin:
            repulsion_y -= (margin - dist_to_left) * 2.0
            
        # If close to Right Boundary (Y < 0), push Left (Y+)
        if dist_to_right < margin:
            repulsion_y += (margin - dist_to_right) * 2.0
            
        # Combine
        final_dx = g_dx  # Goal attraction provides X component
        final_dy = g_dy + repulsion_y
        
        norm = math.hypot(final_dx, final_dy)
        if norm < 0.001:
            return np.array([1.0, 0.0]), 0 # Default forward (X+)
            
        target_vec = np.array([final_dx / norm, final_dy / norm])
        
        # Zone Safety check for Deadband
        # If central area (Y close to 0), safe.
        is_safe = (abs(my_pos[1]) < self.field_width / 3.0)
        
        return target_vec, is_safe

    def run(self):
        if not self.agent.get_if_ball():
            self.logger.info("Lost ball, stopping.")
            self.agent.cmd_vel(0,0,0)
            return

        # 1. Get State
        # Ball in Robot Body Frame [Forward, Left] (X+前, Y+左)
        ball_pos = self.agent.get_ball_pos()
        b_x = ball_pos[0]  # Forward
        b_y = ball_pos[1]  # Left
        
        my_pos = self.agent.get_self_pos()
        my_yaw = self.agent.get_self_yaw()
        
        # [NEW] Ball Behind Check - Turn to face ball first
        # If ball is behind robot (b_x < threshold), we need to turn around
        ball_dist = math.hypot(b_x, b_y)
        ball_angle_to_robot = math.atan2(b_y, b_x)  # Angle from robot forward to ball
        ball_angle_deg = math.degrees(ball_angle_to_robot)
        self.logger.info(
            f"[AdvDribble] ball_angle_to_robot={ball_angle_to_robot:.4f}rad ({ball_angle_deg:.1f}deg), b=({b_x:.3f}, {b_y:.3f})"
        )
        
        prev_turn_to_ball_mode = self.turn_to_ball_mode
        if self.turn_to_ball_mode:
            self.turn_to_ball_mode = b_x < self.turn_to_ball_exit_bx
        else:
            self.turn_to_ball_mode = b_x < self.turn_to_ball_enter_bx

        if self.turn_to_ball_mode != prev_turn_to_ball_mode:
            self.logger.info(
                f"[AdvDribble] MODE_SWITCH turn_to_ball={self.turn_to_ball_mode} b_x={b_x:.3f} (enter<{self.turn_to_ball_enter_bx:.2f}, exit<{self.turn_to_ball_exit_bx:.2f})"
            )

        if self.turn_to_ball_mode:  # Ball behind mode (with hysteresis)
            # Turn towards the ball instead of dribbling
            turn_speed = 1.5 * ball_angle_to_robot  # P-control to face ball
            turn_speed = max(min(turn_speed, 1.5), -1.5)  # Clamp
            
            # Also move slightly towards ball if it's far
            approach_speed = 0.0
            if ball_dist > 0.3:
                # Move forward/backward based on ball position
                # If ball is behind, we should approach after turning
                approach_speed = 0.3 * b_x / (ball_dist + 0.01)  # Will be negative if ball behind
                approach_speed = max(min(approach_speed, 0.5), -0.3)
            
            self.logger.info(f"[AdvDribble] TURN_TO_BALL: BallAngle={ball_angle_deg:.1f} TurnSpd={turn_speed:.2f}")
            self.agent.cmd_vel(approach_speed, 0, turn_speed)
            self.agent.move_head(math.inf, math.inf)
            
            # Log for debugging
            log_data = {
                "time": time.time(),
                "rx": my_pos[0] if my_pos is not None else 0,
                "ry": my_pos[1] if my_pos is not None else 0,
                "ryaw": my_yaw,
                "ball_x": b_x,
                "ball_y": b_y,
                "t_vec_gx": 0, "t_vec_gy": 0,
                "t_ang_local": ball_angle_deg,
                "cmd_x": approach_speed, "cmd_y": 0, "cmd_w": turn_speed,
                "aligned": 0, "safe_zone": 0,
                "err_y": 0, "err_x": 0
            }
            self.recorder.log(log_data)
            return  # Exit early, don't do normal dribble logic
        
        # Target Vector
        target_vec_global, is_safe_zone = self.get_target_vector()
        
        # [DEBUG] Log coordinate values for diagnosis
        self.logger.info(f"[COORD] pos={my_pos}, yaw={my_yaw:.1f}, t_vec={target_vec_global}")
        
        # Rotate Target to Local Frame
        yaw_rad = math.radians(my_yaw)
        # NEW Coordinate System: 
        # Global: X(Forward), Y(Left)
        # Body: X(Forward), Y(Left)
        # Robot Yaw=0 means facing X+ (Forward)
        # Rotation: v_body = R(-yaw) * v_global
        # t_local_x = Gx * cos(yaw) + Gy * sin(yaw)
        # t_local_y = -Gx * sin(yaw) + Gy * cos(yaw)
        
        t_local_x = target_vec_global[0] * math.cos(yaw_rad) + target_vec_global[1] * math.sin(yaw_rad)
        t_local_y = -target_vec_global[0] * math.sin(yaw_rad) + target_vec_global[1] * math.cos(yaw_rad)
        
        target_angle_local = math.atan2(t_local_y, t_local_x)
        target_angle_deg = math.degrees(target_angle_local)
        
        # 2. Omnidirectional Control
        
        # A. Turn (da)
        # Minimize angle to target
        da = -self.bturn_p * target_angle_local  # invert sign to align with TURN_TO_BALL rotation direction
        
        # [NEW] Dampen Turn when very close to ball to prevent oscillation ("Large angle change" issue)
        # If b_x is small (e.g. 0.1), simple turning changes relative geometry fast.
        # Scale down da when close.
        if my_pos is not None: # Ensure we have data
             # Use b_x from previous step or estimate? 
             # Actually we calculated b_x_virt later. 
             # Let's move da clamping/scaling to AFTER b_x_virt calculation or estimate it here.
             # Better: Apply scaling at the end of this block or simply use dist to ball.
             dist_to_ball = math.hypot(b_x, b_y)
             turn_damp = max(0.4, min(1.0, dist_to_ball / 0.4))
             da *= turn_damp
        
        # B. Orbit/Sway (dy)
        # We want the ball to be on the "line" to target.
        # Project Ball Pos onto the Normal of Target Vector?
        # Simpler: Rotate Ball Pos so that Target Vector lies on X-axis (Virtual Frame)
        # In Virtual Frame:
        # Ball Y should be 0.
        # Ball X should be setup_dist.
        
        # Rotation from Robot Body to Virtual Target Frame
        # rot = -target_angle_local
        c_r = math.cos(-target_angle_local)
        s_r = math.sin(-target_angle_local)
        
        b_x_virt = b_x * c_r - b_y * s_r
        b_y_virt = b_x * s_r + b_y * c_r
        
        # PID Controls in Virtual Frame
        # Lateral Error: We want b_y_virt = 0
        err_y = b_y_virt
        cmd_y_virt = self.side_correction_p * err_y 
        
        # Forward Error: We want b_x_virt = setup_dist (or dribble_dist if aligned)
        # Adaptive Deadband
        deadband = self.spread_factor_max if is_safe_zone else self.spread_factor_min
        # [NEW] Require Ball to be In Front (b_x_virt > 0) to consider Aligned
        aligned = abs(target_angle_deg) < deadband and abs(err_y) < 0.1 and b_x_virt > 0.1
        
        # [NEW] 临门一脚 Mode: Near goal, be more aggressive
        # Goal is at X = field_length/2, so threshold is ~80% of the way
        near_goal = my_pos is not None and my_pos[0] > self.field_length * 0.35
        if near_goal:
            # Relax alignment requirement
            aligned = abs(target_angle_deg) < 25.0 and b_x_virt > 0.05
        
        # [NEW] Interpolate Target Distance for smooth transition (Creep)
        target_dist = self.setup_dist
        if aligned:
            target_dist = self.dribble_dist
        elif abs(target_angle_deg) < 25.0:
             # Interpolate between setup_dist (0.4) and dribble_dist (0.2) based on alignment
             # 25 deg -> 0.4, 5 deg -> 0.2
             ratio = (25.0 - abs(target_angle_deg)) / (25.0 - 5.0)
             ratio = max(0.0, min(1.0, ratio))
             target_dist = self.setup_dist - ratio * (self.setup_dist - self.dribble_dist)

        err_x = b_x_virt - target_dist
        
        # Velocity Damping (Anti-Oscillation)
        # Don't rush if not aligned
        forward_factor = 1.0
        if not aligned:
            # Dampen based on angle error but allow "creep" if error is not huge
            angle_err = abs(target_angle_deg)
            if angle_err < 25.0:
                # Creep zone: Interpolate 1.0 -> 0.2
                forward_factor = 1.0 - (angle_err / 25.0) * 0.5 
            else:
                forward_factor = 0.0
        
        # [NEW] Near goal, always push forward aggressively
        if near_goal and b_x_virt > 0.05:
            forward_factor = max(forward_factor, 0.8)  # At least 80% power near goal
            
        cmd_x_virt = self.forward_p * err_x * forward_factor
        
        # [NEW] Minimum Push Velocity when Aligned
        # Always maintain minimum forward velocity when aligned (PUSH mode)
        if aligned:
             if cmd_x_virt < 0.5:  # Always push forward at min 0.5
                 cmd_x_virt = 0.5
        
        # [NEW] Near goal, even if not aligned, keep pushing forward
        if near_goal and b_x_virt > 0.1:
            if cmd_x_virt < 0.4:  # Minimum near-goal push
                cmd_x_virt = 0.4
        
        
        # [NEW] Clamp Virtual Velocities individually first
        cmd_x_virt = max(min(cmd_x_virt, self.max_fw_vel), -0.5)
        # cmd_y_virt is proportional to error, clamping it is key
        # Using a slightly higher limit for lateral correction if needed, but safe to clamp to max velocity
        cmd_y_virt = max(min(cmd_y_virt, self.max_fw_vel), -self.max_fw_vel)
        
        # Transform Commands back to Body Frame
        c = math.cos(target_angle_local)
        s = math.sin(target_angle_local)
        
        cmd_x = cmd_x_virt * c - cmd_y_virt * s
        cmd_y = cmd_x_virt * s + cmd_y_virt * c
        
        # [NEW] Global Clamp on Linear Velocity
        lin_vel_norm = math.hypot(cmd_x, cmd_y)
        if lin_vel_norm > self.max_fw_vel:
            scale = self.max_fw_vel / lin_vel_norm
            cmd_x *= scale
            cmd_y *= scale
            
        # [NEW] Clamp Angular Velocity
        da = max(min(da, 1.5), -1.5)

        # Keep angular direction consistent with TURN_TO_BALL near threshold
        # to avoid opposite commands around mode boundary.
        if b_x < self.direction_consistency_bx and abs(ball_angle_to_robot) > 0.2 and abs(da) > 1e-6:
            da = math.copysign(abs(da), ball_angle_to_robot)
        
        # LOGGING
        log_data = {
            "time": time.time(),
            "rx": my_pos[0] if my_pos is not None else 0,
            "ry": my_pos[1] if my_pos is not None else 0,
            "ryaw": my_yaw,
            "ball_x": b_x,
            "ball_y": b_y,
            "t_vec_gx": target_vec_global[0],
            "t_vec_gy": target_vec_global[1],
            "t_ang_local": target_angle_deg,
            "cmd_x": cmd_x,
            "cmd_y": cmd_y,
            "cmd_w": da,
            "aligned": int(aligned),
            "safe_zone": int(is_safe_zone),
            "err_y": err_y,
            "err_x": err_x
        }
        self.recorder.log(log_data)
        
        # 3. Final Command
        self.logger.info(f"[AdvDribble] Safe:{is_safe_zone} Alg:{aligned} T_Ang:{target_angle_deg:.1f} Cmd:({cmd_x:.2f}, {cmd_y:.2f}, {da:.2f})")
        self.agent.cmd_vel(cmd_x, cmd_y, da)
        self.agent.move_head(math.inf, math.inf)

def init(agent) -> None:
    agent.get_logger().info("[UserEntry] Initializing Logic...")
    
    # Initialize State Machines
    agent.chase_ball_machine = chase_ball.ChaseBallStateMachine(agent)
    agent.find_ball_machine = find_ball.FindBallStateMachine(agent)
    agent.go_back_machine = go_back_to_field.GoBackToFieldStateMachine(agent)
    agent.dribble_machine = dribble.DribbleStateMachine(agent)
    agent.goalkeeper_machine = goalkeeper.GoalkeeperStateMachine(agent)
    
    # Initialize Advanced Dribbler
    agent.adv_dribbler = AdvancedDribbler(agent)

    agent.state_machine_runners = {
        "chase_ball": agent.chase_ball_machine.run,
        "find_ball": agent.find_ball_machine.run,
        "go_back_to_field": agent.go_back_machine.run,
        "dribble": agent.dribble_machine.run,
        "adv_dribble": agent.adv_dribbler.run, # Register new runner
        "stop": agent.stop,
        "goalkeeper": agent.goalkeeper_machine.run,
    }
    
    # Basic Configs
    agent.default_chase_distance = agent.get_config().get("chase",{}).get("default_chase_distance", 0.7)
    
    # Relocalize
    agent.relocate()

def loop(agent) -> None:
    try:
        game(agent)
    except Exception as e:
        agent.get_logger().error(f"Error in user_entry loop: {e}")
        traceback.print_exc()

def game(agent) -> None:
    # # --- Debug Coordinates ---
    # from debug_coords import debug_coords
    # debug_coords(agent)
    
    # --- Select Test to Run ---
    # _playing_logic(agent)        # Default: Full Playing Logic
    # _test_adv_dribble(agent)     # TEST ARGUMENT: Using Advanced Dribble
    # _playing_logic(agent)
    _gc_test_go_back_to_field(agent)

def _gc_test_go_back_to_field(agent):
    """
    GameController test using go_back_to_field.
    """
    gc = agent.gamecontroller
    state = gc.game_state
    
    logger = agent.get_logger()

    
    if state == "STATE_INITIAL" or state == "STATE_READY":
        agent.state_machine_runners['go_back_to_field'](aim_x = 1.3, aim_y = 0.001, aim_yaw = 150.)
        return

    if state == "STATE_SET":
        logger.info("[GC_TEST] Action: STATE_SET -> stop")
        agent.stop()
        return

    elif state in ("STATE_FINISHED", "STATE_STANDBY"):
        logger.info(f"[GC_TEST] Action: {state} -> stop")
        agent.stop()
        return

    if state == "STATE_PLAYING":
        logger.info("[GC_TEST] Action: STATE_PLAYING -> active behavior")
        
        # 1. Look for ball
        if not agent.get_if_ball():
            logger.info("[GC_TEST] -> find_ball (ball not detected)")
            agent.state_machine_runners['find_ball']()
            return

        # 2. Chase Ball
        ball_dist = agent.get_ball_distance()
        if ball_dist > agent.default_chase_distance:
            logger.info(f"[GC_TEST] -> chase_ball (distance={ball_dist:.2f} > {agent.default_chase_distance:.2f})")
            agent.state_machine_runners['chase_ball']()
            return
    
        # 3. Ball Interaction (Close enough) -> NEW Dribble
        logger.info(f"[GC_TEST] -> adv_dribble (distance={ball_dist:.2f} <= {agent.default_chase_distance:.2f})")
        agent.state_machine_runners['adv_dribble']()
        return
    
    logger.warning(f"[GC_TEST] Unknown state: {state} -> stop")
    agent.stop()  

def _test_agents(agent):
    """
    测试各状态机
    """
    # 3调用kick
    agent.state_machine_runners['find_ball']()

    
def _playing_logic(agent):
    """
    Simplified playing logic without GameController.
    """
    # 1. Look for ball
    if not agent.get_if_ball():
        agent.state_machine_runners['find_ball']()
        return

    # 2. Chase Ball
    if agent.get_ball_distance() > agent.default_chase_distance:
        agent.state_machine_runners['chase_ball']()
        return
    
    # 3. Ball Interaction (Close enough) -> NEW Dribble
    agent.state_machine_runners['adv_dribble']()

def _test_adv_dribble(agent) -> None:
    if not agent.get_if_ball():
        agent.state_machine_runners['find_ball']()
    else:
        agent.state_machine_runners['adv_dribble']()

def _test_dribble(agent) -> None:
    if not agent.get_if_ball():
        agent.state_machine_runners['find_ball']()
    else:
        agent.state_machine_runners['dribble'](aim_yaw=0)

def _test_find_ball(agent) -> None:
    agent.state_machine_runners['find_ball']()

def _test_chase_ball(agent) -> None:
    if not agent.get_if_ball():
        agent.state_machine_runners['find_ball']()
    else:
        agent.state_machine_runners['chase_ball']()
