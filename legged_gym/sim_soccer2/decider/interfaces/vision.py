# SPDX-FileCopyrightText: Copyright (c) MOS-Brain Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

# interfaces/vision.py
#
#   @description :   The py files to interpret response from perspection
#
#   @interfaces :
#       1. class Vision
#

import math
import time
import numpy as np
from collections import deque

# Conditional ROS2 Import - allows running without ROS2 in simulation mode
ROS_AVAILABLE = False
try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import Float32MultiArray, Header
    from sensor_msgs.msg import JointState, Imu
    from geometry_msgs.msg import Quaternion, Twist, Pose2D, Point
    from nav_msgs.msg import Odometry
    from thmos_msgs.msg import VisionDetections, VisionObj, HeadPose
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
    ROS_AVAILABLE = True
except ImportError:
    # Running without ROS2 - provide fallback types for Vision
    Node = object
    # Placeholder types for type annotations (never used at runtime in sim mode)
    VisionDetections = None
    Pose2D = None

class Vision(Node):
    # @public variants:
    #   VARIANTS        TYPE        DESCTIPTION
    #   self_pos        np.array    self position in map; already filtered
    #   self_yaw        angle       self orientation in degree, [-180, 180)
    #   head            float       the angle of head
    #   neck            float       the angle of neck
    #   ball_distance   float       the distance from robot to tht ball
    #
    # @public methods:
    #   look_at(args)
    #       disable automatically tracking and force to look_at
    #       use (NaN, NaN) to enable tracking
    #                                       
    #
    # @private variants
    #   _ball_pos_in_vis        the pixel coordinate of ball
    #   _ball_pos               the ball position from robot
    #   _ball_pos_in_map        the ball position in the whole map
    #   _ball_pos_accuracy      the 'accuracy' of ball_pos
    #   _self_pos_accuracy      the 'accuracy' of self_pos, the algorithm
    #                           to measure how 'inaccurary' has to be
    #                           improve. 
    #   _force_look_at          a list (head, neck) to force camera orientation
    #   @@@@ TODO IMPROVE THE ALGORITHM TO MEASURE INACCURACY @@@@
    #   _vision_last_frame_time     the timestamp of last frame of vision
    #   _pos_sub                the handler of /pos_in_map
    #   _vision_sub             the handler of /vision/obj_pos
    #   _soccer_real_sub        the handler of /soccer_real_pos_in_map
    #   _head_pub               the handler of /head_goals
    #   _last_track_ball_time    last timestamp running _track_ball()
    #   _track_ball_stage        the stage ( FSMID ) of _track_ball()
    #   _last_track_ball_phase_time      timestamp for changing phase periodicly
    #
    # @private methods
    #   _head_set(args: float[2])           set head and neck angle
    #   _position_callback(msg)             callback of /pos_in_map
    #   _soccer_real_callback(msg)          callback of /soccer_real_pos_in_map
    #   _vision_callback(target_matrix)     callback of /vision/obj_pos
    #   _track_ball()                        the main algorithm to move head
    #   _track_ball_stage_looking_at_ball()  looing at ball algorithm


    def __init__(self, agent):
        self.agent = agent
        
        self.is_simulation = getattr(agent, "is_simulation", False)

        if self.is_simulation:
            self.logger = agent.get_logger()
            self.logger.info("[Core/Vision] Initializing Vision in Sim Mode")
        else:
            self.logger = agent.get_logger().get_child("vision_node")
        
        self._ball_pos_in_vis = np.array([0, 0]) # 
        self._ball_pos_in_vis_D = np.array([0, 0])
        self._ball_pos_in_vis_I = np.array([0, 0])
        self._ball_pos = np.array([0, 0]) # mm
        self._ball_pos_in_map = np.array([0, 0]) # mm
        self._vision_last_frame_time = 0
        self._last_ball_time = 0
        self.self_pos = np.array([0,0])
        self.self_yaw = 0
        self._self_pos_accuracy = 0
        self._ball_pos_accuracy = 0
        self.ball_distance = 6000
        self._search_ball_phase = 0
        self._ball_history = deque(maxlen=20)
        self._detected_objects = list()
        self.head = 0.0
        self.neck = 0.0

        if self.is_simulation:
            # No ROS Subs/Pubs
            self._location_sub = None
            self._vision_sub = None
            self._relocal_pub = None
            return
 
        # Configure QoS profile for subscriptions
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        relc_qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,  # 可靠传输
            history=HistoryPolicy.KEEP_LAST,         # 只保留最新消息
            depth=1                                 # 队列深度为1
        )
        
        # Create subscribers
        self._location_sub = self.agent.create_subscription(
            Pose2D,
            "/THMOS/location",
            self._position_callback,
            qos_profile
        )
        
        self._vision_sub = self.agent.create_subscription(
            VisionDetections,
            "/THMOS/vision/obj_pos",
            self._vision_callback,
            qos_profile
        )

        self._relocal_pub = self.agent.create_publisher(
            Pose2D,
            "/THMOS/relocalization",
            relc_qos_profile
        )

    def update_from_sim_state(self, state: dict):
        """
        Update vision state directly from Sim dictionary.
        
        Expected state format from sim_server:
        {
            "robots": [
                {"name": "robot" or "rp0", "x": float, "y": float, "theta": float, "team": "red"},
                ...
            ],
            "ball": {"x": float, "y": float, "z": float}
        }
        
        Coordinate Systems (NEW):
        1. Velocity: X-Forward (No transform needed if Sim matches).
        2. Relative Obs: X-Forward, Y-Left.
           No transform needed, same as Sim.
        3. Global Map: Origin Center, X-Axis towards Opponent Goal.
           No transform needed: MOS_Map = Sim.
        """
        # Get robot_id and team color from config
        robot_id = self.agent.get_config().get("id", 0)
        team_color = getattr(self.agent, "color", "red")
        
        # 1. Update Robot Pose - parse from 'robots' array
        robots = state.get("robots", [])
        robot = None
        
        # Determine name prefix based on team color
        # Red team: "robot_rp{id}", Blue team: "robot_bp{id}"
        team_prefix = "rp" if team_color == "red" else "bp"
        expected_name = f"robot_{team_prefix}{robot_id}"
        
        # DEBUG: Log available robots
        self.logger.info(f"[Vision] team={team_color}, id={robot_id}, looking for '{expected_name}'")
        self.logger.info(f"[Vision] Available robots: {[r.get('name') for r in robots]}")
        
        # Find our robot by name
        if robots:
            # Strategy 1: Look for exact name match (e.g., "robot_bp0" for blue team id 0)
            for r in robots:
                if r.get("name") == expected_name:
                    robot = r
                    break
            
            # Strategy 2: Fallback - filter by team and use index
            if robot is None:
                team_robots = [r for r in robots if r.get("team") == team_color]
                if robot_id < len(team_robots):
                    robot = team_robots[robot_id]
                    self.logger.info(f"[Vision] Fallback: using team_robots[{robot_id}] = {robot.get('name')}")
            
            # Strategy 3: Look for standard single-robot name "robot"
            if robot is None and len(robots) == 1:
                robot = robots[0]
                self.logger.warning(f"[Vision] Fallback: only 1 robot, using {robot.get('name')}")
        
        self.logger.info(f"[Vision] Selected robot: {robot.get('name') if robot else 'None'}")
        
        if robot:
            # Sim State (Standard X-Goal)
            sim_x = robot.get("x", 0.0)
            sim_y = robot.get("y", 0.0)
            sim_theta = robot.get("theta", 0.0) # radians
            
            # [NEW] Coordinate Mirroring for Blue Team
            # If we are Blue, mirror positions so we "play as if red" from our perspective.
            # Position mirroring: (-x, -y) rotates the field 180°.
            # Theta is also rotated by π for consistent heading.
            team_color = getattr(self.agent, "color", "red")
            if team_color == "blue":
                self.logger.debug(f"[Vision] Mirroring coordinates for Blue Team")
                sim_x = -sim_x
                sim_y = -sim_y
                sim_theta = sim_theta + math.pi  # Also rotate heading by 180°

            # Transform to MOS Map (X-Forward, Y-Left)
            # MOS X is Forward -> Align with Sim X
            # MOS Y is Left -> Align with Sim Y
            self.self_pos = np.array([sim_x, sim_y])
            
            # Yaw transform: No offset needed
            # Sim 0 (facing +X/Goal) -> MOS 0 (facing forward/X+)
            # MOS Yaw: 0=Forward(X+), 90=Left(Y+), -90=Right
            mos_theta_rad = sim_theta
            self.self_yaw = self.agent.angle_normalize(mos_theta_rad) * 180.0 / math.pi
            
            self.logger.debug(f"[Vision] Updated robot pose: pos={self.self_pos}, yaw={self.self_yaw:.2f}deg")
        else:
            self.logger.warning(f"[Vision] No robot found in state for robot_id={robot_id}")
            
        # 2. Update Ball
        ball = state.get("ball", {})
        if ball:
            # Sim World Coords
            bx, by = ball.get("x"), ball.get("y")
            
            # [NEW] Mirror Ball for Blue Team
            team_color = getattr(self.agent, "color", "red")
            if team_color == "blue":
                bx = -bx
                by = -by
            
            # MOS Map Coords (X-Forward, Y-Left)
            mos_bx = bx
            mos_by = by
            self._ball_pos_in_map = np.array([mos_bx, mos_by])

            # Calculate Relative Position
            # NEW: X-Forward, Y-Left coordinate system
            # First, get relative in Global MOS Frame
            dx_global = mos_bx - self.self_pos[0]
            dy_global = mos_by - self.self_pos[1]
            
            # Rotate into Robot Body Frame
            # Robot Body Frame: X-Forward, Y-Left (same as global)
            
            # Current Yaw (MOS frame)
            yaw_rad = self.self_yaw * math.pi / 180.0
            
            # Rotation Matrix for projecting Global to Body
            # Body X (Right) = [cos(theta-90), sin(theta-90)] ? 
            # Easier: Standard Rotation
            # x_local = dx * cos(-theta) - dy * sin(-theta) -> This aligns X with Forward.
            # But we want X to be Right.
            
            # Let's do Standard X-Forward first
            # "Standard" Frame where X is Forward:
            # Angle of vector = atan2(dy, dx) - theta
            vec_angle = math.atan2(dy_global, dx_global) - yaw_rad
            dist = math.sqrt(dx_global**2 + dy_global**2)
            
            forward_comp = dist * math.cos(vec_angle)
            left_comp = dist * math.sin(vec_angle)
            
            # MOS Observation Frame:
            # X+ = Forward (正前方), Y+ = Left (正左方)
            # Output: [Forward, Left]
            rel_x = forward_comp   # Forward
            rel_y = left_comp      # Left
            
            # DEBUG: Log ball relative calculation
            self.logger.debug(f"[BALL_DEBUG] dx={dx_global:.2f}, dy={dy_global:.2f}, yaw={self.self_yaw:.1f}, vec_ang={math.degrees(vec_angle):.1f}, rel=[{rel_x:.2f}, {rel_y:.2f}]")
            
            self._ball_pos = np.array([rel_x, rel_y])
            self.ball_distance = dist
            self._last_ball_time = time.time()
            
            self._detected_objects = [{
                'label': 'ball',
                'relative_pos': self._ball_pos,
                'absolute_pos': self._ball_pos_in_map,
                'distance': self.ball_distance,
                'confidence': 1.0,
                'bounding_box_center': np.array([0,0]), 
                'timestamp': time.time()
            }]
            self._ball_history.append({
                'pos': self._ball_pos,
                'pos_in_map': self._ball_pos_in_map,
                'pos_in_vis': np.array([0, 0]),
                'distance': self.ball_distance,
                'timestamp': self._last_ball_time,
            })


    def _position_callback(self, msg):
        self.self_pos = np.array([msg.x, msg.y])
        self.self_yaw = self.agent.angle_normalize(msg.theta) * 180 / np.pi


    def _soccer_real_callback(self, msg):
        self._ball_pos_in_map = np.array([msg.x, msg.y])

    def _vision_callback(self, msg: VisionDetections):
        """
        处理机器人位置信息并结合视觉检测计算各种物体的绝对坐标
        
        Args:
            msg (VisionDetections): 包含检测到的物体信息的消息
        """
        # 更新视觉数据时间戳
        self._vision_last_frame_time = time.time()
         
        # 用于存储所有检测到的物体信息
        self._detected_objects = []
        self._ball_pos_accuracy = 0  # 默认无球
        
        # 处理所有检测到的物体
        for obj in msg.detected_objects:
            label = obj.label
            
            # 验证边界框是否有效
            if obj.xmin >= obj.xmax or obj.ymin >= obj.ymax:
                self.logger.warning(f"[Core] Invalid bounding box received for {label}")
                continue
                
            # 计算目标中心坐标
            curr_coord = (np.array([obj.xmin, obj.ymin]) + 
                        np.array([obj.xmax, obj.ymax])) * 0.5
            
            # 获取position_projection (x, y) 坐标
            position_projection = np.array(obj.position_projection)
            # 裁剪前两个值
            position_projection = position_projection[:2]
            if position_projection.shape != (2,):
                self.logger.warning(f"[Core] Invalid position_projection format for {label}, expected 2D coordinates")
                continue

            # 如果有nan值，记录错误并跳过
            if np.isnan(position_projection).any():
                self.logger.warning(f"[Core] NaN values found in position_projection for {label}")
                continue

            # 保存相对坐标（单位：米）
            relative_pos = position_projection / 1000
            
            # 计算到物体的距离
            distance = np.linalg.norm(position_projection) / 1000  # 转换为米
            
            # 计算物体在球场中的绝对坐标
            # 创建旋转矩阵 (假设机器人坐标系与全局坐标系的转换)
            # 注意: 这里假设self_yaw是绕z轴的旋转角（符合ROS惯例）
            rotation_matrix = np.array([
                [np.cos(self.self_yaw/180*math.pi), -np.sin(self.self_yaw/180*math.pi)],
                [np.sin(self.self_yaw/180*math.pi), np.cos(self.self_yaw/180*math.pi)]
            ])
            
            # 将相对坐标旋转到全局坐标系
            rotated_relative = rotation_matrix @ relative_pos
            
            # 计算绝对坐标（全局坐标系，单位：米）
            absolute_coord = self.self_pos + rotated_relative
            
            # 保存该物体的计算结果
            object_info = {
                'label': label,
                'relative_pos': relative_pos,
                'absolute_pos': absolute_coord,
                'distance': distance,
                'confidence': obj.confidence,
                'bounding_box_center': curr_coord / 1000,
                'bound_left_low': np.array(obj.bound_left_low[:2]) if obj.bound_left_low else None,
                'bound_right_low': np.array(obj.bound_right_low[:2]) if obj.bound_right_low else None,
                'timestamp': time.time()
            }
            
            self._detected_objects.append(object_info)
             
        # 特别处理球（如果存在）
        ball_objects = [obj for obj in self._detected_objects if obj['label'] == 'ball']
        if ball_objects:
            # 选择置信度最高的球
            best_ball = max(ball_objects, key=lambda b: b['confidence'])
            
            # 更新球相关的变量
            self._ball_pos = best_ball['relative_pos']
            self._ball_pos_in_map = best_ball['absolute_pos']
            self._ball_pos_in_vis = best_ball['bounding_box_center']
            self._ball_pos_accuracy = best_ball['confidence']
            self.ball_distance = best_ball['distance']
            self._last_ball_time = best_ball['timestamp']
            self._ball_history.append({
                'pos': best_ball['relative_pos'],
                'pos_in_map': best_ball['absolute_pos'],
                'pos_in_vis': best_ball['bounding_box_center'],
                'distance': best_ball['distance'],
                'timestamp': best_ball['timestamp']
            })

    def get_objects(self):
        """
        获取检测到的所有物体信息
        
        Returns:
            list: 包含所有检测到物体的字典列表
        """
        return self._detected_objects

    def get_ball_pos(self):
        return self._ball_pos

    def get_ball_pos_in_vis(self):
        return self._ball_pos_in_vis

    def get_ball_pos_in_map(self):
        return self._ball_pos_in_map

    def get_if_ball(self):
        """
        判断是否成功检测到球
        
        Returns:
            bool: True表示成功检测到球，False表示未检测到
        """
        
        # 检查是否有有效的视觉数据
        if np.all(self._ball_pos_in_vis == 0) and not self.is_simulation:
             # In Sim Mode we might not simulate vision pixels yet,
             # so we skip this check or Ensure update_from_sim_state fills it with dummy
             pass
        
        if np.all(self._ball_pos_in_vis == 0) and not self.is_simulation:
            return False
        
        # In Sim Mode, we trust _last_ball_time
        
        # 检查视觉数据的时间戳，确保数据是最近的
        current_time = time.time()
        if current_time - self._last_ball_time > 0.5:
            return False
        
        # 如果所有检查都通过，返回True表示检测到球
        return True

    def get_ball_history(self):
        """
        获取球的历史位置数据
        
        Returns:
            list: 包含球的历史位置和时间戳的字典列表
        """
        return list(self._ball_history)

    def relocate(self, x=0, y=0, theta=0):
        """
        重新定位，重置视觉数据
        """
        if self.is_simulation or not ROS_AVAILABLE:
            # In simulation mode or without ROS2, just log/skip
            return
            
        msg = Pose2D()
        msg.x = float(x)
        msg.y = float(y) # m
        msg.theta = float(theta) # degree
        
        self._relocal_pub.publish(msg)


