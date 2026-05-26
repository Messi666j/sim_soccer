# SPDX-FileCopyrightText: Copyright (c) MOS-Brain Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

# interfaces/action.py
#
#   @description :   The py files to call actions
#
#   @interfaces :
#       1. class Action
#

import os

# Conditional ROS2 Import - allows running without ROS2 in simulation mode
ROS_AVAILABLE = False
try:
    import rclpy
    from sensor_msgs.msg import JointState
    from geometry_msgs.msg import Quaternion, Twist, Pose2D, Point
    from thmos_msgs.msg import VisionDetections, VisionObj
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
    from std_msgs.msg import Float32MultiArray, Header, Int32
    ROS_AVAILABLE = True
except ImportError:
    # Running without ROS2 - simulation mode only
    pass


class Action:
    # @public variants:
    #   None
    #
    # @public methods;
    #   cmd_vel(vel_x, vel_y, vel_theta) : publish a velocity command
    #   do_kick()                        : publish a kick command
    #
    # @private variants:
    #   _cmd_vel_pub        The publisher for /cmd_vel
    #
    # @private methods
    #   _done_cb()
    #   _feedback_cb()
    #   _action_cb()        Do nothing


    def __init__(self, agent):
        self.agent = agent
        # Check for Sim Mode
        self.is_simulation = getattr(agent, "is_simulation", False)
        
        if self.is_simulation:
            # Sim Mode: Logging to standard logger
            self.logger = agent.get_logger()
            self.logger.info("[Core/Action] Initializing Action in Sim Mode (No ROS)")
            # No publishers created
            self._cmd_vel_pub = None
            self._head_pose_pub = None
            self._kick_pub = None
            self._save_pub = None
        else:
            # ROS Mode
            self.logger = agent.get_logger().get_child("action_node")
            self.logger.info("[Core] Initializing the action module")

            self._cmd_vel_pub = self.agent.create_publisher(Twist, "/THMOS/walk/move", 1)
            self._head_pose_pub = self.agent.create_publisher(
                JointState, "/THMOS/head_control/manual_command", 1
            )
            self._kick_pub = self.agent.create_publisher(Int32, "/THMOS/motion/kick", 1)
            self._save_pub = self.agent.create_publisher(Int32, "/THMOS/motion/save", 1)

    def _move_head(self, pitch, yaw):
        if self.is_simulation:
             # TODO: Send head command to Sim via Agent state?
             # For now just log or ignore
             return

        head_pose_msg = JointState()
        head_pose_msg.header = Header()
        head_pose_msg.header.stamp = self.agent.get_clock().now().to_msg()
        head_pose_msg.position = [0.0, 0.0]
        head_pose_msg.position[0] = float(yaw)  # Set yaw
        head_pose_msg.position[1] = float(pitch)

        self._head_pose_pub.publish(head_pose_msg)

    # cmd_vel(vel_x, vel_y, vel_theta)
    #   publish a velocity command
    def cmd_vel(self, vel_x, vel_y, vel_theta):
        # convert to float
        vel_x = float(vel_x)
        vel_y = float(vel_y)
        vel_theta = float(vel_theta)

        if self.is_simulation:
             # Update Agent's current command state
             self.agent.current_cmd = [vel_x, vel_y, vel_theta]
             return

        cmd = Twist()
        cmd.linear.x = vel_x
        cmd.linear.y = vel_y
        cmd.angular.z = vel_theta
        self._cmd_vel_pub.publish(cmd)

    # kick()
    #   publish a kick command
    def do_kick(self, foot=0, death=0):
        """
        Args:
            foot: 0 for right, 1 for left
            death: 0 for normal kick, 1 for death kick
        """
        if self.is_simulation:
            self.logger.info(f"[Core/Action] Sim Kick: foot={foot}, death={death}")
            # TODO: Implement Kick in Sim
            return

        table = [["r", "l"], ["r_death", "l_death"]]
        try:
            with open("/tmp/THMOS_kick", "w") as fd:
                fd.write(table[death][foot])
                fd.flush()
            self.logger.info("[Core] Pipe kick command published")
        except Exception as e:
            kick_msg = Int32()
            kick_msg.data = foot + death * 10
            self._kick_pub.publish(kick_msg)
            self.logger.warning("[Core] Fallback to ros2 kick command published")

    def save_ball(self, direction):
        if self.is_simulation:
            self.logger.info(f"[Core/Action] Sim Save: dir={direction}")
            return

        save_msg = Int32()
        save_msg.data = direction
        self._save_pub.publish(save_msg)
        self.logger.info("[Core] Save command published")

