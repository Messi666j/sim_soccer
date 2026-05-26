# 兼容公开 motrixsim 的最终版本
import motrixsim as mtx
import numpy as np

class MotrixSoccerSim:
    def __init__(self):
        print("✅ MotrixSim 公开版已初始化")
        self.timestep = 0.01

        # 模拟机器人和球状态（公开版不支持创建物体）
        self.robot_pos = np.array([-8.0, 0.0])
        self.ball_pos = np.array([0.0, 0.0])

    def step(self):
        # 引擎步进
        pass

    def get_state(self):
        return np.concatenate([self.robot_pos, self.ball_pos])

    def send_action(self, action):
        # 模拟运动
        self.robot_pos += np.array(action) * self.timestep

    def reset(self):
        self.robot_pos = np.array([-8.0, 0.0])
        self.ball_pos = np.array([0.0, 0.0])
        print("✅ 仿真已重置")

    def render(self):
        # 公开版无渲染
        pass

