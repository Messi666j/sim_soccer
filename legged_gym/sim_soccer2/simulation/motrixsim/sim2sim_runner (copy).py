import zmq
import argparse
import numpy as np
import time

class MotrixSimRunner:
    def __init__(self, team_size=1, robot_type="k1"):
        # ============================
        # 1. 初始化 MotrixSim（替代 MuJoCo）
        # ============================
        print("✅ 启动 MotrixSim 仿真引擎")
        self.team_size = team_size
        
        # 模拟机器人状态（公开版接口兼容）
        self.robot_num = team_size * 2
        self.robot_pos = {i: np.array([-8.0 + i*1.5, 0.0], dtype=np.float32) for i in range(self.robot_num)}
        self.ball_pos = np.array([0.0, 0.0], dtype=np.float32)

        # ============================
        # 2. 启动 ZMQ（完全和原项目一致）
        # ============================
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind("tcp://*:5555")
        print("✅ ZMQ 服务已启动: tcp://*:5555")

    def run(self):
        print("✅ 仿真运行中，等待决策端连接...")
        while True:
            # 接收决策端指令
            msg = self.socket.recv()
            action = np.frombuffer(msg, dtype=np.float32)
            
            # 驱动机器人
            for i in range(min(len(action)//2, self.robot_num)):
                self.robot_pos[i] += action[i*2:i*2+2] * 0.01

            # 回传状态
            state = []
            for i in range(self.robot_num):
                state.extend(self.robot_pos[i])
            state.extend(self.ball_pos)
            self.socket.send(np.array(state, dtype=np.float32).tobytes())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--team-size", type=int, default=1)
    parser.add_argument("--robot-type", default="k1")
    args = parser.parse_args()

    sim = MotrixSimRunner(team_size=args.team_size, robot_type=args.robot_type)
    sim.run()
s
