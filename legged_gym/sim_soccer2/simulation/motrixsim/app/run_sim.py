# simulation/motrixsim/app/run_sim.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import socket
import numpy as np
from motrixsim.soccer_env import MotrixSoccerSim

def run_server():
    # 1. 启动 MotrixSim 引擎（替换 MuJoCo）
    sim = MotrixSoccerSim()
    sim.reset()

    # 2. 原样开启 Socket（和原项目通信完全一致）
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind(('127.0.0.1', 12345))
    server_socket.listen(1)
    print("✅ MotrixSim 仿真服务已启动，等待决策器连接...")

    conn, addr = server_socket.accept()
    print("✅ 决策器已连接")

    # 3. 主循环（和原项目一模一样）
    while True:
        try:
            # 接收动作
            action = conn.recv(1024)
            if not action:
                break

            action = np.frombuffer(action, dtype=np.float32)
            sim.send_action(action)
            sim.step()

            # 回传状态
            state = sim.get_state()
            conn.sendall(state.tobytes())
        except:
            break

    conn.close()
    server_socket.close()

if __name__ == "__main__":
    run_server()

