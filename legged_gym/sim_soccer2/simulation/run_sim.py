import sys
import os
# 将 simulation 目录加入Python路径（确保能找到 motrixsim 包）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 直接从 motrixsim 包导入（符合包导入规则）
from motrixsim.soccer_env import MotrixSoccerSim

def main():
    sim = MotrixSoccerSim()
    sim.reset()
    print("✅ MotrixSim 仿真启动成功！")
    
    while True:
        sim.send_action([1.0, 0.0])
        sim.step()
        sim.render()

if __name__ == "__main__":
    main()

