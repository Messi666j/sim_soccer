import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.soccer_env_motrix import SoccerMotrixEnv

def main():
    env = SoccerMotrixEnv()
    obs, _ = env.reset()

    print("=" * 60)
    print(" sim_soccer → MotrixArena 引擎切换成功！")
    print("=" * 60)

    # 新增：打印循环开始提示
    print("开始执行环境交互循环...")
    step_count = 0
    for _ in range(1000):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        step_count += 1
        # 新增：每100步打印进度
        if step_count % 100 == 0:
            print(f"已执行 {step_count} 步，当前reward: {reward}, done: {done}")
        env.render()
        if done:
            print(f"环境终止，重置环境（第{step_count}步）")
            obs, _ = env.reset()  # 注意：原代码reset只写了env.reset()，但新版gym返回(obs, info)

    env.close()
    print("循环执行完毕，环境已关闭")

if __name__ == "__main__":
    main()

