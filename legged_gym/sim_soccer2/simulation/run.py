from soccer_env_motrix import SoccerEnvMotrix

# 启动环境
env = SoccerEnvMotrix(render_mode="human")
obs = env.reset()

print("🚀 仿真启动成功！机器人足球 = MotrixLab 版本")

# 开始运行
while True:
    action = env.action_space.sample()  # 随机动作
    obs, reward, done, _ = env.step(action)
    env.render()
    
    if done:
        env.reset()
s
