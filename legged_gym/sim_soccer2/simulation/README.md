# simulation

目录重组后的仿真层入口：

- `isaac_sim/`: Isaac Sim 相关代码与脚本
- `motrixsim/`: MotrixSim 仿真与资产
- `labbridge/`: 独立的 WebView / bridge / sim-manager 模块

常用入口：

```bash
# Isaac Sim
bash simulation/isaac_sim/scripts/launch_sim.sh --headless --webview --task Robocup-Soccer

# MotrixSim
conda run -n motrixsim python simulation/motrixsim/sim2sim_runner.py --team-size 3

# Sim Manager（推荐）
conda run -n motrixsim uvicorn simulation.labbridge.sim_manager:app --host 0.0.0.0 --port 8000
```
