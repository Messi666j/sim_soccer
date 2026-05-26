# sim_manager 模块

文件：`simulation/labbridge/sim_manager.py`

## 作用

管理仿真进程（启动/扫描/停止），暴露 FastAPI 接口。

## 默认行为

- 启动 runner：`<sim_root>/sim2sim_runner.py`
- 默认 `sim_root`：`<project_root>/simulation/motrixsim`
- 默认注册表：`<project_root>/simulation/.labbridge_sim_manager_registry.json`

## 环境变量

- `LABBRIDGE_SIM_ROOT`
- `LABBRIDGE_SIM_RUNNER`
- `LABBRIDGE_REGISTRY`
- `LABBRIDGE_MANAGER_WEB_DIR`

## API

- `GET /`：manager UI
- `GET /manager/docs`：API 说明页
- `GET /healthz`
- `POST /sims/start`
- `GET /sims`
- `POST /sims/stop`
- `POST /sims/stop-all`
- `POST /sims/stop-external`（兼容别名，行为同 `stop-all`）

`POST /sims/start` 支持 `policy_device` 参数（`gpu`/`cpu`，默认 `gpu`）。
当传 `gpu` 但机器无 CUDA 时，MotrixSim 会自动回退到 `cpu` 推理。

## 启动方式

```bash
cd /path/to/mos-brain
uvicorn simulation.labbridge.sim_manager:app --host 0.0.0.0 --port 8000
```

兼容入口（旧命令，仍可用）：

```bash
cd /path/to/mos-brain/simulation/motrixsim
python sim_manager.py --host 0.0.0.0 --port 8000
```

## 自定义初始化

```python
from pathlib import Path
from simulation.labbridge.sim_manager import SimManager, create_app

manager = SimManager(
    registry_path=Path("/tmp/labbridge_registry.json"),
    sim_root=Path("/path/to/mos-brain/simulation/motrixsim"),
    runner=Path("/path/to/mos-brain/simulation/motrixsim/sim2sim_runner.py"),
    manager_index_html=Path("/path/to/mos-brain/simulation/motrixsim/web/manager/index.html"),
    manager_api_docs_html=Path("/path/to/mos-brain/simulation/motrixsim/web/manager/api_docs.html"),
)
app = create_app(manager)
```
