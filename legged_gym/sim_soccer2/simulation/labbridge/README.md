# labbridge

独立的桥接模块目录，和 `motrixsim` 解耦，包含 3 个核心模块：

- `genesis_bridge.py`：Genesis 主循环接入 WebView 的桥接器
- `webview_server.py`：SocketIO + HTML 模板的 WebView 服务
- `sim_manager.py`：仿真进程管理 API（FastAPI）

## 快速导入

```python
from simulation.labbridge import genesis_bridge, GenesisBridge
from simulation.labbridge import LabWebView, WebMsgBuffer
from simulation.labbridge import create_app, SimManager
```

## 文档导航

- `docs/README.genesis_bridge.md`
- `docs/README.webview_server.md`
- `docs/README.sim_manager.md`
