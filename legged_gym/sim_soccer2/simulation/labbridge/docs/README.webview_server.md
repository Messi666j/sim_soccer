# webview_server 模块

文件：`simulation/labbridge/webview_server.py`

## 作用

提供 WebView 服务端（Flask + SocketIO）：

- 提供网页入口 `/`
- 接收前端交互命令
- 推送视频帧 `new_frame`
- 推送仿真状态 `robot_states`

## 核心类型

- `WebMsgBuffer`：前端命令缓冲体
- `LabWebView`：服务类

## 前端命令事件

- `reset_env`
- `set_viewer_point`
- `set_viewer_look_at`
- `set_camera_preset`
- `teleport_entity`
- `set_initial_positions`
- `set_robot_velocity`

## 服务端接口

- `LabWebView.start(port=5811)`
- `LabWebView.poll_commands() -> WebMsgBuffer`
- `LabWebView.emit_frame(rgb)`
- `LabWebView.emit_robot_states(states)`

## 最小用法

```python
from pathlib import Path
from simulation.labbridge import LabWebView

web = LabWebView(template_dir=Path("motrixsim/web/templates"))
web.start(port=5811)

while True:
    cmds = web.poll_commands()
    # apply cmds...
    web.emit_frame(rgb_frame)
    web.emit_robot_states(states)
```
