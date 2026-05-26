# genesis_bridge 模块

文件：`simulation/labbridge/genesis_bridge.py`

## 作用

把你的 Genesis 仿真循环接到 `labbridge.webview_server` 协议上，提供：

- 前端命令轮询：`poll_commands()`
- 画面推送：`emit_frame()`
- 状态推送：`emit_states()`
- 自动限频：按 `web_fps` 节流推送

## 主要接口

- `genesis_bridge(...) -> GenesisBridge`
- `GenesisBridge.poll_commands() -> WebMsgBuffer`
- `GenesisBridge.emit_frame(rgb, force=False)`
- `GenesisBridge.emit_states(states, force=False)`
- `GenesisBridge.publish(rgb=None, states=None, force=False)`

## 最小用法

```python
from simulation.labbridge import genesis_bridge

bridge = genesis_bridge(webview_port=5811, web_fps=20, enable_webview=True)

while True:
    cmds = bridge.poll_commands()
    # 处理 cmds.reset_env / cmds.teleport_cmd / cmds.velocity_cmds ...
    # step your genesis sim...
    bridge.publish(rgb=rgb_frame, states=states_dict)
```

## `states` 数据格式建议

```python
{
  "robot_rp0": {"x":0, "y":0, "z":0.57, "yaw":0.0, "active":True, "team":"red", "cmd_vel":[0,0,0]},
  "robot_bp0": {"x":0, "y":0, "z":0.57, "yaw":3.14, "active":True, "team":"blue", "cmd_vel":[0,0,0]},
  "ball": {"x":0, "y":0, "z":0.11, "yaw":0.0, "active":True, "team":"none"},
  "_last_msg": {"timestamp": 0.0, "id": -1, "source": "genesis"}
}
```

## 使用样例

[`simulation/labbridge/examples/genesis_bridge_example.py`](../examples/genesis_bridge_example.py)
