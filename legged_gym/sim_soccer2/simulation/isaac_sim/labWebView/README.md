# LabWebView

A simple way to render your simulation in isaaclab with web interaction.


Add following code to your play scripts.

```
render_mode = "rgb_array" if args_cli.video or args_cli.web else None

env_cfg.viewer = ViewerCfg(
    eye = (4.0, 4.0, 4.0),
    # eye = (0.0, 0.0, 10.0),
    lookat = (0.0, 0.0, 0.0),
    env_index = 20,
    origin_type = "asset_root",
    # origin_type = "env",
    asset_name = "robot",
)
```

```
if args_cli.web:
    from labWebView.wrapper import RenderEnvWrapper

    if not hasattr(env, "socketio_running"):
        env = RenderEnvWrapper(env)
        env.web_run()
        env.socketio_running = True
        print("[INFO] Running web viewer.")
```