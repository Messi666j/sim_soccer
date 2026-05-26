# SPDX-FileCopyrightText: Copyright (c) MOS-Brain Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import time

import numpy as np

from simulation.labbridge import genesis_bridge


def fake_rgb_frame(width: int = 640, height: int = 360) -> np.ndarray:
    """
    生成一个“假视频帧”（用于验证推流链路）。

    注意：
    - 这不是仿真画面。
    - 实际接入 Genesis 时，请把这里替换成你的相机 RGB 输出。
    - 返回格式必须是 HxWx3 的 uint8 数组。
    """
    img = np.zeros((height, width, 3), dtype=np.uint8)
    t = int(time.time() * 10) % 255
    # 仅让绿色通道随时间变化，方便肉眼确认页面正在刷新。
    img[:, :, 1] = t
    return img


def main():
    """
    最小可运行示例：演示 Genesis <-bridge-> WebView 的完整数据流。

    这份示例做了 4 件事：
    1) 启动 WebView 服务（默认 5811）
    2) 从前端轮询命令（reset / velocity / teleport）
    3) 维护一个简化“仿真状态”（这里只更新 robot_rp0 的 x）
    4) 按固定频率发布帧和状态到页面
    """
    # 创建桥接器：
    # - webview_port: 页面访问端口
    # - web_fps: 帧/状态最大推送频率（内部会自动节流）
    # - enable_webview: 是否开启 Web 服务
    bridge = genesis_bridge(webview_port=5811, web_fps=20, enable_webview=True)

    # ---- 简化的“模拟状态” ----
    # 这里只用一个位置变量 x 表示机器人沿 x 轴移动。
    # 在真实工程中，这部分应由 Genesis 物理与控制器提供。
    x = 0.0
    vx_cmd = 0.0
    vy_cmd = 0.0
    wz_cmd = 0.0

    while True:
        # 1) 轮询前端命令（一次性消息，poll 后会被清空）
        cmds = bridge.poll_commands()

        # reset_env：将示例状态恢复初始值
        if cmds.reset_env:
            x = 0.0
            vx_cmd = 0.0
            vy_cmd = 0.0
            wz_cmd = 0.0

        # velocity_cmds：来自右侧 Velocity Control 的速度命令列表
        # 这里示例只读取“最新一条”并且只处理 robot_rp0
        if cmds.velocity_cmds:
            name, vx, vy, wz = cmds.velocity_cmds[-1]
            if name == "robot_rp0":
                vx_cmd = float(vx)
                vy_cmd = float(vy)
                wz_cmd = float(wz)

        # teleport_cmd：来自小地图拖拽传送
        # 格式：(name, x, y, z_or_none, theta_or_none)
        if cmds.teleport_cmd:
            name, tx, _ty, _z, _theta = cmds.teleport_cmd
            if name == "robot_rp0":
                x = float(tx)

        # 2) 用一个最简“积分器”模拟步进：
        # x_{t+1} = x_t + vx * dt
        # 实际接入时，请替换成真实的 Genesis step()。
        dt = 0.01
        x += vx_cmd * dt

        # 3) 组织前端所需状态结构（必须包含可识别实体）
        # 字段约定：
        # - robot_*: x/y/z/yaw/active/team/cmd_vel
        # - ball: x/y/z/yaw/active/team
        # - _last_msg: 可选，用于页面显示最近命令来源
        states = {
            "robot_rp0": {
                "x": x,
                "y": 0.0,
                "z": 0.57,
                "yaw": 0.0,
                "active": True,
                "team": "red",
                "cmd_vel": [vx_cmd, vy_cmd, wz_cmd],
            },
            "ball": {
                "x": 0.0,
                "y": 0.0,
                "z": 0.11,
                "yaw": 0.0,
                "active": True,
                "team": "none",
            },
            "_last_msg": {
                "timestamp": time.time(),
                "id": 0,
                "source": "genesis",
            },
        }

        # 4) 同时发布帧与状态。
        # bridge 内部会按 web_fps 做节流，不会无限高频发送。
        bridge.publish(rgb=fake_rgb_frame(), states=states)

        # 控制示例循环频率（约 100Hz）
        time.sleep(dt)


if __name__ == "__main__":
    main()
