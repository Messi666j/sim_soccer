# Decider GameController 使用说明

本文说明 `decider` 中 `GameController` 的数据来源、字段语义，以及在 `user_entry.py` 里的推荐使用方式。

## 1. 组件职责

- `simulation/mujoco/app/soccer_referee.py`  
  负责“裁判逻辑本体”：比赛状态机、set play、开球方、比分、球员 penalty 等。
- `decider/interfaces/gamecontroller.py`  
  负责“解析与缓存”：把仿真发送的 `gamecontroller` 数据解析成 `agent.gamecontroller.*` 字段，给策略读取。
- `decider/user_entry.py`  
  负责“策略决策”：根据 `agent.gamecontroller` 决定是否允许进攻、踢球、暂停等。

`decider` 侧不会主动裁判，只读取状态。

## 2. 数据流（仿真模式）

在 `SimAgent.run()` 里每个循环：

1. `user_entry.loop(self)` 先执行一次
2. 向仿真发命令并收状态
3. `self.gamecontroller.update(state.get("gamecontroller", {}))`

注意：你的策略读取到的是“上一帧已更新的数据”（常见 1 tick 延迟），这是当前循环顺序决定的。

## 3. 可用字段

常用字段（`agent.gamecontroller`）：

- `game_state`：如 `STATE_READY/STATE_SET/STATE_PLAYING/...`
- `game_state_int`：对应整数状态
- `set_play_name`：如 `SET_PLAY_KICK_OFF/SET_PLAY_CORNER_KICK/...`
- `kicking_team`：当前发球方 team number
- `kick_off`：当前是否是本队 kickoff
- `can_kick`：当前规则下本机器人是否允许踢
- `penalty` / `player_penalty_name` / `penalized_time`
- `score` / `opponent_score`
- `secs_remaining`：当前半场剩余时间

## 4. 在 user_entry.py 中的推荐写法

示例：把规则门控放在 `game(agent)` 最前面。

```python
def game(agent):
    gc = agent.gamecontroller

    # 1) 非 PLAYING 时，不执行正常攻防
    if gc.game_state != "STATE_PLAYING":
        agent.stop()
        return

    # 2) 被罚下时，不踢球
    if gc.penalty != 0:
        agent.stop()
        return

    # 3) 定位球阶段，非发球方不主动踢
    if gc.set_play_name != "SET_PLAY_NONE" and not gc.can_kick:
        # 可替换为站位策略
        agent.stop()
        return

    # 4) 进入你的正常策略
    if not agent.get_if_ball():
        agent.state_machine_runners["find_ball"]()
    else:
        agent.state_machine_runners["chase_ball"]()
```

## 5. `can_kick` 的当前语义

`can_kick=True` 需同时满足：

1. 本机器人未被 penalize
2. 主状态是 `STATE_PLAYING`
3. 若处于 set play（非 `SET_PLAY_NONE`），必须本队是 `kicking_team`

所以你通常可以直接用 `can_kick` 做动作总开关，不必重复写全部判断。

## 6. 手动裁判指令（联调用）

仿真支持 5-byte 风格命令（通过 ZMQ 消息字段 `game_controller_cmd`）：

```json
{
  "game_controller_cmd": [global, team, player, playerNumber, side]
}
```

- `side`: `0=home(left)`, `1=away(right)`
- 该命令由 `simulation/mujoco/app/multi_robot_sim.py` 转发给 referee 执行

适用于：
- 强制切状态（ready/set/playing/finished）
- 触发定位球
- 给指定球员加/解 penalty

## 7. 排障建议

- 先看 webview 顶部 referee 面板，确认 `state/set_play/kicking team`
- 在 `user_entry.py` 临时打印：
  - `gc.game_state`
  - `gc.set_play_name`
  - `gc.kicking_team`
  - `gc.can_kick`
  - `gc.penalty`
- 若策略“看起来慢一拍”，优先检查第 2 节提到的循环时序

## 8. 相关文件

- `decider/interfaces/gamecontroller.py`
- `decider/decider.py`
- `decider/user_entry.py`
- `simulation/mujoco/app/soccer_referee.py`
- `simulation/mujoco/app/multi_robot_sim.py`
