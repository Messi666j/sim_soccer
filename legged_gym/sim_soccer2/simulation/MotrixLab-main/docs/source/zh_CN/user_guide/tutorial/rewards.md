# 奖励函数设计

奖励函数告诉智能体什么样的行为是期望的，是强化学习环境设计中的核心部分。

## 奖励函数在训练循环中的位置

在 MotrixLab 的 NpEnv 中，奖励计算发生在 `step` 函数的 `update_state` 阶段：

```python
# NpEnv.step() 的执行流程
def step(self, actions: np.ndarray) -> NpEnvState:
    # 1. 准备阶段：清空奖励和状态
    self._prev_physics_step()  # reward = 0.0, terminated = False, truncated = False

    # 2. 应用动作
    self._state = self.apply_action(actions, self._state)

    # 3. 物理仿真
    self.physics_step()  # 执行物理仿真

    # 4. 更新状态 ← 奖励函数在这里计算
    self._state = self.update_state(self._state)  # 计算奖励和观察值

    # 5. 后续处理
    self._update_truncate()  # 检查时间截断
    self._reset_done_envs()  # 重置完成的环境

    return self._state
```

您需要在子类的 `update_state` 方法中实现奖励计算逻辑，具体奖励函数设计思路请参考训练示例。

### 奖励组件设计原则

1. **分离关注点**：每个奖励函数负责一个特定的目标
2. **权重配置**：通过配置文件管理不同组件的权重
3. **归一化**：保持奖励值在合理的范围内
4. **平滑性**：避免硬性阈值，使用指数函数等平滑过渡

这种方法使得奖励函数模块化，便于调试和调整各个组件的权重。

## 设计原则

1. **明确的目标导向**：奖励函数应该直接反映任务目标
2. **合理的奖励范围**：避免过大或过小的奖励值，保持训练稳定
3. **平衡探索与利用**：适当奖励接近目标的行为，避免稀疏奖励
4. **避免奖励漏洞**：检查智能体是否可能通过不期望的方式获得高奖励
5. **调试友好**：在开发阶段输出奖励分解信息，便于调优

通过在 `update_state` 方法中正确实现奖励计算，您可以为各种机器人任务设计有效的学习信号。
