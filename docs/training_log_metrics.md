# 训练日志字段说明

本文解释 `wheel_legged_gym/scripts/train.py` 训练时控制台和 TensorBoard 中常见字段的含义。控制台输出主要来自 `wheel_legged_gym/rsl_rl/runners/on_policy_runner.py` 的 `log()`，奖励分项来自环境里的 `_reward_*` 函数。

## 控制台日志

### `Learning iteration i/N`

当前 PPO 学习迭代编号。一次 iteration 包含：

1. 用当前策略在所有并行环境中 rollout `num_steps_per_env` 步；
2. 计算 returns / advantages；
3. 执行 PPO 更新；
4. 写入 TensorBoard 标量，按 `save_interval` 保存模型。

例如 `Learning iteration 499/500` 表示正在打印第 499 次更新结果；训练结束后会额外保存 `model_500.pt`。

### `Computation`

格式：

```text
Computation: <fps> steps/s (collection: <秒>s, learning <秒>s)
```

- `steps/s`：本轮吞吐量，约等于 `num_envs * num_steps_per_env / (collection_time + learning_time)`。
- `collection`：采样耗时，即环境仿真和策略推理耗时。
- `learning`：PPO 反向传播和优化耗时。

该字段主要用于看训练速度和性能瓶颈，不直接代表策略好坏。

### `Value function loss`

价值函数损失。Critic 预测 return 的均方误差，代码中使用 PPO clipped value loss。通常越小越稳定，但它的绝对值会受奖励尺度影响，不能单独用来判断训练成败。

### `Surrogate loss`

PPO policy surrogate loss。它来自新旧策略动作概率比和 advantage 的 clipped objective。数值常为负，也可能接近零；重点看是否持续异常发散，而不是追求某个固定数值。

### `Mean action noise std`

策略输出动作分布的平均标准差，即探索噪声大小。数值越大，策略探索越强，动作更随机；数值越小，策略更确定。该项目中它来自 actor-critic 的 `std.mean()`。

### `Mean reward`

最近最多 100 个已结束 episode 的总奖励均值。它来自 `rewbuffer`，每个 episode 的 reward 是逐步累计后的总和。

注意：

- 这是 episode 总奖励，不是每秒奖励，也不是单步奖励。
- 训练早期可能很负；只要 episode 长度和跟踪项逐步改善，后续可能回升。
- 不同任务、不同 reward scale 之间不能直接比较绝对值。

### `Mean length`

最近最多 100 个已结束 episode 的平均长度，单位是控制步数。越接近 `max_episode_length`，说明机器人越少提前终止。若任务 `episode_length_s=20` 且控制周期约 `0.01s`，常见最大长度约 `2000` 步。

### `Mean rew_*`

环境上报的奖励分项均值。打印时 key 来自 `extras["episode"]`，格式为：

```text
Mean rew_<reward_name>: <value>
```

它不是单步原始 reward，而是：

```text
episode_sums[reward_name] / max_episode_length_s
```

也就是该分项在已结束 episode 中的累计贡献，再除以 episode 最大秒数。因为 reward scale 已经乘进去，所以正负号代表该项对总奖励的实际贡献方向：

- 正值：该项在鼓励某种行为；
- 负值：该项在惩罚某种行为；
- 接近 0：该项影响小，或对应事件很少发生。

### `Mean a_*_max_command_x`

命令课程学习相关指标，表示某类地形上的最大前向速度指令上限，单位 `m/s`。

常见字段：

- `a_flat_max_command_x`：平地地形的最大前向速度命令；
- `a_smooth_slope_max_command_x`：平滑斜坡；
- `a_rough_slope_max_command_x`：粗糙斜坡；
- `a_stair_up_max_command_x`：上楼梯；
- `a_stair_down_max_command_x`：下楼梯；
- `a_discrete_max_command_x`：离散障碍。

即使当前任务不是 curriculum 地形，只要 `commands.curriculum=True`，仍可能看到 `a_flat_max_command_x`。

### `terrain_level`

地形课程学习等级均值。只在 `terrain.curriculum=True` 时出现。数值越高表示机器人被分配到更难的地形层级。

### `Total timesteps`

从训练开始累计的环境步数：

```text
iterations * num_envs * num_steps_per_env
```

例如 1024 个环境、每轮 48 步，则每轮增加 `49152` 步。

### `Iteration time`

本轮总耗时，等于 `collection + learning`。

### `Total time`

训练累计耗时，单位秒。

### `ETA`

按当前平均速度估算剩余时间，单位秒。公式近似为：

```text
Total time / (当前 iteration + 1) * 剩余 iteration 数
```

## TensorBoard 标量

TensorBoard 日志写在 `logs/<experiment>/<run>/events.out.tfevents.*` 中。常见 tag 如下。

### `Loss/*`

- `Loss/value_function`：同控制台 `Value function loss`。
- `Loss/surrogate`：同控制台 `Surrogate loss`。
- `Loss/encoder`：额外编码器损失，来自 PPO update 返回的 `mean_extra_loss`。若当前算法没有有效 encoder loss，可能长期接近 0。
- `Loss/learning_rate`：当前学习率。若 PPO schedule 为 `adaptive`，会根据 KL 自动升降。

### `Policy/*`

- `Policy/mean_noise_std`：同控制台 `Mean action noise std`。
- `Policy/mean_kl`：新旧策略之间的平均 KL 散度。用于自适应调整学习率；过大说明策略更新步子太大，过小说明更新较保守。

### `Perf/*`

- `Perf/total_fps`：训练吞吐量，等同控制台 `Computation` 的 steps/s。
- `Perf/collection time`：采样耗时。
- `Perf/learning_time`：学习更新耗时。

### `Train/*`

- `Train/mean_reward`：最近最多 100 个已结束 episode 的总奖励均值，同控制台 `Mean reward`。
- `Train/mean_episode_length`：最近最多 100 个已结束 episode 的平均长度，同控制台 `Mean length`。

### `Episode/*`

环境上报的 episode 分项，控制台中显示为 `Mean <key>`。例如：

- TensorBoard：`Episode/rew_tracking_lin_vel`
- 控制台：`Mean rew_tracking_lin_vel`

含义同 `Mean rew_*`。

### `Control/*`

训练 rollout 时即时统计的指令跟踪误差，越小越好：

- `Control/mean_abs_lin_vel_x_error`：前向线速度指令与实际前向速度的平均绝对误差，单位 `m/s`。
- `Control/mean_abs_yaw_vel_error`：yaw 角速度指令与实际 yaw 角速度的平均绝对误差，单位 `rad/s`。
- `Control/mean_abs_height_error`：高度指令与机体高度的平均绝对误差，单位 `m`。只有环境有 `base_height` 时写入。

## 常见奖励分项

下面解释当前 wheel-legged / VMC 任务中常见的 `rew_*` 字段。是否出现取决于任务配置中对应 reward scale 是否非零。

### 跟踪类

- `rew_tracking_lin_vel`：前向线速度跟踪奖励。实际前向速度越接近 `commands[:, 0]` 越高。
- `rew_tracking_lin_vel_enhance`：线速度跟踪增强项，形式为较宽松的指数奖励减 1，通常为非正值，用于塑形。
- `rew_tracking_ang_vel`：yaw 角速度跟踪奖励。实际 yaw 角速度越接近指令越高。
- `rew_tracking_ang_vel_enhance`：yaw 跟踪增强项，通常为非正值。
- `rew_tracking_wheel_vel`：轮速对应线速度的跟踪奖励，用在启用该项的任务中。

### 姿态和高度

- `rew_base_height`：机体高度接近高度指令或目标高度的奖励/惩罚。若 scale 为正，接近目标时为正；若 scale 为负，则通常表现为高度误差惩罚。
- `rew_base_height_enhance`：高度跟踪增强项。
- `rew_orientation`：机体姿态偏离水平的惩罚，主要惩罚 roll/pitch。
- `rew_lin_vel_z`：机体 z 方向线速度惩罚，抑制上下弹跳。
- `rew_ang_vel_xy`：机体 x/y 轴角速度惩罚，抑制 roll/pitch 方向快速旋转。

### 动作和平滑

- `rew_action_rate`：相邻两次动作变化量惩罚，鼓励动作不要突变。
- `rew_action_smooth`：二阶动作差分惩罚，鼓励动作轨迹平滑。
- `rew_torques`：关节/轮电机力矩平方惩罚，鼓励省力。
- `rew_power`：力矩乘关节速度的功率惩罚，鼓励低功耗。

### 关节和约束

- `rew_dof_vel`：关节速度惩罚。
- `rew_dof_acc`：关节加速度惩罚。
- `rew_dof_pos_limits`：关节位置越界或接近限制的惩罚。
- `rew_dof_vel_limits`：关节速度超过软限制的惩罚。
- `rew_torque_limits`：力矩超过软限制的惩罚。

### 碰撞和终止

- `rew_collision`：配置中 `penalize_contacts_on` 指定部件发生接触的惩罚。
- `rew_termination`：非超时终止的惩罚，例如摔倒或触发失败条件。超时结束通常不算该惩罚。
- `rew_stumble`：足端/轮端撞到近似竖直障碍的惩罚。
- `rew_feet_contact_forces`：接触力超过上限的惩罚。

### VMC 虚拟腿相关

- `rew_vmc_leg_action`：虚拟腿相关动作通道幅值惩罚。
- `rew_vmc_leg_motion`：虚拟腿角速度和腿长变化速度惩罚。
- `rew_vmc_leg_symmetry`：左右虚拟腿角度和腿长不对称惩罚。
- `rew_vmc_nominal_leg`：虚拟腿角度和腿长偏离名义姿态的惩罚。
- `rew_vmc_leg_collapse`：腿长低于最小安全腿长时的惩罚。

### 其他行为项

- `rew_nominal_state`：左右腿角度差或名义姿态偏离项。scale 为负时通常是惩罚左右不对称；scale 为正时可作为接近名义状态的奖励。
- `rew_stand_still`：零速度命令下偏离默认关节姿态的惩罚，鼓励静止时保持站姿。

## 阅读训练日志的建议

1. 先看 `Mean length`：如果持续很低，说明机器人频繁提前终止，应优先处理稳定性、碰撞或高度问题。
2. 再看 `Mean reward` 趋势：同一任务内持续上升通常是好信号；不同任务之间不要直接比较绝对值。
3. 查看关键 `rew_*`：定位主要扣分项，例如 `collision`、`orientation`、`action_smooth`、`vmc_leg_collapse`。
4. 查看 `Control/*`：判断策略是否真正跟踪速度、转向和高度指令。
5. 查看 `Policy/mean_kl` 和 `Loss/learning_rate`：判断 PPO 更新是否过激或过保守。
6. 查看 `Mean action noise std`：过高可能动作随机性大，过低可能探索不足；应结合 reward 和 length 一起看。
