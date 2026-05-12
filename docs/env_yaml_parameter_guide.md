# env.yaml 参数配置说明：四足粗糙地形 PPO 环境

> 本文档面向当前上传的 `env.yaml`。它更像 Isaac Lab ManagerBasedRLEnv 的一次环境配置快照，反映了当前四足粗糙地形 PPO baseline 的仿真、场景、机器人、地形、观测、动作、随机化、奖励、终止和数据记录配置。

需要注意：如果这个 `env.yaml` 是训练/评测时导出的配置文件，直接改这个 dump 文件不一定会被训练入口读取。更稳妥的做法是回到对应的 Python env config、agent config、benchmark scenario YAML 或 CLI 参数中修改。这个文档主要告诉你应该改哪些字段、字段含义是什么、改动方向会产生什么影响。

## 1. 整体定位

这个环境配置对应的是一个 ANYmal-C 四足机器人粗糙地形速度跟踪任务。策略接收机体速度、角速度、重力投影、速度指令、关节状态、上一时刻动作和高度扫描；动作是关节位置目标；奖励主要鼓励线速度和 yaw 角速度跟踪，同时惩罚竖直速度、横滚/俯仰角速度、力矩、关节加速度、动作变化和非期望接触。环境带有地形课程学习、物理随机化、质量/质心随机化、reset 随机化和周期性 push 扰动，并且开启了 transition 数据记录。

## 2. 当前最关键的已设置参数

| 类别 | 当前设置 | 含义 |
| --- | --- | --- |
| 并行环境数 | `scene.num_envs = 1024` | 一次并行采样 1024 个环境。越大采样越快，但显存/CPU/仿真负担越高。 |
| 物理步长 | `sim.dt = 0.005` | 物理仿真频率 200 Hz。 |
| 控制降采样 | `decimation = 4` | 策略每 4 个物理步执行一次，所以控制周期是 0.02 s，约 50 Hz。 |
| episode 长度 | `episode_length_s = 20.0` | 每个 episode 最长 20 秒，即约 1000 个策略步。 |
| 机器人 | `ANYmal-C` USD | 使用 ANYbotics ANYmal-C 模型。 |
| 地形 | `terrain_type = generator` | 程序化生成粗糙地形，包括楼梯、倒楼梯、随机方块、随机粗糙面、坡面和倒坡面。 |
| 动作 | `JointPositionAction`, `scale = 0.5` | 策略输出被解释成关节位置目标增量/目标，动作幅度缩放为 0.5。 |
| 指令 | vx/vy/wz 均为 `[-1, 1]` | 训练的是速度跟踪，而不是固定路径或固定目标点导航。 |
| 观测扰动 | `observations.policy.enable_corruption = true` | 训练/运行时给观测加噪声，以增强鲁棒性。 |
| 数据记录 | `recorders` 已开启 | 将 transition 写入 HDF5 数据集，可用于后续 offline RL 或失败轨迹分析。 |

## 3. 仿真与时间尺度

| 参数路径 | 当前值 | 含义 | 怎么设置/影响 |
| --- | --- | --- | --- |
| `sim.device` | `cuda:0` | 使用第 0 张 CUDA GPU。 | 多卡机器上可以改成 `cuda:1`、`cuda:2` 等；也可通过训练脚本的 `--device` 覆盖。 |
| `sim.dt` | `0.005` | 物理仿真步长，200 Hz。 | 减小会更精细但更慢；增大会更快但可能接触不稳定。 |
| `decimation` | `4` | 策略控制间隔为 `dt * decimation = 0.02 s`。 | 增大 decimation 会降低控制频率；减小会让策略更频繁控制。 |
| `sim.render_interval` | `4` | 渲染间隔。当前与 decimation 一致。 | 视频评测时影响渲染帧更新频率。 |
| `sim.gravity` | `[0, 0, -9.81]` | 标准重力。 | 可用于重力偏移实验，但一般不建议训练 baseline 时乱改。 |
| `scene.num_envs` | `1024` | 并行环境数量。 | 训练更快通常增大；显存不足时减小。评测可用 `--num_envs` 覆盖。 |
| `episode_length_s` | `20.0` | episode 最长时间。 | 如果希望观察长时稳定性，可增大；如果只看短期恢复，可减小。 |

## 4. 机器人与动作配置

| 参数路径 | 当前值 | 含义 | 怎么设置/影响 |
| --- | --- | --- | --- |
| `scene.robot.spawn.usd_path` | `.../ANYmal-C/anymal_c.usd` | 机器人模型文件。 | 换机器人时改这里，同时需要适配关节名、初始姿态、actuator、reward/termination 的 body regex。 |
| `scene.robot.init_state.pos` | `[0, 0, 0.6]` | 机器人初始 base 位置。 | z 太低容易初始穿模/触地，太高会掉落。 |
| `scene.robot.init_state.joint_pos` | HAA=0, 前髋 HFE=0.4, 后髋 HFE=-0.4, 前膝=-0.8, 后膝=0.8 | 默认站姿。 | 初始站姿会影响 reset 后稳定性；大改需要保证机器人初始不自碰/不跪地。 |
| `scene.robot.spawn.activate_contact_sensors` | `true` | 启用机器人接触传感器。 | 奖励、终止、足端 air time 依赖该传感器。 |
| `scene.robot.spawn.articulation_props.enabled_self_collisions` | `true` | 启用自碰撞。 | 更真实但仿真负担更高；关闭可能提高速度但失真。 |
| `scene.robot.actuators.legs.class_type` | `ActuatorNetLSTM` | 使用 LSTM actuator network 模拟 ANYdrive 执行器。 | 如果换成理想 PD actuator，动力学特性会变，策略可迁移性不同。 |
| `scene.robot.actuators.legs.effort_limit` | `80.0` | 动作/执行器力矩限制。 | 增大可能更有力但不真实；减小会更难走。 |
| `scene.robot.actuators.legs.velocity_limit` | `7.5` | 关节速度限制。 | 限制过小会影响快速摆腿和恢复。 |
| `scene.robot.actuators.legs.saturation_effort` | `120.0` | 执行器饱和力矩。 | 体现 actuator net 的饱和边界。 |
| `actions.joint_pos.scale` | `0.5` | 策略动作到关节位置目标的缩放。 | 增大动作更激进；减小动作更保守，可能速度跟踪变差。 |
| `actions.joint_pos.joint_names` | `.*` | 所有关节都由策略控制。 | 如果只想控制部分关节，可改 regex，但奖励/机器人结构也要适配。 |

## 5. 地形配置

当前地形是程序化 terrain generator。地形 patch 大小为 8m x 8m，地形网格为 10 行 x 20 列，总共 200 个 terrain cells。课程学习开启，初始最大地形等级为 5，完整难度范围为 0 到 1。

| 参数路径 | 当前值 | 含义 | 怎么设置/影响 |
| --- | --- | --- | --- |
| `scene.terrain.terrain_type` | `generator` | 使用程序生成地形。 | 可改为 plane/usd/importer 等，但当前任务逻辑面向 rough terrain generator。 |
| `scene.terrain.terrain_generator.curriculum` | `true` | 启用地形课程学习。 | 训练初期从较容易地形开始，随后按表现提升难度。关闭后从全分布采样。 |
| `scene.terrain.terrain_generator.difficulty_range` | `[0.0, 1.0]` | 地形难度采样范围。 | 评测 OOD 时可提高上界或固定高难度；训练时过高可能导致早期学不动。 |
| `scene.terrain.max_init_terrain_level` | `5` | 初始可放置的最大地形等级。 | 调低更保守；调高会让初始训练更难。 |
| `scene.terrain.terrain_generator.num_rows` / `num_cols` | `10 / 20` | 地形网格数量。 | 更多格子能容纳更多难度/类型组合，但生成和管理成本更高。 |
| `pyramid_stairs.proportion` | `0.2` | 正金字塔楼梯占 20%。 | 调高表示训练/测试更多上楼梯地形。 |
| `pyramid_stairs.step_height_range` | `[0.05, 0.23]` | 楼梯高度范围。 | 增大可制造更难楼梯，是 long-tail/OOD 的重要旋钮。 |
| `pyramid_stairs_inv.proportion` | `0.2` | 倒金字塔楼梯占 20%。 | 强调下楼梯/反向台阶场景。 |
| `boxes.proportion` | `0.2` | 随机方块地形占 20%。 | 越高越强调离散障碍。 |
| `boxes.grid_height_range` | `[0.05, 0.2]` | 方块高度范围。 | 增大可造成更强的接触冲击和足端摆放困难。 |
| `random_rough.proportion` | `0.2` | 随机粗糙高度场占 20%。 | 强调连续粗糙扰动。 |
| `random_rough.noise_range` | `[0.02, 0.1]` | 随机高度噪声范围。 | 增大后地面更崎岖。 |
| `hf_pyramid_slope.proportion` | `0.1` | 坡面占 10%。 | 强调上坡。 |
| `hf_pyramid_slope.slope_range` | `[0.0, 0.4]` | 坡度范围。 | 增大可构造陡坡 OOD。 |
| `hf_pyramid_slope_inv.proportion` | `0.1` | 倒坡面占 10%。 | 强调下坡。 |

地形类型比例目前加起来正好是 1.0：楼梯 0.2、倒楼梯 0.2、方块 0.2、随机粗糙 0.2、坡面 0.1、倒坡面 0.1。修改比例时建议保持总和约为 1.0，否则采样分布可能不符合预期。

## 6. 传感器与观测配置

| 模块 | 当前设置 | 含义 | 怎么设置/影响 |
| --- | --- | --- | --- |
| 高度扫描 `height_scanner` | `update_period = 0.02`, `size = [1.6, 1.0]`, `resolution = 0.1` | 以机器人 base 为中心向下 ray cast，给策略提供局部地形高度。 | 增大扫描范围能看得更远，但观测维度增加；调粗 resolution 可降维但损失细节。 |
| 接触传感器 `contact_forces` | `update_period = 0.005`, `history_length = 3`, `track_air_time = true` | 记录接触力和足端离地时间。 | feet_air_time 奖励和 base_contact 终止依赖它。 |
| 观测拼接 | `concatenate_terms = true` | 所有 policy observation term 拼成一个向量。 | 通常保持默认。 |
| 观测噪声 | `enable_corruption = true` | 开启观测加噪声。 | 训练鲁棒性需要开启；纯诊断评测时可关闭看上限。 |

| 观测项 | 当前噪声/裁剪 | 含义 | 设置建议 |
| --- | --- | --- | --- |
| `base_lin_vel` | `uniform [-0.1, 0.1]` | 机体坐标系线速度。 | 噪声越大越鲁棒但越难学。 |
| `base_ang_vel` | `uniform [-0.2, 0.2]` | 机体坐标系角速度。 | 对姿态稳定和转向有影响。 |
| `projected_gravity` | `uniform [-0.05, 0.05]` | 重力方向在机体坐标系下的投影，反映姿态。 | 噪声过大可能影响身体姿态判断。 |
| `velocity_commands` | 无噪声 | 目标 vx/vy/wz 或 heading command。 | 一般不加噪声，否则目标本身变模糊。 |
| `joint_pos` | `uniform [-0.01, 0.01]` | 相对关节位置。 | 模拟编码器误差。 |
| `joint_vel` | `uniform [-1.5, 1.5]` | 相对关节速度。 | 当前噪声较大，会增强鲁棒性但增加学习难度。 |
| `actions` | 无噪声 | 上一时刻动作。 | 用于动作平滑和动态记忆。 |
| `height_scan` | `uniform [-0.1, 0.1]`, clip `[-1, 1]` | 局部地形高度观测。 | 噪声越大越能模拟地形感知误差；clip 限制高度异常值。 |

## 7. 随机化与事件配置

| 事件 | 当前设置 | 含义 | 怎么设置/影响 |
| --- | --- | --- | --- |
| `physics_material` | startup；robot body 静摩擦固定 `0.8`，动摩擦固定 `0.6`，restitution `0.0` | 每个环境启动时设置机器人刚体材料。 | 如果要做摩擦随机化，可把范围改成如 `[0.4, 1.2]`。注意这是 robot body 材料，不是 terrain material。 |
| `add_base_mass` | startup；base mass add `[-5, 5]` kg | 随机给 base 增减质量。 | 扩大范围可测试负载变化/动力学偏移；过大会导致训练困难。 |
| `base_com` | startup；x/y `[-0.05,0.05]`，z `[-0.01,0.01]` | 随机 base 质心偏移。 | 适合提高动力学鲁棒性；极端 COM 偏移会显著改变步态。 |
| `base_external_force_torque` | reset；force/torque 都是 0 | 重置时施加外力/力矩。 | 当前等价于关闭。若要 reset 扰动，可设置非零范围。 |
| `reset_base` | 位置 x/y `[-0.5,0.5]`，yaw `[-3.14,3.14]`；速度各轴/姿态角速度约 `[-0.5,0.5]` | 每次 reset 时随机 base 姿态和速度。 | 范围越大，初始状态越复杂。 |
| `reset_robot_joints` | joint position scale `[0.5,1.5]`，velocity `0` | reset 时随机关节位置比例。 | 提高初始姿态多样性；过大可能导致不自然姿态。 |
| `push_robot` | interval；每 `10-15s`，设置 base x/y 速度 `[-0.5,0.5]` | 周期性速度扰动。 | 这是大扰动鲁棒性的重要旋钮。long-tail/OOD 可增大速度范围或缩短间隔。 |

## 8. 指令分布

| 参数路径 | 当前值 | 含义 | 怎么设置/影响 |
| --- | --- | --- | --- |
| `commands.base_velocity.resampling_time_range` | `[10.0, 10.0]` | 每 10 秒重采样一次速度指令。 | 缩短会让命令变化更频繁；加大则每个 episode 内命令更稳定。 |
| `heading_command` | `true` | 启用 heading command，由 heading 控制生成 yaw 角速度目标。 | 如果改为 false，通常直接采样 yaw rate。 |
| `heading_control_stiffness` | `0.5` | heading 到 yaw rate 的控制刚度。 | 越大越积极转向；过大可能导致转向振荡。 |
| `rel_standing_envs` | `0.02` | 约 2% 环境采样站立/零速度指令。 | 提高可增强站立稳定性，但会减少移动样本。 |
| `rel_heading_envs` | `1.0` | 所有环境都使用 heading command。 | 如果要混合 yaw rate command，可降低。 |
| `ranges.lin_vel_x` | `[-1.0, 1.0]` m/s | 前后向速度范围。 | 增大可训练更高速运动，但难度更高。 |
| `ranges.lin_vel_y` | `[-1.0, 1.0]` m/s | 侧向速度范围。 | 四足侧向走能力由此决定；过大可能导致不稳定。 |
| `ranges.ang_vel_z` | `[-1.0, 1.0]` rad/s | yaw 角速度范围。 | 训练转向能力。 |
| `ranges.heading` | `[-pi, pi]` | 目标朝向范围。 | 基本覆盖全方向。 |

## 9. 奖励函数配置

| 奖励项 | 权重 | 含义 | 设置建议 |
| --- | --- | --- | --- |
| `track_lin_vel_xy_exp` | `+1.0` | 鼓励 x/y 线速度跟踪目标命令。 | 主任务奖励。std=0.5，std 越小跟踪误差惩罚越严格。 |
| `track_ang_vel_z_exp` | `+0.5` | 鼓励 yaw 角速度跟踪。 | 转向能力的主奖励。 |
| `lin_vel_z_l2` | `-2.0` | 惩罚竖直方向速度。 | 防止跳跃/弹跳；过大可能抑制跨越障碍。 |
| `ang_vel_xy_l2` | `-0.05` | 惩罚 roll/pitch 角速度。 | 提高身体姿态稳定性。 |
| `dof_torques_l2` | `-1e-5` | 惩罚关节力矩。 | 控制能耗和动作幅度。 |
| `dof_acc_l2` | `-2.5e-7` | 惩罚关节加速度。 | 提高动作平滑性。 |
| `action_rate_l2` | `-0.01` | 惩罚动作变化率。 | 防止动作抖动。 |
| `feet_air_time` | `+0.125` | 鼓励足端有合理摆动时间。 | 促进步态形成；threshold=0.5。 |
| `undesired_contacts` | `-1.0` | 惩罚 THIGH 非期望接触。 | 防止大腿蹭地。 |
| `flat_orientation_l2` | `0.0` | 惩罚非水平姿态，但当前关闭。 | 如果机身倾斜严重，可给负权重；但粗糙地形中过强可能限制适应。 |
| `dof_pos_limits` | `0.0` | 惩罚关节接近限位，但当前关闭。 | 如果发现关节打限位，可打开。 |

## 10. 终止条件

| 终止项 | 当前设置 | 含义 | 设置建议 |
| --- | --- | --- | --- |
| `time_out` | `time_out = true` | episode 到 20 秒正常结束。 | 不应被统计为摔倒。 |
| `base_contact` | base 与地面/物体接触力超过 `1.0` | 非法接触，通常表示机器人摔倒或机体撞地。 | threshold 越小越严格；过小可能误杀，过大可能让明显摔倒继续运行。 |

## 11. 数据记录配置

| 参数路径 | 当前值 | 含义 | 怎么设置/影响 |
| --- | --- | --- | --- |
| `recorders.dataset_file_handler_class_type` | `ChunkedHDF5DatasetFileHandler` | 使用分块 HDF5 写数据。 | 适合长时间采集 transition。 |
| `recorders.dataset_export_dir_path` | `.../datasets/train` | 数据集输出目录。 | 换实验时一定要改，避免覆盖或混淆数据。 |
| `recorders.dataset_filename` | `train_dataset` | 数据集文件名前缀。 | 建议包含任务、算法、日期、是否含扰动等信息。 |
| `recorders.dataset_export_mode` | `EXPORT_ALL` | 导出所有记录数据。 | 如果只想采样部分 episode，需要改 recorder 逻辑或导出模式。 |
| `recorders.transition.command_name` | `base_velocity` | 记录对应速度命令。 | 离线 RL 分析时很重要，因为任务是 command-conditioned。 |
| `record_contact_forces` | `false` | 不记录接触力。 | 如果要分析摔倒/接触事件，建议改成 true，但数据量会增大。 |
| `record_height_scan` | `false` | 不记录高度扫描。 | 如果 offline policy 需要 terrain observation，建议记录，否则离线数据不完整。 |
| `record_critic_obs` | `false` | 不记录 critic observation。 | 如果要复现 asymmetric actor-critic 或做离线 value 学习，可考虑打开。 |

## 12. 可设置项目总表：改哪里、改什么、为什么改

| 目标 | 优先改的参数路径 | 典型改法 | 效果/注意事项 |
| --- | --- | --- | --- |
| 提高训练速度 | `scene.num_envs` | `1024 -> 2048/4096` | 采样吞吐上升，但显存和仿真压力增加。 |
| 显存不足 | `scene.num_envs`，视频 `num_envs` | `1024 -> 256/512` | 降低并行度，训练更慢但更稳定。 |
| 做更强粗糙地形 | `step_height_range`, `grid_height_range`, `noise_range`, `slope_range` | 整体放大 1.2x/1.5x/2.0x | 构造 long-tail/OOD。过强会导致 policy 直接失败。 |
| 做地形分布偏移 | `sub_terrains.*.proportion` | 提高 stairs/boxes/slope 占比 | 改变测试分布，适合研究泛化。 |
| 关闭课程学习 | `terrain_generator.curriculum` | `true -> false` | 从一开始采样全难度；可能更难但更直接。 |
| 提高指令速度难度 | `commands.base_velocity.ranges.*` | vx/vy/wz 从 `[-1,1]` 改到 `[-1.5,1.5]` | 测试高速跟踪与动态稳定性。 |
| 改变命令切换频率 | `resampling_time_range` | `[10,10] -> [3,5]` | 更频繁切换目标，测试快速适应。 |
| 增强扰动 | `events.push_robot.params.velocity_range`, `interval_range_s` | 速度范围变大，间隔变短 | 最直接的大扰动鲁棒性测试。 |
| 做动力学随机化 | `add_base_mass`, `base_com`, `physics_material` | 扩大 mass / COM / friction range | 增强 sim2real/动力学泛化，但训练更难。 |
| 提高观测鲁棒性 | `observations.policy.*.noise` | 适度增大 n_min/n_max | 模拟传感器噪声；过大会导致策略无法精确控制。 |
| 降低观测难度 | `enable_corruption` 或各 noise | 关闭 corruption 或减小噪声 | 用于诊断 policy 上限，不适合鲁棒训练。 |
| 调整动作激进程度 | `actions.joint_pos.scale` | `0.5 -> 0.25/0.75` | 小 scale 保守，大 scale 激进。 |
| 强调平滑/省力 | `action_rate_l2`, `dof_torques_l2`, `dof_acc_l2` 权重 | 负权重绝对值增大 | 动作更平滑省力，但可能牺牲速度跟踪。 |
| 强调速度跟踪 | `track_lin_vel_xy_exp`, `track_ang_vel_z_exp` 权重/`std` | 增大权重或减小 std | 跟踪更严格，但可能带来激进行为。 |
| 调整摔倒判定 | `terminations.base_contact.threshold` | `1.0 -> 更高/更低` | 越低越严格，越高越宽松。 |
| 采集更完整离线数据 | `record_contact_forces`, `record_height_scan`, `record_critic_obs` | `false -> true` | 便于 offline RL/失败分析，但数据量显著增加。 |

## 13. 修改示例

### 13.1 在训练环境配置中直接改关键字段

如果你是在 Python config 里改，常见写法类似：

```python
env_cfg.scene.num_envs = 2048
env_cfg.sim.device = "cuda:1"
env_cfg.commands.base_velocity.ranges.lin_vel_x = (-1.5, 1.5)
env_cfg.commands.base_velocity.ranges.lin_vel_y = (-1.0, 1.0)
env_cfg.commands.base_velocity.ranges.ang_vel_z = (-1.5, 1.5)
env_cfg.events.push_robot.interval_range_s = (5.0, 8.0)
env_cfg.events.push_robot.params["velocity_range"]["x"] = (-1.0, 1.0)
env_cfg.events.push_robot.params["velocity_range"]["y"] = (-1.0, 1.0)
```

### 13.2 在 benchmark scenario YAML 里改评测分布

如果你使用前面那个 `benchmark_quadruped_rough.py`，更推荐在 scenario YAML 里写 override。例如：

```yaml
scenarios:
  - name: long_tail_push_rough
    group: long_tail
    description: "stronger pushes and rougher terrain than training distribution"
    num_envs: 32
    num_episodes: 64
    overrides:
      command:
        lin_vel_x: [-1.0, 1.0]
        lin_vel_y: [-1.0, 1.0]
        ang_vel_z: [-1.0, 1.0]
      push:
        enabled: true
        interval_s: [5.0, 8.0]
        vel_x: [-1.2, 1.2]
        vel_y: [-1.2, 1.2]
      terrain:
        difficulty_range: [0.7, 1.0]
        curriculum: false
        stairs_height_scale: 1.5
        boxes_height_scale: 1.5
        rough_noise_scale: 1.5
        slope_scale: 1.5
      friction:
        static: [0.4, 1.2]
        dynamic: [0.3, 1.0]
      mass:
        add_base_mass: [-8.0, 8.0]
```

这个 override 对应的就是评测阶段改变 command、push、terrain、friction、mass，而不动训练流程。它特别适合用来构造 ID / long-tail / OOD 对比。

### 13.3 构造三个常用测试分布

| 分布 | 推荐设置 | 目的 |
| --- | --- | --- |
| ID | 保持接近训练设置：速度 `[-1,1]`，push `[-0.5,0.5]`，terrain scale `1.0`，friction/mass 保持训练范围。 | 确认 baseline 正常能力。 |
| Long-tail | 地形 scale 1.3-1.7，push 速度 1.0-1.5，摩擦范围稍微拉宽，质量范围稍微拉宽。 | 找训练分布边缘的性能退化。 |
| OOD | 地形 scale 2.0 以上，push 明显增强，低摩擦/高质量偏移/高坡度等组合出现。 | 暴露 PPO+domain randomization 的明确短板。 |

## 14. 当前配置的研究含义

这份配置不是一个极简 locomotion baseline，而是一个已经比较完整的粗糙地形鲁棒 locomotion baseline：它有局部高度感知、观测噪声、课程地形、动力学随机化、push 扰动和数据记录。因此，如果该 baseline 在 ID 场景表现不错，而在 long-tail/OOD 场景出现 return 下降、摔倒率上升、tracking error 增大、recovery 变慢，就可以比较有力地说明：普通 PPO + domain randomization 能覆盖常规粗糙地形，但面对更强扰动、更强地形偏移和动力学偏移时仍有系统性短板。这正好可以作为后续引入 offline data、failure data mining、memory mechanism、world model 或 offline-to-online adaptation 的实验动机。
