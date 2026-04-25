你现在是我的 Isaac Lab / Isaac Sim 研究工程搭档。请你直接在当前仓库中执行，不要只给建议；先审计项目，再按计划逐步修改代码并汇报结果。

【首先必须做的事情】
在开始任何实现之前，你必须先完整阅读以下背景文件，并把其中与你的实现相关的信息提炼出来作为后续实现约束：

/workspace/amadeus/amadeus/prj_background.md

请同时审计当前仓库，尤其是：
1. Isaac Sim 生成的官方 external 模板目录结构
2. 当前 external 项目的 source / scripts / configs / logs / outputs 等组织方式
3. 当前是否已经存在 manager-based 环境模板、训练入口、RL backend 配置
4. 当前是否已经存在四足机器人资产或可以直接复用的 locomotion 相关配置
5. 当前项目与 Isaac Lab 的连接方式（依赖方式、python path、启动脚本等）

在完成审计前，不要盲目开始大规模写代码。

==================================================
【总体目标】
我现在要先做一个 baseline。

我要在 Isaac Sim 生成的官方 external 模板工程中，基于 manager-based 训练模式，搭建一个“四足机器人在粗糙/崎岖地形上的强化学习 baseline”，并且在训练过程中同步采集离线数据，供后续 offline RL / memory / skill abstraction 使用。

因为当前是在服务器 Docker 中运行，我希望整个程序是面向 headless 训练设计的，但同时能够：
1. 每训练一段时间自动执行一次 evaluate
2. 在 evaluate 时保存一小段可视化视频
3. 持续记录并可视化 loss、reward、episode length、成功率/跌倒率等指标
4. 训练、评估、数据采集的输出目录清晰可管理
5. 尽量少改官方模板，优先在 external 项目内部扩展

==================================================
【重要实现原则】
请严格遵守：

1. 优先使用 manager-based workflow，不要走 direct workflow。
2. 不要修改 Isaac Lab 核心源码，除非 absolutely necessary；优先在 external project 内扩展。
3. 优先最小可运行版本，不要一开始做复杂重构。
4. 不要一开始加入 memory / skill / offline RL 方法创新；当前只做 baseline + 数据采集基础设施。
5. 如果仓库中已有现成的四足 robot、rough terrain、manager-based locomotion 配置，请优先复用。
6. 如果没有完全现成的任务，请在 external 模板内构造一个最小合理版本，而不是从零魔改大系统。
7. 所有修改都要工程化、可调试、可扩展。
8. 所有新增脚本都必须支持服务器 headless 运行。
9. 所有实现都要尽量保持与 Isaac Lab / official template 风格一致。

==================================================
【baseline 目标定义】
我要的 baseline 是：

- 四足机器人
- 粗糙 / 崎岖地形 locomotion
- manager-based 训练模式
- 强化学习 baseline
- 训练时同步采集离线数据
- 定期 evaluate
- 定期导出一小段 evaluate 视频
- 记录并可视化训练指标
- 可以保存 checkpoint
- 可以加载 checkpoint 做评估
- 可以导出 trajectory / rollout 数据
- 作为后续 memory / offline RL / skill learning 的底座

==================================================
【你需要优先帮我做的技术决策】
在开始实现前，请你先自动判断并汇报以下问题：

1. 当前 external 模板中，最适合做 startup baseline 的四足机器人是谁？
   - 优先已有资产
   - 优先已有 rough terrain 兼容性
   - 优先稳定、易训练
   - 不要盲目选最复杂机器人

2. 当前 external 模板里最适合使用哪个 RL backend 做 baseline？
   - 优先考虑 RSL-RL 或 RL-Games
   - 如果模板已有默认官方推荐，就优先沿用
   - baseline 第一版优先 PPO，不要一开始上复杂算法

3. 训练时同步采集离线数据，最合理的切入点在哪？
   - environment step 后记录
   - wrapper / callback
   - trainer hook
   - manager 外围 recorder
   你需要选择最稳妥、最少侵入的一种方案。

4. 定期 evaluate + 保存视频，最合理的实现方式是什么？
   - 单独评估脚本周期性调用
   - 训练内部 callback 触发
   - 独立 evaluator
   你需要给出最适合服务器 headless 运行的设计。

5. loss / reward 等可视化，优先用什么？
   - TensorBoard
   - CSV + plotting
   - 其他现成 logger
   要求服务器上可稳定使用，并尽量与现有训练框架兼容。

==================================================
【请按顺序完成的任务】
请不要跳步，严格按以下阶段推进。

====================
阶段 A：项目审计与实施计划
====================
请先完成：
1. 阅读 /workspace/amadeus/amadeus/prj_background.md
2. 审计当前 external template 项目结构
3. 找出关键训练入口、环境配置入口、日志入口、checkpoint 入口
4. 判断当前可复用的 robot / terrain / manager-based 组件
5. 输出一份“审计结论 + 实施计划”

注意：
- 这一步先不要大规模生成代码
- 先告诉我你准备怎么做
- 明确指出不确定项和风险点

====================
阶段 B：manager-based 四足 rough terrain baseline
====================
请你实现一个最小可运行的 manager-based baseline，要求：

1. 构造或补全一个 manager-based 的四足 rough terrain locomotion 任务
2. 尽量复用现成 manager 配置：
   - action manager
   - command manager
   - observation manager
   - reward manager
   - event / reset / termination manager
3. baseline 第一版优先采用 PPO
4. 提供 agent config
5. 提供训练入口脚本
6. 支持 headless 运行
7. 支持 checkpoint 保存与恢复

如果 rough terrain 组件已存在，优先直接接入；
如果不存在，请构造最小必要版本，不要一开始做复杂地形库。

====================
阶段 C：训练时同步采集离线数据
====================
请实现一个训练时 recorder / dataset writer，要求：

1. 在训练过程中同步记录离线数据
2. 最少记录以下字段：
   - obs
   - action
   - reward
   - next_obs
   - done
   - episode_id
   - env_id
   - timestep
   - command（如果环境有速度指令）
3. 如果存在 privileged obs / contact / terrain info，尽量设计为可选记录项
4. 数据写入不要只存成一个无限增长的大文件
5. 支持 chunked 保存或按 episode 保存
6. 优先使用一种简单稳妥的数据格式（例如 npz/hdf5），并明确 schema
7. 再提供一个后处理脚本：
   - 将 transition chunks 整理为 trajectory-level dataset
   - 方便后续 offline RL / skill segmentation 使用

要求：
- 优先最少侵入训练主流程
- 不要明显拖慢训练
- 数据结构要清晰可读

====================
阶段 D：周期性 evaluate + 视频保存
====================
请实现一个周期性评估机制，要求：

1. 每训练一段时间自动执行 evaluate
2. evaluate 使用当前 checkpoint 或当前策略参数
3. 每次 evaluate 保存一小段视频
4. 视频长度不要太长，只保留一个短片段，足够看 gait、稳定性、rough terrain 表现
5. 适配服务器 / docker / headless 环境
6. 如果 headless 直接录视频有难点，请在审计后选择最稳妥方案并说明原因
7. 视频输出目录要清晰：
   - 按 iteration / step / timestamp 组织
8. 如果训练框架无法优雅内嵌 evaluate，请实现一个训练外部可调用的 evaluator，但要和训练流程配套

====================
阶段 E：训练指标记录与可视化
====================
请实现训练指标的记录与可视化，要求：

1. 记录至少以下内容：
   - policy loss
   - value loss
   - entropy / kl（如果算法中存在）
   - episode reward
   - episode length
   - velocity tracking error（如果可获取）
   - fall rate / termination stats（如果可获取）
2. 优先接入 TensorBoard；如果现有框架已自带日志记录，优先复用
3. 同时尽量把关键指标导出为 csv/json
4. 请给出启动 TensorBoard 的命令
5. 如果框架本身已经记录部分指标，请说明哪些可复用，哪些需要补充

====================
阶段 F：评估脚本与文档
====================
请补全：
1. 单独的评估脚本
   - 加载 checkpoint
   - 固定 seed
   - 运行若干 episode
   - 输出基础指标
2. rollout 导出脚本
   - 基于训练好的策略导出完整轨迹
3. README 或 STARTUP 文档
   - baseline 训练命令
   - headless 训练命令
   - 评估命令
   - 视频导出命令
   - 数据采集命令
   - trajectory 后处理命令
   - TensorBoard 命令
4. 输出目录说明
   - checkpoint 在哪
   - logs 在哪
   - dataset 在哪
   - videos 在哪

==================================================
【推荐实现风格】
请优先按下面思路组织代码，但允许根据当前 external 模板实际情况调整：

- source/<my_project>/...
- scripts/...
- configs/...
- datasets/...
- logs/...
- outputs/...

新增代码尽量拆分清楚，例如：
- env/config
- agent/config
- recorder
- evaluator
- dataset/postprocess
- plotting / metrics

不要把所有逻辑塞到一个脚本里。

==================================================
【你最终必须给我的输出】
在完成修改后，你必须完整输出以下内容，不要只说“完成了”：

1. 审计结论
   - 你从 prj_background.md 中读取到了哪些对实现重要的信息
   - 当前 external 模板的关键入口是什么
   - 你最终选了哪个四足机器人
   - 你最终选了哪个 RL backend
   - 你为什么这样选

2. 新增/修改文件清单
   - 每个文件路径
   - 每个文件作用
   - 为什么需要它

3. 运行命令
   - baseline 训练命令
   - headless 训练命令
   - checkpoint 恢复训练命令
   - evaluate 命令
   - rollout 导出命令
   - 数据后处理命令
   - TensorBoard 命令

4. 输出目录结构说明
   - checkpoints
   - logs
   - datasets
   - videos
   - metrics

5. 后续扩展建议
   - 如果后续我要加 history encoder / memory / skill abstraction，最适合插入哪些模块
   - 现在哪些文件未来最可能要扩展
   - 当前 baseline 对后续 offline RL 最重要的中间产物是什么

==================================================
【执行要求】
- 先审计，后改代码
- 先给计划，再实施
- 优先最小可运行版本
- 优先复用现有 manager-based 组件
- 优先服务器 headless 可跑
- 不要直接做复杂方法
- 如果遇到冲突，请明确写出冲突点并给出最稳妥替代方案
- 如果某一步你不确定，请明确说明，不要假装确定