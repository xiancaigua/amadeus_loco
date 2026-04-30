# Quadruped Rough Baseline 改动说明（PPO + SAC）

更新时间：2026-04-25（UTC）

本文档用于完整记录本轮在 `Template-Amadeus-Quadruped-Rough` 任务上的改动，覆盖：
- 既有 PPO baseline 的增强与兼容修复
- 新增 SAC baseline（训练/评估/配置/工具）
- 本轮定位并修复的 `train_metrics.csv` 未落盘问题

---

## 1. 改动目标

本轮改动的总体目标有四类：

1. 统一并稳定输出目录结构，减少恢复训练和后处理脚本在不同 run 布局下的失败概率。
2. 增强评估视频能力（周期触发、跟随相机、速度 marker 可视化）并打通到训练中的周期评估。
3. 增加可复用的数据/指标链路（CSV/JSONL metrics、rollout 导出、绘图工具）。
4. 引入完整 SAC 训练与评估流程，并与现有任务注册和文档体系对齐。

---

## 2. 文件变更总览

### 2.1 现有文件改动（PPO/任务注册/文档）

| 文件 | 变更类型 | 核心内容 |
|---|---|---|
| `scripts/baseline/train_quadruped_rough.py` | 增强 | 新增周期评估视频参数透传；恢复路径兼容 `checkpoints/ckpt/旧布局`；`output_root` 描述更新 |
| `scripts/baseline/eval_quadruped_rough.py` | 增强 | checkpoint 父目录推断兼容 `ckpt/checkpoints`；评估数据导出目录默认值增加 `datasets` 与 `data` 双布局兼容 |
| `scripts/baseline/export_rollouts.py` | 增强 | rollout 输出目录推断兼容 `datasets` 与 `data` 两套布局；checkpoint 目录兼容 `ckpt/checkpoints` |
| `source/amadeus/amadeus/tasks/manager_based/quadruped_rough/__init__.py` | 扩展 | 为 train/play 两个 env 注册 `skrl_sac_cfg_entry_point` |
| `docs/quadruped_rough_baseline/README.md` | 文档更新 | run 布局改为 `<output_root>/rsl_rl/<experiment_name>/<timestamp>[_run_name]`；TensorBoard logdir 更新 |
| `scripts/baseline/train_quadruped_rough.md` | 文档更新 | `output_root` 说明和 run_dir 示例更新 |
| `scripts/baseline/eval_quadruped_rough.md` | 文档更新 | 评估视频触发行为说明、checkpoint 示例路径、视频输出示例路径更新 |
| `scripts/baseline/export_rollouts.md` | 文档更新 | `ckpt/checkpoints` 兼容说明与示例路径更新 |

### 2.2 新增文件（SAC/分析工具）

| 文件 | 作用 |
|---|---|
| `scripts/sac/train_quadruped_rough_sac.py` | SAC 训练入口（skrl backend），支持周期评估、视频、checkpoint、dataset、metrics |
| `scripts/sac/eval_quadruped_rough_sac.py` | SAC 评估入口，支持多片段视频触发、跟随相机、速度 marker、metrics、dataset 导出 |
| `source/amadeus/amadeus/tasks/manager_based/quadruped_rough/agents/skrl_sac_cfg.yaml` | SAC 网络与训练超参配置（policy/critics/memory/trainer/runtime） |
| `source/amadeus/amadeus/algorithms/sac/utils.py` | SAC 通用工具（checkpoint 解析、run_dir 推断、eval metrics 导出等） |
| `source/amadeus/amadeus/algorithms/sac/train_metrics.py` | SAC 训练 tracking_data 到标准行结构的映射 |
| `source/amadeus/amadeus/algorithms/sac/__init__.py` | SAC 工具导出入口 |
| `scripts/baseline/plot_metrics.py` | 训练/评估 CSV 指标绘图工具（平滑、列名容错、占位图容错） |

---

## 3. PPO 相关改动详解

## 3.1 `train_quadruped_rough.py`

### A. 周期评估视频能力增强

新增参数：
- `--eval_video_start_step`
- `--eval_follow_robot_camera` / `--no-eval_follow_robot_camera`
- `--eval_camera_eye`
- `--eval_camera_lookat`
- `--eval_camera_offset`
- `--eval_camera_robot_env_id`
- `--eval_show_velocity_markers` / `--no-eval_show_velocity_markers`

训练脚本在调用 `eval_quadruped_rough.py` 的周期评估子进程时，会把上述参数透传给评估脚本，实现：
- 非固定 step=0 的视频触发
- 机器人跟随相机
- 速度命令/实际速度 marker 可视化

### B. Resume/checkpoint 路径兼容增强

`_resolve_resume_path(...)` 逻辑升级为多 root 回退查找：
- 首选新布局 root：`<output_root>/rsl_rl/<experiment_name>/...`
- 兼容旧短布局 root：`<output_root>/...`
- 目录名兼容 `checkpoints` 与 `ckpt`
- 继续兼容“checkpoint 直接在 run 根目录”的历史布局

当所有路径都找不到时，抛出包含搜索根目录与 `load_run/load_checkpoint` 的明确错误，便于定位。

### C. run_dir 决策增强

`_make_run_dir(...)` 在 resume 场景下优先检测：
- 新布局候选目录
- 旧短布局候选目录

存在即复用，不存在时再按新布局构造。

### D. 输出目录说明更新

`--output_root` 帮助文本明确为：
`<output_root>/rsl_rl/<experiment_name>/<run_name>/...`

---

## 3.2 `eval_quadruped_rough.py`

### A. checkpoint 父目录推断兼容

`_infer_run_dir_from_checkpoint(...)` 兼容两种 checkpoint 目录名：
- `checkpoints`
- `ckpt`

### B. 评估 dataset 默认目录选择器

新增 `_default_eval_dataset_dir(...)`，默认路径优先级：
1. 若存在 `<run_dir>/datasets`，用 `<run_dir>/datasets/eval`
2. 否则若存在 `<run_dir>/data`，用 `<run_dir>/data/eval`
3. 都不存在时回退 `<run_dir>/datasets/eval`

这样可覆盖 PPO 新布局与短布局的历史 run。

---

## 3.3 `export_rollouts.py`

导出目录推断增强：
- checkpoint 上级目录兼容 `checkpoints` 和 `ckpt`
- 若未显式传 `--output_dir`，按以下顺序选择：
1. `<run_dir>/datasets/rollouts`
2. `<run_dir>/data/rollouts`
3. 回退 `<run_dir>/datasets/rollouts`

---

## 3.4 PPO 文档同步

`README.md`、`train_quadruped_rough.md`、`eval_quadruped_rough.md`、`export_rollouts.md` 统一更新了：
- 新 run 路径表达
- TensorBoard `--logdir` 推荐
- 示例命令路径
- 评估视频行为描述

---

## 4. SAC 新增实现详解

## 4.1 任务注册扩展

`source/.../quadruped_rough/__init__.py` 给 train/play 两个环境都新增：
- `skrl_sac_cfg_entry_point: <agents_package>:skrl_sac_cfg.yaml`

这样可通过 Hydra 的 `--agent skrl_sac_cfg_entry_point` 直接切到 SAC 配置。

## 4.2 SAC 配置 `skrl_sac_cfg.yaml`

配置包含：
- `models`: Gaussian policy + twin critic + target critics
- `memory`: `RandomMemory`, `memory_size=1000000`
- `agent`: SAC 核心超参（`batch_size`, `polyak`, entropy 配置等）
- `trainer`: `SequentialTrainer`, 默认 `timesteps=2000000`
- `sac_runtime`: 外层运行控制（`update_frequency`, `save_interval`, `eval_interval_steps`, `eval_num_envs`, `eval_episodes`）

本轮新增：
- `sac_runtime.train_metrics_interval: 2000`

## 4.3 SAC 通用工具 `algorithms/sac/utils.py`

提供了统一工具函数：
- `parse_model_step`：从 `model_XXXX.pt` 提取步数
- `latest_model_checkpoint`：取目录内最新 step checkpoint
- `infer_run_dir_from_checkpoint`：兼容 `ckpt/checkpoints`
- `default_eval_dataset_dir`：`data` 与 `datasets` 双布局回退
- `write_eval_metrics`：`eval_metrics.jsonl + eval_metrics.csv` 双写

## 4.4 SAC 训练指标映射 `algorithms/sac/train_metrics.py`

`build_sac_train_metrics_row(...)` 将 `agent.tracking_data` 映射到统一字段：
- 奖励/长度
- policy/value/entropy loss 与 entropy_coef
- 学习率
- fail/timeout rate
- tracking reward
- `tracking_data_json`（保留原始均值快照）

## 4.5 SAC 训练脚本 `scripts/sac/train_quadruped_rough_sac.py`

主要能力：
- 单脚本启动 SAC 训练（skrl）
- run 目录标准化：`ckpt/tb/metrics/data/video/params/logs`
- 支持 resume（`--resume/--load_run/--checkpoint`）
- `update_frequency` 控制采样与更新解耦
- 周期评估通过子进程调用 `eval_quadruped_rough_sac.py`
- 支持训练视频与评估视频
- 导出 params 与 run_info 便于复现实验

本轮修复过的两个关键点：

1. `SkrlBaseAgent` 拼写错误修复为 `SkrlAgentBase`
- 位置：非更新 step 分支调用 `post_interaction` 的地方。
- 影响：避免运行时 `NameError`。

2. `train_metrics.csv` 未落盘修复（详见第 5 节）

## 4.6 SAC 评估脚本 `scripts/sac/eval_quadruped_rough_sac.py`

主要能力：
- 加载 checkpoint 做多 episode 评估
- 支持固定相机与跟随相机
- 支持多片段视频触发（`video_start_step + video_interval_steps + max_video_clips`）
- 支持速度 marker 可视化（命令速度 vs 实际速度）
- 导出 `eval_metrics.csv/jsonl`
- 支持 `dataset_dir` 导出评估轨迹

---

## 5. 本轮缺陷修复：`train_metrics.csv` 未落盘

## 5.1 现象

在短训练（例如几百 step）中，SAC run 目录下常出现：
- `metrics/eval_metrics.csv` 存在
- `metrics/train_metrics.csv` 不存在

## 5.2 根因

SAC 配置默认 `agent.experiment.write_interval = 2000`。  
skrl 仅在 `timestep % write_interval == 0` 时触发 `write_tracking_data`。  
短训练（如 320 step）达不到 2000，因此不会触发 train metrics 写出。

## 5.3 修复方案

已在 `scripts/sac/train_quadruped_rough_sac.py` 落地：

1. 新增 CLI 参数：
- `--train_metrics_interval`

2. 运行时解析策略：
- 优先 CLI
- 否则用 `sac_runtime.train_metrics_interval`
- 再回退 `agent.experiment.write_interval`
- 当值为正且大于 `max_steps` 时自动 clamp 到 `max_steps`

3. 将解析后的值写回：
- `agent_cfg["agent"]["experiment"]["write_interval"]`

4. 兜底保障：
- 若训练结束时仍未写出任何 train metrics 且 interval>0，强制写一条 final snapshot 到 `train_metrics.csv/jsonl`。

5. `run_info.json` 增加字段：
- `train_metrics_interval`

6. `skrl_sac_cfg.yaml` 增加：
- `sac_runtime.train_metrics_interval: 2000`

## 5.4 修复后行为

- 短训练也会生成 `metrics/train_metrics.csv`。
- 默认长训练行为不变（仍是 2000 步周期写）。
- 若用户显式传 `--train_metrics_interval <= 0`，可禁用 train metrics 导出。

---

## 6. 验证记录（不影响长期训练）

执行策略：
- 所有验证均使用独立输出目录（`outputs/tmp_*`）
- 使用独立 GPU 设备参数
- 每次验证后都确认长期训练进程仍在

关键验证结果：

1. SAC 全链路验证通过（训练/周期评估/视频/checkpoint/dataset/eval metrics）
- 曾出现一次高 `num_envs` 场景的显存不足（属于运行参数资源问题，不是逻辑错误）。
- 下调 `num_envs` 后完整跑通。

2. 独立评估链路验证通过
- `eval_quadruped_rough_sac.py` 的视频、marker、metrics、dataset 均正常。

3. `train_metrics.csv` 修复回归通过
- 使用 `max_steps=40` 短跑验证，成功生成：
  - `metrics/train_metrics.csv`
  - `metrics/train_metrics.jsonl`

4. 验证过程未中断长期训练任务
- 目标长期进程保持运行态（`R`）。

---

## 7. 使用建议

1. 若只想保证“至少有 train metrics”，可保持默认配置不改；短跑也会自动落盘。
2. 若希望更密集观察 SAC 训练曲线，可显式传：
- `--train_metrics_interval 100`（或更小）
3. 若遇到 SAC OOM，优先降低：
- `--num_envs`
- `memory.memory_size`
- `batch_size`

---

## 8. 后续可选改进（未在本轮实现）

1. 给 `scripts/sac/` 补充独立 `.md` 使用文档（训练脚本与评估脚本各一份）。
2. 在 `plot_metrics.py` 增加 SAC 专用图组模板（目前已具备通用列名容错能力）。
3. 增加自动 smoke CI（超小步数）覆盖 `train/eval/metrics` 基础链路。

