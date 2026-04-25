# `train_quadruped_rough.py` 脚本说明

## 1. 脚本定位
该脚本是 baseline 主训练入口，负责在 manager-based 四足 rough terrain 任务上执行 RSL-RL PPO 训练，并且在训练过程中周期性触发评估脚本。

核心能力：
- 启动 `Template-Amadeus-Quadruped-Rough-v0` 训练环境
- 配置并运行 `OnPolicyRunner`（PPO）
- 记录 TensorBoard + JSONL/CSV 指标
- 导出训练阶段 HDF5 数据集
- 每 `eval_interval` 迭代自动调用评估脚本并导出评估视频

## 2. 代码主流程（从上到下）
1. 解析命令行参数（本脚本参数 + RSL-RL 参数 + AppLauncher 参数）。
2. 启动 Isaac Sim app（支持 headless）。
3. 通过 Hydra 加载：
   - 环境配置：`amadeus.tasks.manager_based.quadruped_rough.env_cfg:AmadeusQuadrupedRoughEnvCfg`
   - Agent 配置：`...rsl_rl_ppo_cfg:AmadeusQuadrupedRoughPPORunnerCfg`
4. 根据 CLI 覆盖配置（如 `num_envs`、`max_iterations`、`device`）。
5. 准备 run 目录，设置：
   - `checkpoints/`
   - `tensorboard/`
   - `logs/`
   - `datasets/train/`
   - `datasets/eval/`
   - `metrics/`
   - `videos/train/`
   - `videos/eval/`
   - `params/`
6. 创建环境并按需套 `gym.wrappers.RecordVideo`（训练视频）。
7. 创建 `OnPolicyRunner`，并包装 logger，额外写入 `train_metrics.csv/jsonl`。
8. 如配置了 resume，则加载 checkpoint。
9. 分 chunk 训练：每次训练 `min(eval_interval, remaining)` 个迭代。
10. 每个 chunk 结束后调用评估脚本 `eval_quadruped_rough.py`：
    - 输入最新 checkpoint
    - 输出评估指标与短视频
11. 训练结束，关闭环境与仿真 app。

## 3. 参数接口（本脚本直接定义）

| 参数 | 默认值 | 含义 | 常用建议 |
|---|---:|---|---|
| `--task` | `Template-Amadeus-Quadruped-Rough-v0` | 训练任务名 | 一般保持默认 |
| `--play_task` | `Template-Amadeus-Quadruped-Rough-Play-v0` | 周期评估任务名 | 一般保持默认 |
| `--agent` | `rsl_rl_cfg_entry_point` | agent 配置入口 | 一般保持默认 |
| `--num_envs` | `None` | 训练并行环境数 | baseline: 512~2048；冒烟: 32~128 |
| `--seed` | `None` | 随机种子 | 固定为 42 便于复现 |
| `--max_iterations` | `None` | PPO 总迭代数 | baseline 可先 1000~3000 |
| `--eval_interval` | `50` | 每 N 迭代自动评估 | baseline 建议 25~100 |
| `--eval_num_envs` | `32` | 评估并行环境数 | 4~32 |
| `--eval_episodes` | `8` | 每次评估 episode 数 | baseline 4~16 |
| `--eval_video_length` | `300` | 每次评估视频帧步数 | 80~300 足够观察 gait |
| `--eval_strict` | `False` | 周期评估失败时是否中断训练 | 默认关：评估失败会告警并继续训练 |
| `--dataset_chunk_episodes` | `128` | 每个 HDF5 shard 的 episode 数 | 64~256 |
| `--output_root` | `outputs/quadruped_rough_baseline` | 输出根目录 | 建议按实验分目录 |
| `--video` | `False` | 是否录制训练视频 | 通常关，减少开销 |
| `--video_interval` | `5000` | 训练视频触发步间隔 | 仅训练录像时使用 |
| `--video_length` | `200` | 训练视频长度 | 50~200 |
| `--distributed` | `False` | 分布式训练开关 | 单机通常关 |

补充参数来源：
- `cli_args.add_rsl_rl_args(parser)`：RSL-RL 通用参数（如 `--resume`、`--load_run`、`--checkpoint`、`--run_name` 等）。
- `AppLauncher.add_app_launcher_args(parser)`：Isaac Sim 启动参数（如 `--headless`、`--device`）。

## 4. 输出内容
以 `run_dir=<output_root>/rsl_rl/<experiment>/<timestamp_run>/` 为例：
- `checkpoints/model_*.pt`：checkpoint
- `tensorboard/events.out.tfevents.*`：TensorBoard
- `logs/git/*.diff`：git 状态与 diff 快照
- `metrics/train_metrics.csv`、`metrics/train_metrics.jsonl`：训练指标
- `metrics/eval_metrics.csv`、`metrics/eval_metrics.jsonl`：评估指标
- `datasets/train/*.hdf5`：训练阶段同步采集数据
- `datasets/eval/*.hdf5`：评估/rollout 阶段导出数据
- `videos/train/*.mp4`：训练阶段视频（启用 `--video` 时）
- `videos/eval/iter_xxxx/*.mp4`：周期评估视频
- `params/env.yaml`、`params/agent.yaml`：实际配置快照

## 5. 推荐命令
headless baseline（示例）：

```bash
/isaac-sim/python.sh scripts/baseline/train_quadruped_rough.py \
  --headless \
  --num_envs 1024 \
  --max_iterations 2000 \
  --eval_interval 50 \
  --eval_num_envs 16 \
  --eval_episodes 8 \
  --eval_video_length 160 \
  --dataset_chunk_episodes 128 \
  --seed 42 \
  --output_root outputs/quadruped_rough_baseline
```

## 6. 离线/弱联网注意事项
如果容器无法稳定访问远端 `IsaacLab` 资产（ANYmal USD 或 actuator net），可配置本地资产根目录：

```bash
export AMADEUS_ISAACLAB_ASSETS_ROOT=/path/to/isaaclab_assets/data
```

该目录下至少需要：
- `Robots/ANYbotics/ANYmal-C/anymal_c.usd`
- `ActuatorNets/ANYbotics/anydrive_3_lstm_jit.pt`

周期评估默认是“容错模式”：
- 先尝试“带视频评估”
- 失败后自动降级为“无视频评估”
- 若仍失败，默认继续训练（`--eval_strict` 未开启时）

resume（示例）：

```bash
/isaac-sim/python.sh scripts/baseline/train_quadruped_rough.py \
  --headless \
  --resume \
  --load_run <run_name> \
  --checkpoint <abs_or_rel_model_path> \
  --output_root outputs/quadruped_rough_baseline
```
