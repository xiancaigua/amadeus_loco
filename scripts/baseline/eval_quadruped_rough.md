# `eval_quadruped_rough.py` 脚本说明

## 1. 脚本定位
该脚本是统一评估入口，用于加载训练得到的 checkpoint，在 manager-based rough terrain 环境中执行策略推理，输出评估指标，并可选导出短视频与评估数据集。

该脚本既可单独调用，也被 `train_quadruped_rough.py` 周期性自动调用。

## 2. 代码主流程
1. 解析参数（含 RSL-RL 和 AppLauncher 参数）。
2. 检查 `--checkpoint` 必填。
3. 若指定 `--video_folder`，自动开启摄像头渲染能力。
4. 通过 Hydra 加载 play 环境配置与 agent 配置。
5. 根据 CLI 覆盖 `num_envs`、`seed`、`device`，并设置 recorder 导出目录。
6. 创建环境（可选套 `RecordVideo`，只在 step=0 触发一次短视频采样）。
7. 创建 `OnPolicyRunner` 并加载 checkpoint，提取推理策略。
8. 循环执行策略：
   - 收集 `reward/length` 与速度跟踪误差
   - 按 `done` 统计已完成 episode 的 reward/length/fall/timeout
9. 达到 `num_episodes` 后汇总结果，打印 JSON，并可选写入 `eval_metrics.csv/jsonl`。
10. 关闭环境与仿真 app。

实现细节说明：
- 现在 `video_length` 只控制视频片段长度，不再提前中断评估循环；这样可以同时满足“保留短视频”和“跑完指定评估 episode 数”。

## 3. 参数接口

| 参数 | 默认值 | 含义 | 常用建议 |
|---|---:|---|---|
| `--checkpoint` | 必填 | 待评估模型路径 | 使用 `model_x.pt` |
| `--task` | `Template-Amadeus-Quadruped-Rough-Play-v0` | 评估任务名 | 一般保持默认 |
| `--agent` | `rsl_rl_cfg_entry_point` | agent 配置入口 | 一般保持默认 |
| `--num_envs` | `32` | 评估并行环境数 | 4~32 |
| `--num_episodes` | `8` | 需要完成的 episode 数 | 冒烟 1，正式 8~64 |
| `--seed` | `42` | 随机种子 | 固定用于可复现 |
| `--video_folder` | `None` | 视频输出目录 | 周期评估建议开启 |
| `--video_length` | `300` | 视频步数长度 | 80~300 |
| `--metrics_dir` | `None` | 指标输出目录 | 建议始终设置 |
| `--dataset_dir` | `None` | 评估 rollout 导出目录 | 离线数据导出时设置 |
| `--real-time` | `False` | 是否按真实时钟节奏运行 | 服务器评估通常关 |
| `--disable_fabric` | `False` | 关闭 fabric | 仅排障时使用 |

补充参数来源：
- `cli_args.add_rsl_rl_args(parser)`：RSL-RL 相关参数（包括 `--device` 等）。
- `AppLauncher.add_app_launcher_args(parser)`：Isaac Sim 启动参数（如 `--headless`）。

## 4. 输出内容
- 终端 JSON（评估摘要）：
  - `mean_episode_reward`
  - `mean_episode_length`
  - `mean_lin_vel_tracking_error`
  - `mean_yaw_vel_tracking_error`
  - `fall_rate`
  - `timeout_rate`
- 可选写盘：
  - `<metrics_dir>/eval_metrics.csv`
  - `<metrics_dir>/eval_metrics.jsonl`
  - `<video_folder>/*.mp4`
  - `<dataset_dir>/eval_rollouts_*.hdf5`

## 5. 推荐命令
单次评估 + 导出短视频：

```bash
/isaac-sim/python.sh scripts/baseline/eval_quadruped_rough.py \
  --headless \
  --checkpoint outputs/quadruped_rough_baseline/.../model_200.pt \
  --num_envs 8 \
  --num_episodes 8 \
  --video_folder outputs/quadruped_rough_baseline/eval_videos/run_200 \
  --video_length 160 \
  --metrics_dir outputs/quadruped_rough_baseline/eval_metrics \
  --seed 42
```

