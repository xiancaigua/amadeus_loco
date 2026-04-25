# `export_rollouts.py` 脚本说明

## 1. 脚本定位
该脚本是 rollout 数据导出入口。它本身不直接做环境循环，而是组装参数后调用 `eval_quadruped_rough.py`，复用评估逻辑导出轨迹数据和指标。

定位为“离线数据导出快捷入口”。

## 2. 代码主流程
1. 解析导出参数（checkpoint、episodes、envs、seed、task 等）。
2. 推导输出目录：
   - 若未传 `--output_dir`，默认落到 run 目录下的 `datasets/rollouts/`。
   - 若 `--checkpoint` 位于 `<run_dir>/checkpoints/model_x.pt`，会自动回推到 `<run_dir>/datasets/rollouts/`。
3. 组装子进程命令，调用 `eval_quadruped_rough.py`：
   - `--dataset_dir <output_dir>`
   - `--metrics_dir <output_dir>/metrics`
   - 强制 `--headless`
4. 执行完成后在 stdout 打印实际输出目录。

## 3. 参数接口

| 参数 | 默认值 | 含义 | 常用建议 |
|---|---:|---|---|
| `--checkpoint` | 必填 | 策略 checkpoint 路径 | 指向 `model_x.pt` |
| `--num_episodes` | `16` | 导出 episode 数 | 离线数据可用 64~512 |
| `--num_envs` | `32` | 并行导出环境数 | 8~64 |
| `--seed` | `42` | 随机种子 | 固定便于可复现 |
| `--task` | `Template-Amadeus-Quadruped-Rough-Play-v0` | 导出任务 | 一般保持默认 |
| `--output_dir` | `None` | 导出目录 | 建议显式指定实验目录 |
| `--device` | `None` | 仿真设备 | 多卡环境可显式指定 |

## 4. 输出内容
`output_dir` 下典型内容：
- `eval_rollouts_*.hdf5`：轨迹数据分片
- `metrics/eval_metrics.csv`
- `metrics/eval_metrics.jsonl`

## 5. 推荐命令

```bash
/isaac-sim/python.sh scripts/baseline/export_rollouts.py \
  --checkpoint outputs/quadruped_rough_baseline/.../<run_name>/checkpoints/model_500.pt \
  --num_episodes 128 \
  --num_envs 32 \
  --seed 42 \
  --output_dir outputs/quadruped_rough_baseline/rollouts/model_500
```
