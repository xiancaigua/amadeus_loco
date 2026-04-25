# Baseline 交接闭环（`scripts/baseline`）

本目录下的脚本与介绍文档一一对应：

- `train_quadruped_rough.py` -> `train_quadruped_rough.md`
- `eval_quadruped_rough.py` -> `eval_quadruped_rough.md`
- `export_rollouts.py` -> `export_rollouts.md`

## 本次关键代码改动
- `eval_quadruped_rough.py`
  - 移除“达到 `video_length` 后提前退出评估循环”的逻辑。
  - 现在行为为：视频只录制短片段，但评估继续执行直到完成 `num_episodes`。
  - 目的：同时满足“保留短视频”和“得到有效 episode 级评估指标”。
- `train_quadruped_rough.py`
  - 周期评估改为两段式容错：
    - 先尝试带视频评估
    - 若失败自动重试无视频评估
    - 若仍失败且未开启 `--eval_strict`，训练继续进行并打印告警
  - 目的：避免评估子进程偶发失败导致长训练中断。
- `source/amadeus/amadeus/tasks/manager_based/quadruped_rough/env_cfg.py`
  - 新增本地资产覆盖环境变量：`AMADEUS_ISAACLAB_ASSETS_ROOT`
  - 可在离线/弱联网环境下将 ANYmal USD 和 actuator net 指向本地路径。
  - 新增默认本地资产目录回退：`assets/isaaclab_data`
  - 若该目录下存在完整文件，将自动优先使用本地资产，减少远端 S3 依赖。
- `scripts/baseline/eval_quadruped_rough.py`
  - 修复 `num_envs=1` 场景下的 `time_outs`/`dones` 0-dim 索引异常。
  - 现在对 `rewards/dones/time_outs` 统一 `reshape(-1)` 后再统计 episode 指标。

## 本地资产镜像（已落盘）
- `assets/isaaclab_data/Robots/ANYbotics/ANYmal-C/anymal_c.usd`
- `assets/isaaclab_data/Robots/ANYbotics/ANYmal-C/Props/instanceable_meshes.usd`
- `assets/isaaclab_data/ActuatorNets/ANYbotics/anydrive_3_lstm_jit.pt`

## 本次更接近真实冒烟记录
执行命令：

```bash
/isaac-sim/python.sh scripts/baseline/train_quadruped_rough.py \
  --headless \
  --num_envs 64 \
  --max_iterations 1 \
  --eval_interval 1 \
  --eval_num_envs 4 \
  --eval_episodes 1 \
  --eval_video_length 80 \
  --dataset_chunk_episodes 8 \
  --seed 42 \
  --output_root outputs/quadruped_rough_smoke
```

执行过程中的关键产物（清理前确认存在）：
- `checkpoints/model_0.pt`
- `datasets/train/train_dataset_*.hdf5`
- `datasets/eval/eval_rollouts_0000.hdf5`
- `metrics/train_metrics.csv/jsonl`
- `metrics/eval_metrics.csv/jsonl`
- `videos/eval/iter_0000/rl-video-step-0.mp4`

本次复跑评估结果摘要（来自评估 JSON）：
- `num_episodes = 1`
- `mean_episode_reward = 1.788787603378296`
- `mean_episode_length = 1000.0`
- `mean_lin_vel_tracking_error = 0.7770501375198364`
- `mean_yaw_vel_tracking_error = 0.9384233355522156`
- `fall_rate = 0.0`
- `timeout_rate = 1.0`

清理动作：

```bash
rm -rf /workspace/amadeus/amadeus/outputs/quadruped_rough_smoke
```

清理结果：`REMOVED`
