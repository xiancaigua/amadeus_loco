# Script Doc: train_quadruped_rough_sac.py

## Script Path

- [scripts/sac/train_quadruped_rough_sac.py](../../../scripts/sac/train_quadruped_rough_sac.py)

## One-Line Purpose

Train quadruped rough-terrain locomotion using SAC (skrl backend), with checkpointing, metrics export, periodic evaluation, and dataset export.

## What Problem It Solves

Provides an off-policy SAC training path in the existing external project without changing the PPO baseline path.

## Pipeline Stage

Primary SAC training entry script.

## Direct Dependencies

- SAC config (Hydra entry point):
  - [source/amadeus/amadeus/tasks/manager_based/quadruped_rough/agents/skrl_sac_cfg.yaml](../../../source/amadeus/amadeus/tasks/manager_based/quadruped_rough/agents/skrl_sac_cfg.yaml)
- Evaluation script called periodically:
  - [scripts/sac/eval_quadruped_rough_sac.py](../../../scripts/sac/eval_quadruped_rough_sac.py)
- SAC utility modules:
  - [source/amadeus/amadeus/algorithms/sac/utils.py](../../../source/amadeus/amadeus/algorithms/sac/utils.py)
  - [source/amadeus/amadeus/algorithms/sac/train_metrics.py](../../../source/amadeus/amadeus/algorithms/sac/train_metrics.py)
- Shared metrics writer:
  - `amadeus.baselines.quadruped_rough.metrics.MetricsWriter`
- Isaac Lab + skrl wrappers:
  - `SkrlVecEnvWrapper`, `Runner`

## Inputs

### CLI Inputs (important)

- Core:
  - `--task`
  - `--play_task`
  - `--agent` (default `skrl_sac_cfg_entry_point`)
  - `--headless`
  - `--device`
  - `--num_envs`
  - `--seed`
- Training duration / update:
  - `--max_steps`
  - `--update_frequency`
  - `--save_interval`
  - `--train_metrics_interval`
  - `--log_interval`
- Periodic eval:
  - `--eval_interval`
  - `--eval_episodes`
  - `--eval_num_envs`
  - `--eval_video_length`
  - `--eval_video_start_step`
  - camera and marker args (`--eval_follow_robot_camera`, `--eval_camera_*`, `--eval_show_velocity_markers`)
  - `--eval_strict`
- Resume:
  - `--resume`
  - `--checkpoint`
  - `--load_run`
- Output / logging:
  - `--output_root` (default `outputs/train_quadruped_rough`)
  - `--run_name`
  - `--video`, `--video_length`, `--video_interval`
  - `--dataset_chunk_episodes`

### Config Inputs

- `skrl_sac_cfg.yaml` defines:
  - actor/critic/target critic networks
  - replay memory size
  - SAC optimizer/hyperparameters
  - trainer timesteps
  - `sac_runtime` defaults (eval/save/update intervals)

## Outputs

Per run:

`<output_root>/sac/<run_name>/<timestamp>/`

- `ckpt/model_<step>.pt` checkpoints
- `tb/` TensorBoard logs (via skrl experiment config)
- `metrics/train_metrics.csv` + `train_metrics.jsonl`
- `metrics/eval_metrics.csv` + `eval_metrics.jsonl` (from periodic eval subprocess)
- `data/train/` offline dataset shards (recorder export)
- `data/eval/` eval rollout dataset shards
- `video/train/` training clips (if `--video`)
- `video/eval/iter_<step>/` periodic eval videos
- `params/env.yaml`, `params/agent.yaml`
- `logs/run_info.json`
- terminal heartbeat every `--log_interval` steps, with reward, episode length, losses, entropy coefficient, termination rates, speed, and ETA

## Internal Execution Flow

1. Parse CLI and launch Isaac app.
2. Resolve runtime values from CLI > `sac_runtime` > config defaults.
3. Build run directory and subfolders.
4. Set env recorder output path and dataset shard policy.
5. Save config snapshots (`env.yaml`, `agent.yaml`) and `run_info.json`.
6. Build gym env, optional training video wrapper.
7. Wrap env with `SkrlVecEnvWrapper`, create skrl `Runner`, get SAC agent.
8. Hook agent tracking writer to additionally write CSV/JSONL train metrics.
9. Optionally load resume checkpoint:
   - explicit `--checkpoint`, else latest from `ckpt/` (or old `checkpoints/` fallback).
10. Main loop for `max_steps`:
    - `act -> env.step -> record_transition`
    - update every `update_frequency` steps
    - print terminal heartbeat every `log_interval` steps
    - save checkpoint every `save_interval`
    - periodic eval subprocess every `eval_interval`
11. Ensure final metric row exists if periodic writer never fired.
12. Save final checkpoint at `model_<max_steps>.pt`.

## Calls To Other Modules

- `build_sac_train_metrics_row(...)` for metrics schema extraction
- `latest_model_checkpoint(...)`, `parse_model_step(...)` for resume
- subprocess call to `scripts/sac/eval_quadruped_rough_sac.py`

## Most Important Notes

- This script uses **skrl SAC** (not RSL-RL).
- Checkpoint naming convention is `model_<step>.pt`.
- Resume logic supports both new `ckpt/` and legacy `checkpoints/`.
- Periodic eval failure is tolerated by default; set `--eval_strict` to fail-fast.

## Common Failure Points / Pitfalls

- Wrong checkpoint type or task mismatch on resume.
- Too aggressive `num_envs` + long video settings can cause heavy runtime/memory load.
- If `--eval_interval` is enabled and assets/network have issues, periodic eval may fail (training continues unless `--eval_strict`).
- If `train_metrics_interval` is too large, only final fallback snapshot may be written.
