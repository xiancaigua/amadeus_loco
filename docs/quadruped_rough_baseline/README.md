# Quadruped Rough Baseline

This directory is the handoff hub for the Isaac Lab / Isaac Sim quadruped rough-terrain baseline added to the Amadeus external template.

## Audit Summary

- Background constraints came from the deleted-but-tracked `prj_background.md` and the live `prj_background_for_container.md`.
- The project is an Isaac Lab external template installed as editable package `amadeus`.
- Existing registered tasks before this change were:
  - `Template-Amadeus-Direct-v0`
  - `Template-Amadeus-Marl-Direct-v0`
  - `Template-Amadeus-v0`
- Existing manager-based task in the external project was a cartpole template only.
- Existing RL entry scripts are in `scripts/rsl_rl`, `scripts/rl_games`, `scripts/skrl`, `scripts/sb3`.
- The external project connects to Isaac Lab through installed packages and task registry import in `source/amadeus/amadeus/tasks/__init__.py`.
- Isaac Lab core already provides reusable manager-based quadruped rough locomotion tasks and configs in:
  - `/workspace/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/...`

## Baseline Decisions

- Robot: `ANYmal-C`
  - Reason: official manager-based rough locomotion config exists, official RSL-RL PPO config exists, and it is a conservative default for a first external-project baseline.
- RL backend: `RSL-RL PPO`
  - Reason: the external template already ships RSL-RL training/play scripts, checkpointing, TensorBoard logging, and official ANYmal-C rough config.
- Offline data collection: `RecorderManager` inside the manager-based env
  - Reason: lowest-intrusion hook point; it already exposes pre-step, post-step, pre-reset, and post-reset lifecycle hooks.
- Periodic evaluate: training orchestration calls standalone evaluator after every training chunk.
  - Reason: avoids patching Isaac Lab core or RSL-RL internals while remaining stable in headless Docker execution.
- Visualization/logging: TensorBoard remains primary; CSV/JSONL metrics are exported alongside it.

## New Files

- `source/amadeus/amadeus/tasks/manager_based/quadruped_rough/__init__.py`
  - Registers the new baseline train/play tasks.
- `source/amadeus/amadeus/tasks/manager_based/quadruped_rough/env_cfg.py`
  - Reuses official ANYmal-C rough env config and attaches recorder config.
- `source/amadeus/amadeus/tasks/manager_based/quadruped_rough/recorders.py`
  - Defines transition recorder terms and chunked HDF5 dataset export.
- `source/amadeus/amadeus/tasks/manager_based/quadruped_rough/agents/*`
  - RSL-RL PPO config and optional RL-Games config entry.
- `source/amadeus/amadeus/baselines/quadruped_rough/metrics.py`
  - CSV/JSONL metric export helpers.
- `scripts/baseline/train_quadruped_rough.py`
  - Headless-oriented training entry with checkpointing, periodic evaluation, dataset export, and metric export.
- `scripts/baseline/eval_quadruped_rough.py`
  - Standalone evaluation script with optional video recording.
- `scripts/baseline/export_rollouts.py`
  - Rollout export entry built on the evaluator path.
- `scripts/datasets/postprocess_quadruped_dataset.py`
  - Chunk-to-trajectory dataset summary postprocess.

## Runtime Output Layout

Default output root:

```text
outputs/quadruped_rough_baseline/
```

Per run layout:

```text
outputs/quadruped_rough_baseline/rsl_rl/amadeus_quadruped_rough/<timestamp_run>/
  model_*.pt
  params/
  git/
  events.out.tfevents.*
  datasets/
    train/
      train_dataset_0000.hdf5
      train_dataset_0001.hdf5
      ...
    eval/
      eval_rollouts_0000.hdf5
  videos/
    train/
    eval/
      iter_XXXX/
  metrics/
    train_metrics.csv
    train_metrics.jsonl
    eval_metrics.csv
    eval_metrics.jsonl
```

## Commands

Training:

```bash
/isaac-sim/python.sh scripts/baseline/train_quadruped_rough.py --headless
```

Training with explicit headless and custom env count:

```bash
/isaac-sim/python.sh scripts/baseline/train_quadruped_rough.py --headless --num_envs 2048
```

Resume training:

```bash
/isaac-sim/python.sh scripts/baseline/train_quadruped_rough.py --headless --resume --load_run <run_dir_name>
```

Evaluate checkpoint:

```bash
/isaac-sim/python.sh scripts/baseline/eval_quadruped_rough.py --headless --checkpoint <run_dir>/model_<iter>.pt
```

Export rollouts:

```bash
/isaac-sim/python.sh scripts/baseline/export_rollouts.py --checkpoint <run_dir>/model_<iter>.pt
```

Postprocess chunked dataset:

```bash
/isaac-sim/python.sh scripts/datasets/postprocess_quadruped_dataset.py \
  --input_dir <run_dir>/datasets/train \
  --output_path <run_dir>/datasets/train/trajectory_summary.json
```

TensorBoard:

```bash
/isaac-sim/python.sh -m tensorboard.main --logdir outputs/quadruped_rough_baseline/rsl_rl
```

## Extension Points

- History encoder / memory:
  - Insert after policy observation creation or inside the policy network config path.
- Skill abstraction:
  - Extend task-level commands and recorder schema first, then add policy-side latent heads.
- Offline RL:
  - Reuse the exported `datasets/train/*.hdf5` transition shards and `trajectory_summary.json`.
- Most likely files to extend next:
  - `env_cfg.py`
  - `recorders.py`
  - `train_quadruped_rough.py`
  - `eval_quadruped_rough.py`
