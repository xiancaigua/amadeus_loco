# Script Doc: eval_quadruped_rough_sac.py

## Script Path

- [scripts/sac/eval_quadruped_rough_sac.py](../../../scripts/sac/eval_quadruped_rough_sac.py)

## One-Line Purpose

Load a trained SAC checkpoint, run evaluation episodes, optionally record video, and export eval metrics.

## What Problem It Solves

Provides standalone validation for SAC checkpoints and serves as periodic-eval backend called by training.

## Pipeline Stage

Evaluation stage (both manual and periodic).

## Direct Dependencies

- SAC utility module:
  - [source/amadeus/amadeus/algorithms/sac/utils.py](../../../source/amadeus/amadeus/algorithms/sac/utils.py)
- skrl runner/wrapper:
  - `Runner`, `SkrlVecEnvWrapper`
- Task registration:
  - `amadeus.tasks`

## Inputs

### Required CLI

- `--checkpoint`

### Frequent CLI

- `--task` (default play task)
- `--agent`
- `--num_envs`
- `--num_episodes`
- `--seed`
- `--headless`
- `--device`

### Video / Camera CLI

- `--video_folder`
- `--video_length`
- `--video_start_step`
- `--video_interval_steps`
- `--max_video_clips`
- `--camera_eye`, `--camera_lookat`
- `--follow_robot_camera`
- `--camera_offset`
- `--camera_robot_env_id`
- `--show_velocity_markers`
- marker options (`--velocity_marker_*`)

### Export CLI

- `--metrics_dir`
- `--dataset_dir` (optional; otherwise inferred from checkpoint run dir)

## Outputs

- Terminal JSON summary of eval result
- If `--metrics_dir` set:
  - `eval_metrics.csv`
  - `eval_metrics.jsonl`
- If recorder enabled:
  - eval rollout dataset under `--dataset_dir` (or inferred default)
- If `--video_folder` set:
  - video clips (e.g. `rl-video-step-<n>.mp4`)

## Internal Execution Flow

1. Parse CLI and launch Isaac app.
2. Resolve checkpoint absolute path and infer run dir.
3. Apply eval env settings (`num_envs`, `seed`, device).
4. Set eval dataset export dir (explicit or inferred).
5. Create env (with render mode if video).
6. Optional camera/viewer setup and `RecordVideo` wrapping.
7. Wrap env with `SkrlVecEnvWrapper`.
8. Build runner, load checkpoint, disable training mode.
9. Rollout until `num_episodes` are completed.
10. Compute aggregate metrics:
    - mean reward, episode length
    - lin/yaw tracking errors
    - fall/timeout rates
11. Print JSON result and write eval metrics files if requested.

## Camera Behavior Notes

- Supports fixed camera and follow-robot camera.
- Follow camera updates every step when enabled.
- Includes fallback path if one camera API is unavailable.

## Common Failure Points / Pitfalls

- Invalid checkpoint path or incompatible checkpoint format.
- Missing assets during env creation can fail before evaluation loop.
- Multi-env eval with video may produce distant views unless camera params are tuned.
- If `--metrics_dir` is omitted, results are printed but not persisted to csv/jsonl.
