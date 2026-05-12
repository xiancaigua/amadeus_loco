# Benchmark Config Guide

This guide describes the **actual config system** used by the structured benchmark runner.

## 1) Primary Config File

Canonical suite config:

- [scripts/benchmark/configs/quadruped_rough_benchmark_suites.yaml](../../scripts/benchmark/configs/quadruped_rough_benchmark_suites.yaml)

Parsed by:

- [source/amadeus/amadeus/benchmarking/quadruped_rough/config.py](../../source/amadeus/amadeus/benchmarking/quadruped_rough/config.py)

## 2) Top-Level YAML Layout

The file has two major sections:

- `global`
- `suites`

### `global` section

Controls defaults for all cases:

- task / device / seed
- default `num_envs`, `num_episodes`, `max_eval_steps`
- trace settings (`trace.env_id`, `trace.max_steps`)
- recovery metrics thresholds (`recovery.trigger_error`, `clear_error`, `min_hold_steps`)
- video defaults (camera, clip length, marker settings)

### `suites` section

Map from `suite_name -> suite_config`.
Each suite contains:

- `description`
- `enabled`
- `cases` (list)

Each case contains:

- `name` (unique within suite)
- `bucket` (`id`, `long_tail`, `ood`)
- optional `description`, `tags`
- optional case-local `num_envs`, `num_episodes`, `seed`, `max_eval_steps`, `save_video`
- `overrides` (actual perturbation knobs)

## 3) How ID / Long-tail / OOD Is Expressed

The runner uses each case’s `bucket` field as semantic label:

- `bucket: id` -> in-distribution reference
- `bucket: long_tail` -> boundary/rare region
- `bucket: ood` -> out-of-distribution stress

No hidden rule infers these labels. They are explicitly declared per case in YAML.

## 4) Supported Override Groups (Current Implementation)

Applied by:

- [source/amadeus/amadeus/benchmarking/quadruped_rough/overrides.py](../../source/amadeus/amadeus/benchmarking/quadruped_rough/overrides.py)

Supported keys inside `case.overrides`:

- `command`
  - `lin_vel_x`, `lin_vel_y`, `ang_vel_z`, `heading`
  - `resampling_time_s`, `rel_standing_envs`, `rel_heading_envs`, `heading_command`, `heading_control_stiffness`
- `push`
  - static env-event push tuning (`enabled`, `interval_s`, velocity ranges)
- `friction`
  - static/dynamic/restitution ranges
- `mass`
  - `add_base_mass`
- `com`
  - COM x/y/z ranges
- `init`
  - base pose/velocity reset ranges
  - joint position/velocity reset ranges
- `terrain`
  - difficulty range
  - curriculum toggle
  - sub-terrain scales (`stairs_height_scale`, `boxes_height_scale`, `rough_noise_scale`, `slope_scale`)
  - `max_init_terrain_level`
- `observation`
  - corruption enable, noise scaling, optional height scan disable
- `dynamics`
  - actuator stiffness/damping scale
  - action scale

## 5) Runtime-Only Perturbation Groups

Parsed and executed at rollout runtime by:

- [source/amadeus/amadeus/benchmarking/quadruped_rough/runtime.py](../../source/amadeus/amadeus/benchmarking/quadruped_rough/runtime.py)

Use these in `case.overrides`:

- `observation_runtime`
  - `delay_steps`
  - `additive_noise_std`
  - `drop_prob`
- `push_runtime`
  - `enabled`
  - `pattern`: `single` / `periodic` / `random_interval`
  - timing and magnitude controls:
    - `start_step`, `interval_steps`, `random_interval_range`, `duration_steps`
    - `magnitude_range`, `direction_deg_range`, `env_fraction`

## 6) Selection/Filtering Behavior

Selection logic:

- [select_cases(...) in config.py](../../source/amadeus/amadeus/benchmarking/quadruped_rough/config.py)

CLI filters:

- `--suite_names a,b,c`
- `--case_names x,y,z`

If no suite filter is passed, only suites with `enabled: true` are selected.

## 7) Minimal Example: Add One New Case

Add in `suites.command_distribution.cases`:

```yaml
- name: cmd_tail_diag_test
  bucket: long_tail
  description: "Boundary command stress for debugging."
  overrides:
    command:
      lin_vel_x: [-1.3, 1.3]
      lin_vel_y: [-1.3, 1.3]
      ang_vel_z: [-1.3, 1.3]
      resampling_time_s: [3.0, 5.0]
```

Run only this case:

```bash
/isaac-sim/python.sh scripts/benchmark/run_quadruped_rough_benchmark.py \
  --task Template-Amadeus-Quadruped-Rough-v0 \
  --checkpoint <ckpt> \
  --headless \
  --case_names cmd_tail_diag_test
```

## 8) Case Override Precedence

Order used by runner:

1. Start from base environment config loaded by Hydra task.
2. Apply case `overrides`.
3. Apply CLI-level runtime overrides (e.g. `--num_envs`, `--num_episodes`, `--seed`, video args).

CLI arguments can override per-case/global defaults for run-time control.
