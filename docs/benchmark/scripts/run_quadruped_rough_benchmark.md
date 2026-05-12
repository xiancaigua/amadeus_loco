# Script Doc: run_quadruped_rough_benchmark.py

## Script Path

- [scripts/benchmark/run_quadruped_rough_benchmark.py](../../../scripts/benchmark/run_quadruped_rough_benchmark.py)

## One-Line Purpose

Run structured suite/case benchmark evaluation for one PPO checkpoint and export metrics/videos/traces/summaries.

## Problem It Solves

Turns a checkpoint + benchmark config into reproducible robustness evidence across multiple controlled cases, without changing training logic.

## Benchmark Stage

Primary execution stage (single-process runner).

## Depends On

- YAML suite config:
  - [scripts/benchmark/configs/quadruped_rough_benchmark_suites.yaml](../../../scripts/benchmark/configs/quadruped_rough_benchmark_suites.yaml)
- Benchmark modules:
  - [config.py](../../../source/amadeus/amadeus/benchmarking/quadruped_rough/config.py)
  - [overrides.py](../../../source/amadeus/amadeus/benchmarking/quadruped_rough/overrides.py)
  - [runtime.py](../../../source/amadeus/amadeus/benchmarking/quadruped_rough/runtime.py)
  - [analysis.py](../../../source/amadeus/amadeus/benchmarking/quadruped_rough/analysis.py)
  - [outputs.py](../../../source/amadeus/amadeus/benchmarking/quadruped_rough/outputs.py)
- Isaac Lab task registration (`amadeus.tasks`)
- RSL-RL runner (`OnPolicyRunner`)

## Inputs

### CLI Inputs (main)

- `--checkpoint` (required)
- `--task`, `--agent`
- `--benchmark_cfg`
- `--suite_names`, `--case_names`, `--list_cases`
- `--output_root`, `--benchmark_run_name`
- runtime overrides: `--num_envs`, `--num_episodes`, `--seed`, `--max_eval_steps`
- video overrides: `--video/--no-video`, `--video_length`, `--video_start_step`, `--video_max_clips`
- trace override: `--trace_env_id`, `--trace_max_steps`
- report: `--save_plots/--no-save_plots`
- app/runtime: `--headless`, `--device`, RSL-RL CLI args

### Config Inputs

- Parsed `global` defaults and per-case `overrides` from suite YAML.

### Runtime Inputs

- Loaded checkpoint policy weights.
- Env state streams during rollouts.

## Outputs

Under `<output_root>/<run_name>/`:

- `RUN_SUMMARY.md`
- `config_snapshot/*`
- `raw_metrics/<suite>/<case>/*`
- `traces/<suite>/<case>/trace_env.csv`
- `videos/<suite>/<case>/*.mp4` (if enabled)
- `summary_metrics/case_summary.csv|json`
- `summary_metrics/index.json`
- `plots/*` and `reports/*` (if `--save_plots`)

## Internal Execution Flow

1. Parse CLI, bootstrap Isaac Sim app.
2. Load benchmark config and select cases.
3. Resolve checkpoint path.
4. Create run directory layout.
5. Save base config snapshots and selected case list.
6. For each case:
   - deep-copy env cfg
   - apply static overrides (`apply_env_cfg_overrides`)
   - resolve num_envs/episodes/seed/video settings
   - create env (`gym.make`)
   - optional camera + `RecordVideo` wrapping
   - wrap with `RslRlVecEnvWrapper`
   - load policy via `OnPolicyRunner`
   - create runtime perturbators:
     - `ObservationPerturbator`
     - `RuntimePushScheduler`
   - rollout loop until enough episodes or max steps
   - collect episode metrics, termination stats, traces
   - save case raw artifacts
7. Aggregate all case summaries.
8. Save summary CSV/JSON + plots/reports.
9. Save `summary_metrics/index.json` and top-level `RUN_SUMMARY.md`.

## Calls Out To

- `load_benchmark_config`, `select_cases`
- `apply_env_cfg_overrides`
- `ObservationPerturbator`, `RuntimePushScheduler`
- `save_summary_artifacts`

## Most Important Operational Notes

- `--checkpoint` must be valid and compatible with selected task.
- Video mode can force smaller env/episode settings (from video config) unless explicitly overridden.
- `--list_cases` exits before app-heavy evaluation.

## Common Failure Points / Pitfalls

- Asset path/network issues during env creation (USD not found) can fail run before rollout.
- Mismatch between checkpoint and task/obs/action shape causes load/inference errors.
- Very large `num_envs * num_episodes` with long videos can be slow/heavy.
- Runtime observation perturbation now supports tensor-like dict observations; older outputs may have failed this path.
