# Quadruped Robustness Benchmark

Detailed engineering documentation for this benchmark stack now lives in:

- [`docs/benchmark/README.md`](../../docs/benchmark/README.md)

This folder provides two benchmark paths on top of the existing PPO baseline:

- `run_quadruped_rough_benchmark.py` (recommended, suite/case modular benchmark)
- `benchmark_quadruped_rough.py` (legacy scenario runner kept for backward compatibility)

## New Structured Runner

### Core files

- `run_quadruped_rough_benchmark.py`
  - Configuration-driven benchmark runner with batch suite execution.
  - Supports runtime perturbations (push schedule and observation delay/noise/drop).
  - Exports raw metrics, traces, videos, summary tables, plots, and research summary.
- `configs/quadruped_rough_benchmark_suites.yaml`
  - Canonical benchmark suites:
    - initialization_sensitivity
    - disturbance_recovery
    - terrain_generalization
    - dynamics_mismatch
    - observation_mismatch
    - command_distribution
    - combined_ood

### List suites/cases

```bash
/isaac-sim/python.sh scripts/benchmark/run_quadruped_rough_benchmark.py \
  --benchmark_cfg scripts/benchmark/configs/quadruped_rough_benchmark_suites.yaml \
  --list_cases
```

### Run a full suite set

```bash
/isaac-sim/python.sh scripts/benchmark/run_quadruped_rough_benchmark.py \
  --task Template-Amadeus-Quadruped-Rough-v0 \
  --checkpoint <ckpt> \
  --headless \
  --suite_names command_distribution,disturbance_recovery,terrain_generalization \
  --num_envs 32 \
  --num_episodes 32 \
  --output_root outputs/quadruped_rough_benchmark
```

### Run one case with longer video evidence

```bash
/isaac-sim/python.sh scripts/benchmark/run_quadruped_rough_benchmark.py \
  --task Template-Amadeus-Quadruped-Rough-v0 \
  --checkpoint <ckpt> \
  --headless \
  --case_names combo_ood_grid_2 \
  --num_envs 1 \
  --num_episodes 2 \
  --video \
  --video_length 1500 \
  --video_max_clips 2 \
  --output_root outputs/quadruped_rough_benchmark
```

## Output structure (new runner)

```text
<run_dir>/
  config_snapshot/
    env_base.yaml
    agent_base.yaml
    runtime_args.json
    benchmark_cfg_resolved.json
    selected_cases.json
  raw_metrics/
    <suite>/<case>/
      episodes.csv
      episodes.json
      summary.json
      termination_stats.csv
      case_overrides_snapshot.json
  traces/
    <suite>/<case>/trace_env.csv
  videos/
    <suite>/<case>/*.mp4
  summary_metrics/
    case_summary.csv
    case_summary.json
    index.json
  plots/
    case_bar_*.png
    bucket_bar_*.png
    combined_ood_fall_rate_heatmap.png (if grid metadata exists)
  reports/
    research_summary.md
    manifest.json
```

## Legacy tools

- `benchmark_quadruped_rough.py`: previous ID/long-tail/OOD scenario benchmark runner.
- `build_robustness_report.py`: report merger for isolated legacy runs.

## Regenerate plots/report from existing run

```bash
/isaac-sim/python.sh scripts/benchmark/generate_quadruped_benchmark_report.py \
  --run_dir <benchmark_run_dir>
```
