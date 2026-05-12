# Script Doc: run_quadruped_benchmark_isolated.py

## Script Path

- [scripts/benchmark/run_quadruped_benchmark_isolated.py](../../../scripts/benchmark/run_quadruped_benchmark_isolated.py)

## One-Line Purpose

Execute benchmark cases in isolated subprocesses (one case per process), then merge per-case summaries.

## Problem It Solves

Mitigates long single-process Isaac Sim instability by isolating each case execution boundary.

## Benchmark Stage

Batch orchestration stage for robust large-suite runs.

## Depends On

- Case definitions from suite YAML.
- Per-case runner:
  - [run_quadruped_rough_benchmark.py](../../../scripts/benchmark/run_quadruped_rough_benchmark.py)
- Post-merge report:
  - [generate_quadruped_benchmark_report.py](../../../scripts/benchmark/generate_quadruped_benchmark_report.py)

## Inputs

### CLI Inputs

- `--checkpoint` (required)
- `--task`, `--benchmark_cfg`
- filters: `--suite_names`, `--case_names`
- run output: `--output_root`, `--master_run_name`
- rollout controls: `--num_envs`, `--num_episodes`, `--max_eval_steps`, `--seed`, `--device`
- video controls: `--video/--no-video`, `--video_length`, `--video_start_step`, `--video_max_clips`
- behavior controls: `--headless`, `--stop_on_error`

### External Inputs

- valid `run_quadruped_rough_benchmark.py` script path
- valid checkpoint

## Outputs

Under `<output_root>/<master_run_name>/`:

- `RUN_SUMMARY.md`
- `selected_cases.json`
- `manifest.json`
- `runs/<index_suite__case>/...` (full per-case benchmark run outputs)
- `summary_metrics/case_summary.csv|json`
- `summary_metrics/failures.json`
- merged `plots/*` and `reports/*` (from report generator)

## Internal Execution Flow

1. Load YAML and resolve selected `(suite, case)` list.
2. Create master output directories.
3. Save selected case index.
4. For each selected case:
   - construct subprocess command invoking `run_quadruped_rough_benchmark.py`
   - pass suite/case filters and common overrides
   - execute subprocess
   - collect success/failure status
   - read per-case `summary_metrics/case_summary.csv` on success
5. Merge all successful case rows into master summary CSV/JSON.
6. Save failure list.
7. Invoke `generate_quadruped_benchmark_report.py` on master directory.
8. Save `manifest.json` and `RUN_SUMMARY.md`.

## Calls Out To

- `/isaac-sim/python.sh scripts/benchmark/run_quadruped_rough_benchmark.py ...`
- `/isaac-sim/python.sh scripts/benchmark/generate_quadruped_benchmark_report.py --run_dir <master_dir>`

## Most Important Operational Notes

- Per-case output directories are named:
  - `<index:02d>_<suite_name>__<case_name>`
- `failures.json` is authoritative for missing/broken cases.
- With `--stop_on_error`, batch stops on first failed case.

## Common Failure Points / Pitfalls

- If runner command path changes, orchestration fails.
- If a per-case run exits 0 but summary CSV is missing, case is marked as failed.
- Master summary includes only successful rows.
