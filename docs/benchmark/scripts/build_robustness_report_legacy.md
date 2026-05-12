# Script Doc (Legacy): build_robustness_report.py

## Script Path

- [scripts/benchmark/build_robustness_report.py](../../../scripts/benchmark/build_robustness_report.py)

## Status

Legacy reporting utility for old isolated benchmark outputs.

## One-Line Purpose

Merge exactly three legacy runs (ID, long-tail, OOD) into one report directory.

## Problem It Solves

Creates a single comparison package from separately executed legacy scenario groups.

## Benchmark Stage

Legacy post-processing stage.

## Depends On

- three legacy run directories each containing:
  - `summary/scenario_summary.csv`

## Inputs

### CLI Inputs

- `--id_run_dir` (required)
- `--long_tail_run_dir` (required)
- `--ood_run_dir` (required)
- optional video roots:
  - `--id_video_run_dir`
  - `--long_tail_video_run_dir`
  - `--ood_video_run_dir`
- optional `--out_dir`

## Outputs

Under `<out_dir>`:

- `tables/scenario_summary.csv`
- `tables/scenario_delta_vs_id.csv`
- `plots/*.png`
- `video_index.json`
- `analysis_summary.md`
- `manifest.json`

## Internal Execution Flow

1. Read one-row summary CSV from each of ID/long-tail/OOD runs.
2. Convert key fields to numeric.
3. Compute deltas vs ID baseline.
4. Save summary and delta tables.
5. Save comparison plots.
6. Index optional video files.
7. Write markdown analysis and manifest.

## Most Important Operational Notes

- This script expects exactly one summary row per input run.
- It is designed for the old `quadruped_rough_benchmark_isolated` format.
