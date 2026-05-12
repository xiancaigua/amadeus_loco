# Script Doc: generate_quadruped_benchmark_report.py

## Script Path

- [scripts/benchmark/generate_quadruped_benchmark_report.py](../../../scripts/benchmark/generate_quadruped_benchmark_report.py)

## One-Line Purpose

Regenerate plots and summary markdown from an existing benchmark summary CSV.

## Problem It Solves

Decouples expensive simulation from light report generation, so you can rebuild plots/reports without rerunning environments.

## Benchmark Stage

Post-processing report stage.

## Depends On

- Existing `summary_metrics/case_summary.csv` in a benchmark run directory.

## Inputs

### CLI Inputs

- `--run_dir` (required)
- `--summary_csv` (optional override; default is `<run_dir>/summary_metrics/case_summary.csv`)

## Outputs

Under `<run_dir>`:

- `plots/case_bar_*.png`
- `plots/bucket_bar_*.png`
- `reports/research_summary.md`
- `reports/manifest.json`

## Internal Execution Flow

1. Read summary CSV rows.
2. Build case-level bar charts (`mean_return`, `episode_length`, `fall_rate`, tracking error).
3. Build bucket-level bar charts.
4. Auto-generate compact vulnerability summary markdown.
5. Save report manifest.

## Calls Out To

No subprocess calls; pure Python plotting and file I/O.

## Most Important Operational Notes

- This script is intentionally standalone (no Isaac Sim app bootstrap).
- Works for both single-run and merged summaries if CSV schema is compatible.

## Common Failure Points / Pitfalls

- Missing summary CSV -> hard failure.
- If required numeric fields are absent or malformed, some plots may show `nan` bars.
