# Script Doc: merge_quadruped_benchmark_runs.py

## Script Path

- [scripts/benchmark/merge_quadruped_benchmark_runs.py](../../../scripts/benchmark/merge_quadruped_benchmark_runs.py)

## One-Line Purpose

Merge multiple isolated benchmark masters into one deduplicated summary run.

## Problem It Solves

Allows stitching interrupted/resumed/repair batches into a single comparison-ready result set.

## Benchmark Stage

Post-processing merge stage.

## Depends On

- Source run directories with:
  - `runs/*/summary_metrics/case_summary.csv`

## Inputs

### CLI Inputs

- `--run_dirs` (required, comma-separated)
- `--out_dir` (required)

## Outputs

Under `--out_dir`:

- `RUN_SUMMARY.md`
- `merge_manifest.json`
- `summary_metrics/case_summary.csv`
- `summary_metrics/case_summary.json`
- empty `plots/` and `reports/` directories (prepared for report generation)

## Internal Execution Flow

1. Parse source run directories list.
2. Scan each run for per-case `summary_metrics/case_summary.csv`.
3. Concatenate all rows.
4. Deduplicate by `(suite_name, case_name)`; later rows overwrite earlier rows.
5. Write merged summary csv/json.
6. Write `merge_manifest.json`.
7. Write `RUN_SUMMARY.md`.

## Calls Out To

No subprocess calls; pure file processing.

## Most Important Operational Notes

- Dedup key is `(suite_name, case_name)`.
- If the same case appears in multiple sources, the row from the last processed source wins.

## Common Failure Points / Pitfalls

- Missing or malformed source summary CSVs lead to partial or empty merge.
- Merge does not automatically regenerate plots/report; run `generate_quadruped_benchmark_report.py` afterward.
