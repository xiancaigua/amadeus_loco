# Experiment Overview Template

Use this template for short run handoff notes.  
Current benchmark scripts now auto-generate a lightweight version as `RUN_SUMMARY.md`.

## Template

```markdown
# Benchmark Run Summary

## Overview
- Benchmark type / suite:
- Run directory:
- Checkpoint:
- Task:
- Seed(s):
- Num envs:
- Episodes per case:

## Case Coverage
- Number of cases:
- Buckets covered: id / long_tail / ood
- Case list:

## Key Outputs
- Summary CSV:
- Raw metrics root:
- Traces root:
- Videos root:
- Plots root:
- Research summary:

## Quick Metrics
- Mean return across cases:
- Mean fall rate across cases:
- Mean episode length across cases:

## One-line Conclusion
- (fill after review)
```

## Where It Is Auto-Generated

- `scripts/benchmark/run_quadruped_rough_benchmark.py`
  - writes `<run_dir>/RUN_SUMMARY.md`
- `scripts/benchmark/run_quadruped_benchmark_isolated.py`
  - writes `<master_dir>/RUN_SUMMARY.md`
- `scripts/benchmark/merge_quadruped_benchmark_runs.py`
  - writes `<merged_dir>/RUN_SUMMARY.md`

## Notes

- Auto-generated summary is intentionally concise.
- For exact reproducibility, always pair it with:
  - `config_snapshot/runtime_args.json`
  - `config_snapshot/selected_cases.json`
  - `summary_metrics/case_summary.csv`
