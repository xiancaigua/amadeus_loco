# Quadruped Rough Benchmark Docs

This document set explains the **existing benchmark implementation** in this repository.  
Scope is limited to understanding, operating, and maintaining the current code and outputs.

## 1) Benchmark Goal

The benchmark evaluates an already-trained PPO locomotion checkpoint on structured robustness buckets:

- initialization sensitivity
- disturbance recovery
- terrain generalization
- dynamics mismatch
- observation mismatch
- command distribution (ID / long-tail / OOD)
- combined OOD

The objective is not training new methods; it is to produce reproducible evidence of where the baseline degrades.

## 2) Where The Benchmark Code Lives

- Main scripts: [scripts/benchmark](../../scripts/benchmark)
- Benchmark support package: [source/amadeus/amadeus/benchmarking/quadruped_rough](../../source/amadeus/amadeus/benchmarking/quadruped_rough)
- Canonical suite config: [scripts/benchmark/configs/quadruped_rough_benchmark_suites.yaml](../../scripts/benchmark/configs/quadruped_rough_benchmark_suites.yaml)

## 3) Core Modules

- `config.py`: parses YAML suites/cases into typed config objects.
- `overrides.py`: applies case overrides into `ManagerBasedRLEnvCfg`.
- `runtime.py`: runtime perturbations (push scheduler, observation delay/noise/drop).
- `analysis.py`: writes summary csv/json, plots, research summary markdown.
- `outputs.py`: standard output folder layout helper.

## 4) Main Script Roles

- `run_quadruped_rough_benchmark.py`
  - Single-process structured runner.
  - Executes selected suite/case set, writes raw metrics + summaries + plots + report.
- `run_quadruped_benchmark_isolated.py`
  - Runs each case in a separate subprocess for stability, merges at master level.
- `merge_quadruped_benchmark_runs.py`
  - Merges multiple isolated masters into one deduplicated summary.
- `generate_quadruped_benchmark_report.py`
  - Regenerates plots/report from an existing `summary_metrics/case_summary.csv`.

Legacy scripts are still present and documented separately:

- `benchmark_quadruped_rough.py`
- `build_robustness_report.py`

## 5) End-to-End Flow (Checkpoint -> Outputs)

1. Load benchmark suite YAML.
2. Select cases by suite/case filter.
3. For each case:
   - clone base env cfg
   - apply static overrides (`overrides.py`)
   - create env and load checkpoint policy
   - apply runtime perturbations (`runtime.py`) during rollout
   - collect per-episode metrics + traces + termination stats
   - optionally record video
4. Aggregate case summaries.
5. Save summary tables, plots, research markdown.
6. Save `RUN_SUMMARY.md` for fast run triage.

## 6) Current Real Output Roots

Structured benchmark outputs currently under:

- [outputs/quadruped_rough_benchmark](../../outputs/quadruped_rough_benchmark)

Legacy isolated outputs currently under:

- [outputs/quadruped_rough_benchmark_isolated](../../outputs/quadruped_rough_benchmark_isolated)

Detailed output semantics: [outputs_guide.md](./outputs_guide.md)

## 7) Reading Order Recommendation

1. [outputs_guide.md](./outputs_guide.md)  
2. [config_guide.md](./config_guide.md)  
3. [scripts/run_quadruped_rough_benchmark.md](./scripts/run_quadruped_rough_benchmark.md)  
4. [how_to_read_results.md](./how_to_read_results.md)

## 8) Script Docs Index

- [scripts/run_quadruped_rough_benchmark.md](./scripts/run_quadruped_rough_benchmark.md)
- [scripts/run_quadruped_benchmark_isolated.md](./scripts/run_quadruped_benchmark_isolated.md)
- [scripts/merge_quadruped_benchmark_runs.md](./scripts/merge_quadruped_benchmark_runs.md)
- [scripts/generate_quadruped_benchmark_report.md](./scripts/generate_quadruped_benchmark_report.md)
- [scripts/benchmark_quadruped_rough_legacy.md](./scripts/benchmark_quadruped_rough_legacy.md)
- [scripts/build_robustness_report_legacy.md](./scripts/build_robustness_report_legacy.md)

## 9) Experiment Overview Template

- [experiment_overview_template.md](./experiment_overview_template.md)

This template is now partially automated: benchmark scripts generate `RUN_SUMMARY.md` in run directories.
