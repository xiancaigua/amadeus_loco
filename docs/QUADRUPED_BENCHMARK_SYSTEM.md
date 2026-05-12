# Quadruped Rough Benchmark System (PPO Baseline)

## 1. Existing capability audit (before this refactor)

- Baseline training/eval already available:
  - `scripts/baseline/train_quadruped_rough.py`
  - `scripts/baseline/eval_quadruped_rough.py`
- Existing benchmark existed but was scenario-centric and monolithic:
  - `scripts/benchmark/benchmark_quadruped_rough.py`
  - `scripts/benchmark/configs/quadruped_rough_robustness.yaml`
- Existing output artifacts:
  - `outputs/quadruped_rough_benchmark_isolated/*`
  - `outputs/quadruped_rough_benchmark/*`

## 2. Gaps identified

- Missing suite-oriented benchmark abstraction for:
  - initialization sensitivity
  - disturbance recovery pattern/timing
  - terrain generalization
  - dynamics mismatch
  - observation mismatch
  - command long-tail/OOD
  - representative combined OOD
- Missing clear modular layering (config/overrides/runtime/report) for reuse by future methods.
- Limited explicit support for runtime observation mismatch (delay/drop) and runtime push schedules.

## 3. Implemented benchmark modules

- `source/amadeus/amadeus/benchmarking/quadruped_rough/config.py`
  - suite/case schema
  - loader with backward compatibility (`suites` and legacy `scenarios`)
- `source/amadeus/amadeus/benchmarking/quadruped_rough/overrides.py`
  - env_cfg override applicator:
    - command / push / friction / mass / com / init / terrain / observation / dynamics
- `source/amadeus/amadeus/benchmarking/quadruped_rough/runtime.py`
  - runtime observation mismatch:
    - delay / additive noise / drop-frame
  - runtime push scheduler:
    - single / periodic / random_interval
- `source/amadeus/amadeus/benchmarking/quadruped_rough/outputs.py`
  - unified output layout helper
- `source/amadeus/amadeus/benchmarking/quadruped_rough/analysis.py`
  - summary table export
  - plots:
    - case bars
    - bucket bars
    - combined OOD heatmap (when grid metadata exists)
  - research summary markdown exporter

## 4. Runner and configs

- New runner:
  - `scripts/benchmark/run_quadruped_rough_benchmark.py`
  - supports:
    - `--suite_names`
    - `--case_names`
    - `--list_cases`
    - `--checkpoint`
    - `--num_envs`, `--num_episodes`, `--seed`, `--max_eval_steps`
    - `--video/--no-video`
    - `--output_root`
- New suite config:
  - `scripts/benchmark/configs/quadruped_rough_benchmark_suites.yaml`
  - includes 7 benchmark buckets and ID/long-tail/OOD case levels.

## 5. Output structure

```text
<run_dir>/
  config_snapshot/
  raw_metrics/<suite>/<case>/
  traces/<suite>/<case>/
  videos/<suite>/<case>/
  summary_metrics/
  plots/
  reports/
```

## 6. Verification

- compile check:
  - `/isaac-sim/python.sh -m compileall source/amadeus/amadeus scripts/benchmark`
- lightweight case list check:
  - `/isaac-sim/python.sh scripts/benchmark/run_quadruped_rough_benchmark.py --list_cases --benchmark_cfg scripts/benchmark/configs/quadruped_rough_benchmark_suites.yaml`

## 7. Deferred items (explicit)

- Full phase-aware gait push timing:
  - currently approximated by step-based scheduling (single/periodic/random interval).
- Terrain transition within episode:
  - currently represented by per-case terrain distribution shifts.
- Exhaustive multi-factor grid search:
  - currently constrained to representative combined OOD cases for tractability.
