# Script Doc (Legacy): benchmark_quadruped_rough.py

## Script Path

- [scripts/benchmark/benchmark_quadruped_rough.py](../../../scripts/benchmark/benchmark_quadruped_rough.py)

## Status

Legacy path retained for backward compatibility.  
For new work, prefer `run_quadruped_rough_benchmark.py`.

## One-Line Purpose

Run earlier ID/long-tail/OOD scenario benchmark from `quadruped_rough_robustness.yaml`.

## Problem It Solves

Provides coarse three-group robustness checks with older scenario layout.

## Benchmark Stage

Legacy execution stage.

## Depends On

- [scripts/benchmark/configs/quadruped_rough_robustness.yaml](../../../scripts/benchmark/configs/quadruped_rough_robustness.yaml)
- RSL-RL PPO checkpoint
- Isaac Lab task registration

## Inputs

Main CLI patterns:

- `--checkpoint` (required)
- `--scenario_cfg`
- `--scenario_group` (`all`, `in_distribution`, `long_tail`, `ood`)
- `--scenario_names`
- rollout/video/trace overrides similar to structured runner

## Outputs

Under run directory (legacy style):

- `config/*`
- `scenarios/<scenario_name>/metrics/*`
- `scenarios/<scenario_name>/traces/*`
- `scenarios/<scenario_name>/videos/*` (if enabled)
- `summary/scenario_summary.csv|json`
- `plots/*`

## Internal Execution Flow

1. Load scenario config.
2. Select scenarios by group/name.
3. For each scenario:
   - apply env overrides
   - load checkpoint policy
   - rollout and collect metrics
   - optionally record video
4. Aggregate scenario summaries and plots.

## Why It Is Legacy

- Scenario model is less modular than suite/case structure.
- Output schema differs from current structured benchmark.
- New benchmark tooling (merge/report/docs) targets structured runner first.
