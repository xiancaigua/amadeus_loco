# Quadruped Robustness Benchmark

This folder adds a reproducible robustness benchmark on top of the existing PPO baseline.

## Files

- `benchmark_quadruped_rough.py`
  - Runs scenario-based evaluation for the quadruped rough baseline.
  - Supports configurable distribution shifts (commands, pushes, friction, mass, terrain difficulty/scales).
  - Exports per-episode metrics, termination stats, traces, and optional videos.
- `configs/quadruped_rough_robustness.yaml`
  - Default scenario set with:
    - `id_reference`
    - `long_tail_push_command_terrain`
    - `ood_extreme_shift`
- `build_robustness_report.py`
  - Consolidates isolated scenario runs into one report folder with merged tables and plots.

## Typical Usage

Quantitative (no video):

```bash
/isaac-sim/python.sh scripts/benchmark/benchmark_quadruped_rough.py \
  --task Template-Amadeus-Quadruped-Rough-v0 \
  --checkpoint <ckpt> \
  --headless \
  --num_envs 16 \
  --num_episodes 16 \
  --no-video \
  --scenario_names id_reference \
  --benchmark_run_name benchmark_id \
  --output_root outputs/quadruped_rough_benchmark_isolated
```

Video evidence (single env):

```bash
/isaac-sim/python.sh scripts/benchmark/benchmark_quadruped_rough.py \
  --task Template-Amadeus-Quadruped-Rough-v0 \
  --checkpoint <ckpt> \
  --headless \
  --num_envs 1 \
  --num_episodes 3 \
  --video \
  --video_length 700 \
  --video_max_clips 3 \
  --video_start_step 20 \
  --scenario_names ood_extreme_shift \
  --benchmark_run_name benchmark_video_ood \
  --output_root outputs/quadruped_rough_benchmark_isolated
```

Build consolidated report:

```bash
/isaac-sim/python.sh scripts/benchmark/build_robustness_report.py \
  --id_run_dir <id_run_dir> \
  --long_tail_run_dir <long_tail_run_dir> \
  --ood_run_dir <ood_run_dir> \
  --id_video_run_dir <id_video_run_dir> \
  --long_tail_video_run_dir <long_tail_video_run_dir> \
  --ood_video_run_dir <ood_video_run_dir> \
  --out_dir outputs/quadruped_rough_benchmark_isolated/report_<date>
```

## Output Structure (per run)

```text
<run_dir>/
  config/
  scenarios/<scenario_name>/
    metrics/
      episodes.csv
      episodes.json
      summary.json
      termination_stats.csv
    traces/
      trace_env.csv
    videos/                # when --video
  summary/
    scenario_summary.csv
    scenario_summary.json
    index.json
  plots/
```

## Notes

- For stability, running scenarios in isolated processes is recommended (`--scenario_names` one at a time).
- `max_eval_steps` in YAML prevents endless runs in extreme shifts.
- Recorder manager is disabled in this benchmark path to reduce I/O overhead.
