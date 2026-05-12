# How To Read Benchmark Results

This is a practical reading sequence for one benchmark run.

## 1) Recommended Reading Order

1. `RUN_SUMMARY.md`
2. `summary_metrics/case_summary.csv`
3. `plots/` (case bars then bucket bars)
4. `raw_metrics/<suite>/<case>/summary.json` for suspicious cases
5. `raw_metrics/<suite>/<case>/episodes.csv` to inspect failure distribution
6. `traces/<suite>/<case>/trace_env.csv` for time-series behavior
7. `videos/<suite>/<case>/` to confirm qualitative failure mode

## 2) Fast Interpretation Checklist

### Step A: Locate overall degradation

Compare across `bucket`:

- `mean_return`
- `fall_rate`
- `mean_episode_length`
- `mean_lin_vel_tracking_error`

If OOD bucket shows large return drop + fall-rate spike, robustness gap is confirmed.

### Step B: Distinguish failure type

Use per-case rows and traces:

- recovery weakness signature:
  - high `fall_rate`
  - short `mean_episode_length`
  - high `mean_recovery_time_s`
  - low `recovery_success_rate`
- command tracking boundary issue:
  - moderate fall-rate increase
  - clear rise in `mean_lin_vel_tracking_error` / `mean_yaw_vel_tracking_error`
- progressive instability:
  - traces show roll/pitch and tracking error growth before termination
- abrupt collapse:
  - traces and video show sudden failure after perturbation onset

### Step C: Map issue to benchmark family

- initialization sensitivity:
  - fails early with perturbed initial states
- disturbance recovery:
  - cannot recover from push schedule/magnitude changes
- terrain generalization:
  - behavior degrades as roughness/difficulty scales
- dynamics mismatch:
  - friction/mass/actuator shifts cause persistent instability
- observation mismatch:
  - delay/noise/drop causes policy lag and unstable corrections
- command distribution:
  - long-tail/OOD commands create tracking or stability failures
- combined OOD:
  - interaction effects exceed single-factor degradations

## 3) Which Files To Compare Across Experiments

For run-to-run comparison:

- `summary_metrics/case_summary.csv` (primary)
- `plots/bucket_bar_*.png` (quick)
- `reports/research_summary.md` (auto text)

For case deep-dive:

- `raw_metrics/<suite>/<case>/episodes.csv`
- `raw_metrics/<suite>/<case>/termination_stats.csv`
- `traces/<suite>/<case>/trace_env.csv`
- `videos/<suite>/<case>/*.mp4`

## 4) Mapping To Future Method Work (History/Memory/Skill)

Use these benchmark signals as method-selection evidence:

- high recovery-time + low recovery-success in disturbance cases:
  - prioritize memory/history for temporal recovery control
- strong degradation in observation delay/drop:
  - prioritize temporal filtering/history encoder
- combined OOD collapse with moderate single-factor degradation:
  - prioritize compositional skill abstractions and robustness conditioning
- command OOD tracking drift without immediate falls:
  - prioritize command-conditioned memory + stabilization regularization

## 5) Minimum Evidence Package For A Claim

For each claimed weakness, keep:

- 1 table entry from `case_summary.csv`
- 1 episode-level subset from `episodes.csv`
- 1 trace plot or trace snippet
- 1 representative failure video

This prevents over-claiming from aggregate numbers alone.
