# SAC Baseline Docs

This section documents the existing SAC baseline implementation in this repository.

## Scope

- Training entry: `scripts/sac/train_quadruped_rough_sac.py`
- Evaluation entry: `scripts/sac/eval_quadruped_rough_sac.py`
- SAC utilities: `source/amadeus/amadeus/algorithms/sac/*`
- SAC agent config: `source/amadeus/amadeus/tasks/manager_based/quadruped_rough/agents/skrl_sac_cfg.yaml`

## Quick File Map

- Script docs index: [scripts/README.md](./scripts/README.md)
- Train script doc: [scripts/train_quadruped_rough_sac.md](./scripts/train_quadruped_rough_sac.md)
- Eval script doc: [scripts/eval_quadruped_rough_sac.md](./scripts/eval_quadruped_rough_sac.md)
- Config guide: [config_guide.md](./config_guide.md)

## Default Output Root

SAC training script currently defaults to:

- `outputs/train_quadruped_rough/sac/<run_name>/<timestamp>/`

With subdirectories:

- `ckpt/`
- `tb/`
- `metrics/`
- `data/train/`
- `data/eval/`
- `video/train/`
- `video/eval/`
- `params/`
- `logs/`

## One-Command Training (example)

```bash
/isaac-sim/python.sh scripts/sac/train_quadruped_rough_sac.py \
  --task Template-Amadeus-Quadruped-Rough-v0 \
  --play_task Template-Amadeus-Quadruped-Rough-Play-v0 \
  --headless \
  --device cuda:0 \
  --num_envs 256 \
  --max_steps 2000000 \
  --save_interval 20000 \
  --eval_interval 50000 \
  --eval_episodes 4 \
  --eval_num_envs 8 \
  --run_name amadeus_quadruped_rough_sac
```
