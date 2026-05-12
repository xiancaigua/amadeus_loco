# SAC Config Guide

## Config File

- [source/amadeus/amadeus/tasks/manager_based/quadruped_rough/agents/skrl_sac_cfg.yaml](../../source/amadeus/amadeus/tasks/manager_based/quadruped_rough/agents/skrl_sac_cfg.yaml)

This is the Hydra entry (`skrl_sac_cfg_entry_point`) used by SAC train/eval scripts.

## Main Sections

### `seed`

- global seed default.

### `models`

- `policy`:
  - Gaussian actor, tanh output, hidden layers `[512, 256, 256]`, `elu`.
- `critic_1`, `critic_2`:
  - deterministic Q networks on `[obs, action]`.
- `target_critic_1`, `target_critic_2`:
  - target Q networks mirroring critic architecture.

### `memory`

- replay buffer (`RandomMemory`) with `memory_size: 1000000`.

### `agent` (SAC hyperparameters)

- `gradient_steps`
- `batch_size`
- `discount_factor` (gamma)
- `polyak` (target update rate tau)
- `learning_rate` (policy + critics)
- `random_timesteps`
- `learning_starts`
- `grad_norm_clip`
- entropy settings:
  - `learn_entropy`
  - `initial_entropy_value`
  - `target_entropy`
- preprocessing:
  - `observation_preprocessor: RunningStandardScaler`
- experiment logging:
  - directory/name/write interval/checkpoint interval

### `trainer`

- `timesteps`
- `headless`
- `disable_progressbar`

### `sac_runtime` (script-level runtime control defaults)

- `update_frequency`
- `save_interval`
- `train_metrics_interval`
- `eval_interval_steps`
- `eval_num_envs`
- `eval_episodes`

These are consumed by `train_quadruped_rough_sac.py` and can be overridden by CLI.

## Override Priority

Runtime precedence in training script:

1. CLI args
2. `sac_runtime` defaults
3. other config defaults (e.g. `trainer.timesteps`)

## Typical Knob Locations

- Change actor/critic network width/depth:
  - `models.policy.network.layers`
  - `models.critic_1.network.layers`
  - `models.critic_2.network.layers`
- Change replay buffer size:
  - `memory.memory_size`
- Change SAC optimization:
  - `agent.batch_size`
  - `agent.gradient_steps`
  - `agent.learning_rate`
  - `agent.discount_factor`
  - `agent.polyak`
- Change periodic eval frequency:
  - `sac_runtime.eval_interval_steps`
- Change checkpoint save interval:
  - `sac_runtime.save_interval`
