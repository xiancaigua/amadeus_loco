# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.envs.mdp.events import push_by_setting_velocity


@dataclass
class ObservationMismatchRuntimeCfg:
    delay_steps: int = 0
    additive_noise_std: float = 0.0
    drop_prob: float = 0.0

    @classmethod
    def from_overrides(cls, overrides: dict) -> "ObservationMismatchRuntimeCfg":
        cfg = overrides.get("observation_runtime", {})
        return cls(
            delay_steps=int(cfg.get("delay_steps", 0)),
            additive_noise_std=float(cfg.get("additive_noise_std", 0.0)),
            drop_prob=float(cfg.get("drop_prob", 0.0)),
        )


class ObservationPerturbator:
    """Apply runtime observation mismatch (delay/noise/drop-frame) on policy inputs."""

    def __init__(self, cfg: ObservationMismatchRuntimeCfg):
        self.cfg = cfg
        self._history: deque[torch.Tensor] = deque(maxlen=max(1, cfg.delay_steps + 1))
        self._prev_output: torch.Tensor | None = None

    @property
    def enabled(self) -> bool:
        return self.cfg.delay_steps > 0 or self.cfg.additive_noise_std > 0.0 or self.cfg.drop_prob > 0.0

    def reset(self):
        self._history.clear()
        self._prev_output = None

    def _is_tensordict_like(self, obs) -> bool:
        return hasattr(obs, "items") and hasattr(obs, "clone") and not isinstance(obs, torch.Tensor)

    def _iter_tensor_items(self, obs_like):
        if not self._is_tensordict_like(obs_like):
            return []
        return [(k, v) for k, v in obs_like.items() if isinstance(v, torch.Tensor)]

    def _apply_noise_drop_to_tensor(self, tensor: torch.Tensor, prev_tensor: torch.Tensor | None) -> torch.Tensor:
        out = tensor
        if self.cfg.additive_noise_std > 0.0:
            out = out + torch.randn_like(out) * self.cfg.additive_noise_std
        if self.cfg.drop_prob > 0.0 and prev_tensor is not None:
            mask = (torch.rand((out.shape[0], 1), device=out.device) < self.cfg.drop_prob).expand_as(out)
            out = torch.where(mask, prev_tensor, out)
        return out

    def _apply_noise_drop(self, obs_like, prev_like=None):
        if isinstance(obs_like, torch.Tensor):
            prev_tensor = prev_like if isinstance(prev_like, torch.Tensor) else None
            return self._apply_noise_drop_to_tensor(obs_like, prev_tensor)
        if self._is_tensordict_like(obs_like):
            out = obs_like.clone()
            prev_td = prev_like if self._is_tensordict_like(prev_like) else None
            for key, value in self._iter_tensor_items(out):
                prev_value = prev_td.get(key) if prev_td is not None and key in prev_td.keys() else None
                new_value = self._apply_noise_drop_to_tensor(value, prev_value)
                if hasattr(out, "set"):
                    out.set(key, new_value)
                else:
                    out[key] = new_value
            return out
        return obs_like

    def _apply_done_reset(self, obs_like, source_like, done_flat: torch.Tensor):
        if isinstance(obs_like, torch.Tensor) and isinstance(source_like, torch.Tensor):
            obs_like[done_flat] = source_like[done_flat]
            return
        if self._is_tensordict_like(obs_like) and self._is_tensordict_like(source_like):
            for key, value in self._iter_tensor_items(obs_like):
                src_value = source_like.get(key) if key in source_like.keys() else None
                if src_value is None:
                    continue
                value[done_flat] = src_value[done_flat]
                if hasattr(obs_like, "set"):
                    obs_like.set(key, value)
                else:
                    obs_like[key] = value

    def apply(self, obs: torch.Tensor, done_mask: torch.Tensor | None = None) -> torch.Tensor:
        if not self.enabled:
            return obs

        if len(self._history) == 0:
            for _ in range(max(1, self.cfg.delay_steps + 1)):
                self._history.append(obs.clone())
        else:
            self._history.append(obs.clone())

        delayed_obs = self._history[0].clone() if self.cfg.delay_steps > 0 else obs.clone()
        out = self._apply_noise_drop(delayed_obs, prev_like=self._prev_output)

        if done_mask is not None and len(self._history) > 0:
            done_flat = done_mask.reshape(-1).bool()
            if torch.any(done_flat):
                for hist in self._history:
                    self._apply_done_reset(hist, obs, done_flat)
                self._apply_done_reset(out, obs, done_flat)
                if self._prev_output is not None:
                    self._apply_done_reset(self._prev_output, obs, done_flat)

        self._prev_output = out.clone()
        return out


@dataclass
class PushRuntimeCfg:
    enabled: bool = False
    pattern: str = "disabled"  # disabled | single | periodic | random_interval
    start_step: int = 100
    interval_steps: int = 200
    random_interval_range: tuple[int, int] = (120, 260)
    duration_steps: int = 1
    magnitude_range: tuple[float, float] = (0.3, 0.8)
    direction_deg_range: tuple[float, float] = (-180.0, 180.0)
    env_fraction: float = 1.0

    @classmethod
    def from_overrides(cls, overrides: dict) -> "PushRuntimeCfg":
        raw = overrides.get("push_runtime", {})
        random_range = raw.get("random_interval_range", [120, 260])
        return cls(
            enabled=bool(raw.get("enabled", False)),
            pattern=str(raw.get("pattern", "disabled")),
            start_step=int(raw.get("start_step", 100)),
            interval_steps=int(raw.get("interval_steps", 200)),
            random_interval_range=(int(random_range[0]), int(random_range[1])),
            duration_steps=max(1, int(raw.get("duration_steps", 1))),
            magnitude_range=(float(raw.get("magnitude_range", [0.3, 0.8])[0]), float(raw.get("magnitude_range", [0.3, 0.8])[1])),
            direction_deg_range=(
                float(raw.get("direction_deg_range", [-180.0, 180.0])[0]),
                float(raw.get("direction_deg_range", [-180.0, 180.0])[1]),
            ),
            env_fraction=float(raw.get("env_fraction", 1.0)),
        )


class RuntimePushScheduler:
    """Apply runtime pushes by directly injecting root velocity impulses."""

    def __init__(self, cfg: PushRuntimeCfg, num_envs: int, device: torch.device):
        self.cfg = cfg
        self.num_envs = num_envs
        self.device = device
        self._single_fired = False
        self._next_random_trigger = cfg.start_step + self._sample_random_interval()
        self._active_events: list[dict[str, object]] = []

    @property
    def enabled(self) -> bool:
        return self.cfg.enabled and self.cfg.pattern != "disabled"

    def _sample_random_interval(self) -> int:
        lo, hi = self.cfg.random_interval_range
        if hi < lo:
            lo, hi = hi, lo
        return int(torch.randint(lo, hi + 1, (1,), device=self.device).item())

    def _sample_push_vector(self) -> tuple[float, float]:
        magnitude = float(
            torch.empty(1, device=self.device).uniform_(self.cfg.magnitude_range[0], self.cfg.magnitude_range[1]).item()
        )
        direction_deg = float(
            torch.empty(1, device=self.device).uniform_(
                self.cfg.direction_deg_range[0], self.cfg.direction_deg_range[1]
            ).item()
        )
        direction_rad = math.radians(direction_deg)
        return magnitude * math.cos(direction_rad), magnitude * math.sin(direction_rad)

    def _pick_env_ids(self) -> torch.Tensor:
        fraction = min(max(self.cfg.env_fraction, 0.0), 1.0)
        if fraction <= 0.0:
            return torch.zeros((0,), dtype=torch.long, device=self.device)
        if fraction >= 1.0:
            return torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        count = max(1, int(round(self.num_envs * fraction)))
        perm = torch.randperm(self.num_envs, device=self.device)
        return perm[:count].to(torch.long)

    def _should_trigger(self, step: int) -> bool:
        if step < self.cfg.start_step:
            return False
        if self.cfg.pattern == "single":
            if self._single_fired:
                return False
            self._single_fired = True
            return True
        if self.cfg.pattern == "periodic":
            return (step - self.cfg.start_step) % max(1, self.cfg.interval_steps) == 0
        if self.cfg.pattern == "random_interval":
            if step < self._next_random_trigger:
                return False
            self._next_random_trigger = step + self._sample_random_interval()
            return True
        return False

    def on_step(self, base_env, step: int):
        if not self.enabled:
            return

        if self._should_trigger(step):
            env_ids = self._pick_env_ids()
            if env_ids.numel() > 0:
                vel_x, vel_y = self._sample_push_vector()
                self._active_events.append(
                    {
                        "remaining_steps": self.cfg.duration_steps,
                        "env_ids": env_ids,
                        "vel_x": vel_x,
                        "vel_y": vel_y,
                    }
                )

        if not self._active_events:
            return

        next_active: list[dict[str, object]] = []
        for event in self._active_events:
            env_ids = event["env_ids"]
            push_by_setting_velocity(
                base_env,
                env_ids=env_ids,
                velocity_range={"x": (event["vel_x"], event["vel_x"]), "y": (event["vel_y"], event["vel_y"])},
                asset_cfg=SceneEntityCfg("robot"),
            )
            event["remaining_steps"] = int(event["remaining_steps"]) - 1
            if int(event["remaining_steps"]) > 0:
                next_active.append(event)
        self._active_events = next_active
