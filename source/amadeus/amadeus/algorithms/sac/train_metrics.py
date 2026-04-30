# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import json
from statistics import mean
from typing import Any


def _tracking_mean(tracking_data: dict[str, list[float]], *candidates: str) -> float | None:
    for key in candidates:
        values = tracking_data.get(key)
        if values:
            return float(mean(values))
    return None


def build_sac_train_metrics_row(tracking_data: dict[str, list[float]], step: int) -> dict[str, Any]:
    row = {
        "step": int(step),
        "mean_reward": _tracking_mean(
            tracking_data,
            "Reward / Total reward (mean)",
            "Reward / Instantaneous reward (mean)",
        ),
        "mean_episode_length": _tracking_mean(
            tracking_data,
            "Episode / Total timesteps (mean)",
        ),
        "policy_loss": _tracking_mean(tracking_data, "Loss / Policy loss"),
        "value_loss": _tracking_mean(tracking_data, "Loss / Critic loss"),
        "entropy_loss": _tracking_mean(tracking_data, "Loss / Entropy loss"),
        "entropy_coef": _tracking_mean(tracking_data, "Coefficient / Entropy coefficient"),
        "policy_learning_rate": _tracking_mean(tracking_data, "Learning / Policy learning rate"),
        "critic_learning_rate": _tracking_mean(tracking_data, "Learning / Critic learning rate"),
        "fall_rate": _tracking_mean(
            tracking_data,
            "Episode_Termination/base_contact",
            "Info / Episode_Termination/base_contact",
        ),
        "time_out_rate": _tracking_mean(
            tracking_data,
            "Episode_Termination/time_out",
            "Info / Episode_Termination/time_out",
        ),
        "track_lin_vel_reward": _tracking_mean(
            tracking_data,
            "Episode_Reward/track_lin_vel_xy_exp",
            "Info / Episode_Reward/track_lin_vel_xy_exp",
        ),
        "track_ang_vel_reward": _tracking_mean(
            tracking_data,
            "Episode_Reward/track_ang_vel_z_exp",
            "Info / Episode_Reward/track_ang_vel_z_exp",
        ),
        "tracking_data_json": json.dumps(
            {
                k: float(mean(v))
                for k, v in tracking_data.items()
                if isinstance(v, list) and len(v) > 0
            },
            ensure_ascii=True,
            sort_keys=True,
        ),
    }
    return row

