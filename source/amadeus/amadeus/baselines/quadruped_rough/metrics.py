# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import csv
import json
import os
import statistics
from typing import Any

import torch


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def summarize_ep_extras(ep_extras: list[dict[str, Any]], device: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    if not ep_extras:
        return metrics

    all_keys = set()
    for item in ep_extras:
        all_keys.update(item.keys())

    for key in sorted(all_keys):
        infotensor = torch.tensor([], device=device)
        for ep_info in ep_extras:
            if key not in ep_info:
                continue
            value = ep_info[key]
            if not isinstance(value, torch.Tensor):
                value = torch.tensor([value], device=device)
            if len(value.shape) == 0:
                value = value.unsqueeze(0)
            infotensor = torch.cat((infotensor, value.to(device)))
        if infotensor.numel() > 0:
            metrics[key] = torch.mean(infotensor).item()
    return metrics


class MetricsWriter:
    """Append-only CSV/JSONL metric exporter."""

    def __init__(self, output_dir: str):
        ensure_dir(output_dir)
        self.output_dir = output_dir
        self.train_jsonl = os.path.join(output_dir, "train_metrics.jsonl")
        self.eval_jsonl = os.path.join(output_dir, "eval_metrics.jsonl")
        self.train_csv = os.path.join(output_dir, "train_metrics.csv")
        self.eval_csv = os.path.join(output_dir, "eval_metrics.csv")

    def write_train_row(self, row: dict[str, Any]) -> None:
        self._write_jsonl(self.train_jsonl, row)
        self._write_csv(self.train_csv, row)

    def write_eval_row(self, row: dict[str, Any]) -> None:
        self._write_jsonl(self.eval_jsonl, row)
        self._write_csv(self.eval_csv, row)

    def _write_jsonl(self, path: str, row: dict[str, Any]) -> None:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")

    def _write_csv(self, path: str, row: dict[str, Any]) -> None:
        file_exists = os.path.exists(path)
        with open(path, "a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)


def build_train_metrics_row(logger, loss_dict: dict[str, float], learning_rate: float, action_std: torch.Tensor, it: int,
                            collect_time: float, learn_time: float) -> dict[str, Any]:
    extras = summarize_ep_extras(logger.ep_extras, logger.device)
    collection_size = logger.cfg["num_steps_per_env"] * logger.num_envs * logger.gpu_world_size
    fps = int(collection_size / max(collect_time + learn_time, 1.0e-6))

    return {
        "iteration": it,
        "total_steps": logger.tot_timesteps + collection_size,
        "fps": fps,
        "collect_time_s": collect_time,
        "learn_time_s": learn_time,
        "learning_rate": learning_rate,
        "mean_action_std": action_std.mean().item(),
        "mean_reward": statistics.mean(logger.rewbuffer) if len(logger.rewbuffer) > 0 else None,
        "mean_episode_length": statistics.mean(logger.lenbuffer) if len(logger.lenbuffer) > 0 else None,
        "policy_loss": loss_dict.get("surrogate_loss"),
        "value_loss": loss_dict.get("value_loss"),
        "entropy_loss": loss_dict.get("entropy_loss"),
        "kl": loss_dict.get("kl"),
        "fall_rate": extras.get("Episode_Termination/base_contact"),
        "time_out_rate": extras.get("Episode_Termination/time_out"),
        "track_lin_vel_reward": extras.get("Episode_Reward/track_lin_vel_xy_exp"),
        "track_ang_vel_reward": extras.get("Episode_Reward/track_ang_vel_z_exp"),
        "extras_json": json.dumps(extras, ensure_ascii=True, sort_keys=True),
        "losses_json": json.dumps(loss_dict, ensure_ascii=True, sort_keys=True),
    }
