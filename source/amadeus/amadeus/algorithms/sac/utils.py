# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import csv
import glob
import json
import os
import re
from typing import Any


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def parse_model_step(checkpoint_path: str) -> int | None:
    match = re.search(r"model_(\d+)\.pt$", os.path.basename(checkpoint_path))
    if match is None:
        return None
    return int(match.group(1))


def latest_model_checkpoint(checkpoint_dir: str) -> str | None:
    candidates = glob.glob(os.path.join(checkpoint_dir, "model_*.pt"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: parse_model_step(p) or -1)


def infer_run_dir_from_checkpoint(checkpoint_path: str) -> str:
    checkpoint_dir = os.path.dirname(os.path.abspath(checkpoint_path))
    if os.path.basename(checkpoint_dir) in {"ckpt", "checkpoints"}:
        return os.path.dirname(checkpoint_dir)
    return checkpoint_dir


def default_eval_dataset_dir(run_dir: str) -> str:
    data_dir = os.path.join(run_dir, "data")
    datasets_dir = os.path.join(run_dir, "datasets")
    if os.path.isdir(data_dir):
        return os.path.join(data_dir, "eval")
    if os.path.isdir(datasets_dir):
        return os.path.join(datasets_dir, "eval")
    return os.path.join(data_dir, "eval")


def write_eval_metrics(metrics_dir: str, row: dict[str, Any]) -> None:
    ensure_dir(metrics_dir)
    jsonl_path = os.path.join(metrics_dir, "eval_metrics.jsonl")
    csv_path = os.path.join(metrics_dir, "eval_metrics.csv")
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=True) + "\n")
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

