# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Convert chunked transition datasets into a trajectory-level summary file."""

from __future__ import annotations

import argparse
import json
import os
from glob import glob

import h5py
import numpy as np


def _load_dataset(dataset_path: str) -> list[dict]:
    trajectories = []
    with h5py.File(dataset_path, "r") as f:
        for demo_name, demo_group in f["data"].items():
            transition = demo_group["transition"]
            rewards = np.asarray(transition["reward"]).reshape(-1)
            dones = np.asarray(transition["done"]).reshape(-1)
            lengths = len(rewards)
            trajectories.append(
                {
                    "source_file": os.path.basename(dataset_path),
                    "demo_name": demo_name,
                    "num_steps": int(lengths),
                    "episode_reward": float(rewards.sum()),
                    "terminated": bool(dones[-1]) if lengths > 0 else False,
                    "env_id": int(np.asarray(transition["env_id"])[0][0]) if lengths > 0 else None,
                    "episode_id": int(np.asarray(transition["episode_id"])[0][0]) if lengths > 0 else None,
                    "mean_lin_vel_tracking_error": float(np.asarray(transition["lin_vel_tracking_error"]).mean())
                    if "lin_vel_tracking_error" in transition
                    else None,
                    "mean_yaw_vel_tracking_error": float(np.asarray(transition["yaw_vel_tracking_error"]).mean())
                    if "yaw_vel_tracking_error" in transition
                    else None,
                }
            )
    return trajectories


def main():
    parser = argparse.ArgumentParser(description="Postprocess chunked quadruped datasets into trajectory summaries.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing chunked HDF5 files.")
    parser.add_argument("--output_path", type=str, required=True, help="Output JSON path.")
    args = parser.parse_args()

    dataset_paths = sorted(glob(os.path.join(args.input_dir, "*.hdf5")))
    summaries = []
    for dataset_path in dataset_paths:
        summaries.extend(_load_dataset(dataset_path))

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "num_files": len(dataset_paths),
                "num_trajectories": len(summaries),
                "trajectories": summaries,
            },
            f,
            indent=2,
            ensure_ascii=True,
        )

    print(args.output_path)


if __name__ == "__main__":
    main()
