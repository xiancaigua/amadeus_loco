# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Export rollout trajectories from a trained quadruped rough-terrain policy."""

import argparse
import os
import subprocess

parser = argparse.ArgumentParser(description="Export rollout trajectories from a trained policy.")
parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint path.")
parser.add_argument("--num_episodes", type=int, default=16, help="Number of episodes to export.")
parser.add_argument("--num_envs", type=int, default=32, help="Number of rollout environments.")
parser.add_argument("--seed", type=int, default=42, help="Seed for rollout export.")
parser.add_argument("--task", type=str, default="Template-Amadeus-Quadruped-Rough-Play-v0", help="Play task.")
parser.add_argument("--output_dir", type=str, default=None, help="Output directory for rollout datasets.")
parser.add_argument("--device", type=str, default=None, help="Simulation device.")
args = parser.parse_args()

checkpoint_path = os.path.abspath(args.checkpoint)
checkpoint_dir = os.path.dirname(checkpoint_path)
run_dir = os.path.dirname(checkpoint_dir) if os.path.basename(checkpoint_dir) in {"checkpoints", "ckpt"} else checkpoint_dir
if args.output_dir:
    output_dir = args.output_dir
else:
    if os.path.isdir(os.path.join(run_dir, "datasets")):
        output_dir = os.path.join(run_dir, "datasets", "rollouts")
    elif os.path.isdir(os.path.join(run_dir, "data")):
        output_dir = os.path.join(run_dir, "data", "rollouts")
    else:
        output_dir = os.path.join(run_dir, "datasets", "rollouts")
os.makedirs(output_dir, exist_ok=True)

cmd = [
    "/isaac-sim/python.sh",
    os.path.join(os.getcwd(), "scripts/baseline/eval_quadruped_rough.py"),
    "--task",
    args.task,
    "--checkpoint",
    checkpoint_path,
    "--num_envs",
    str(args.num_envs),
    "--num_episodes",
    str(args.num_episodes),
    "--seed",
    str(args.seed),
    "--dataset_dir",
    output_dir,
    "--metrics_dir",
    os.path.join(output_dir, "metrics"),
    "--headless",
]
if args.device:
    cmd.extend(["--device", args.device])

env = os.environ.copy()
subprocess.run(cmd, check=True, env=env)
print(output_dir)
