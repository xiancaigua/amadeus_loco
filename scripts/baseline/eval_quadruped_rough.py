# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate a trained quadruped rough-terrain checkpoint and optionally record a short video."""

import argparse
import csv
import json
import os
import sys
from datetime import datetime

from isaaclab.app import AppLauncher

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "rsl_rl"))
import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Evaluate the quadruped rough-terrain baseline.")
parser.add_argument("--task", type=str, default="Template-Amadeus-Quadruped-Rough-Play-v0", help="Play task name.")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point", help="Agent config entry point.")
parser.add_argument("--num_envs", type=int, default=32, help="Number of eval environments.")
parser.add_argument("--num_episodes", type=int, default=8, help="Episodes to evaluate.")
parser.add_argument("--seed", type=int, default=42, help="Random seed.")
parser.add_argument("--video_length", type=int, default=300, help="Video length in env steps.")
parser.add_argument("--video_folder", type=str, default=None, help="Video output directory.")
parser.add_argument("--metrics_dir", type=str, default=None, help="Metrics output directory.")
parser.add_argument("--dataset_dir", type=str, default=None, help="Dataset export directory.")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.checkpoint is None:
    parser.error("--checkpoint is required")

if args_cli.video_folder:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import importlib.metadata as metadata  # noqa: E402
import time  # noqa: E402

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from packaging import version  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab.envs import ManagerBasedRLEnvCfg  # noqa: E402
from isaaclab.utils.assets import retrieve_file_path  # noqa: E402
from isaaclab.utils.dict import print_dict  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

import amadeus.tasks  # noqa: F401, E402


def _write_eval_metrics(metrics_dir: str, row: dict):
    os.makedirs(metrics_dir, exist_ok=True)
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


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    installed_version = metadata.version("rsl-rl-lib")
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.recorders.dataset_export_dir_path = args_cli.dataset_dir or os.path.join(
        os.path.dirname(retrieve_file_path(args_cli.checkpoint)), "datasets", "eval"
    )
    env_cfg.recorders.dataset_filename = "eval_rollouts"

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video_folder else None)
    if args_cli.video_folder:
        video_kwargs = {
            "video_folder": args_cli.video_folder,
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording evaluation video.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    checkpoint_path = retrieve_file_path(args_cli.checkpoint)
    runner.load(checkpoint_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    ep_reward = torch.zeros(env.num_envs, device=env.device)
    ep_length = torch.zeros(env.num_envs, device=env.device)
    ep_lin_error = torch.zeros(env.num_envs, device=env.device)
    ep_yaw_error = torch.zeros(env.num_envs, device=env.device)
    completed_rewards = []
    completed_lengths = []
    completed_lin_errors = []
    completed_yaw_errors = []
    completed_falls = []
    completed_timeouts = []

    obs = env.get_observations()
    completed_episodes = 0

    while simulation_app.is_running() and completed_episodes < args_cli.num_episodes:
        start_time = time.time()
        with torch.inference_mode():
            actions = policy(obs)
            obs, rewards, dones, extras = env.step(actions)
            command = env.unwrapped.command_manager.get_command("base_velocity")
            robot = env.unwrapped.scene["robot"]
            lin_error = torch.linalg.norm(command[:, :2] - robot.data.root_lin_vel_b[:, :2], dim=1)
            yaw_error = torch.abs(command[:, 2] - robot.data.root_ang_vel_b[:, 2])

            rewards_flat = torch.atleast_1d(rewards).reshape(-1)
            dones_flat = torch.atleast_1d(dones).reshape(-1)
            ep_reward += rewards_flat
            ep_length += 1
            ep_lin_error += lin_error
            ep_yaw_error += yaw_error

            raw_time_outs = extras.get("time_outs", torch.zeros_like(dones_flat, dtype=torch.bool))
            time_outs = torch.atleast_1d(raw_time_outs).reshape(-1).bool()
            done_ids = (dones_flat > 0).nonzero(as_tuple=False).reshape(-1)
            for idx in done_ids.tolist():
                completed_rewards.append(ep_reward[idx].item())
                completed_lengths.append(ep_length[idx].item())
                completed_lin_errors.append((ep_lin_error[idx] / torch.clamp(ep_length[idx], min=1)).item())
                completed_yaw_errors.append((ep_yaw_error[idx] / torch.clamp(ep_length[idx], min=1)).item())
                completed_falls.append(float(not time_outs[idx].item()))
                completed_timeouts.append(float(time_outs[idx].item()))
                ep_reward[idx] = 0
                ep_length[idx] = 0
                ep_lin_error[idx] = 0
                ep_yaw_error[idx] = 0
                completed_episodes += 1
                if completed_episodes >= args_cli.num_episodes:
                    break

            if version.parse(installed_version) >= version.parse("4.0.0"):
                policy.reset(dones)

        sleep_time = env.unwrapped.step_dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    result = {
        "timestamp": datetime.utcnow().isoformat(),
        "checkpoint": checkpoint_path,
        "num_episodes": len(completed_rewards),
        "mean_episode_reward": float(np.mean(completed_rewards)) if completed_rewards else None,
        "mean_episode_length": float(np.mean(completed_lengths)) if completed_lengths else None,
        "mean_lin_vel_tracking_error": float(np.mean(completed_lin_errors)) if completed_lin_errors else None,
        "mean_yaw_vel_tracking_error": float(np.mean(completed_yaw_errors)) if completed_yaw_errors else None,
        "fall_rate": float(np.mean(completed_falls)) if completed_falls else None,
        "timeout_rate": float(np.mean(completed_timeouts)) if completed_timeouts else None,
    }
    print(json.dumps(result, indent=2, ensure_ascii=True))
    if args_cli.metrics_dir:
        _write_eval_metrics(args_cli.metrics_dir, result)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
