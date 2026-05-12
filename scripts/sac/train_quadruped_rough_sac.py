# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train quadruped rough-terrain locomotion with SAC (skrl backend)."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train Template-Amadeus-Quadruped-Rough SAC baseline.")
parser.add_argument("--task", type=str, default="Template-Amadeus-Quadruped-Rough-v0", help="Training task name.")
parser.add_argument("--play_task", type=str, default="Template-Amadeus-Quadruped-Rough-Play-v0", help="Eval task name.")
parser.add_argument("--agent", type=str, default="skrl_sac_cfg_entry_point", help="Hydra SAC config entry point.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of training environments.")
parser.add_argument("--seed", type=int, default=None, help="Random seed.")
parser.add_argument("--max_steps", type=int, default=None, help="Total environment interaction steps.")
parser.add_argument(
    "--memory_size",
    type=int,
    default=None,
    help=(
        "Replay buffer memory_size (first buffer dimension in skrl). "
        "Actual transition capacity is memory_size * num_envs."
    ),
)
parser.add_argument(
    "--target_replay_transitions",
    type=int,
    default=2000000,
    help=(
        "If --memory_size is not set, auto-scale memory_size ~= target_replay_transitions / num_envs. "
        "Helps avoid replay-buffer OOM on large num_envs."
    ),
)
parser.add_argument("--batch_size", type=int, default=None, help="Override SAC batch_size.")
parser.add_argument("--learning_starts", type=int, default=None, help="Override SAC learning_starts.")
parser.add_argument("--update_frequency", type=int, default=None, help="Perform SAC update every N env steps.")
parser.add_argument("--save_interval", type=int, default=None, help="Save checkpoint every N env steps.")
parser.add_argument(
    "--train_metrics_interval",
    type=int,
    default=None,
    help=(
        "Write train metrics every N environment steps. "
        "<=0 disables train-metrics export; if positive and larger than max_steps, it is clamped to max_steps."
    ),
)
parser.add_argument(
    "--log_interval",
    type=int,
    default=2000,
    help="Print one-line SAC training progress to terminal every N environment steps. <=0 disables console heartbeat.",
)
parser.add_argument("--eval_interval", type=int, default=None, help="Run periodic eval every N env steps.")
parser.add_argument("--eval_episodes", type=int, default=None, help="Number of episodes per periodic eval.")
parser.add_argument("--eval_num_envs", type=int, default=None, help="Number of eval environments.")
parser.add_argument("--eval_video_length", type=int, default=300, help="Periodic eval video length in env steps.")
parser.add_argument(
    "--eval_video_start_step",
    type=int,
    default=20,
    help="Environment step index to start periodic-eval video capture.",
)
parser.add_argument(
    "--eval_follow_robot_camera",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Use follow-camera mode during periodic evaluation videos.",
)
parser.add_argument(
    "--eval_camera_eye",
    type=float,
    nargs=3,
    default=[4.0, 4.0, 3.0],
    metavar=("X", "Y", "Z"),
    help="Fallback fixed camera eye for periodic eval video.",
)
parser.add_argument(
    "--eval_camera_lookat",
    type=float,
    nargs=3,
    default=[0.0, 0.0, 0.5],
    metavar=("X", "Y", "Z"),
    help="Fallback fixed camera lookat for periodic eval video.",
)
parser.add_argument(
    "--eval_camera_offset",
    type=float,
    nargs=3,
    default=[3.0, 3.0, 2.0],
    metavar=("X", "Y", "Z"),
    help="Follow-camera offset relative to robot target for periodic eval video.",
)
parser.add_argument(
    "--eval_camera_robot_env_id",
    type=int,
    default=0,
    help="Environment index used by follow-camera mode during periodic eval video.",
)
parser.add_argument(
    "--eval_show_velocity_markers",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Show command-vs-actual velocity markers in periodic eval video.",
)
parser.add_argument(
    "--eval_strict",
    action="store_true",
    default=False,
    help="Abort training when periodic evaluation fails. By default, evaluation failures are tolerated.",
)
parser.add_argument("--resume", action="store_true", default=False, help="Resume from the latest checkpoint.")
parser.add_argument("--checkpoint", type=str, default=None, help="Explicit checkpoint path for resume.")
parser.add_argument("--load_run", type=str, default=None, help="Run folder name to resume from.")
parser.add_argument("--run_name", type=str, default="amadeus_quadruped_rough_sac", help="Run name suffix.")
parser.add_argument("--video", action="store_true", default=False, help="Record short videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Training video length in env steps.")
parser.add_argument("--video_interval", type=int, default=5000, help="Training video interval in env steps.")
parser.add_argument("--dataset_chunk_episodes", type=int, default=128, help="Episodes per dataset shard.")
parser.add_argument(
    "--output_root",
    type=str,
    default="outputs/train_quadruped_rough",
    help=(
        "Output root directory. "
        "Default layout: <output_root>/sac/<run_name>/<timestamp>/{ckpt,tb,metrics,data,video,params,logs}."
    ),
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from skrl.agents.torch.base import Agent as SkrlAgentBase  # noqa: E402
from skrl.utils.runner.torch import Runner  # noqa: E402

from isaaclab.envs import ManagerBasedRLEnvCfg  # noqa: E402
from isaaclab.utils.dict import print_dict  # noqa: E402
from isaaclab.utils.io import dump_yaml  # noqa: E402
from isaaclab_rl.skrl import SkrlVecEnvWrapper  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

import amadeus.tasks  # noqa: F401, E402
from amadeus.algorithms.sac.train_metrics import build_sac_train_metrics_row  # noqa: E402
from amadeus.algorithms.sac.utils import ensure_dir, latest_model_checkpoint, parse_model_step  # noqa: E402
from amadeus.baselines.quadruped_rough.metrics import MetricsWriter  # noqa: E402
from amadeus.tasks.manager_based.quadruped_rough.recorders import ChunkedHDF5DatasetFileHandler  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def _build_run_paths(run_dir: str) -> dict[str, str]:
    return {
        "ckpt": os.path.join(run_dir, "ckpt"),
        "tb": os.path.join(run_dir, "tb"),
        "metrics": os.path.join(run_dir, "metrics"),
        "data_train": os.path.join(run_dir, "data", "train"),
        "data_eval": os.path.join(run_dir, "data", "eval"),
        "video_train": os.path.join(run_dir, "video", "train"),
        "video_eval": os.path.join(run_dir, "video", "eval"),
        "params": os.path.join(run_dir, "params"),
        "logs": os.path.join(run_dir, "logs"),
    }


def _latest_run_dir(output_root: str) -> str | None:
    if not os.path.isdir(output_root):
        return None
    run_dirs: list[str] = []
    for root, dirs, _ in os.walk(output_root):
        if "ckpt" in dirs or "checkpoints" in dirs:
            run_dirs.append(root)
    if not run_dirs:
        # fallback for flat legacy layouts
        run_dirs = [os.path.join(output_root, item) for item in os.listdir(output_root)]
        run_dirs = [path for path in run_dirs if os.path.isdir(path)]
        if not run_dirs:
            return None
    return max(run_dirs, key=os.path.getmtime)


def _resolve_resume_checkpoint(output_root: str, run_dir: str) -> str | None:
    if args_cli.checkpoint:
        return os.path.abspath(args_cli.checkpoint)
    if not args_cli.resume:
        return None

    method_root = os.path.join(output_root, "sac")
    experiment_root = os.path.join(method_root, args_cli.run_name)
    legacy_root = os.path.abspath(os.path.join("outputs", "loco_sac"))

    candidate_run_dirs: list[str] = []
    if args_cli.load_run:
        if os.path.isabs(args_cli.load_run):
            candidate_run_dirs.append(args_cli.load_run)
        else:
            legacy_match = re.match(r"^(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})_(.+)$", args_cli.load_run)
            if legacy_match:
                legacy_ts = legacy_match.group(1)
                legacy_exp = legacy_match.group(2)
                candidate_run_dirs.extend(
                    [
                        os.path.join(method_root, legacy_exp, legacy_ts),
                        os.path.join(experiment_root, legacy_ts),
                    ]
                )
            candidate_run_dirs.extend(
                [
                    os.path.join(experiment_root, args_cli.load_run),
                    os.path.join(method_root, args_cli.load_run),
                    os.path.join(output_root, args_cli.load_run),
                    os.path.join(legacy_root, args_cli.load_run),
                ]
            )
    else:
        candidate_run_dirs.append(run_dir)
        for root in (experiment_root, method_root, output_root, legacy_root):
            latest = _latest_run_dir(root)
            if latest is not None:
                candidate_run_dirs.append(latest)

    # de-duplicate while preserving order
    deduped_candidate_dirs: list[str] = []
    seen = set()
    for path in candidate_run_dirs:
        abspath = os.path.abspath(path)
        if abspath in seen:
            continue
        seen.add(abspath)
        deduped_candidate_dirs.append(abspath)

    for target_run_dir in deduped_candidate_dirs:
        for ckpt_dir_name in ("ckpt", "checkpoints"):
            ckpt_dir = os.path.join(target_run_dir, ckpt_dir_name)
            if not os.path.isdir(ckpt_dir):
                continue
            checkpoint_path = latest_model_checkpoint(ckpt_dir)
            if checkpoint_path is not None:
                return checkpoint_path
    return None


def _run_periodic_eval(run_paths: dict[str, str], checkpoint_path: str, step: int) -> None:
    metrics_dir = run_paths["metrics"]
    eval_video_dir = os.path.join(run_paths["video_eval"], f"iter_{step:07d}")
    base_cmd = [
        "/isaac-sim/python.sh",
        os.path.join(os.getcwd(), "scripts/sac/eval_quadruped_rough_sac.py"),
        "--task",
        args_cli.play_task,
        "--agent",
        args_cli.agent,
        "--checkpoint",
        checkpoint_path,
        "--num_envs",
        str(_runtime_eval_num_envs),
        "--num_episodes",
        str(_runtime_eval_episodes),
        "--seed",
        str(_runtime_seed),
        "--metrics_dir",
        metrics_dir,
        "--dataset_dir",
        run_paths["data_eval"],
        "--headless",
    ]
    if args_cli.device:
        base_cmd.extend(["--device", args_cli.device])

    cmd_with_video = [
        *base_cmd,
        "--video_folder",
        eval_video_dir,
        "--video_length",
        str(args_cli.eval_video_length),
        "--video_start_step",
        str(args_cli.eval_video_start_step),
        "--camera_eye",
        *[str(v) for v in args_cli.eval_camera_eye],
        "--camera_lookat",
        *[str(v) for v in args_cli.eval_camera_lookat],
        "--camera_offset",
        *[str(v) for v in args_cli.eval_camera_offset],
        "--camera_robot_env_id",
        str(args_cli.eval_camera_robot_env_id),
    ]
    if args_cli.eval_follow_robot_camera:
        cmd_with_video.append("--follow_robot_camera")
    if args_cli.eval_show_velocity_markers:
        cmd_with_video.append("--show_velocity_markers")

    try:
        subprocess.run(cmd_with_video, check=True)
        return
    except subprocess.CalledProcessError as video_err:
        print(
            "[WARN] SAC periodic evaluation with video failed at "
            f"step {step}: {video_err}. Retrying once without video."
        )
    try:
        subprocess.run(base_cmd, check=True)
    except subprocess.CalledProcessError as eval_err:
        if args_cli.eval_strict:
            raise
        print(
            "[WARN] SAC periodic evaluation failed again without video at "
            f"step {step}: {eval_err}. Continuing training (eval_strict=False)."
        )


_runtime_seed = 42
_runtime_eval_interval = 50000
_runtime_save_interval = 20000
_runtime_eval_episodes = 4
_runtime_eval_num_envs = 8
_runtime_update_frequency = 1
_runtime_max_steps = 2000000
_runtime_train_metrics_interval = 2000


def _resolve_memory_size(agent_cfg: dict, num_envs: int, batch_size: int) -> tuple[int, str]:
    memory_cfg = agent_cfg.get("memory", {})
    base_memory_size = int(memory_cfg.get("memory_size", 1000000))
    if args_cli.memory_size is not None:
        return max(int(args_cli.memory_size), max(1, batch_size)), "cli"

    target_total = max(int(args_cli.target_replay_transitions), 1)
    auto_scaled = max(target_total // max(num_envs, 1), 1)
    resolved = min(base_memory_size, auto_scaled)
    resolved = max(resolved, max(1, batch_size))
    return resolved, "auto"


def _fmt_metric(value) -> str:
    if value is None:
        return "nan"
    try:
        return f"{float(value):.4g}"
    except (TypeError, ValueError):
        return "nan"


def _print_train_heartbeat(row: dict, *, step: int, start_time: float, max_steps: int):
    elapsed_s = max(time.time() - start_time, 1e-6)
    steps_per_s = step / elapsed_s
    eta_s = (max_steps - step) / steps_per_s if steps_per_s > 0 else float("nan")
    eta_min = eta_s / 60.0 if eta_s == eta_s else float("nan")
    print(
        "[TRAIN] "
        f"step={step}/{max_steps} "
        f"reward={_fmt_metric(row.get('mean_reward'))} "
        f"ep_len={_fmt_metric(row.get('mean_episode_length'))} "
        f"policy_loss={_fmt_metric(row.get('policy_loss'))} "
        f"value_loss={_fmt_metric(row.get('value_loss'))} "
        f"entropy_loss={_fmt_metric(row.get('entropy_loss'))} "
        f"entropy_coef={_fmt_metric(row.get('entropy_coef'))} "
        f"fall_rate={_fmt_metric(row.get('fall_rate'))} "
        f"timeout_rate={_fmt_metric(row.get('time_out_rate'))} "
        f"lin_track_rew={_fmt_metric(row.get('track_lin_vel_reward'))} "
        f"sps={steps_per_s:.1f} "
        f"eta_min={eta_min:.1f}",
        flush=True,
    )


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: dict):
    global _runtime_seed
    global _runtime_eval_interval
    global _runtime_save_interval
    global _runtime_eval_episodes
    global _runtime_eval_num_envs
    global _runtime_update_frequency
    global _runtime_max_steps
    global _runtime_train_metrics_interval

    runtime_cfg = agent_cfg.get("sac_runtime", {})
    _runtime_seed = int(args_cli.seed if args_cli.seed is not None else agent_cfg.get("seed", 42))
    _runtime_eval_interval = int(
        args_cli.eval_interval if args_cli.eval_interval is not None else runtime_cfg.get("eval_interval_steps", 50000)
    )
    _runtime_save_interval = int(
        args_cli.save_interval if args_cli.save_interval is not None else runtime_cfg.get("save_interval", 20000)
    )
    _runtime_eval_episodes = int(
        args_cli.eval_episodes if args_cli.eval_episodes is not None else runtime_cfg.get("eval_episodes", 4)
    )
    _runtime_eval_num_envs = int(
        args_cli.eval_num_envs if args_cli.eval_num_envs is not None else runtime_cfg.get("eval_num_envs", 8)
    )
    _runtime_update_frequency = int(
        args_cli.update_frequency if args_cli.update_frequency is not None else runtime_cfg.get("update_frequency", 1)
    )
    _runtime_update_frequency = max(_runtime_update_frequency, 1)
    _runtime_max_steps = int(
        args_cli.max_steps if args_cli.max_steps is not None else agent_cfg.get("trainer", {}).get("timesteps", 2000000)
    )
    default_metrics_interval = runtime_cfg.get(
        "train_metrics_interval",
        agent_cfg.get("agent", {}).get("experiment", {}).get("write_interval", 2000),
    )
    _runtime_train_metrics_interval = int(
        args_cli.train_metrics_interval if args_cli.train_metrics_interval is not None else default_metrics_interval
    )
    if _runtime_train_metrics_interval > 0:
        _runtime_train_metrics_interval = min(_runtime_train_metrics_interval, _runtime_max_steps)

    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = _runtime_seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    resolved_num_envs = int(env_cfg.scene.num_envs)

    ChunkedHDF5DatasetFileHandler.max_episodes_per_file = args_cli.dataset_chunk_episodes

    output_root = os.path.abspath(args_cli.output_root)
    method_root = os.path.join(output_root, "sac")
    experiment_root = os.path.join(method_root, args_cli.run_name)
    legacy_root = os.path.abspath(os.path.join("outputs", "loco_sac"))
    ensure_dir(output_root)
    ensure_dir(method_root)
    ensure_dir(experiment_root)

    if args_cli.resume and args_cli.load_run:
        if os.path.isabs(args_cli.load_run):
            run_dir = args_cli.load_run
        else:
            legacy_match = re.match(r"^(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})_(.+)$", args_cli.load_run)
            legacy_mapped = None
            if legacy_match:
                legacy_ts = legacy_match.group(1)
                legacy_exp = legacy_match.group(2)
                candidate_legacy_mapped = os.path.join(method_root, legacy_exp, legacy_ts)
                if os.path.isdir(candidate_legacy_mapped):
                    legacy_mapped = candidate_legacy_mapped
                elif legacy_exp == args_cli.run_name:
                    candidate_same_exp = os.path.join(experiment_root, legacy_ts)
                    if os.path.isdir(candidate_same_exp):
                        legacy_mapped = candidate_same_exp
            candidate_new = os.path.join(experiment_root, args_cli.load_run)
            candidate_method = os.path.join(method_root, args_cli.load_run)
            candidate_flat = os.path.join(output_root, args_cli.load_run)
            candidate_legacy = os.path.join(legacy_root, args_cli.load_run)
            if legacy_mapped is not None:
                run_dir = legacy_mapped
            elif os.path.isdir(candidate_new):
                run_dir = candidate_new
            elif os.path.isdir(candidate_method):
                run_dir = candidate_method
            elif os.path.isdir(candidate_flat):
                run_dir = candidate_flat
            elif os.path.isdir(candidate_legacy):
                run_dir = candidate_legacy
            else:
                run_dir = candidate_new
    elif args_cli.resume and not args_cli.load_run:
        run_dir = (
            _latest_run_dir(experiment_root)
            or _latest_run_dir(method_root)
            or _latest_run_dir(output_root)
            or _latest_run_dir(legacy_root)
            or os.path.join(experiment_root, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        )
    else:
        run_dir = os.path.join(experiment_root, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

    run_paths = _build_run_paths(run_dir)
    for path in run_paths.values():
        ensure_dir(path)

    env_cfg.log_dir = run_dir
    env_cfg.recorders.dataset_export_dir_path = run_paths["data_train"]
    env_cfg.recorders.dataset_filename = "train_dataset"

    agent_cfg["seed"] = _runtime_seed
    agent_cfg.setdefault("trainer", {})
    agent_cfg["trainer"]["timesteps"] = _runtime_max_steps
    agent_cfg["trainer"]["headless"] = bool(args_cli.headless)
    agent_cfg["trainer"]["close_environment_at_exit"] = False
    agent_cfg.setdefault("memory", {})
    agent_cfg.setdefault("agent", {})
    agent_cfg["agent"].setdefault("experiment", {})
    if args_cli.batch_size is not None:
        agent_cfg["agent"]["batch_size"] = int(args_cli.batch_size)
    if args_cli.learning_starts is not None:
        ls = int(args_cli.learning_starts)
        agent_cfg["agent"]["learning_starts"] = ls
        agent_cfg["agent"]["random_timesteps"] = ls
    resolved_batch_size = int(agent_cfg["agent"].get("batch_size", 2048))
    resolved_memory_size, memory_size_source = _resolve_memory_size(agent_cfg, resolved_num_envs, resolved_batch_size)
    agent_cfg["memory"]["memory_size"] = int(resolved_memory_size)
    agent_cfg["agent"]["experiment"]["directory"] = run_dir
    agent_cfg["agent"]["experiment"]["experiment_name"] = "tb"
    agent_cfg["agent"]["experiment"]["write_interval"] = _runtime_train_metrics_interval
    agent_cfg["agent"]["experiment"]["checkpoint_interval"] = 0

    dump_yaml(os.path.join(run_paths["params"], "env.yaml"), env_cfg)
    dump_yaml(os.path.join(run_paths["params"], "agent.yaml"), agent_cfg)
    with open(os.path.join(run_paths["logs"], "run_info.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "task": args_cli.task,
                "agent_entry_point": args_cli.agent,
                "seed": _runtime_seed,
                "num_envs": resolved_num_envs,
                "max_steps": _runtime_max_steps,
                "update_frequency": _runtime_update_frequency,
                "save_interval": _runtime_save_interval,
                "train_metrics_interval": _runtime_train_metrics_interval,
                "log_interval": int(args_cli.log_interval),
                "eval_interval": _runtime_eval_interval,
                "eval_episodes": _runtime_eval_episodes,
                "eval_num_envs": _runtime_eval_num_envs,
                "memory_size": resolved_memory_size,
                "memory_size_source": memory_size_source,
                "target_replay_transitions": int(args_cli.target_replay_transitions),
                "batch_size": int(agent_cfg["agent"].get("batch_size", -1)),
                "learning_starts": int(agent_cfg["agent"].get("learning_starts", -1)),
            },
            f,
            indent=2,
            ensure_ascii=True,
        )

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if args_cli.video:
        video_kwargs = {
            "video_folder": run_paths["video_train"],
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording training videos.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = SkrlVecEnvWrapper(env, ml_framework="torch")
    runner = Runner(env, agent_cfg)
    agent = runner.agent

    metrics_writer = MetricsWriter(run_paths["metrics"])
    original_write_tracking_data = agent.write_tracking_data
    train_metrics_rows_written = 0

    def _write_tracking_data_with_csv(*, timestep: int, timesteps: int):
        nonlocal train_metrics_rows_written
        row = build_sac_train_metrics_row(agent.tracking_data, timestep)
        metrics_writer.write_train_row(row)
        train_metrics_rows_written += 1
        original_write_tracking_data(timestep=timestep, timesteps=timesteps)

    agent.write_tracking_data = _write_tracking_data_with_csv
    agent.enable_training_mode(True)

    resume_path = _resolve_resume_checkpoint(output_root, run_dir)
    start_step = 0
    if resume_path is not None:
        print(f"[INFO] Loading SAC checkpoint from: {resume_path}")
        agent.load(resume_path)
        parsed_step = parse_model_step(resume_path)
        if parsed_step is not None:
            start_step = parsed_step

    obs, infos = env.reset()
    states = env.state()

    print(f"[INFO] Logging experiment root: {method_root}")
    print(f"[INFO] SAC experiment root: {experiment_root}")
    print(f"[INFO] Current SAC run directory: {run_dir}")
    print(
        f"[INFO] SAC runtime: max_steps={_runtime_max_steps}, update_frequency={_runtime_update_frequency}, "
        f"save_interval={_runtime_save_interval}, train_metrics_interval={_runtime_train_metrics_interval}, "
        f"log_interval={args_cli.log_interval}, eval_interval={_runtime_eval_interval}."
    )
    print(
        "[INFO] SAC replay buffer config: "
        f"memory_size={resolved_memory_size} ({memory_size_source}), "
        f"num_envs={resolved_num_envs}, "
        f"effective_capacity={resolved_memory_size * resolved_num_envs}, "
        f"batch_size={resolved_batch_size}."
    )

    start_time = time.time()
    last_saved_step = start_step

    for timestep in range(start_step, _runtime_max_steps):
        agent.pre_interaction(timestep=timestep, timesteps=_runtime_max_steps)

        with torch.no_grad():
            actions, _ = agent.act(obs, states, timestep=timestep, timesteps=_runtime_max_steps)
            next_obs, rewards, terminated, truncated, infos = env.step(actions)
            next_states = env.state()
            agent.record_transition(
                observations=obs,
                states=states,
                actions=actions,
                rewards=rewards,
                next_observations=next_obs,
                next_states=next_states,
                terminated=terminated,
                truncated=truncated,
                infos=infos,
                timestep=timestep,
                timesteps=_runtime_max_steps,
            )

        if (timestep + 1) % _runtime_update_frequency == 0:
            agent.post_interaction(timestep=timestep, timesteps=_runtime_max_steps)
        else:
            # Keep writer/checkpoint hooks active while skipping gradient updates at this step.
            SkrlAgentBase.post_interaction(agent, timestep=timestep, timesteps=_runtime_max_steps)

        obs, states = next_obs, next_states

        step = timestep + 1
        if args_cli.log_interval > 0 and step % args_cli.log_interval == 0:
            row = build_sac_train_metrics_row(agent.tracking_data, step)
            _print_train_heartbeat(row, step=step, start_time=start_time, max_steps=_runtime_max_steps)

        if _runtime_save_interval > 0 and step % _runtime_save_interval == 0:
            checkpoint_path = os.path.join(run_paths["ckpt"], f"model_{step}.pt")
            agent.save(checkpoint_path)
            last_saved_step = step

        if _runtime_eval_interval > 0 and _runtime_eval_episodes > 0 and step % _runtime_eval_interval == 0:
            checkpoint_path = os.path.join(run_paths["ckpt"], f"model_{step}.pt")
            if not os.path.isfile(checkpoint_path):
                agent.save(checkpoint_path)
                last_saved_step = step
            _run_periodic_eval(run_paths, checkpoint_path, step)

    final_step = _runtime_max_steps
    if _runtime_train_metrics_interval > 0 and train_metrics_rows_written == 0:
        metrics_writer.write_train_row(build_sac_train_metrics_row(agent.tracking_data, final_step))
        train_metrics_rows_written += 1
        print(
            "[INFO] No periodic SAC train metrics were emitted during training. "
            f"Wrote a final metrics snapshot at step {final_step}."
        )

    final_checkpoint = os.path.join(run_paths["ckpt"], f"model_{final_step}.pt")
    if last_saved_step != final_step or not os.path.isfile(final_checkpoint):
        agent.save(final_checkpoint)

    print(f"Training time: {round(time.time() - start_time, 2)} seconds")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
