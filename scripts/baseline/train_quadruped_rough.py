# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train the Amadeus manager-based quadruped rough-terrain baseline with periodic evaluation."""

import argparse
import os
import subprocess
import sys
from datetime import datetime

from isaaclab.app import AppLauncher

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "rsl_rl"))
import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Train the Amadeus quadruped rough-terrain baseline with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record short videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Recorded training video length in env steps.")
parser.add_argument("--video_interval", type=int, default=5000, help="Training video interval in env steps.")
parser.add_argument("--task", type=str, default="Template-Amadeus-Quadruped-Rough-v0", help="Task name.")
parser.add_argument("--play_task", type=str, default="Template-Amadeus-Quadruped-Rough-Play-v0", help="Eval task name.")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point", help="Agent config entry point.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of training environments.")
parser.add_argument("--eval_num_envs", type=int, default=32, help="Number of evaluation environments.")
parser.add_argument("--seed", type=int, default=None, help="Random seed.")
parser.add_argument("--max_iterations", type=int, default=None, help="Total PPO iterations.")
parser.add_argument("--eval_interval", type=int, default=50, help="Evaluate every N training iterations.")
parser.add_argument("--eval_episodes", type=int, default=8, help="Number of evaluation episodes.")
parser.add_argument("--eval_video_length", type=int, default=300, help="Evaluation video length in env steps.")
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
    help="Use follow-camera mode during periodic evaluation videos (default: enabled).",
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
parser.add_argument("--dataset_chunk_episodes", type=int, default=128, help="Episodes per dataset shard.")
parser.add_argument(
    "--output_root",
    type=str,
    default="outputs/quadruped_rough_baseline",
    help="Output root directory. Layout: <output_root>/rsl_rl/<experiment_name>/<run_name>/...",
)
parser.add_argument("--distributed", action="store_true", default=False, help="Run distributed training.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import importlib.metadata as metadata  # noqa: E402
import logging  # noqa: E402
import time  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from packaging import version  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab.envs import ManagerBasedRLEnvCfg  # noqa: E402
from isaaclab.utils.dict import print_dict  # noqa: E402
from isaaclab.utils.io import dump_yaml  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.utils import get_checkpoint_path  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

import amadeus.tasks  # noqa: F401, E402
from amadeus.baselines.quadruped_rough.metrics import MetricsWriter, build_train_metrics_row, ensure_dir  # noqa: E402
from amadeus.tasks.manager_based.quadruped_rough.recorders import ChunkedHDF5DatasetFileHandler  # noqa: E402

logger = logging.getLogger(__name__)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def _resolve_resume_path(log_root_path: str, agent_cfg: RslRlBaseRunnerCfg) -> str | None:
    if not args_cli.resume:
        return None
    if args_cli.checkpoint:
        return args_cli.checkpoint
    load_run = args_cli.load_run or agent_cfg.load_run
    if not load_run:
        return None

    search_roots: list[str] = [log_root_path]
    # Transitional compatibility with short-layout runs created previously.
    compat_short_root = os.path.abspath(args_cli.output_root)
    if compat_short_root not in search_roots:
        search_roots.append(compat_short_root)

    for root in search_roots:
        try:
            return get_checkpoint_path(
                root,
                load_run,
                agent_cfg.load_checkpoint,
                other_dirs=["ckpt", "checkpoints"],
            )
        except ValueError:
            pass
        try:
            # Backward compatibility with older run layout where checkpoints lived in run root.
            return get_checkpoint_path(root, load_run, agent_cfg.load_checkpoint)
        except ValueError:
            pass
    raise ValueError(
        f"Unable to resolve checkpoint for resume. Tried roots={search_roots}, load_run={load_run}, "
        f"load_checkpoint={agent_cfg.load_checkpoint}"
    )


def _make_run_dir(agent_cfg: RslRlBaseRunnerCfg) -> tuple[str, str]:
    log_root_path = os.path.abspath(os.path.join(args_cli.output_root, "rsl_rl", agent_cfg.experiment_name))
    ensure_dir(log_root_path)
    if args_cli.resume and args_cli.load_run:
        candidate_default = os.path.join(log_root_path, args_cli.load_run)
        compat_short = os.path.abspath(os.path.join(args_cli.output_root, args_cli.load_run))
        if os.path.isdir(candidate_default):
            return log_root_path, candidate_default
        if os.path.isdir(compat_short):
            return os.path.abspath(args_cli.output_root), compat_short
        return log_root_path, candidate_default

    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M")
    if agent_cfg.run_name:
        run_name += f"_{agent_cfg.run_name}"
    return log_root_path, os.path.join(log_root_path, run_name)


def _build_run_paths(run_dir: str) -> dict[str, str]:
    return {
        "checkpoints": os.path.join(run_dir, "checkpoints"),
        "tensorboard": os.path.join(run_dir, "tensorboard"),
        "metrics": os.path.join(run_dir, "metrics"),
        "datasets_train": os.path.join(run_dir, "datasets", "train"),
        "datasets_eval": os.path.join(run_dir, "datasets", "eval"),
        "videos_train": os.path.join(run_dir, "videos", "train"),
        "videos_eval": os.path.join(run_dir, "videos", "eval"),
        "params": os.path.join(run_dir, "params"),
        "logs": os.path.join(run_dir, "logs"),
    }


def _latest_checkpoint_path(checkpoint_dir: str, current_iteration: int) -> str:
    return os.path.join(checkpoint_dir, f"model_{current_iteration}.pt")


def _run_periodic_eval(run_dir: str, checkpoint_path: str, iteration: int):
    metrics_dir = os.path.join(run_dir, "metrics")
    eval_video_dir = os.path.join(run_dir, "videos", "eval", f"iter_{iteration:04d}")
    base_cmd = [
        "/isaac-sim/python.sh",
        os.path.join(os.getcwd(), "scripts/baseline/eval_quadruped_rough.py"),
        "--task",
        args_cli.play_task,
        "--checkpoint",
        checkpoint_path,
        "--num_envs",
        str(args_cli.eval_num_envs),
        "--num_episodes",
        str(args_cli.eval_episodes),
        "--seed",
        str(args_cli.seed if args_cli.seed is not None else 42),
        "--metrics_dir",
        metrics_dir,
        "--headless",
    ]
    if args_cli.device:
        base_cmd.extend(["--device", args_cli.device])

    # First attempt: evaluate with short video capture.
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
            "[WARN] Periodic evaluation with video failed at "
            f"iteration {iteration}: {video_err}. Retrying once without video."
        )

    # Fallback attempt: evaluate without video (less renderer/network pressure).
    try:
        subprocess.run(base_cmd, check=True)
    except subprocess.CalledProcessError as eval_err:
        if args_cli.eval_strict:
            raise
        print(
            "[WARN] Periodic evaluation failed again without video at "
            f"iteration {iteration}: {eval_err}. Continuing training (eval_strict=False)."
        )


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    installed_version = metadata.version("rsl-rl-lib")
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.max_iterations is not None:
        agent_cfg.max_iterations = args_cli.max_iterations

    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    ChunkedHDF5DatasetFileHandler.max_episodes_per_file = args_cli.dataset_chunk_episodes

    log_root_path, run_dir = _make_run_dir(agent_cfg)
    run_paths = _build_run_paths(run_dir)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    print(f"Exact experiment name requested from command line: {os.path.basename(run_dir)}")

    for path in run_paths.values():
        ensure_dir(path)

    env_cfg.log_dir = run_dir
    env_cfg.recorders.dataset_export_dir_path = run_paths["datasets_train"]
    env_cfg.recorders.dataset_filename = "train_dataset"

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if args_cli.video:
        video_kwargs = {
            "video_folder": run_paths["videos_train"],
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording training videos.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=run_paths["tensorboard"], device=agent_cfg.device)

    # Keep TensorBoard in run_dir/tensorboard and force checkpoints into run_dir/checkpoints.
    original_save = runner.save

    def save_with_checkpoint_redirect(path: str, infos: dict | None = None):
        file_name = os.path.basename(path)
        target_path = path
        if file_name.startswith("model_") and file_name.endswith(".pt"):
            target_path = os.path.join(run_paths["checkpoints"], file_name)
        return original_save(target_path, infos)

    runner.save = save_with_checkpoint_redirect

    # Keep RSL-RL git/code-state snapshots under run_dir/logs/git.
    original_store_code_state = runner.logger._store_code_state

    def store_code_state_in_logs():
        original_log_dir = runner.logger.log_dir
        try:
            runner.logger.log_dir = run_paths["logs"]
            return original_store_code_state()
        finally:
            runner.logger.log_dir = original_log_dir

    runner.logger._store_code_state = store_code_state_in_logs
    runner.add_git_repo_to_log(__file__)

    metrics_writer = MetricsWriter(run_paths["metrics"])
    original_log = runner.logger.log

    def wrapped_log(*, it, start_it, total_it, collect_time, learn_time, loss_dict, learning_rate, action_std,
                    rnd_weight, print_minimal=False, width=80, pad=40):
        row = build_train_metrics_row(
            runner.logger,
            loss_dict=loss_dict,
            learning_rate=learning_rate,
            action_std=action_std,
            it=it,
            collect_time=collect_time,
            learn_time=learn_time,
        )
        metrics_writer.write_train_row(row)
        original_log(
            it=it,
            start_it=start_it,
            total_it=total_it,
            collect_time=collect_time,
            learn_time=learn_time,
            loss_dict=loss_dict,
            learning_rate=learning_rate,
            action_std=action_std,
            rnd_weight=rnd_weight,
            print_minimal=print_minimal,
            width=width,
            pad=pad,
        )

    runner.logger.log = wrapped_log

    resume_path = _resolve_resume_path(log_root_path, agent_cfg)
    if resume_path:
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        runner.load(resume_path)
        runner.current_learning_iteration += 1

    dump_yaml(os.path.join(run_paths["params"], "env.yaml"), env_cfg)
    dump_yaml(os.path.join(run_paths["params"], "agent.yaml"), agent_cfg)

    start_time = time.time()
    remaining = agent_cfg.max_iterations - runner.current_learning_iteration
    init_random_ep_len = not args_cli.resume

    while remaining > 0:
        chunk = min(args_cli.eval_interval, remaining)
        runner.learn(num_learning_iterations=chunk, init_at_random_ep_len=init_random_ep_len)
        if getattr(runner.logger, "writer", None) is not None and hasattr(runner.logger.writer, "close"):
            runner.logger.writer.close()
        checkpoint_path = _latest_checkpoint_path(run_paths["checkpoints"], runner.current_learning_iteration)
        _run_periodic_eval(run_dir, checkpoint_path, runner.current_learning_iteration)
        remaining = agent_cfg.max_iterations - (runner.current_learning_iteration + 1)
        init_random_ep_len = False
        if remaining > 0:
            runner.current_learning_iteration += 1

    print(f"Training time: {round(time.time() - start_time, 2)} seconds")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
