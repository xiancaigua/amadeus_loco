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
parser.add_argument(
    "--camera_eye",
    type=float,
    nargs=3,
    default=[4.0, 4.0, 3.0],
    metavar=("X", "Y", "Z"),
    help="Viewer camera eye position used for video recording.",
)
parser.add_argument(
    "--camera_lookat",
    type=float,
    nargs=3,
    default=[0.0, 0.0, 0.5],
    metavar=("X", "Y", "Z"),
    help="Viewer camera look-at target used for video recording.",
)
parser.add_argument("--video_start_step", type=int, default=0, help="Environment step index to start recording video.")
parser.add_argument(
    "--video_interval_steps",
    type=int,
    default=0,
    help="Start a new video every N environment steps after video_start_step (<=0 means single clip only).",
)
parser.add_argument(
    "--max_video_clips",
    type=int,
    default=1,
    help="Maximum number of video clips to save (<=0 means no explicit clip limit).",
)
parser.add_argument(
    "--follow_robot_camera",
    action="store_true",
    default=False,
    help="If set, camera follows one robot during evaluation video recording.",
)
parser.add_argument(
    "--camera_offset",
    type=float,
    nargs=3,
    default=[3.0, 3.0, 2.0],
    metavar=("X", "Y", "Z"),
    help="Camera offset from followed robot target when --follow_robot_camera is enabled.",
)
parser.add_argument(
    "--camera_robot_env_id",
    type=int,
    default=0,
    help="Environment index whose robot is used as follow-camera target.",
)
parser.add_argument(
    "--show_velocity_markers",
    action="store_true",
    default=False,
    help="Visualize command vs actual base velocity markers during evaluation/video.",
)
parser.add_argument(
    "--velocity_marker_scale",
    type=float,
    default=3.0,
    help="Scale factor for velocity marker length.",
)
parser.add_argument(
    "--velocity_marker_height",
    type=float,
    default=0.5,
    help="Vertical offset above robot base for velocity markers.",
)
parser.add_argument(
    "--velocity_marker_env_id",
    type=int,
    default=-1,
    help="Environment index used for velocity markers (-1 means camera_robot_env_id).",
)
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

import isaaclab.sim as sim_utils  # noqa: E402
import isaaclab.utils.math as math_utils  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnvCfg  # noqa: E402
from isaaclab.markers import CUBOID_MARKER_CFG, VisualizationMarkers  # noqa: E402
from isaaclab.utils.assets import retrieve_file_path  # noqa: E402
from isaaclab.utils.dict import print_dict  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

import amadeus.tasks  # noqa: F401, E402


def _infer_run_dir_from_checkpoint(checkpoint_path: str) -> str:
    checkpoint_dir = os.path.dirname(checkpoint_path)
    if os.path.basename(checkpoint_dir) == "checkpoints":
        return os.path.dirname(checkpoint_dir)
    # Backward compatibility with older layout where checkpoints lived directly under run_dir.
    return checkpoint_dir


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


def _make_video_step_trigger(args):
    """Create RecordVideo step trigger with optional periodic multi-clip capture."""
    trigger_state = {"clips_started": 0}

    def _trigger(step: int) -> bool:
        if step < args.video_start_step:
            return False
        if args.max_video_clips > 0 and trigger_state["clips_started"] >= args.max_video_clips:
            return False

        if args.video_interval_steps > 0:
            should_start = ((step - args.video_start_step) % args.video_interval_steps) == 0
        else:
            should_start = step == args.video_start_step

        if should_start:
            trigger_state["clips_started"] += 1
            return True
        return False

    return _trigger


def _resolve_xy_velocity_to_marker(xy_velocity: torch.Tensor, base_quat_w: torch.Tensor, scale_factor: float):
    """Convert XY velocity vectors into marker scales and quaternions in world frame."""
    marker_scale = torch.ones((xy_velocity.shape[0], 3), device=xy_velocity.device)
    marker_scale[:, 0] *= torch.linalg.norm(xy_velocity, dim=1) * scale_factor
    heading = torch.atan2(xy_velocity[:, 1], xy_velocity[:, 0])
    zeros = torch.zeros_like(heading)
    marker_quat = math_utils.quat_from_euler_xyz(zeros, zeros, heading)
    marker_quat = math_utils.quat_mul(base_quat_w, marker_quat)
    return marker_scale, marker_quat


def _create_velocity_markers():
    """Create geometry-only markers for command/actual velocity visualization."""
    cmd_cfg = CUBOID_MARKER_CFG.copy()
    cmd_cfg.prim_path = "/Visuals/EvalVelocity/command"
    cmd_cfg.markers["cuboid"].size = (0.15, 0.04, 0.04)
    cmd_cfg.markers["cuboid"].visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0))

    act_cfg = CUBOID_MARKER_CFG.copy()
    act_cfg.prim_path = "/Visuals/EvalVelocity/actual"
    act_cfg.markers["cuboid"].size = (0.15, 0.04, 0.04)
    act_cfg.markers["cuboid"].visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0))

    return {
        "command": VisualizationMarkers(cmd_cfg),
        "actual": VisualizationMarkers(act_cfg),
    }


def _update_velocity_markers(env, args, markers):
    """Update command/actual velocity markers for a selected environment."""
    base_env = env.unwrapped if hasattr(env, "unwrapped") else env
    robot = base_env.scene["robot"]

    marker_env_id = args.velocity_marker_env_id if args.velocity_marker_env_id >= 0 else args.camera_robot_env_id
    num_envs = int(robot.data.root_pos_w.shape[0])
    marker_env_id = max(0, min(marker_env_id, num_envs - 1))

    base_pos_w = robot.data.root_pos_w.clone()
    base_pos_w[:, 2] += float(args.velocity_marker_height)
    base_quat_w = robot.data.root_quat_w

    command = base_env.command_manager.get_command("base_velocity")[:, :2]
    actual = robot.data.root_lin_vel_b[:, :2]
    cmd_scale, cmd_quat = _resolve_xy_velocity_to_marker(command, base_quat_w, float(args.velocity_marker_scale))
    act_scale, act_quat = _resolve_xy_velocity_to_marker(actual, base_quat_w, float(args.velocity_marker_scale))

    env_slice = slice(marker_env_id, marker_env_id + 1)
    markers["command"].visualize(
        translations=base_pos_w[env_slice],
        orientations=cmd_quat[env_slice],
        scales=cmd_scale[env_slice],
    )
    markers["actual"].visualize(
        translations=base_pos_w[env_slice],
        orientations=act_quat[env_slice],
        scales=act_scale[env_slice],
    )


def _set_camera_view_with_fallback(base_env, eye: tuple[float, float, float], target: tuple[float, float, float]):
    """Set render camera using the best available API in this Isaac Lab runtime."""
    sim = getattr(base_env, "sim", None)
    if sim is not None and hasattr(sim, "set_camera_view"):
        try:
            sim.set_camera_view(eye=eye, target=target)
            return "env.unwrapped.sim.set_camera_view(eye=..., target=...)", None
        except TypeError:
            sim.set_camera_view(eye, target)
            return "env.unwrapped.sim.set_camera_view(eye, target)", None
        except Exception as exc:  # pragma: no cover - runtime dependent
            return None, exc

    try:
        from isaacsim.core.utils.viewports import set_camera_view as viewport_set_camera_view  # noqa: PLC0415

        viewport_set_camera_view(eye, target)
        return "isaacsim.core.utils.viewports.set_camera_view(eye, target)", None
    except Exception as exc:  # pragma: no cover - runtime dependent
        return None, exc


def set_eval_camera(env, args, *, log: bool = True):
    """Apply evaluation camera pose to the active render viewport."""
    base_env = env.unwrapped if hasattr(env, "unwrapped") else env
    eye = tuple(float(v) for v in args.camera_eye)
    target = tuple(float(v) for v in args.camera_lookat)
    robot_root_pos = None
    robot_env_id = args.camera_robot_env_id

    if args.follow_robot_camera:
        try:
            robot = base_env.scene["robot"]
            num_robot_envs = int(robot.data.root_pos_w.shape[0])
            if num_robot_envs <= 0:
                raise ValueError("robot.data.root_pos_w has zero environments.")
            robot_env_id = max(0, min(robot_env_id, num_robot_envs - 1))
            root_pos = robot.data.root_pos_w[robot_env_id].detach().cpu().tolist()
            robot_root_pos = tuple(float(v) for v in root_pos)
            target = (
                robot_root_pos[0],
                robot_root_pos[1],
                robot_root_pos[2] + 0.4,
            )
            offset = tuple(float(v) for v in args.camera_offset)
            eye = (
                target[0] + offset[0],
                target[1] + offset[1],
                target[2] + offset[2],
            )
        except Exception as exc:
            if log:
                print(f"[WARN] Failed to compute follow camera from robot root pose: {exc}. Falling back to fixed camera.")
            eye = tuple(float(v) for v in args.camera_eye)
            target = tuple(float(v) for v in args.camera_lookat)

    api_name, api_error = _set_camera_view_with_fallback(base_env, eye, target)
    if log:
        camera_log = {
            "follow_robot_camera": bool(args.follow_robot_camera),
            "camera_eye": eye,
            "camera_lookat": target,
            "camera_api": api_name if api_name is not None else "unavailable",
        }
        if args.follow_robot_camera:
            camera_log["camera_robot_env_id"] = int(robot_env_id)
            camera_log["camera_offset"] = tuple(float(v) for v in args.camera_offset)
            camera_log["robot_root_position"] = robot_root_pos
        if api_error is not None:
            camera_log["camera_api_error"] = str(api_error)
            print("[WARN] Unable to apply camera pose with available runtime APIs.")
        print("[INFO] Applied evaluation camera pose.")
        print_dict(camera_log, nesting=4)

    return api_name is not None


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    installed_version = metadata.version("rsl-rl-lib")
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    checkpoint_path = retrieve_file_path(args_cli.checkpoint)
    run_dir = _infer_run_dir_from_checkpoint(checkpoint_path)

    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.recorders.dataset_export_dir_path = args_cli.dataset_dir or os.path.join(run_dir, "datasets", "eval")
    env_cfg.recorders.dataset_filename = "eval_rollouts"
    if args_cli.video_folder:
        viewer_cfg = getattr(env_cfg, "viewer", None)
        if viewer_cfg is not None:
            eye_tuple = tuple(args_cli.camera_eye)
            lookat_tuple = tuple(args_cli.camera_lookat)
            if hasattr(viewer_cfg, "eye"):
                viewer_cfg.eye = eye_tuple
            elif hasattr(viewer_cfg, "cam_eye"):
                viewer_cfg.cam_eye = eye_tuple
            if hasattr(viewer_cfg, "lookat"):
                viewer_cfg.lookat = lookat_tuple
            elif hasattr(viewer_cfg, "cam_target"):
                viewer_cfg.cam_target = lookat_tuple

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video_folder else None)
    velocity_markers = None
    if args_cli.show_velocity_markers:
        try:
            velocity_markers = _create_velocity_markers()
            marker_log_kwargs = {
                "show_velocity_markers": True,
                "velocity_marker_scale": args_cli.velocity_marker_scale,
                "velocity_marker_height": args_cli.velocity_marker_height,
                "velocity_marker_env_id": args_cli.velocity_marker_env_id,
                "legend": {
                    "command": "green cuboid",
                    "actual": "blue cuboid",
                },
            }
            print("[INFO] Velocity marker visualization enabled.")
            print_dict(marker_log_kwargs, nesting=4)
        except Exception as exc:
            print(f"[WARN] Failed to initialize velocity markers: {exc}. Continuing without markers.")
            velocity_markers = None
    if args_cli.video_folder:
        set_eval_camera(env, args_cli, log=True)
        step_trigger = _make_video_step_trigger(args_cli)
        video_kwargs = {
            "video_folder": args_cli.video_folder,
            "step_trigger": step_trigger,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        camera_kwargs = {
            "camera_eye": tuple(args_cli.camera_eye),
            "camera_lookat": tuple(args_cli.camera_lookat),
            "follow_robot_camera": args_cli.follow_robot_camera,
            "camera_offset": tuple(args_cli.camera_offset),
            "camera_robot_env_id": args_cli.camera_robot_env_id,
        }
        video_log_kwargs = {
            "video_folder": args_cli.video_folder,
            "video_start_step": args_cli.video_start_step,
            "video_interval_steps": args_cli.video_interval_steps,
            "max_video_clips": args_cli.max_video_clips,
            "video_length": args_cli.video_length,
            "disable_logger": True,
            "step_trigger": (
                f"step == {args_cli.video_start_step}"
                if args_cli.video_interval_steps <= 0
                else f"(step - {args_cli.video_start_step}) % {args_cli.video_interval_steps} == 0"
            ),
        }
        print("[INFO] Evaluation camera configuration.")
        print_dict(camera_kwargs, nesting=4)
        print("[INFO] Recording evaluation video.")
        print_dict(video_log_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
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
            if args_cli.video_folder and args_cli.follow_robot_camera:
                set_eval_camera(env, args_cli, log=False)
            if velocity_markers is not None:
                _update_velocity_markers(env, args_cli, velocity_markers)
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
