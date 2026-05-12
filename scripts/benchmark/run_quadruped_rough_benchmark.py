# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Structured benchmark runner for quadruped rough-terrain PPO baseline.

This runner is suite/case configuration-driven and designed for batch execution.
It evaluates one checkpoint over multiple benchmark buckets, saves raw metrics,
summary metrics, traces, videos, plots, and a short research-facing summary.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from isaaclab.app import AppLauncher

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "rsl_rl"))
import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Run structured robustness benchmark on quadruped PPO checkpoint.")
parser.add_argument("--task", type=str, default="Template-Amadeus-Quadruped-Rough-v0", help="Task name.")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point", help="Agent config entry point.")
parser.add_argument(
    "--benchmark_cfg",
    type=str,
    default=os.path.join("scripts", "benchmark", "configs", "quadruped_rough_benchmark_suites.yaml"),
    help="Suite/case benchmark YAML config.",
)
parser.add_argument(
    "--suite_names",
    type=str,
    default=None,
    help="Optional comma-separated suite names. Default: all enabled suites.",
)
parser.add_argument(
    "--case_names",
    type=str,
    default=None,
    help="Optional comma-separated case names.",
)
parser.add_argument("--list_cases", action="store_true", default=False, help="List available suites/cases and exit.")
parser.add_argument(
    "--output_root",
    type=str,
    default="outputs/quadruped_rough_benchmark",
    help="Output root. Run layout: <output_root>/<run_name>/{config_snapshot,raw_metrics,summary_metrics,videos,traces,plots,reports}",
)
parser.add_argument("--benchmark_run_name", type=str, default=None, help="Optional run folder name.")
parser.add_argument("--num_envs", type=int, default=None, help="Override num_envs for all cases.")
parser.add_argument("--num_episodes", type=int, default=None, help="Override num_episodes for all cases.")
parser.add_argument("--seed", type=int, default=None, help="Override seed for all cases.")
parser.add_argument("--max_eval_steps", type=int, default=None, help="Override maximum eval steps for all cases.")
parser.add_argument(
    "--video",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Enable/disable benchmark video recording globally.",
)
parser.add_argument("--video_length", type=int, default=None, help="Override video length.")
parser.add_argument("--video_start_step", type=int, default=None, help="Override video start step.")
parser.add_argument("--video_max_clips", type=int, default=None, help="Override max clips per case.")
parser.add_argument("--trace_env_id", type=int, default=None, help="Override trace environment id.")
parser.add_argument("--trace_max_steps", type=int, default=None, help="Override trace max steps.")
parser.add_argument(
    "--save_plots",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Generate summary plots after benchmark run.",
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

cfg_path = os.path.abspath(args_cli.benchmark_cfg)
if args_cli.list_cases:
    with open(cfg_path, encoding="utf-8") as f:
        raw_cfg = yaml.safe_load(f)
    suites = raw_cfg.get("suites", {})
    payload = {
        name: {
            "enabled": bool(suite.get("enabled", True)),
            "description": suite.get("description", ""),
            "cases": [case.get("name") for case in suite.get("cases", [])],
        }
        for name, suite in suites.items()
    }
    print(json.dumps(payload, ensure_ascii=True, indent=2))
    raise SystemExit(0)

if args_cli.checkpoint is None:
    parser.error("--checkpoint is required")

if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import importlib.metadata as metadata  # noqa: E402
import math  # noqa: E402

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
from isaaclab.utils.io import dump_yaml  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

import amadeus.tasks  # noqa: F401, E402
from amadeus.benchmarking.quadruped_rough import (  # noqa: E402
    BenchmarkOutputLayout,
    ObservationMismatchRuntimeCfg,
    ObservationPerturbator,
    PushRuntimeCfg,
    RuntimePushScheduler,
    apply_env_cfg_overrides,
    load_benchmark_config,
    save_summary_artifacts,
    select_cases,
)

benchmark_cfg = load_benchmark_config(cfg_path)


def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _write_csv(path: str, rows: list[dict[str, Any]]):
    if not rows:
        return
    keys = sorted({k for row in rows for k in row.keys()})
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_json(path: str, payload: Any):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)


def _make_run_name(checkpoint_path: str) -> str:
    ckpt_stem = Path(checkpoint_path).stem
    return f"{datetime.utcnow().strftime('%Y-%m-%d_%H-%M-%S')}_{ckpt_stem}"


def _fmt_float(value: Any, digits: int = 4) -> str:
    try:
        number = float(value)
    except Exception:
        return "n/a"
    if math.isnan(number):
        return "n/a"
    return f"{number:.{digits}f}"


def _write_run_summary_md(
    run_dir: str,
    manifest: dict[str, Any],
    selected: list[tuple[str, Any]],
    summary_rows: list[dict[str, Any]],
):
    """Write a concise experiment overview for quick run triage."""
    out_path = os.path.join(run_dir, "RUN_SUMMARY.md")
    case_list = [f"{suite}:{case.name}" for suite, case in selected]
    buckets = sorted({str(row.get("bucket", "unknown")) for row in summary_rows})
    num_cases = len(summary_rows)
    mean_return = np.nan
    mean_fall_rate = np.nan
    mean_episode_length = np.nan
    if summary_rows:
        mean_return = float(np.nanmean([float(row.get("mean_return", np.nan)) for row in summary_rows]))
        mean_fall_rate = float(np.nanmean([float(row.get("fall_rate", np.nan)) for row in summary_rows]))
        mean_episode_length = float(np.nanmean([float(row.get("mean_episode_length", np.nan)) for row in summary_rows]))

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# Benchmark Run Summary\n\n")
        f.write("## Overview\n\n")
        f.write(f"- Run directory: `{run_dir}`\n")
        f.write(f"- Checkpoint: `{manifest.get('checkpoint')}`\n")
        f.write(f"- Benchmark config: `{manifest.get('benchmark_cfg')}`\n")
        f.write(f"- Number of cases: {num_cases}\n")
        f.write(f"- Buckets covered: {', '.join(buckets) if buckets else 'n/a'}\n")
        f.write(f"- Elapsed wall time (s): {manifest.get('elapsed_s', 'n/a')}\n")
        f.write("\n## Key Result Pointers\n\n")
        f.write(f"- Summary CSV: `{os.path.join(run_dir, 'summary_metrics', 'case_summary.csv')}`\n")
        f.write(f"- Summary JSON: `{os.path.join(run_dir, 'summary_metrics', 'case_summary.json')}`\n")
        f.write(f"- Raw episode metrics: `{os.path.join(run_dir, 'raw_metrics')}`\n")
        f.write(f"- Traces: `{os.path.join(run_dir, 'traces')}`\n")
        f.write(f"- Videos: `{os.path.join(run_dir, 'videos')}`\n")
        f.write(f"- Plots: `{os.path.join(run_dir, 'plots')}`\n")
        f.write(f"- Research summary: `{os.path.join(run_dir, 'reports', 'research_summary.md')}`\n")
        f.write("\n## Quick Aggregates (case-level means)\n\n")
        f.write(f"- Mean return across cases: {_fmt_float(mean_return)}\n")
        f.write(f"- Mean fall rate across cases: {_fmt_float(mean_fall_rate)}\n")
        f.write(f"- Mean episode length across cases: {_fmt_float(mean_episode_length)}\n")
        f.write("\n## Selected Cases\n\n")
        for item in case_list:
            f.write(f"- `{item}`\n")
        f.write("\n## Notes\n\n")
        f.write("- This file is auto-generated for quick experiment handoff.\n")
        f.write("- For per-case details, open `raw_metrics/<suite>/<case>/summary.json`.\n")


def _set_camera_view_with_fallback(base_env, eye: tuple[float, float, float], target: tuple[float, float, float]):
    sim = getattr(base_env, "sim", None)
    if sim is not None and hasattr(sim, "set_camera_view"):
        try:
            sim.set_camera_view(eye=eye, target=target)
            return "env.unwrapped.sim.set_camera_view(eye=..., target=...)", None
        except TypeError:
            sim.set_camera_view(eye, target)
            return "env.unwrapped.sim.set_camera_view(eye, target)", None
        except Exception as exc:  # pragma: no cover
            return None, exc

    try:
        from isaacsim.core.utils.viewports import set_camera_view as viewport_set_camera_view  # noqa: PLC0415

        viewport_set_camera_view(eye, target)
        return "isaacsim.core.utils.viewports.set_camera_view(eye, target)", None
    except Exception as exc:  # pragma: no cover
        return None, exc


def _apply_camera_from_video_cfg(env, video_cfg: dict[str, Any], *, log: bool = True):
    base_env = env.unwrapped if hasattr(env, "unwrapped") else env
    follow = bool(video_cfg.get("follow_robot_camera", False))
    eye = tuple(float(v) for v in video_cfg.get("camera_eye", [4.0, 4.0, 3.0]))
    target = tuple(float(v) for v in video_cfg.get("camera_lookat", [0.0, 0.0, 0.5]))
    robot_root_pos = None
    robot_env_id = int(video_cfg.get("camera_robot_env_id", 0))

    if follow:
        try:
            robot = base_env.scene["robot"]
            num_robot_envs = int(robot.data.root_pos_w.shape[0])
            robot_env_id = max(0, min(robot_env_id, num_robot_envs - 1))
            root_pos = robot.data.root_pos_w[robot_env_id].detach().cpu().tolist()
            robot_root_pos = tuple(float(v) for v in root_pos)
            target = (robot_root_pos[0], robot_root_pos[1], robot_root_pos[2] + 0.4)
            offset = tuple(float(v) for v in video_cfg.get("camera_offset", [3.0, 3.0, 2.0]))
            eye = (target[0] + offset[0], target[1] + offset[1], target[2] + offset[2])
        except Exception as exc:
            if log:
                print(f"[WARN] Follow-camera target failed: {exc}. Falling back to fixed camera.")
            follow = False

    api_name, api_error = _set_camera_view_with_fallback(base_env, eye, target)
    if log:
        payload = {
            "follow_robot_camera": follow,
            "camera_eye": eye,
            "camera_lookat": target,
            "camera_api": api_name if api_name is not None else "unavailable",
        }
        if follow:
            payload["camera_robot_env_id"] = robot_env_id
            payload["camera_offset"] = tuple(float(v) for v in video_cfg.get("camera_offset", [3.0, 3.0, 2.0]))
            payload["robot_root_position"] = robot_root_pos
        if api_error is not None:
            payload["camera_api_error"] = str(api_error)
        print("[INFO] Benchmark camera configuration:")
        print_dict(payload, nesting=4)


def _make_video_step_trigger(start_step: int, interval_steps: int, max_clips: int):
    state = {"clips_started": 0}

    def _trigger(step: int) -> bool:
        if step < start_step:
            return False
        if max_clips > 0 and state["clips_started"] >= max_clips:
            return False
        if interval_steps > 0:
            should_start = ((step - start_step) % interval_steps) == 0
        else:
            should_start = step == start_step
        if should_start:
            state["clips_started"] += 1
            return True
        return False

    return _trigger


def _resolve_xy_velocity_to_marker(xy_velocity: torch.Tensor, base_quat_w: torch.Tensor, scale_factor: float):
    marker_scale = torch.ones((xy_velocity.shape[0], 3), device=xy_velocity.device)
    marker_scale[:, 0] *= torch.linalg.norm(xy_velocity, dim=1) * scale_factor
    heading = torch.atan2(xy_velocity[:, 1], xy_velocity[:, 0])
    zeros = torch.zeros_like(heading)
    marker_quat = math_utils.quat_from_euler_xyz(zeros, zeros, heading)
    marker_quat = math_utils.quat_mul(base_quat_w, marker_quat)
    return marker_scale, marker_quat


def _create_velocity_markers():
    cmd_cfg = CUBOID_MARKER_CFG.copy()
    cmd_cfg.prim_path = "/Visuals/BenchmarkVelocity/command"
    cmd_cfg.markers["cuboid"].size = (0.15, 0.04, 0.04)
    cmd_cfg.markers["cuboid"].visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0))

    act_cfg = CUBOID_MARKER_CFG.copy()
    act_cfg.prim_path = "/Visuals/BenchmarkVelocity/actual"
    act_cfg.markers["cuboid"].size = (0.15, 0.04, 0.04)
    act_cfg.markers["cuboid"].visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0))
    return {"command": VisualizationMarkers(cmd_cfg), "actual": VisualizationMarkers(act_cfg)}


def _update_velocity_markers(env, video_cfg: dict[str, Any], markers):
    base_env = env.unwrapped if hasattr(env, "unwrapped") else env
    robot = base_env.scene["robot"]

    marker_env_id = int(video_cfg.get("velocity_marker_env_id", -1))
    if marker_env_id < 0:
        marker_env_id = int(video_cfg.get("camera_robot_env_id", 0))
    num_envs = int(robot.data.root_pos_w.shape[0])
    marker_env_id = max(0, min(marker_env_id, num_envs - 1))

    base_pos_w = robot.data.root_pos_w.clone()
    base_pos_w[:, 2] += float(video_cfg.get("velocity_marker_height", 0.5))
    base_quat_w = robot.data.root_quat_w
    command = base_env.command_manager.get_command("base_velocity")[:, :2]
    actual = robot.data.root_lin_vel_b[:, :2]
    cmd_scale, cmd_quat = _resolve_xy_velocity_to_marker(command, base_quat_w, float(video_cfg.get("velocity_marker_scale", 3.0)))
    act_scale, act_quat = _resolve_xy_velocity_to_marker(actual, base_quat_w, float(video_cfg.get("velocity_marker_scale", 3.0)))

    env_slice = slice(marker_env_id, marker_env_id + 1)
    markers["command"].visualize(translations=base_pos_w[env_slice], orientations=cmd_quat[env_slice], scales=cmd_scale[env_slice])
    markers["actual"].visualize(translations=base_pos_w[env_slice], orientations=act_quat[env_slice], scales=act_scale[env_slice])


def _safe_float(value) -> float:
    if value is None:
        return float("nan")
    return float(value)


def _build_trace_row(step: int, env_id: int, command: torch.Tensor, robot) -> dict[str, float]:
    root_pos = robot.data.root_pos_w[env_id]
    root_lin = robot.data.root_lin_vel_b[env_id]
    root_ang = robot.data.root_ang_vel_b[env_id]
    roll, pitch, yaw = math_utils.euler_xyz_from_quat(robot.data.root_quat_w[env_id].unsqueeze(0))
    return {
        "step": int(step),
        "env_id": int(env_id),
        "command_vx": float(command[env_id, 0].item()),
        "command_vy": float(command[env_id, 1].item()),
        "command_wz": float(command[env_id, 2].item()),
        "actual_vx": float(root_lin[0].item()),
        "actual_vy": float(root_lin[1].item()),
        "actual_wz": float(root_ang[2].item()),
        "base_height": float(root_pos[2].item()),
        "roll": float(roll[0].item()),
        "pitch": float(pitch[0].item()),
        "yaw": float(yaw[0].item()),
    }


def _summarize_episodes(episode_rows: list[dict[str, Any]]) -> dict[str, float]:
    if not episode_rows:
        return {}

    def arr(key):
        return np.array([_safe_float(r.get(key, np.nan)) for r in episode_rows], dtype=np.float64)

    out = {
        "num_episodes": len(episode_rows),
        "mean_return": float(np.nanmean(arr("episode_return"))),
        "mean_episode_length": float(np.nanmean(arr("episode_length"))),
        "mean_survival_time_s": float(np.nanmean(arr("survival_time_s"))),
        "mean_lin_vel_tracking_error": float(np.nanmean(arr("lin_vel_tracking_error"))),
        "mean_yaw_vel_tracking_error": float(np.nanmean(arr("yaw_vel_tracking_error"))),
        "fall_rate": float(np.nanmean(arr("fell"))),
        "timeout_rate": float(np.nanmean(arr("timed_out"))),
        "mean_action_smoothness": float(np.nanmean(arr("action_smoothness"))),
        "mean_energy_proxy": float(np.nanmean(arr("energy_proxy"))),
        "mean_base_height": float(np.nanmean(arr("mean_base_height"))),
        "mean_abs_roll": float(np.nanmean(arr("mean_abs_roll"))),
        "mean_abs_pitch": float(np.nanmean(arr("mean_abs_pitch"))),
    }
    recovery_values = arr("recovery_time_s")
    valid_recovery = recovery_values[~np.isnan(recovery_values)]
    out["mean_recovery_time_s"] = float(np.mean(valid_recovery)) if len(valid_recovery) > 0 else float("nan")
    out["recovery_event_count"] = int(np.nansum(arr("recovery_event_count")))
    out["recovery_success_count"] = int(np.nansum(arr("recovery_success_count")))
    out["recovery_success_rate"] = (
        out["recovery_success_count"] / out["recovery_event_count"] if out["recovery_event_count"] > 0 else float("nan")
    )
    return out


def _run_case(
    suite_name: str,
    case,
    base_env_cfg: ManagerBasedRLEnvCfg,
    agent_cfg: RslRlBaseRunnerCfg,
    checkpoint_path: str,
    cfg,
    output_layout: BenchmarkOutputLayout,
) -> dict[str, Any]:
    case_raw_dir = output_layout.case_raw_metrics_dir(suite_name, case.name)
    case_trace_dir = output_layout.case_traces_dir(suite_name, case.name)
    case_video_dir = output_layout.case_videos_dir(suite_name, case.name)
    _ensure_dir(case_raw_dir)
    _ensure_dir(case_trace_dir)
    _ensure_dir(case_video_dir)

    env_cfg = deepcopy(base_env_cfg)
    apply_env_cfg_overrides(env_cfg, case.overrides)

    global_cfg = cfg.global_cfg
    video_cfg = deepcopy(global_cfg.video.__dict__)
    if args_cli.video_length is not None:
        video_cfg["video_length"] = int(args_cli.video_length)
    if args_cli.video_start_step is not None:
        video_cfg["video_start_step"] = int(args_cli.video_start_step)
    if args_cli.video_max_clips is not None:
        video_cfg["max_clips_per_case"] = int(args_cli.video_max_clips)

    num_envs = int(args_cli.num_envs or case.num_envs or global_cfg.num_envs)
    num_episodes = int(args_cli.num_episodes or case.num_episodes or global_cfg.num_episodes)
    seed = int(args_cli.seed if args_cli.seed is not None else (case.seed if case.seed is not None else global_cfg.seed))
    max_eval_steps = int(args_cli.max_eval_steps or case.max_eval_steps or global_cfg.max_eval_steps)
    if case.save_video is not None:
        video_enabled = bool(case.save_video and args_cli.video and video_cfg.get("enabled", True))
    else:
        video_enabled = bool(args_cli.video and video_cfg.get("enabled", True))

    if video_enabled and args_cli.num_envs is None:
        num_envs = int(video_cfg.get("num_envs", num_envs))
    if video_enabled and args_cli.num_episodes is None:
        num_episodes = int(video_cfg.get("num_episodes", num_episodes))

    env_cfg.scene.num_envs = num_envs
    env_cfg.seed = seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else global_cfg.device
    env_cfg.observations.policy.enable_corruption = True
    env_cfg.recorders = None

    render_mode = "rgb_array" if video_enabled else None
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=render_mode)
    markers = None
    if video_enabled and bool(video_cfg.get("show_velocity_markers", False)):
        try:
            markers = _create_velocity_markers()
        except Exception as exc:
            print(f"[WARN] Failed to create velocity markers: {exc}")
            markers = None

    if video_enabled:
        _apply_camera_from_video_cfg(env, video_cfg, log=True)
        video_kwargs = {
            "video_folder": case_video_dir,
            "step_trigger": _make_video_step_trigger(
                start_step=int(video_cfg.get("video_start_step", 20)),
                interval_steps=int(video_cfg.get("video_interval_steps", 0)),
                max_clips=int(video_cfg.get("max_clips_per_case", 2)),
            ),
            "video_length": int(video_cfg.get("video_length", 900)),
            "disable_logger": True,
        }
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(checkpoint_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    obs_pert = ObservationPerturbator(ObservationMismatchRuntimeCfg.from_overrides(case.overrides))
    push_sched = RuntimePushScheduler(
        cfg=PushRuntimeCfg.from_overrides(case.overrides),
        num_envs=env.num_envs,
        device=env.unwrapped.device,
    )

    trace_env_id = int(args_cli.trace_env_id if args_cli.trace_env_id is not None else global_cfg.trace.env_id)
    trace_max_steps = int(args_cli.trace_max_steps if args_cli.trace_max_steps is not None else global_cfg.trace.max_steps)
    trace_rows: list[dict[str, Any]] = []

    recovery_trigger = float(global_cfg.recovery.trigger_error)
    recovery_clear = float(global_cfg.recovery.clear_error)
    recovery_hold_steps = int(global_cfg.recovery.min_hold_steps)

    ep_reward = torch.zeros(env.num_envs, device=env.device)
    ep_length = torch.zeros(env.num_envs, device=env.device)
    ep_lin_error = torch.zeros(env.num_envs, device=env.device)
    ep_yaw_error = torch.zeros(env.num_envs, device=env.device)
    ep_action_smooth = torch.zeros(env.num_envs, device=env.device)
    ep_energy = torch.zeros(env.num_envs, device=env.device)
    ep_base_height = torch.zeros(env.num_envs, device=env.device)
    ep_abs_roll = torch.zeros(env.num_envs, device=env.device)
    ep_abs_pitch = torch.zeros(env.num_envs, device=env.device)
    prev_actions = None

    in_recovery = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    recovery_start_step = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    recovery_hold = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    recovery_sum = torch.zeros(env.num_envs, device=env.device)
    recovery_event_count = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    recovery_success_count = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

    term_names = list(env.unwrapped.termination_manager.active_terms)
    termination_counter = {name: 0 for name in term_names}
    termination_counter["time_out"] = 0
    termination_counter["unknown"] = 0

    episode_rows: list[dict[str, Any]] = []
    completed_episodes = 0
    global_step = 0

    obs = env.get_observations()
    obs = obs_pert.apply(obs, done_mask=None)

    while simulation_app.is_running() and completed_episodes < num_episodes and global_step < max_eval_steps:
        with torch.inference_mode():
            actions = policy(obs)
            obs_next, rewards, dones, extras = env.step(actions)
            base_env = env.unwrapped
            done_mask = base_env.reset_buf.reshape(-1).bool()

            if push_sched.enabled:
                push_sched.on_step(base_env, global_step)

            if video_enabled and bool(video_cfg.get("follow_robot_camera", False)):
                _apply_camera_from_video_cfg(env, video_cfg, log=False)
            if markers is not None:
                _update_velocity_markers(env, video_cfg, markers)

            obs_next = obs_pert.apply(obs_next, done_mask=done_mask)
            obs = obs_next

            robot = base_env.scene["robot"]
            command = base_env.command_manager.get_command("base_velocity")

            lin_error = torch.linalg.norm(command[:, :2] - robot.data.root_lin_vel_b[:, :2], dim=1)
            yaw_error = torch.abs(command[:, 2] - robot.data.root_ang_vel_b[:, 2])
            roll, pitch, _ = math_utils.euler_xyz_from_quat(robot.data.root_quat_w)
            if prev_actions is None:
                smooth = torch.zeros(actions.shape[0], device=actions.device)
            else:
                smooth = torch.linalg.norm(actions - prev_actions, dim=1)
            energy = torch.sum(torch.abs(robot.data.applied_torque * robot.data.joint_vel), dim=1)

            ep_reward += rewards.reshape(-1)
            ep_length += 1
            ep_lin_error += lin_error
            ep_yaw_error += yaw_error
            ep_action_smooth += smooth
            ep_energy += energy
            ep_base_height += robot.data.root_pos_w[:, 2]
            ep_abs_roll += torch.abs(roll)
            ep_abs_pitch += torch.abs(pitch)
            prev_actions = actions.clone()

            event_begin = (~in_recovery) & (lin_error > recovery_trigger)
            recovery_start_step[event_begin] = ep_length[event_begin].long()
            recovery_event_count[event_begin] += 1
            in_recovery[event_begin] = True
            recovery_hold[event_begin] = 0

            low_error = lin_error < recovery_clear
            recovery_hold[in_recovery & low_error] += 1
            recovery_hold[in_recovery & (~low_error)] = 0
            recovered = in_recovery & (recovery_hold >= recovery_hold_steps)
            if torch.any(recovered):
                duration = (ep_length[recovered] - recovery_start_step[recovered].float()) * base_env.step_dt
                recovery_sum[recovered] += duration
                recovery_success_count[recovered] += 1
                in_recovery[recovered] = False
                recovery_hold[recovered] = 0

            if global_step < trace_max_steps:
                safe_trace_env_id = max(0, min(trace_env_id, env.num_envs - 1))
                trace_rows.append(_build_trace_row(global_step, safe_trace_env_id, command, robot))

            timeout_mask = base_env.reset_time_outs.reshape(-1).bool()
            for env_id in done_mask.nonzero(as_tuple=False).reshape(-1).tolist():
                if completed_episodes >= num_episodes:
                    break
                length = torch.clamp(ep_length[env_id], min=1.0)
                timed_out = bool(timeout_mask[env_id].item())
                fell = 0.0 if timed_out else 1.0

                primary_reason = "time_out" if timed_out else "unknown"
                if timed_out:
                    termination_counter["time_out"] += 1
                else:
                    matched = False
                    for term in term_names:
                        term_val = bool(base_env.termination_manager.get_term(term)[env_id].item())
                        if term_val:
                            termination_counter[term] += 1
                            if not matched:
                                primary_reason = term
                                matched = True
                    if not matched:
                        termination_counter["unknown"] += 1

                recovery_time = (
                    (recovery_sum[env_id] / max(int(recovery_success_count[env_id].item()), 1)).item()
                    if int(recovery_success_count[env_id].item()) > 0
                    else float("nan")
                )
                episode_rows.append(
                    {
                        "suite_name": suite_name,
                        "case_name": case.name,
                        "bucket": case.bucket,
                        "episode_index": completed_episodes,
                        "env_id": env_id,
                        "episode_return": float(ep_reward[env_id].item()),
                        "episode_length": float(ep_length[env_id].item()),
                        "survival_time_s": float(ep_length[env_id].item() * base_env.step_dt),
                        "lin_vel_tracking_error": float((ep_lin_error[env_id] / length).item()),
                        "yaw_vel_tracking_error": float((ep_yaw_error[env_id] / length).item()),
                        "action_smoothness": float((ep_action_smooth[env_id] / length).item()),
                        "energy_proxy": float((ep_energy[env_id] / length).item()),
                        "mean_base_height": float((ep_base_height[env_id] / length).item()),
                        "mean_abs_roll": float((ep_abs_roll[env_id] / length).item()),
                        "mean_abs_pitch": float((ep_abs_pitch[env_id] / length).item()),
                        "recovery_event_count": int(recovery_event_count[env_id].item()),
                        "recovery_success_count": int(recovery_success_count[env_id].item()),
                        "recovery_time_s": float(recovery_time),
                        "timed_out": float(timed_out),
                        "fell": float(fell),
                        "termination_reason": primary_reason,
                    }
                )

                ep_reward[env_id] = 0
                ep_length[env_id] = 0
                ep_lin_error[env_id] = 0
                ep_yaw_error[env_id] = 0
                ep_action_smooth[env_id] = 0
                ep_energy[env_id] = 0
                ep_base_height[env_id] = 0
                ep_abs_roll[env_id] = 0
                ep_abs_pitch[env_id] = 0
                in_recovery[env_id] = False
                recovery_start_step[env_id] = 0
                recovery_hold[env_id] = 0
                recovery_sum[env_id] = 0
                recovery_event_count[env_id] = 0
                recovery_success_count[env_id] = 0
                completed_episodes += 1

            if version.parse(metadata.version("rsl-rl-lib")) >= version.parse("4.0.0"):
                policy.reset(done_mask)
            global_step += 1

    summary = _summarize_episodes(episode_rows)
    grid_cfg = case.overrides.get("grid", {}) if isinstance(case.overrides, dict) else {}
    summary.update(
        {
            "suite_name": suite_name,
            "case_name": case.name,
            "bucket": case.bucket,
            "description": case.description,
            "checkpoint": checkpoint_path,
            "num_envs": num_envs,
            "seed": seed,
            "target_num_episodes": num_episodes,
            "completed_episodes": completed_episodes,
            "max_eval_steps": max_eval_steps,
            "early_stop": bool(completed_episodes < num_episodes),
            "runtime_obs_delay_steps": obs_pert.cfg.delay_steps,
            "runtime_obs_noise_std": obs_pert.cfg.additive_noise_std,
            "runtime_obs_drop_prob": obs_pert.cfg.drop_prob,
            "runtime_push_pattern": push_sched.cfg.pattern,
            "runtime_push_enabled": push_sched.cfg.enabled,
            "grid_x": grid_cfg.get("x"),
            "grid_y": grid_cfg.get("y"),
        }
    )

    _write_csv(os.path.join(case_raw_dir, "episodes.csv"), episode_rows)
    _write_json(os.path.join(case_raw_dir, "episodes.json"), episode_rows)
    _write_json(os.path.join(case_raw_dir, "summary.json"), summary)
    term_rows = [{"suite_name": suite_name, "case_name": case.name, "termination_reason": k, "count": v} for k, v in termination_counter.items()]
    _write_csv(os.path.join(case_raw_dir, "termination_stats.csv"), term_rows)
    _write_csv(os.path.join(case_trace_dir, "trace_env.csv"), trace_rows)
    _write_json(os.path.join(case_raw_dir, "case_overrides_snapshot.json"), case.overrides)

    print(f"[INFO] Case finished: {suite_name}:{case.name}")
    print_dict(summary, nesting=4)
    env.close()
    return summary


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    installed_version = metadata.version("rsl-rl-lib")
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    checkpoint_path = retrieve_file_path(args_cli.checkpoint)
    suite_names = [s.strip() for s in args_cli.suite_names.split(",") if s.strip()] if args_cli.suite_names else None
    case_names = [s.strip() for s in args_cli.case_names.split(",") if s.strip()] if args_cli.case_names else None
    selected = select_cases(benchmark_cfg, suite_names=suite_names, case_names=case_names)

    if args_cli.seed is not None:
        env_cfg.seed = args_cli.seed
    if args_cli.device:
        env_cfg.sim.device = args_cli.device
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = int(args_cli.num_envs)

    run_name = args_cli.benchmark_run_name or _make_run_name(checkpoint_path)
    run_dir = os.path.abspath(os.path.join(args_cli.output_root, run_name))
    output_layout = BenchmarkOutputLayout.create(run_dir)

    dump_yaml(os.path.join(output_layout.config_snapshot_dir, "env_base.yaml"), env_cfg)
    dump_yaml(os.path.join(output_layout.config_snapshot_dir, "agent_base.yaml"), agent_cfg)
    _write_json(os.path.join(output_layout.config_snapshot_dir, "runtime_args.json"), vars(args_cli))
    _write_json(os.path.join(output_layout.config_snapshot_dir, "benchmark_cfg_resolved.json"), benchmark_cfg.to_dict())
    _write_json(
        os.path.join(output_layout.config_snapshot_dir, "selected_cases.json"),
        [{"suite_name": suite_name, "case_name": case.name, "bucket": case.bucket} for suite_name, case in selected],
    )

    print("[INFO] Benchmark run directory:")
    print(run_dir)
    print("[INFO] Selected cases:")
    print_dict({"cases": [f"{suite_name}:{case.name}" for suite_name, case in selected]}, nesting=4)

    summary_rows: list[dict[str, Any]] = []
    wall_start = time.time()
    for suite_name, case in selected:
        case_start = time.time()
        summary = _run_case(
            suite_name=suite_name,
            case=case,
            base_env_cfg=env_cfg,
            agent_cfg=agent_cfg,
            checkpoint_path=checkpoint_path,
            cfg=benchmark_cfg,
            output_layout=output_layout,
        )
        summary["wall_time_s"] = round(time.time() - case_start, 2)
        summary_rows.append(summary)

    manifest = {
        "run_dir": run_dir,
        "checkpoint": checkpoint_path,
        "benchmark_cfg": cfg_path,
        "num_cases": len(summary_rows),
        "elapsed_s": round(time.time() - wall_start, 2),
    }

    if args_cli.save_plots:
        generated = save_summary_artifacts(summary_rows, output_layout)
        manifest.update(generated)
    else:
        _write_csv(os.path.join(output_layout.summary_metrics_dir, "case_summary.csv"), summary_rows)
        _write_json(os.path.join(output_layout.summary_metrics_dir, "case_summary.json"), summary_rows)

    _write_json(os.path.join(output_layout.summary_metrics_dir, "index.json"), manifest)
    _write_run_summary_md(run_dir, manifest, selected, summary_rows)
    print("[INFO] Benchmark completed.")
    print_dict(manifest, nesting=4)


if __name__ == "__main__":
    main()
    simulation_app.close()
