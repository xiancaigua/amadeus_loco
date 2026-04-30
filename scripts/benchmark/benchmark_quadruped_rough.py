# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Robustness benchmark for the existing quadruped rough-terrain PPO baseline.

This script evaluates a trained PPO checkpoint under three scenario groups:
- in_distribution
- long_tail
- ood

It reuses the existing manager-based task and only applies scenario-specific overrides
at evaluation time, without touching the training pipeline.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from isaaclab.app import AppLauncher

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "rsl_rl"))
import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Run robustness benchmark on quadruped PPO checkpoint.")
parser.add_argument("--task", type=str, default="Template-Amadeus-Quadruped-Rough-v0", help="Task name.")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point", help="Agent config entry point.")
parser.add_argument(
    "--scenario_cfg",
    type=str,
    default=os.path.join("scripts", "benchmark", "configs", "quadruped_rough_robustness.yaml"),
    help="Benchmark scenario YAML config.",
)
parser.add_argument(
    "--scenario_group",
    type=str,
    default="all",
    choices=["all", "in_distribution", "long_tail", "ood"],
    help="Scenario group to run.",
)
parser.add_argument(
    "--scenario_names",
    type=str,
    default=None,
    help="Optional comma-separated scenario names (overrides --scenario_group filter).",
)
parser.add_argument(
    "--output_root",
    type=str,
    default="outputs/quadruped_rough_benchmark",
    help="Benchmark output root directory.",
)
parser.add_argument("--benchmark_run_name", type=str, default=None, help="Optional benchmark run folder name.")
parser.add_argument("--num_envs", type=int, default=None, help="Override eval num_envs for all scenarios.")
parser.add_argument("--num_episodes", type=int, default=None, help="Override eval episodes for all scenarios.")
parser.add_argument("--seed", type=int, default=None, help="Override seed.")
parser.add_argument(
    "--video",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Export representative videos per scenario.",
)
parser.add_argument(
    "--video_length",
    type=int,
    default=None,
    help="Override video length in steps.",
)
parser.add_argument(
    "--video_start_step",
    type=int,
    default=None,
    help="Override video capture start step.",
)
parser.add_argument(
    "--video_max_clips",
    type=int,
    default=None,
    help="Override max clips per scenario.",
)
parser.add_argument(
    "--trace_env_id",
    type=int,
    default=None,
    help="Override trace environment index.",
)
parser.add_argument(
    "--trace_max_steps",
    type=int,
    default=None,
    help="Override max step-trace length.",
)
parser.add_argument(
    "--save_plots",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Save summary comparison plots.",
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.checkpoint is None:
    parser.error("--checkpoint is required")

if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import importlib.metadata as metadata  # noqa: E402

import gymnasium as gym  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from packaging import version  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import isaaclab.utils.math as math_utils  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnvCfg  # noqa: E402
from isaaclab.utils.assets import retrieve_file_path  # noqa: E402
from isaaclab.utils.dict import print_dict  # noqa: E402
from isaaclab.utils.io import dump_yaml  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

import amadeus.tasks  # noqa: F401, E402


def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _make_run_name(ckpt_path: str) -> str:
    ckpt_stem = Path(ckpt_path).stem
    return f"{datetime.utcnow().strftime('%Y-%m-%d_%H-%M-%S')}_{ckpt_stem}"


def _load_scenario_cfg(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Scenario config must be a mapping: {path}")
    if "scenarios" not in data:
        raise ValueError(f"Scenario config missing 'scenarios': {path}")
    return data


def _select_scenarios(all_scenarios: list[dict[str, Any]], group: str, names_csv: str | None) -> list[dict[str, Any]]:
    if names_csv:
        names = {name.strip() for name in names_csv.split(",") if name.strip()}
        selected = [s for s in all_scenarios if s.get("name") in names]
    elif group == "all":
        selected = list(all_scenarios)
    else:
        selected = [s for s in all_scenarios if s.get("group") == group]
    if not selected:
        raise ValueError(f"No scenarios selected. group={group}, names={names_csv}")
    return selected


def _scale_range(range_pair: tuple[float, float] | list[float], scale: float) -> tuple[float, float]:
    return float(range_pair[0]) * scale, float(range_pair[1]) * scale


def _set_pair_if_present(obj: Any, field: str, values: list[float] | tuple[float, float] | None):
    if values is None:
        return
    if hasattr(obj, field):
        setattr(obj, field, (float(values[0]), float(values[1])))


def _apply_overrides(env_cfg: ManagerBasedRLEnvCfg, overrides: dict[str, Any]):
    """Apply scenario-level distribution overrides to env cfg in-place."""
    if not overrides:
        return

    # Command distribution
    cmd_cfg = overrides.get("command", {})
    if hasattr(env_cfg.commands, "base_velocity") and cmd_cfg:
        ranges = env_cfg.commands.base_velocity.ranges
        _set_pair_if_present(ranges, "lin_vel_x", cmd_cfg.get("lin_vel_x"))
        _set_pair_if_present(ranges, "lin_vel_y", cmd_cfg.get("lin_vel_y"))
        _set_pair_if_present(ranges, "ang_vel_z", cmd_cfg.get("ang_vel_z"))
        _set_pair_if_present(ranges, "heading", cmd_cfg.get("heading"))

    # Disturbance pushes
    push_cfg = overrides.get("push", {})
    if push_cfg:
        enabled = push_cfg.get("enabled", True)
        if not enabled:
            env_cfg.events.push_robot = None
        elif getattr(env_cfg.events, "push_robot", None) is not None:
            if "interval_s" in push_cfg:
                env_cfg.events.push_robot.interval_range_s = (
                    float(push_cfg["interval_s"][0]),
                    float(push_cfg["interval_s"][1]),
                )
            velocity_range = env_cfg.events.push_robot.params.get("velocity_range", {})
            if "vel_x" in push_cfg:
                velocity_range["x"] = (float(push_cfg["vel_x"][0]), float(push_cfg["vel_x"][1]))
            if "vel_y" in push_cfg:
                velocity_range["y"] = (float(push_cfg["vel_y"][0]), float(push_cfg["vel_y"][1]))
            env_cfg.events.push_robot.params["velocity_range"] = velocity_range

    # Friction randomization ranges
    friction_cfg = overrides.get("friction", {})
    if friction_cfg and getattr(env_cfg.events, "physics_material", None) is not None:
        params = env_cfg.events.physics_material.params
        if "static" in friction_cfg:
            params["static_friction_range"] = (float(friction_cfg["static"][0]), float(friction_cfg["static"][1]))
        if "dynamic" in friction_cfg:
            params["dynamic_friction_range"] = (float(friction_cfg["dynamic"][0]), float(friction_cfg["dynamic"][1]))

    # Mass randomization ranges
    mass_cfg = overrides.get("mass", {})
    if mass_cfg and getattr(env_cfg.events, "add_base_mass", None) is not None:
        params = env_cfg.events.add_base_mass.params
        if "add_base_mass" in mass_cfg:
            params["mass_distribution_params"] = (
                float(mass_cfg["add_base_mass"][0]),
                float(mass_cfg["add_base_mass"][1]),
            )

    # Terrain distribution and roughness scale
    terrain_cfg = overrides.get("terrain", {})
    if terrain_cfg:
        terrain_generator = getattr(env_cfg.scene.terrain, "terrain_generator", None)
        if terrain_generator is not None:
            if "difficulty_range" in terrain_cfg:
                terrain_generator.difficulty_range = (
                    float(terrain_cfg["difficulty_range"][0]),
                    float(terrain_cfg["difficulty_range"][1]),
                )
            if "curriculum" in terrain_cfg:
                terrain_generator.curriculum = bool(terrain_cfg["curriculum"])
            if "stairs_height_scale" in terrain_cfg:
                scale = float(terrain_cfg["stairs_height_scale"])
                for key in ("pyramid_stairs", "pyramid_stairs_inv"):
                    if key in terrain_generator.sub_terrains:
                        st_cfg = terrain_generator.sub_terrains[key]
                        st_cfg.step_height_range = _scale_range(st_cfg.step_height_range, scale)
            if "boxes_height_scale" in terrain_cfg and "boxes" in terrain_generator.sub_terrains:
                scale = float(terrain_cfg["boxes_height_scale"])
                box_cfg = terrain_generator.sub_terrains["boxes"]
                box_cfg.grid_height_range = _scale_range(box_cfg.grid_height_range, scale)
            if "rough_noise_scale" in terrain_cfg and "random_rough" in terrain_generator.sub_terrains:
                scale = float(terrain_cfg["rough_noise_scale"])
                rough_cfg = terrain_generator.sub_terrains["random_rough"]
                rough_cfg.noise_range = _scale_range(rough_cfg.noise_range, scale)
            if "slope_scale" in terrain_cfg:
                scale = float(terrain_cfg["slope_scale"])
                for key in ("hf_pyramid_slope", "hf_pyramid_slope_inv"):
                    if key in terrain_generator.sub_terrains:
                        slope_cfg = terrain_generator.sub_terrains[key]
                        slope_cfg.slope_range = _scale_range(slope_cfg.slope_range, scale)

        if "max_init_terrain_level" in terrain_cfg:
            env_cfg.scene.terrain.max_init_terrain_level = terrain_cfg["max_init_terrain_level"]


def _set_camera_view_with_fallback(base_env, eye: tuple[float, float, float], target: tuple[float, float, float]):
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


def _apply_camera(env, video_cfg: dict[str, Any], *, log: bool = True):
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
            eye = tuple(float(v) for v in video_cfg.get("camera_eye", [4.0, 4.0, 3.0]))
            target = tuple(float(v) for v in video_cfg.get("camera_lookat", [0.0, 0.0, 0.5]))

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
            payload["robot_root_position"] = robot_root_pos
            payload["camera_offset"] = tuple(float(v) for v in video_cfg.get("camera_offset", [3.0, 3.0, 2.0]))
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


def _safe_float(value) -> float:
    if value is None:
        return float("nan")
    return float(value)


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


def _plot_summary(summary_rows: list[dict[str, Any]], out_dir: str):
    if not summary_rows:
        return []
    _ensure_dir(out_dir)
    scenario_names = [row["scenario_name"] for row in summary_rows]
    groups = [row["scenario_group"] for row in summary_rows]

    metric_specs = [
        ("mean_return", "Average Return"),
        ("mean_episode_length", "Episode Length"),
        ("fall_rate", "Fall Rate"),
        ("mean_lin_vel_tracking_error", "LinVel Tracking Error"),
        ("mean_recovery_time_s", "Recovery Time (s)"),
    ]
    color_map = {"in_distribution": "#4C78A8", "long_tail": "#F58518", "ood": "#E45756"}
    generated = []

    for metric_key, title in metric_specs:
        values = [row.get(metric_key, float("nan")) for row in summary_rows]
        fig = plt.figure(figsize=(10, 4))
        colors = [color_map.get(group, "#999999") for group in groups]
        x = np.arange(len(values))
        plt.bar(x, values, color=colors)
        plt.xticks(x, scenario_names, rotation=15, ha="right")
        plt.ylabel(metric_key)
        plt.title(title)
        plt.tight_layout()
        path = os.path.join(out_dir, f"{metric_key}.png")
        fig.savefig(path, dpi=140)
        plt.close(fig)
        generated.append(path)
    return generated


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


def _run_scenario(
    scenario: dict[str, Any],
    base_env_cfg: ManagerBasedRLEnvCfg,
    agent_cfg: RslRlBaseRunnerCfg,
    checkpoint_path: str,
    global_cfg: dict[str, Any],
    run_dir: str,
):
    scenario_name = scenario["name"]
    scenario_group = scenario.get("group", "unknown")
    scenario_dir = _ensure_dir(os.path.join(run_dir, "scenarios", scenario_name))
    metrics_dir = _ensure_dir(os.path.join(scenario_dir, "metrics"))
    traces_dir = _ensure_dir(os.path.join(scenario_dir, "traces"))
    videos_dir = _ensure_dir(os.path.join(scenario_dir, "videos"))

    env_cfg = deepcopy(base_env_cfg)
    overrides = scenario.get("overrides", {})
    _apply_overrides(env_cfg, overrides)

    num_envs = int(args_cli.num_envs or scenario.get("num_envs", global_cfg.get("num_envs", 32)))
    num_episodes = int(args_cli.num_episodes or scenario.get("num_episodes", global_cfg.get("num_episodes", 32)))
    seed = int(args_cli.seed if args_cli.seed is not None else global_cfg.get("seed", 42))
    env_cfg.scene.num_envs = num_envs
    env_cfg.seed = seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else global_cfg.get("device", env_cfg.sim.device)
    env_cfg.observations.policy.enable_corruption = True
    # Disable the recorder manager for benchmark eval to avoid unnecessary HDF5 overhead.
    env_cfg.recorders = None

    video_cfg = deepcopy(global_cfg.get("video", {}))
    video_cfg.update(scenario.get("video_overrides", {}))
    if args_cli.video_length is not None:
        video_cfg["video_length"] = args_cli.video_length
    if args_cli.video_start_step is not None:
        video_cfg["video_start_step"] = args_cli.video_start_step
    if args_cli.video_max_clips is not None:
        video_cfg["max_clips_per_scenario"] = args_cli.video_max_clips
    video_enabled = bool(args_cli.video and video_cfg.get("enabled", True))
    # For representative videos, prefer a small dedicated env count to keep camera framing stable.
    if video_enabled and "num_envs" in video_cfg and args_cli.num_envs is None:
        num_envs = int(video_cfg["num_envs"])

    render_mode = "rgb_array" if video_enabled else None
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=render_mode)

    if video_enabled:
        _apply_camera(env, video_cfg, log=True)
        step_trigger = _make_video_step_trigger(
            start_step=int(video_cfg.get("video_start_step", 20)),
            interval_steps=int(video_cfg.get("video_length", 900)),
            max_clips=int(video_cfg.get("max_clips_per_scenario", 3)),
        )
        video_kwargs = {
            "video_folder": videos_dir,
            "step_trigger": step_trigger,
            "video_length": int(video_cfg.get("video_length", 900)),
            "disable_logger": True,
        }
        print("[INFO] Benchmark video config:")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(checkpoint_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    trace_env_id = int(args_cli.trace_env_id if args_cli.trace_env_id is not None else global_cfg.get("step_trace_env_id", 0))
    trace_max_steps = int(
        args_cli.trace_max_steps if args_cli.trace_max_steps is not None else global_cfg.get("step_trace_max_steps", 1200)
    )
    trace_rows: list[dict[str, Any]] = []

    recovery_trigger = float(global_cfg.get("recovery_trigger_error", 0.75))
    recovery_clear = float(global_cfg.get("recovery_clear_error", 0.25))
    recovery_hold_steps = int(global_cfg.get("recovery_min_hold_steps", 5))

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
    max_eval_steps = int(scenario.get("max_eval_steps", global_cfg.get("max_eval_steps", 12000)))
    while simulation_app.is_running() and completed_episodes < num_episodes and global_step < max_eval_steps:
        with torch.inference_mode():
            actions = policy(obs)
            obs, rewards, dones, extras = env.step(actions)

            if video_enabled and bool(video_cfg.get("follow_robot_camera", False)):
                _apply_camera(env, video_cfg, log=False)

            base_env = env.unwrapped
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

            # recovery tracking from tracking-error spikes
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

            # step traces for one env
            if global_step < trace_max_steps:
                safe_trace_env_id = max(0, min(trace_env_id, env.num_envs - 1))
                trace_rows.append(_build_trace_row(global_step, safe_trace_env_id, command, robot))

            done_mask = base_env.reset_buf.reshape(-1).bool()
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
                        "scenario_name": scenario_name,
                        "scenario_group": scenario_group,
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
    summary.update(
        {
            "scenario_name": scenario_name,
            "scenario_group": scenario_group,
            "description": scenario.get("description", ""),
            "checkpoint": checkpoint_path,
            "num_envs": num_envs,
            "seed": seed,
            "target_num_episodes": num_episodes,
            "completed_episodes": completed_episodes,
            "max_eval_steps": max_eval_steps,
            "early_stop": bool(completed_episodes < num_episodes),
        }
    )
    if completed_episodes < num_episodes:
        print(
            f"[WARN] Scenario '{scenario_name}' reached max_eval_steps={max_eval_steps} "
            f"before collecting all episodes ({completed_episodes}/{num_episodes})."
        )

    _write_csv(os.path.join(metrics_dir, "episodes.csv"), episode_rows)
    _write_json(os.path.join(metrics_dir, "episodes.json"), episode_rows)
    _write_json(os.path.join(metrics_dir, "summary.json"), summary)
    term_rows = [{"scenario_name": scenario_name, "termination_reason": k, "count": v} for k, v in termination_counter.items()]
    _write_csv(os.path.join(metrics_dir, "termination_stats.csv"), term_rows)
    _write_csv(os.path.join(traces_dir, "trace_env.csv"), trace_rows)

    print(f"[INFO] Scenario finished: {scenario_name}")
    print_dict(summary, nesting=4)
    env.close()
    return summary


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    installed_version = metadata.version("rsl-rl-lib")
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    checkpoint_path = retrieve_file_path(args_cli.checkpoint)
    scenario_cfg_path = os.path.abspath(args_cli.scenario_cfg)
    scenario_cfg = _load_scenario_cfg(scenario_cfg_path)
    global_cfg = scenario_cfg.get("global", {})
    all_scenarios = scenario_cfg.get("scenarios", [])
    scenarios = _select_scenarios(all_scenarios, args_cli.scenario_group, args_cli.scenario_names)

    if args_cli.seed is not None:
        env_cfg.seed = args_cli.seed
    if args_cli.device:
        env_cfg.sim.device = args_cli.device
    env_cfg.scene.num_envs = int(args_cli.num_envs or global_cfg.get("num_envs", env_cfg.scene.num_envs))

    run_name = args_cli.benchmark_run_name or _make_run_name(checkpoint_path)
    run_dir = _ensure_dir(os.path.abspath(os.path.join(args_cli.output_root, run_name)))
    _ensure_dir(os.path.join(run_dir, "scenarios"))
    _ensure_dir(os.path.join(run_dir, "summary"))
    _ensure_dir(os.path.join(run_dir, "plots"))
    _ensure_dir(os.path.join(run_dir, "config"))

    dump_yaml(os.path.join(run_dir, "config", "env_base.yaml"), env_cfg)
    dump_yaml(os.path.join(run_dir, "config", "agent_base.yaml"), agent_cfg)
    _write_json(os.path.join(run_dir, "config", "benchmark_runtime_args.json"), vars(args_cli))
    _write_json(os.path.join(run_dir, "config", "benchmark_scenarios.json"), scenario_cfg)

    print("[INFO] Benchmark run directory:")
    print(run_dir)
    print("[INFO] Selected scenarios:")
    print_dict({"scenario_names": [s.get("name") for s in scenarios]}, nesting=4)

    summary_rows: list[dict[str, Any]] = []
    wall_start = time.time()
    for scenario in scenarios:
        scenario_start = time.time()
        summary = _run_scenario(
            scenario=scenario,
            base_env_cfg=env_cfg,
            agent_cfg=agent_cfg,
            checkpoint_path=checkpoint_path,
            global_cfg=global_cfg,
            run_dir=run_dir,
        )
        summary["wall_time_s"] = round(time.time() - scenario_start, 2)
        summary_rows.append(summary)

    summary_csv = os.path.join(run_dir, "summary", "scenario_summary.csv")
    summary_json = os.path.join(run_dir, "summary", "scenario_summary.json")
    _write_csv(summary_csv, summary_rows)
    _write_json(summary_json, summary_rows)

    generated_plots = []
    if args_cli.save_plots:
        generated_plots = _plot_summary(summary_rows, os.path.join(run_dir, "plots"))

    index_payload = {
        "run_dir": run_dir,
        "checkpoint": checkpoint_path,
        "scenario_cfg": scenario_cfg_path,
        "num_scenarios": len(summary_rows),
        "elapsed_s": round(time.time() - wall_start, 2),
        "summary_csv": summary_csv,
        "summary_json": summary_json,
        "generated_plots": generated_plots,
    }
    _write_json(os.path.join(run_dir, "summary", "index.json"), index_payload)
    print("[INFO] Benchmark completed.")
    print_dict(index_payload, nesting=4)


if __name__ == "__main__":
    main()
    simulation_app.close()
