# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

from isaaclab.envs import ManagerBasedRLEnvCfg


def _scale_range(range_pair: tuple[float, float] | list[float], scale: float) -> tuple[float, float]:
    return float(range_pair[0]) * scale, float(range_pair[1]) * scale


def _set_pair_if_present(obj: Any, field: str, values: list[float] | tuple[float, float] | None):
    if values is None:
        return
    if hasattr(obj, field):
        setattr(obj, field, (float(values[0]), float(values[1])))


def _scale_numeric_container(value: Any, scale: float):
    if isinstance(value, (int, float)):
        return float(value) * scale
    if isinstance(value, Mapping):
        return {k: _scale_numeric_container(v, scale) for k, v in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return type(value)(_scale_numeric_container(v, scale) for v in value)
    return value


def _apply_command_overrides(env_cfg: ManagerBasedRLEnvCfg, cmd_cfg: dict[str, Any]):
    if not cmd_cfg or not hasattr(env_cfg.commands, "base_velocity"):
        return

    command = env_cfg.commands.base_velocity
    ranges = command.ranges
    _set_pair_if_present(ranges, "lin_vel_x", cmd_cfg.get("lin_vel_x"))
    _set_pair_if_present(ranges, "lin_vel_y", cmd_cfg.get("lin_vel_y"))
    _set_pair_if_present(ranges, "ang_vel_z", cmd_cfg.get("ang_vel_z"))
    _set_pair_if_present(ranges, "heading", cmd_cfg.get("heading"))

    if "resampling_time_s" in cmd_cfg:
        window = cmd_cfg["resampling_time_s"]
        command.resampling_time_range = (float(window[0]), float(window[1]))
    if "rel_standing_envs" in cmd_cfg:
        command.rel_standing_envs = float(cmd_cfg["rel_standing_envs"])
    if "rel_heading_envs" in cmd_cfg:
        command.rel_heading_envs = float(cmd_cfg["rel_heading_envs"])
    if "heading_command" in cmd_cfg:
        command.heading_command = bool(cmd_cfg["heading_command"])
    if "heading_control_stiffness" in cmd_cfg:
        command.heading_control_stiffness = float(cmd_cfg["heading_control_stiffness"])


def _apply_push_overrides(env_cfg: ManagerBasedRLEnvCfg, push_cfg: dict[str, Any]):
    if not push_cfg:
        return
    enabled = bool(push_cfg.get("enabled", True))
    if not enabled:
        env_cfg.events.push_robot = None
        return
    if getattr(env_cfg.events, "push_robot", None) is None:
        return

    if "interval_s" in push_cfg:
        env_cfg.events.push_robot.interval_range_s = (
            float(push_cfg["interval_s"][0]),
            float(push_cfg["interval_s"][1]),
        )

    velocity_range = deepcopy(env_cfg.events.push_robot.params.get("velocity_range", {}))
    if "vel_x" in push_cfg:
        velocity_range["x"] = (float(push_cfg["vel_x"][0]), float(push_cfg["vel_x"][1]))
    if "vel_y" in push_cfg:
        velocity_range["y"] = (float(push_cfg["vel_y"][0]), float(push_cfg["vel_y"][1]))
    if "vel_z" in push_cfg:
        velocity_range["z"] = (float(push_cfg["vel_z"][0]), float(push_cfg["vel_z"][1]))
    if "vel_roll" in push_cfg:
        velocity_range["roll"] = (float(push_cfg["vel_roll"][0]), float(push_cfg["vel_roll"][1]))
    if "vel_pitch" in push_cfg:
        velocity_range["pitch"] = (float(push_cfg["vel_pitch"][0]), float(push_cfg["vel_pitch"][1]))
    if "vel_yaw" in push_cfg:
        velocity_range["yaw"] = (float(push_cfg["vel_yaw"][0]), float(push_cfg["vel_yaw"][1]))
    env_cfg.events.push_robot.params["velocity_range"] = velocity_range


def _apply_friction_overrides(env_cfg: ManagerBasedRLEnvCfg, friction_cfg: dict[str, Any]):
    if not friction_cfg or getattr(env_cfg.events, "physics_material", None) is None:
        return
    params = env_cfg.events.physics_material.params
    if "static" in friction_cfg:
        params["static_friction_range"] = (float(friction_cfg["static"][0]), float(friction_cfg["static"][1]))
    if "dynamic" in friction_cfg:
        params["dynamic_friction_range"] = (float(friction_cfg["dynamic"][0]), float(friction_cfg["dynamic"][1]))
    if "restitution" in friction_cfg:
        params["restitution_range"] = (float(friction_cfg["restitution"][0]), float(friction_cfg["restitution"][1]))


def _apply_mass_and_com_overrides(env_cfg: ManagerBasedRLEnvCfg, mass_cfg: dict[str, Any], com_cfg: dict[str, Any]):
    if mass_cfg and getattr(env_cfg.events, "add_base_mass", None) is not None:
        params = env_cfg.events.add_base_mass.params
        if "add_base_mass" in mass_cfg:
            params["mass_distribution_params"] = (
                float(mass_cfg["add_base_mass"][0]),
                float(mass_cfg["add_base_mass"][1]),
            )
    if com_cfg and getattr(env_cfg.events, "base_com", None) is not None:
        params = env_cfg.events.base_com.params
        com_range = params.get("com_range", {})
        for key in ("x", "y", "z"):
            if key in com_cfg:
                com_range[key] = (float(com_cfg[key][0]), float(com_cfg[key][1]))
        params["com_range"] = com_range


def _apply_init_overrides(env_cfg: ManagerBasedRLEnvCfg, init_cfg: dict[str, Any]):
    if not init_cfg:
        return

    reset_base = init_cfg.get("reset_base", {})
    if reset_base and getattr(env_cfg.events, "reset_base", None) is not None:
        params = env_cfg.events.reset_base.params
        if "pose_range" in reset_base:
            pose_range = params.get("pose_range", {})
            for key, val in reset_base["pose_range"].items():
                pose_range[key] = (float(val[0]), float(val[1]))
            params["pose_range"] = pose_range
        if "velocity_range" in reset_base:
            velocity_range = params.get("velocity_range", {})
            for key, val in reset_base["velocity_range"].items():
                velocity_range[key] = (float(val[0]), float(val[1]))
            params["velocity_range"] = velocity_range

    reset_joints = init_cfg.get("reset_joints", {})
    if reset_joints and getattr(env_cfg.events, "reset_robot_joints", None) is not None:
        params = env_cfg.events.reset_robot_joints.params
        if "position_range" in reset_joints:
            params["position_range"] = (
                float(reset_joints["position_range"][0]),
                float(reset_joints["position_range"][1]),
            )
        if "velocity_range" in reset_joints:
            params["velocity_range"] = (
                float(reset_joints["velocity_range"][0]),
                float(reset_joints["velocity_range"][1]),
            )


def _apply_terrain_overrides(env_cfg: ManagerBasedRLEnvCfg, terrain_cfg: dict[str, Any]):
    if not terrain_cfg:
        return
    terrain_generator = getattr(env_cfg.scene.terrain, "terrain_generator", None)
    if terrain_generator is not None:
        if "difficulty_range" in terrain_cfg:
            terrain_generator.difficulty_range = (
                float(terrain_cfg["difficulty_range"][0]),
                float(terrain_cfg["difficulty_range"][1]),
            )
        if "curriculum" in terrain_cfg:
            terrain_generator.curriculum = bool(terrain_cfg["curriculum"])
        if "num_rows" in terrain_cfg:
            terrain_generator.num_rows = int(terrain_cfg["num_rows"])
        if "num_cols" in terrain_cfg:
            terrain_generator.num_cols = int(terrain_cfg["num_cols"])

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


def _apply_observation_overrides(env_cfg: ManagerBasedRLEnvCfg, obs_cfg: dict[str, Any]):
    if not obs_cfg:
        return
    policy_cfg = getattr(env_cfg.observations, "policy", None)
    if policy_cfg is None:
        return

    if "enable_corruption" in obs_cfg:
        policy_cfg.enable_corruption = bool(obs_cfg["enable_corruption"])

    if bool(obs_cfg.get("disable_height_scan", False)) and hasattr(policy_cfg, "height_scan"):
        policy_cfg.height_scan = None

    noise_scale = obs_cfg.get("noise_scale")
    if noise_scale is not None:
        noise_scale = float(noise_scale)
        for field_name in dir(policy_cfg):
            if field_name.startswith("_"):
                continue
            term = getattr(policy_cfg, field_name, None)
            if term is None or not hasattr(term, "noise"):
                continue
            noise = getattr(term, "noise", None)
            if noise is None:
                continue
            if hasattr(noise, "n_min"):
                noise.n_min = float(noise.n_min) * noise_scale
            if hasattr(noise, "n_max"):
                noise.n_max = float(noise.n_max) * noise_scale


def _apply_dynamics_overrides(env_cfg: ManagerBasedRLEnvCfg, dynamics_cfg: dict[str, Any]):
    if not dynamics_cfg:
        return
    robot_cfg = getattr(env_cfg.scene, "robot", None)
    if robot_cfg is None:
        return

    scale_stiffness = dynamics_cfg.get("actuator_stiffness_scale")
    scale_damping = dynamics_cfg.get("actuator_damping_scale")
    if (scale_stiffness is None and scale_damping is None) or not hasattr(robot_cfg, "actuators"):
        pass
    else:
        for actuator in robot_cfg.actuators.values():
            if scale_stiffness is not None and hasattr(actuator, "stiffness"):
                actuator.stiffness = _scale_numeric_container(actuator.stiffness, float(scale_stiffness))
            if scale_damping is not None and hasattr(actuator, "damping"):
                actuator.damping = _scale_numeric_container(actuator.damping, float(scale_damping))

    if "action_scale" in dynamics_cfg and hasattr(env_cfg.actions, "joint_pos"):
        if hasattr(env_cfg.actions.joint_pos, "scale"):
            env_cfg.actions.joint_pos.scale = float(dynamics_cfg["action_scale"])


def apply_env_cfg_overrides(env_cfg: ManagerBasedRLEnvCfg, overrides: dict[str, Any]):
    """Apply benchmark case overrides to env cfg in-place."""
    if not overrides:
        return

    _apply_command_overrides(env_cfg, dict(overrides.get("command", {})))
    _apply_push_overrides(env_cfg, dict(overrides.get("push", {})))
    _apply_friction_overrides(env_cfg, dict(overrides.get("friction", {})))
    _apply_mass_and_com_overrides(
        env_cfg=env_cfg,
        mass_cfg=dict(overrides.get("mass", {})),
        com_cfg=dict(overrides.get("com", {})),
    )
    _apply_init_overrides(env_cfg, dict(overrides.get("init", {})))
    _apply_terrain_overrides(env_cfg, dict(overrides.get("terrain", {})))
    _apply_observation_overrides(env_cfg, dict(overrides.get("observation", {})))
    _apply_dynamics_overrides(env_cfg, dict(overrides.get("dynamics", {}))
    )
