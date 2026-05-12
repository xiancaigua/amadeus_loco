# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class VideoCfg:
    enabled: bool = True
    num_envs: int = 1
    num_episodes: int = 1
    video_length: int = 900
    video_start_step: int = 20
    video_interval_steps: int = 0
    max_clips_per_case: int = 2
    follow_robot_camera: bool = True
    camera_eye: tuple[float, float, float] = (4.0, 4.0, 3.0)
    camera_lookat: tuple[float, float, float] = (0.0, 0.0, 0.5)
    camera_offset: tuple[float, float, float] = (3.0, 3.0, 2.0)
    camera_robot_env_id: int = 0
    show_velocity_markers: bool = True
    velocity_marker_scale: float = 3.0
    velocity_marker_height: float = 0.5
    velocity_marker_env_id: int = -1


@dataclass
class TraceCfg:
    env_id: int = 0
    max_steps: int = 1600


@dataclass
class RecoveryCfg:
    trigger_error: float = 0.75
    clear_error: float = 0.25
    min_hold_steps: int = 5


@dataclass
class GlobalBenchmarkCfg:
    task: str = "Template-Amadeus-Quadruped-Rough-v0"
    num_envs: int = 64
    num_episodes: int = 64
    seed: int = 42
    device: str = "cuda:0"
    max_eval_steps: int = 6000
    trace: TraceCfg = field(default_factory=TraceCfg)
    video: VideoCfg = field(default_factory=VideoCfg)
    recovery: RecoveryCfg = field(default_factory=RecoveryCfg)


@dataclass
class BenchmarkCase:
    name: str
    bucket: str
    description: str = ""
    tags: list[str] = field(default_factory=list)
    num_envs: int | None = None
    num_episodes: int | None = None
    seed: int | None = None
    max_eval_steps: int | None = None
    save_video: bool | None = None
    overrides: dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkSuite:
    name: str
    description: str = ""
    enabled: bool = True
    cases: list[BenchmarkCase] = field(default_factory=list)


@dataclass
class BenchmarkConfig:
    global_cfg: GlobalBenchmarkCfg
    suites: dict[str, BenchmarkSuite]
    source_path: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "global": asdict(self.global_cfg),
            "suites": {name: asdict(suite) for name, suite in self.suites.items()},
            "source_path": self.source_path,
        }


def _as_tuple3(values: Any, default: tuple[float, float, float]) -> tuple[float, float, float]:
    if values is None:
        return default
    if not isinstance(values, (list, tuple)) or len(values) != 3:
        raise ValueError(f"Expected 3-vector, got: {values}")
    return float(values[0]), float(values[1]), float(values[2])


def _load_global_cfg(raw: dict[str, Any]) -> GlobalBenchmarkCfg:
    trace_raw = raw.get("trace", {})
    video_raw = raw.get("video", {})
    recovery_raw = raw.get("recovery", {})

    trace_cfg = TraceCfg(
        env_id=int(trace_raw.get("env_id", trace_raw.get("step_trace_env_id", 0))),
        max_steps=int(trace_raw.get("max_steps", trace_raw.get("step_trace_max_steps", 1600))),
    )
    video_cfg = VideoCfg(
        enabled=bool(video_raw.get("enabled", True)),
        num_envs=int(video_raw.get("num_envs", 1)),
        num_episodes=int(video_raw.get("num_episodes", 1)),
        video_length=int(video_raw.get("video_length", 900)),
        video_start_step=int(video_raw.get("video_start_step", 20)),
        video_interval_steps=int(video_raw.get("video_interval_steps", 0)),
        max_clips_per_case=int(video_raw.get("max_clips_per_case", video_raw.get("max_clips_per_scenario", 2))),
        follow_robot_camera=bool(video_raw.get("follow_robot_camera", True)),
        camera_eye=_as_tuple3(video_raw.get("camera_eye"), (4.0, 4.0, 3.0)),
        camera_lookat=_as_tuple3(video_raw.get("camera_lookat"), (0.0, 0.0, 0.5)),
        camera_offset=_as_tuple3(video_raw.get("camera_offset"), (3.0, 3.0, 2.0)),
        camera_robot_env_id=int(video_raw.get("camera_robot_env_id", 0)),
        show_velocity_markers=bool(video_raw.get("show_velocity_markers", True)),
        velocity_marker_scale=float(video_raw.get("velocity_marker_scale", 3.0)),
        velocity_marker_height=float(video_raw.get("velocity_marker_height", 0.5)),
        velocity_marker_env_id=int(video_raw.get("velocity_marker_env_id", -1)),
    )
    recovery_cfg = RecoveryCfg(
        trigger_error=float(recovery_raw.get("trigger_error", raw.get("recovery_trigger_error", 0.75))),
        clear_error=float(recovery_raw.get("clear_error", raw.get("recovery_clear_error", 0.25))),
        min_hold_steps=int(recovery_raw.get("min_hold_steps", raw.get("recovery_min_hold_steps", 5))),
    )
    return GlobalBenchmarkCfg(
        task=str(raw.get("task", "Template-Amadeus-Quadruped-Rough-v0")),
        num_envs=int(raw.get("num_envs", 64)),
        num_episodes=int(raw.get("num_episodes", 64)),
        seed=int(raw.get("seed", 42)),
        device=str(raw.get("device", "cuda:0")),
        max_eval_steps=int(raw.get("max_eval_steps", 6000)),
        trace=trace_cfg,
        video=video_cfg,
        recovery=recovery_cfg,
    )


def _load_case(raw: dict[str, Any], fallback_bucket: str) -> BenchmarkCase:
    return BenchmarkCase(
        name=str(raw["name"]),
        bucket=str(raw.get("bucket", raw.get("group", fallback_bucket))),
        description=str(raw.get("description", "")),
        tags=[str(x) for x in raw.get("tags", [])],
        num_envs=int(raw["num_envs"]) if raw.get("num_envs") is not None else None,
        num_episodes=int(raw["num_episodes"]) if raw.get("num_episodes") is not None else None,
        seed=int(raw["seed"]) if raw.get("seed") is not None else None,
        max_eval_steps=int(raw["max_eval_steps"]) if raw.get("max_eval_steps") is not None else None,
        save_video=bool(raw["save_video"]) if raw.get("save_video") is not None else None,
        overrides=dict(raw.get("overrides", {})),
    )


def _load_suites(raw: dict[str, Any]) -> dict[str, BenchmarkSuite]:
    suites_raw = raw.get("suites")
    if suites_raw is None:
        # Backward compatibility with legacy `scenarios: [...]` format.
        scenarios = raw.get("scenarios", [])
        if not scenarios:
            raise ValueError("Config must contain either 'suites' or legacy 'scenarios'.")
        grouped: dict[str, list[dict[str, Any]]] = {}
        for scenario in scenarios:
            group = str(scenario.get("group", "legacy"))
            grouped.setdefault(group, []).append(scenario)
        suites: dict[str, BenchmarkSuite] = {}
        for group, rows in grouped.items():
            suite_name = f"legacy_{group}"
            suites[suite_name] = BenchmarkSuite(
                name=suite_name,
                description=f"Legacy imported suite for group '{group}'.",
                enabled=True,
                cases=[_load_case(item, fallback_bucket=group) for item in rows],
            )
        return suites

    suites: dict[str, BenchmarkSuite] = {}
    if not isinstance(suites_raw, dict):
        raise ValueError("'suites' must be a mapping.")

    for suite_name, suite_raw in suites_raw.items():
        if not isinstance(suite_raw, dict):
            raise ValueError(f"Suite '{suite_name}' must be a mapping.")
        cases_raw = suite_raw.get("cases", [])
        if not isinstance(cases_raw, list):
            raise ValueError(f"Suite '{suite_name}' cases must be a list.")
        suites[suite_name] = BenchmarkSuite(
            name=str(suite_name),
            description=str(suite_raw.get("description", "")),
            enabled=bool(suite_raw.get("enabled", True)),
            cases=[_load_case(case_raw, fallback_bucket=suite_name) for case_raw in cases_raw],
        )
    return suites


def load_benchmark_config(path: str) -> BenchmarkConfig:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"Benchmark config must be a mapping: {path}")

    global_cfg = _load_global_cfg(raw.get("global", {}))
    suites = _load_suites(raw)
    return BenchmarkConfig(global_cfg=global_cfg, suites=suites, source_path=str(Path(path).resolve()))


def select_cases(
    cfg: BenchmarkConfig,
    suite_names: list[str] | None = None,
    case_names: list[str] | None = None,
) -> list[tuple[str, BenchmarkCase]]:
    suite_filter = set(suite_names or [])
    case_filter = set(case_names or [])

    selected: list[tuple[str, BenchmarkCase]] = []
    for suite_name, suite in cfg.suites.items():
        if not suite.enabled and not suite_filter:
            continue
        if suite_filter and suite_name not in suite_filter:
            continue
        for case in suite.cases:
            if case_filter and case.name not in case_filter:
                continue
            selected.append((suite_name, case))
    if not selected:
        raise ValueError(
            f"No benchmark cases selected. suites={sorted(suite_filter) or '<all-enabled>'}, "
            f"cases={sorted(case_filter) or '<all>'}"
        )
    return selected
