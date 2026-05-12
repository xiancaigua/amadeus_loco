# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run benchmark cases in isolated subprocesses, then merge summaries.

This avoids long-running multi-case single-process instability observed in Isaac Sim.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _read_csv(path: str) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


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


def _write_run_summary_md(master_dir: str, manifest: dict[str, Any], selected: list[tuple[str, str]], failures: list[dict[str, Any]]):
    out_path = os.path.join(master_dir, "RUN_SUMMARY.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# Isolated Benchmark Master Summary\n\n")
        f.write("## Overview\n\n")
        f.write(f"- Master run directory: `{master_dir}`\n")
        f.write(f"- Selected cases: {manifest.get('num_selected_cases')}\n")
        f.write(f"- Succeeded cases: {manifest.get('num_success_cases')}\n")
        f.write(f"- Failed cases: {manifest.get('num_failed_cases')}\n")
        f.write(f"- Per-case run root: `{manifest.get('per_case_root')}`\n")
        f.write(f"- Summary metrics dir: `{manifest.get('summary_dir')}`\n")
        f.write("\n## Key Result Pointers\n\n")
        f.write(f"- merged summary CSV: `{os.path.join(master_dir, 'summary_metrics', 'case_summary.csv')}`\n")
        f.write(f"- merged summary JSON: `{os.path.join(master_dir, 'summary_metrics', 'case_summary.json')}`\n")
        f.write(f"- failure list: `{os.path.join(master_dir, 'summary_metrics', 'failures.json')}`\n")
        f.write(f"- selected cases: `{os.path.join(master_dir, 'selected_cases.json')}`\n")
        f.write(f"- plots: `{os.path.join(master_dir, 'plots')}`\n")
        f.write(f"- report: `{os.path.join(master_dir, 'reports', 'research_summary.md')}`\n")
        f.write("\n## Selected Cases\n\n")
        for suite_name, case_name in selected:
            f.write(f"- `{suite_name}:{case_name}`\n")
        if failures:
            f.write("\n## Failures\n\n")
            for item in failures:
                f.write(f"- `{item.get('suite_name')}:{item.get('case_name')}` returncode={item.get('returncode')} run_dir=`{item.get('run_dir')}`\n")
        f.write("\n## Notes\n\n")
        f.write("- This master run executes each case in a separate subprocess to improve robustness.\n")


def _select_cases(raw_cfg: dict[str, Any], suite_filter: set[str], case_filter: set[str]) -> list[tuple[str, str]]:
    suites = raw_cfg.get("suites", {})
    selected: list[tuple[str, str]] = []
    for suite_name, suite in suites.items():
        if suite_filter and suite_name not in suite_filter:
            continue
        if not suite_filter and not bool(suite.get("enabled", True)):
            continue
        for case in suite.get("cases", []):
            case_name = case.get("name")
            if case_filter and case_name not in case_filter:
                continue
            selected.append((suite_name, case_name))
    return selected


def main():
    parser = argparse.ArgumentParser(description="Run quadruped benchmark in isolated per-case subprocesses.")
    parser.add_argument("--task", type=str, default="Template-Amadeus-Quadruped-Rough-v0")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--benchmark_cfg", type=str, default="scripts/benchmark/configs/quadruped_rough_benchmark_suites.yaml")
    parser.add_argument("--output_root", type=str, default="outputs/quadruped_rough_benchmark")
    parser.add_argument("--master_run_name", type=str, default=None)
    parser.add_argument("--suite_names", type=str, default=None)
    parser.add_argument("--case_names", type=str, default=None)
    parser.add_argument("--num_envs", type=int, default=None)
    parser.add_argument("--num_episodes", type=int, default=None)
    parser.add_argument("--max_eval_steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--video", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--video_length", type=int, default=None)
    parser.add_argument("--video_start_step", type=int, default=None)
    parser.add_argument("--video_max_clips", type=int, default=None)
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--stop_on_error", action="store_true", default=False)
    args = parser.parse_args()

    with open(args.benchmark_cfg, encoding="utf-8") as f:
        raw_cfg = yaml.safe_load(f)

    suite_filter = {x.strip() for x in (args.suite_names or "").split(",") if x.strip()}
    case_filter = {x.strip() for x in (args.case_names or "").split(",") if x.strip()}
    selected = _select_cases(raw_cfg, suite_filter, case_filter)
    if not selected:
        raise ValueError("No benchmark cases selected.")

    master_run_name = args.master_run_name or f"isolated_{datetime.utcnow().strftime('%Y-%m-%d_%H-%M-%S')}"
    master_dir = _ensure_dir(os.path.abspath(os.path.join(args.output_root, master_run_name)))
    per_case_root = _ensure_dir(os.path.join(master_dir, "runs"))
    summary_dir = _ensure_dir(os.path.join(master_dir, "summary_metrics"))
    plots_dir = _ensure_dir(os.path.join(master_dir, "plots"))
    reports_dir = _ensure_dir(os.path.join(master_dir, "reports"))

    _write_json(os.path.join(master_dir, "selected_cases.json"), [{"suite_name": s, "case_name": c} for s, c in selected])

    case_results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    runner_script = os.path.join("scripts", "benchmark", "run_quadruped_rough_benchmark.py")
    for idx, (suite_name, case_name) in enumerate(selected, start=1):
        run_name = f"{idx:02d}_{suite_name}__{case_name}"
        cmd = [
            "/isaac-sim/python.sh",
            runner_script,
            "--task",
            args.task,
            "--checkpoint",
            args.checkpoint,
            "--benchmark_cfg",
            args.benchmark_cfg,
            "--suite_names",
            suite_name,
            "--case_names",
            case_name,
            "--output_root",
            per_case_root,
            "--benchmark_run_name",
            run_name,
        ]
        if args.headless:
            cmd.append("--headless")
        if args.num_envs is not None:
            cmd.extend(["--num_envs", str(args.num_envs)])
        if args.num_episodes is not None:
            cmd.extend(["--num_episodes", str(args.num_episodes)])
        if args.max_eval_steps is not None:
            cmd.extend(["--max_eval_steps", str(args.max_eval_steps)])
        if args.seed is not None:
            cmd.extend(["--seed", str(args.seed)])
        if args.device:
            cmd.extend(["--device", args.device])
        cmd.append("--video" if args.video else "--no-video")
        if args.video_length is not None:
            cmd.extend(["--video_length", str(args.video_length)])
        if args.video_start_step is not None:
            cmd.extend(["--video_start_step", str(args.video_start_step)])
        if args.video_max_clips is not None:
            cmd.extend(["--video_max_clips", str(args.video_max_clips)])

        print(f"[INFO] Running case {idx}/{len(selected)}: {suite_name}:{case_name}")
        print("[INFO] Command:", " ".join(cmd))
        ret = subprocess.run(cmd)
        run_dir = os.path.join(per_case_root, run_name)
        summary_csv = os.path.join(run_dir, "summary_metrics", "case_summary.csv")
        if ret.returncode != 0:
            failures.append(
                {
                    "suite_name": suite_name,
                    "case_name": case_name,
                    "returncode": ret.returncode,
                    "run_dir": run_dir,
                }
            )
            if args.stop_on_error:
                break
            continue
        if not os.path.isfile(summary_csv):
            failures.append(
                {
                    "suite_name": suite_name,
                    "case_name": case_name,
                    "returncode": 0,
                    "error": "missing_summary_csv",
                    "run_dir": run_dir,
                }
            )
            if args.stop_on_error:
                break
            continue
        rows = _read_csv(summary_csv)
        case_results.extend(rows)

    _write_csv(os.path.join(summary_dir, "case_summary.csv"), case_results)
    _write_json(os.path.join(summary_dir, "case_summary.json"), case_results)
    _write_json(os.path.join(summary_dir, "failures.json"), failures)

    report_cmd = [
        "/isaac-sim/python.sh",
        os.path.join("scripts", "benchmark", "generate_quadruped_benchmark_report.py"),
        "--run_dir",
        master_dir,
    ]
    print("[INFO] Building merged plots/report...")
    subprocess.run(report_cmd, check=False)

    manifest = {
        "master_dir": master_dir,
        "num_selected_cases": len(selected),
        "num_success_cases": len(case_results),
        "num_failed_cases": len(failures),
        "per_case_root": per_case_root,
        "summary_dir": summary_dir,
        "plots_dir": plots_dir,
        "reports_dir": reports_dir,
    }
    _write_json(os.path.join(master_dir, "manifest.json"), manifest)
    _write_run_summary_md(master_dir, manifest, selected, failures)
    print(json.dumps(manifest, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
