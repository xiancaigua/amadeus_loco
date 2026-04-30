# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Build one consolidated robustness report directory from isolated scenario runs."""

from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def _read_single_summary_csv(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if len(rows) != 1:
        raise ValueError(f"Expected exactly one row in {path}, got {len(rows)}")
    return rows[0]


def _to_float(v: Any) -> float:
    if v is None:
        return float("nan")
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip()
    if s == "" or s.lower() in {"none", "nan"}:
        return float("nan")
    return float(s)


def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _list_videos(run_dir: str) -> list[str]:
    if not run_dir:
        return []
    root = Path(run_dir)
    if not root.exists():
        return []
    return sorted(str(p) for p in root.rglob("*.mp4"))


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


def _plot_compare(rows: list[dict[str, Any]], out_dir: str) -> list[str]:
    names = [r["scenario_name"] for r in rows]
    specs = [
        ("mean_return", "Average Return"),
        ("mean_episode_length", "Episode Length"),
        ("fall_rate", "Fall Rate"),
        ("mean_lin_vel_tracking_error", "LinVel Tracking Error"),
        ("mean_yaw_vel_tracking_error", "Yaw Tracking Error"),
        ("mean_recovery_time_s", "Recovery Time (s)"),
        ("recovery_success_rate", "Recovery Success Rate"),
    ]
    colors = {
        "in_distribution": "#4C78A8",
        "long_tail": "#F58518",
        "ood": "#E45756",
    }
    generated = []

    for key, title in specs:
        vals = [_to_float(r.get(key)) for r in rows]
        fig = plt.figure(figsize=(10, 4))
        x = np.arange(len(vals))
        bar_colors = [colors.get(r.get("scenario_group", ""), "#999999") for r in rows]
        plt.bar(x, vals, color=bar_colors)
        plt.xticks(x, names, rotation=12, ha="right")
        plt.ylabel(key)
        plt.title(title)
        plt.tight_layout()
        out_path = os.path.join(out_dir, f"{key}.png")
        fig.savefig(out_path, dpi=140)
        plt.close(fig)
        generated.append(out_path)
    return generated


def _relative_change(base: float, value: float) -> float:
    if np.isnan(base) or base == 0:
        return float("nan")
    return (value - base) / abs(base)


def main():
    parser = argparse.ArgumentParser(description="Consolidate isolated robustness runs into one report directory.")
    parser.add_argument("--id_run_dir", type=str, required=True)
    parser.add_argument("--long_tail_run_dir", type=str, required=True)
    parser.add_argument("--ood_run_dir", type=str, required=True)
    parser.add_argument("--id_video_run_dir", type=str, default=None)
    parser.add_argument("--long_tail_video_run_dir", type=str, default=None)
    parser.add_argument("--ood_video_run_dir", type=str, default=None)
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output report directory. Default: outputs/quadruped_rough_benchmark_isolated/report_<timestamp>.",
    )
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.join(
        "outputs",
        "quadruped_rough_benchmark_isolated",
        f"report_{datetime.utcnow().strftime('%Y-%m-%d_%H-%M-%S')}",
    )
    _ensure_dir(out_dir)
    _ensure_dir(os.path.join(out_dir, "tables"))
    _ensure_dir(os.path.join(out_dir, "plots"))

    def load(run_dir: str):
        summary_path = os.path.join(run_dir, "summary", "scenario_summary.csv")
        row = _read_single_summary_csv(summary_path)
        row["run_dir"] = run_dir
        return row

    rows = [load(args.id_run_dir), load(args.long_tail_run_dir), load(args.ood_run_dir)]

    # numeric conversion
    numeric_cols = [
        "mean_return",
        "mean_episode_length",
        "fall_rate",
        "timeout_rate",
        "mean_lin_vel_tracking_error",
        "mean_yaw_vel_tracking_error",
        "mean_recovery_time_s",
        "recovery_success_rate",
        "mean_action_smoothness",
        "mean_energy_proxy",
        "mean_abs_roll",
        "mean_abs_pitch",
    ]
    for row in rows:
        for key in numeric_cols:
            if key in row:
                row[key] = _to_float(row[key])

    # degradation table relative to ID
    id_row = rows[0]
    deltas = []
    for row in rows[1:]:
        deltas.append(
            {
                "scenario_name": row["scenario_name"],
                "scenario_group": row["scenario_group"],
                "return_change_vs_id": _relative_change(id_row["mean_return"], row["mean_return"]),
                "episode_length_change_vs_id": _relative_change(id_row["mean_episode_length"], row["mean_episode_length"]),
                "fall_rate_delta_vs_id": row["fall_rate"] - id_row["fall_rate"],
                "lin_tracking_error_change_vs_id": _relative_change(
                    id_row["mean_lin_vel_tracking_error"], row["mean_lin_vel_tracking_error"]
                ),
                "yaw_tracking_error_change_vs_id": _relative_change(
                    id_row["mean_yaw_vel_tracking_error"], row["mean_yaw_vel_tracking_error"]
                ),
                "recovery_success_rate_delta_vs_id": row["recovery_success_rate"] - id_row["recovery_success_rate"],
            }
        )

    summary_csv = os.path.join(out_dir, "tables", "scenario_summary.csv")
    delta_csv = os.path.join(out_dir, "tables", "scenario_delta_vs_id.csv")
    _write_csv(summary_csv, rows)
    _write_csv(delta_csv, deltas)

    plot_files = _plot_compare(rows, os.path.join(out_dir, "plots"))

    video_index = {
        "id_videos": _list_videos(args.id_video_run_dir) if args.id_video_run_dir else [],
        "long_tail_videos": _list_videos(args.long_tail_video_run_dir) if args.long_tail_video_run_dir else [],
        "ood_videos": _list_videos(args.ood_video_run_dir) if args.ood_video_run_dir else [],
    }
    _write_json(os.path.join(out_dir, "video_index.json"), video_index)

    report_md = os.path.join(out_dir, "analysis_summary.md")
    with open(report_md, "w", encoding="utf-8") as f:
        f.write("# Quadruped Rough Robustness Benchmark Report\n\n")
        f.write("## Scenario Summary\n\n")
        f.write(f"- ID run: `{args.id_run_dir}`\n")
        f.write(f"- Long-tail run: `{args.long_tail_run_dir}`\n")
        f.write(f"- OOD run: `{args.ood_run_dir}`\n\n")
        f.write("## Key Findings (Auto)\n\n")
        for d in deltas:
            f.write(
                f"- `{d['scenario_name']}`: "
                f"return change vs ID = {d['return_change_vs_id']:.3f}, "
                f"episode length change vs ID = {d['episode_length_change_vs_id']:.3f}, "
                f"fall-rate delta = {d['fall_rate_delta_vs_id']:.3f}, "
                f"lin-vel error change = {d['lin_tracking_error_change_vs_id']:.3f}, "
                f"yaw-vel error change = {d['yaw_tracking_error_change_vs_id']:.3f}.\n"
            )
        f.write("\n## Files\n\n")
        f.write(f"- scenario table: `{summary_csv}`\n")
        f.write(f"- delta table: `{delta_csv}`\n")
        f.write(f"- plots: `{os.path.join(out_dir, 'plots')}`\n")
        f.write(f"- video index: `{os.path.join(out_dir, 'video_index.json')}`\n")

    manifest = {
        "out_dir": out_dir,
        "summary_csv": summary_csv,
        "delta_csv": delta_csv,
        "plots": plot_files,
        "video_index": os.path.join(out_dir, "video_index.json"),
        "analysis_markdown": report_md,
    }
    _write_json(os.path.join(out_dir, "manifest.json"), manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
