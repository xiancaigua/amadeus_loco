# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Regenerate benchmark plots/report from an existing benchmark run directory.

This utility is intentionally standalone (no `import amadeus`) so it can run
without Isaac-Sim app bootstrapping.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from datetime import datetime
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def _read_csv(path: str) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _safe_float(v: Any) -> float:
    if v is None:
        return float("nan")
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip()
    if s == "" or s.lower() in {"none", "nan"}:
        return float("nan")
    try:
        return float(s)
    except ValueError:
        return float("nan")


def _write_json(path: str, payload: Any):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)


def _plot_case_bars(rows: list[dict[str, Any]], out_dir: str) -> list[str]:
    specs = [
        ("mean_return", "Average Return"),
        ("mean_episode_length", "Episode Length"),
        ("fall_rate", "Fall Rate"),
        ("mean_lin_vel_tracking_error", "Linear Velocity Tracking Error"),
    ]
    names = [f"{r.get('suite_name')}:{r.get('case_name')}" for r in rows]
    buckets = [str(r.get("bucket", "unknown")) for r in rows]
    color_map = {"id": "#4C78A8", "in_distribution": "#4C78A8", "long_tail": "#F58518", "ood": "#E45756"}
    outputs: list[str] = []
    for key, title in specs:
        values = [_safe_float(r.get(key)) for r in rows]
        fig = plt.figure(figsize=(max(10, len(values) * 0.55), 4.2))
        x = np.arange(len(values))
        plt.bar(x, values, color=[color_map.get(bucket, "#999999") for bucket in buckets])
        plt.xticks(x, names, rotation=20, ha="right")
        plt.ylabel(key)
        plt.title(title)
        plt.tight_layout()
        out_path = os.path.join(out_dir, f"case_bar_{key}.png")
        fig.savefig(out_path, dpi=140)
        plt.close(fig)
        outputs.append(out_path)
    return outputs


def _plot_bucket_bars(rows: list[dict[str, Any]], out_dir: str) -> list[str]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("bucket", "unknown"))].append(row)
    buckets = sorted(grouped.keys())
    specs = [
        ("mean_return", "Bucket Average Return"),
        ("fall_rate", "Bucket Fall Rate"),
        ("mean_lin_vel_tracking_error", "Bucket Linear Velocity Tracking Error"),
    ]
    outputs: list[str] = []
    for key, title in specs:
        values = []
        for bucket in buckets:
            nums = [_safe_float(item.get(key)) for item in grouped[bucket]]
            nums = [x for x in nums if not math.isnan(x)]
            values.append(float(np.mean(nums)) if nums else float("nan"))
        fig = plt.figure(figsize=(8, 4))
        x = np.arange(len(values))
        plt.bar(x, values, color="#5B8FF9")
        plt.xticks(x, buckets)
        plt.ylabel(key)
        plt.title(title)
        plt.tight_layout()
        out_path = os.path.join(out_dir, f"bucket_bar_{key}.png")
        fig.savefig(out_path, dpi=140)
        plt.close(fig)
        outputs.append(out_path)
    return outputs


def _build_summary_md(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "No benchmark rows are available."
    id_rows = [r for r in rows if str(r.get("bucket", "")).lower() in {"id", "in_distribution"}]
    ref = id_rows[0] if id_rows else rows[0]
    ref_return = _safe_float(ref.get("mean_return"))
    ref_fall = _safe_float(ref.get("fall_rate"))
    ref_track = _safe_float(ref.get("mean_lin_vel_tracking_error"))
    rank = []
    for row in rows:
        d_return = _safe_float(row.get("mean_return")) - ref_return
        d_fall = _safe_float(row.get("fall_rate")) - ref_fall
        d_track = _safe_float(row.get("mean_lin_vel_tracking_error")) - ref_track
        score = (-d_return) + (3.0 * d_fall) + (2.0 * max(d_track, 0.0))
        rank.append((score, row, d_return, d_fall, d_track))
    rank.sort(key=lambda x: x[0], reverse=True)
    lines = []
    lines.append("# PPO Baseline Robustness Summary")
    lines.append("")
    lines.append(f"- Generated at: {datetime.utcnow().isoformat()} UTC")
    lines.append(
        f"- Reference: `{ref.get('suite_name')}:{ref.get('case_name')}` "
        f"(return={ref_return:.3f}, fall={ref_fall:.3f}, lin_err={ref_track:.3f})"
    )
    lines.append("")
    lines.append("## Most Vulnerable Cases")
    lines.append("")
    for _, row, d_ret, d_fall, d_track in rank[: min(5, len(rank))]:
        lines.append(
            f"- `{row.get('suite_name')}:{row.get('case_name')}` | bucket={row.get('bucket')} | "
            f"Δreturn={d_ret:.3f}, Δfall_rate={d_fall:.3f}, Δlin_err={d_track:.3f}"
        )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Regenerate benchmark summary plots/report from case_summary.csv.")
    parser.add_argument("--run_dir", type=str, required=True, help="Benchmark run directory.")
    parser.add_argument(
        "--summary_csv",
        type=str,
        default=None,
        help="Optional summary csv path. Default: <run_dir>/summary_metrics/case_summary.csv",
    )
    args = parser.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    summary_csv = args.summary_csv or os.path.join(run_dir, "summary_metrics", "case_summary.csv")
    if not os.path.isfile(summary_csv):
        raise FileNotFoundError(f"Summary CSV not found: {summary_csv}")

    rows = _read_csv(summary_csv)
    plots_dir = _ensure_dir(os.path.join(run_dir, "plots"))
    reports_dir = _ensure_dir(os.path.join(run_dir, "reports"))
    outputs = []
    outputs.extend(_plot_case_bars(rows, plots_dir))
    outputs.extend(_plot_bucket_bars(rows, plots_dir))

    summary_md_path = os.path.join(reports_dir, "research_summary.md")
    with open(summary_md_path, "w", encoding="utf-8") as f:
        f.write(_build_summary_md(rows))

    manifest = {
        "run_dir": run_dir,
        "summary_csv": summary_csv,
        "plots": outputs,
        "research_summary_md": summary_md_path,
    }
    _write_json(os.path.join(reports_dir, "manifest.json"), manifest)
    print(json.dumps(manifest, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
