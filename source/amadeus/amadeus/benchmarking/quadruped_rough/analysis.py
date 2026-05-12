# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import csv
import json
import math
import os
from collections import defaultdict
from datetime import datetime
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from .outputs import BenchmarkOutputLayout, ensure_dir


def _safe_float(v: Any) -> float:
    if v is None:
        return float("nan")
    if isinstance(v, (int, float)):
        return float(v)
    text = str(v).strip()
    if text == "" or text.lower() in {"none", "nan"}:
        return float("nan")
    try:
        return float(text)
    except ValueError:
        return float("nan")


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


def _plot_case_bars(summary_rows: list[dict[str, Any]], out_dir: str) -> list[str]:
    if not summary_rows:
        return []
    ensure_dir(out_dir)
    names = [f"{r['suite_name']}:{r['case_name']}" for r in summary_rows]
    buckets = [r.get("bucket", "unknown") for r in summary_rows]
    color_map = {
        "id": "#4C78A8",
        "in_distribution": "#4C78A8",
        "long_tail": "#F58518",
        "ood": "#E45756",
    }
    specs = [
        ("mean_return", "Average Return"),
        ("mean_episode_length", "Episode Length"),
        ("fall_rate", "Fall Rate"),
        ("mean_lin_vel_tracking_error", "Linear Velocity Tracking Error"),
        ("mean_recovery_time_s", "Recovery Time (s)"),
    ]
    generated: list[str] = []
    for metric_key, title in specs:
        vals = [_safe_float(r.get(metric_key)) for r in summary_rows]
        fig = plt.figure(figsize=(max(10, len(vals) * 0.55), 4.2))
        x = np.arange(len(vals))
        colors = [color_map.get(bucket, "#999999") for bucket in buckets]
        plt.bar(x, vals, color=colors)
        plt.xticks(x, names, rotation=20, ha="right")
        plt.ylabel(metric_key)
        plt.title(title)
        plt.tight_layout()
        out_path = os.path.join(out_dir, f"case_bar_{metric_key}.png")
        fig.savefig(out_path, dpi=140)
        plt.close(fig)
        generated.append(out_path)
    return generated


def _plot_bucket_bars(summary_rows: list[dict[str, Any]], out_dir: str) -> list[str]:
    if not summary_rows:
        return []
    ensure_dir(out_dir)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in summary_rows:
        grouped[str(row.get("bucket", "unknown"))].append(row)
    buckets = sorted(grouped.keys())

    def bucket_mean(metric: str, bucket: str):
        vals = [_safe_float(r.get(metric)) for r in grouped[bucket]]
        vals = [v for v in vals if not math.isnan(v)]
        return float(np.mean(vals)) if vals else float("nan")

    specs = [
        ("mean_return", "Bucket Average Return"),
        ("mean_episode_length", "Bucket Episode Length"),
        ("fall_rate", "Bucket Fall Rate"),
        ("mean_lin_vel_tracking_error", "Bucket Linear Velocity Tracking Error"),
    ]
    generated: list[str] = []
    for metric_key, title in specs:
        vals = [bucket_mean(metric_key, b) for b in buckets]
        fig = plt.figure(figsize=(8, 4))
        x = np.arange(len(vals))
        plt.bar(x, vals, color="#5B8FF9")
        plt.xticks(x, buckets, rotation=0)
        plt.ylabel(metric_key)
        plt.title(title)
        plt.tight_layout()
        out_path = os.path.join(out_dir, f"bucket_bar_{metric_key}.png")
        fig.savefig(out_path, dpi=140)
        plt.close(fig)
        generated.append(out_path)
    return generated


def _plot_combined_ood_heatmap(summary_rows: list[dict[str, Any]], out_dir: str) -> list[str]:
    """Build a simple heatmap for cases tagged with `combined_ood` and explicit grid metadata."""
    ensure_dir(out_dir)
    rows = [r for r in summary_rows if str(r.get("suite_name", "")).startswith("combined_ood")]
    if not rows:
        return []

    x_vals = []
    y_vals = []
    z_vals = []
    for row in rows:
        grid_x = _safe_float(row.get("grid_x"))
        grid_y = _safe_float(row.get("grid_y"))
        if math.isnan(grid_x) or math.isnan(grid_y):
            continue
        x_vals.append(grid_x)
        y_vals.append(grid_y)
        z_vals.append(_safe_float(row.get("fall_rate")))
    if not x_vals:
        return []

    xs = sorted(set(x_vals))
    ys = sorted(set(y_vals))
    mat = np.full((len(ys), len(xs)), np.nan, dtype=np.float64)
    for x, y, z in zip(x_vals, y_vals, z_vals):
        ix = xs.index(x)
        iy = ys.index(y)
        mat[iy, ix] = z

    fig = plt.figure(figsize=(7, 5))
    im = plt.imshow(mat, cmap="magma", aspect="auto", origin="lower")
    plt.colorbar(im, label="fall_rate")
    plt.xticks(np.arange(len(xs)), [f"{v:.2f}" for v in xs])
    plt.yticks(np.arange(len(ys)), [f"{v:.2f}" for v in ys])
    plt.xlabel("grid_x")
    plt.ylabel("grid_y")
    plt.title("Combined OOD Fall Rate Heatmap")
    plt.tight_layout()
    out_path = os.path.join(out_dir, "combined_ood_fall_rate_heatmap.png")
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return [out_path]


def build_research_summary(summary_rows: list[dict[str, Any]]) -> str:
    if not summary_rows:
        return "No benchmark rows are available."

    # Pick a reference ID row and compare others against it.
    id_rows = [r for r in summary_rows if str(r.get("bucket", "")).lower() in {"id", "in_distribution"}]
    ref = id_rows[0] if id_rows else summary_rows[0]
    ref_return = _safe_float(ref.get("mean_return"))
    ref_fall = _safe_float(ref.get("fall_rate"))
    ref_track = _safe_float(ref.get("mean_lin_vel_tracking_error"))

    rank = []
    for row in summary_rows:
        d_return = _safe_float(row.get("mean_return")) - ref_return
        d_fall = _safe_float(row.get("fall_rate")) - ref_fall
        d_track = _safe_float(row.get("mean_lin_vel_tracking_error")) - ref_track
        score = (-d_return) + (3.0 * d_fall) + (2.0 * max(d_track, 0.0))
        rank.append((score, row, d_return, d_fall, d_track))
    rank.sort(key=lambda x: x[0], reverse=True)

    worst = rank[0]
    top_failures = rank[: min(5, len(rank))]

    lines = []
    lines.append("# PPO Baseline Robustness Summary")
    lines.append("")
    lines.append(f"- Generated at: {datetime.utcnow().isoformat()} UTC")
    lines.append(
        f"- Reference case: `{ref.get('suite_name')}:{ref.get('case_name')}` "
        f"(return={ref_return:.3f}, fall_rate={ref_fall:.3f}, lin_err={ref_track:.3f})"
    )
    lines.append("")
    lines.append("## Most Vulnerable Scenarios")
    lines.append("")
    for _, row, d_ret, d_fall, d_track in top_failures:
        lines.append(
            f"- `{row.get('suite_name')}:{row.get('case_name')}` | bucket={row.get('bucket')} | "
            f"Δreturn={d_ret:.3f}, Δfall_rate={d_fall:.3f}, Δlin_err={d_track:.3f}"
        )
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        f"- Worst degradation appears in `{worst[1].get('suite_name')}:{worst[1].get('case_name')}`, "
        "indicating current PPO policy is sensitive to distribution shift in that bucket."
    )
    lines.append(
        "- If fall-rate rises faster than tracking error, the dominant failure mode is likely abrupt stability loss "
        "(recovery weakness)."
    )
    lines.append(
        "- If tracking error rises with moderate fall-rate increase, the dominant failure mode is command-tracking "
        "generalization drift near boundary/OOD commands."
    )
    lines.append(
        "- Scenarios with high recovery-time and low recovery-success strongly motivate history/memory mechanisms."
    )
    return "\n".join(lines)


def save_summary_artifacts(summary_rows: list[dict[str, Any]], output_layout: BenchmarkOutputLayout) -> dict[str, Any]:
    ensure_dir(output_layout.summary_metrics_dir)
    ensure_dir(output_layout.plots_dir)
    ensure_dir(output_layout.reports_dir)

    summary_csv = os.path.join(output_layout.summary_metrics_dir, "case_summary.csv")
    summary_json = os.path.join(output_layout.summary_metrics_dir, "case_summary.json")
    _write_csv(summary_csv, summary_rows)
    _write_json(summary_json, summary_rows)

    plot_paths = []
    plot_paths.extend(_plot_case_bars(summary_rows, output_layout.plots_dir))
    plot_paths.extend(_plot_bucket_bars(summary_rows, output_layout.plots_dir))
    plot_paths.extend(_plot_combined_ood_heatmap(summary_rows, output_layout.plots_dir))

    summary_md_text = build_research_summary(summary_rows)
    summary_md = os.path.join(output_layout.reports_dir, "research_summary.md")
    with open(summary_md, "w", encoding="utf-8") as f:
        f.write(summary_md_text)

    manifest = {
        "summary_csv": summary_csv,
        "summary_json": summary_json,
        "plots": plot_paths,
        "research_summary_md": summary_md,
    }
    _write_json(os.path.join(output_layout.reports_dir, "manifest.json"), manifest)
    return manifest
