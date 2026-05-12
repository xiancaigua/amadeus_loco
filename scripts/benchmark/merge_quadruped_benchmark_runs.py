# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Merge multiple isolated benchmark master directories into one summary."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any


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


def _write_run_summary_md(out_dir: str, manifest: dict[str, Any]):
    out_path = os.path.join(out_dir, "RUN_SUMMARY.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# Merged Benchmark Summary\n\n")
        f.write("## Overview\n\n")
        f.write(f"- Merged output directory: `{out_dir}`\n")
        f.write(f"- Source run dirs: {len(manifest.get('merged_from_run_dirs', []))}\n")
        f.write(f"- Merged case count (deduplicated): {manifest.get('merged_case_count')}\n")
        f.write(f"- Raw row count before dedup: {manifest.get('raw_row_count')}\n")
        f.write("\n## Key Files\n\n")
        f.write(f"- merge manifest: `{os.path.join(out_dir, 'merge_manifest.json')}`\n")
        f.write(f"- summary CSV: `{manifest.get('summary_csv')}`\n")
        f.write(f"- summary JSON: `{manifest.get('summary_json')}`\n")
        f.write(f"- plots dir (after report generation): `{os.path.join(out_dir, 'plots')}`\n")
        f.write(f"- report dir (after report generation): `{os.path.join(out_dir, 'reports')}`\n")
        f.write("\n## Source Runs\n\n")
        for src in manifest.get("merged_from_run_dirs", []):
            f.write(f"- `{src}`\n")


def _collect_rows(run_dir: str) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    sources: list[str] = []
    root = Path(run_dir)
    for summary in sorted(root.glob("runs/*/summary_metrics/case_summary.csv")):
        source_case_dir = str(summary.parent.parent)
        case_rows = _read_csv(str(summary))
        for row in case_rows:
            row["_source_case_dir"] = source_case_dir
        rows.extend(case_rows)
        sources.append(source_case_dir)
    return rows, sources


def main():
    parser = argparse.ArgumentParser(description="Merge isolated benchmark masters into one merged summary.")
    parser.add_argument(
        "--run_dirs",
        type=str,
        required=True,
        help="Comma-separated isolated benchmark master directories.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Merged output directory containing summary_metrics for report generation.",
    )
    args = parser.parse_args()

    run_dirs = [x.strip() for x in args.run_dirs.split(",") if x.strip()]
    if not run_dirs:
        raise ValueError("No run_dirs provided.")

    merged_rows: list[dict[str, Any]] = []
    merged_sources: list[str] = []
    for run_dir in run_dirs:
        rows, sources = _collect_rows(run_dir)
        merged_rows.extend(rows)
        merged_sources.extend(sources)

    dedup: dict[tuple[str, str], dict[str, Any]] = {}
    for row in merged_rows:
        key = (str(row.get("suite_name")), str(row.get("case_name")))
        dedup[key] = row
    final_rows = [dedup[k] for k in sorted(dedup.keys())]

    out_dir = os.path.abspath(args.out_dir)
    summary_dir = _ensure_dir(os.path.join(out_dir, "summary_metrics"))
    _ensure_dir(os.path.join(out_dir, "plots"))
    _ensure_dir(os.path.join(out_dir, "reports"))

    summary_csv = os.path.join(summary_dir, "case_summary.csv")
    summary_json = os.path.join(summary_dir, "case_summary.json")
    _write_csv(summary_csv, final_rows)
    _write_json(summary_json, final_rows)

    manifest = {
        "merged_from_run_dirs": run_dirs,
        "merged_case_count": len(final_rows),
        "raw_row_count": len(merged_rows),
        "source_case_dirs": sorted(set(merged_sources)),
        "summary_csv": summary_csv,
        "summary_json": summary_json,
    }
    _write_json(os.path.join(out_dir, "merge_manifest.json"), manifest)
    _write_run_summary_md(out_dir, manifest)
    print(json.dumps(manifest, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
