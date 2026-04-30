# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Plot training/evaluation metrics from a baseline run directory."""

from __future__ import annotations

import argparse
import os
from typing import Sequence

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _warn(msg: str) -> None:
    print(f"[WARN] {msg}")


def _info(msg: str) -> None:
    print(f"[INFO] {msg}")


def _read_csv_if_exists(path: str) -> pd.DataFrame | None:
    if not os.path.isfile(path):
        _warn(f"Metrics file not found: {path}")
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:  # pragma: no cover - file-dependent
        _warn(f"Failed to read CSV {path}: {exc}")
        return None


def _smooth_series(series: pd.Series, window: int) -> pd.Series:
    if window <= 1:
        return series
    return series.rolling(window=window, min_periods=1).mean()


def _resolve_x(df: pd.DataFrame) -> tuple[pd.Series, str]:
    for candidate in ("iteration", "total_steps", "step"):
        if candidate in df.columns:
            return pd.to_numeric(df[candidate], errors="coerce"), candidate
    return pd.Series(range(len(df))), "index"


def _resolve_column(df: pd.DataFrame, candidates: Sequence[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _plot_group(
    df: pd.DataFrame,
    *,
    title: str,
    x: pd.Series,
    x_label: str,
    curves: list[tuple[str, Sequence[str]]],
    smooth: int,
    out_path: str,
) -> bool:
    plt.figure(figsize=(10, 5))
    plotted_any = False
    for label, candidates in curves:
        col = _resolve_column(df, candidates)
        if col is None:
            _warn(f"[{title}] missing columns for '{label}': {list(candidates)}")
            continue
        y = pd.to_numeric(df[col], errors="coerce")
        y = _smooth_series(y, smooth)
        mask = x.notna() & y.notna()
        if not mask.any():
            _warn(f"[{title}] column '{col}' has no numeric data after cleanup.")
            continue
        plt.plot(x[mask], y[mask], label=f"{label} ({col})")
        plotted_any = True

    if not plotted_any:
        _warn(f"[{title}] no plottable curves found; saving placeholder figure: {out_path}")
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel("value")
        plt.text(0.5, 0.5, "No available numeric columns", ha="center", va="center", transform=plt.gca().transAxes)
        plt.grid(True, alpha=0.25)
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        return True

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel("value")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return True


def main():
    parser = argparse.ArgumentParser(description="Plot train/eval metrics for a baseline run directory.")
    parser.add_argument("--run_dir", type=str, required=True, help="Run directory containing metrics/*.csv.")
    parser.add_argument("--out_dir", type=str, default=None, help="Plot output directory (default: <run_dir>/plots).")
    parser.add_argument("--smooth", type=int, default=10, help="Rolling average window size.")
    parser.add_argument("--show", action="store_true", default=False, help="Display figures after saving.")
    args = parser.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    out_dir = os.path.abspath(args.out_dir) if args.out_dir else os.path.join(run_dir, "plots")
    os.makedirs(out_dir, exist_ok=True)

    train_csv = os.path.join(run_dir, "metrics", "train_metrics.csv")
    eval_csv = os.path.join(run_dir, "metrics", "eval_metrics.csv")
    train_df = _read_csv_if_exists(train_csv)
    eval_df = _read_csv_if_exists(eval_csv)

    generated_paths: list[str] = []

    if train_df is not None and len(train_df) > 0:
        x_train, x_label_train = _resolve_x(train_df)
        train_specs = [
            (
                "train_reward_length.png",
                "Train Reward / Episode Length",
                [
                    ("mean_reward", ("mean_reward",)),
                    ("mean_episode_length", ("mean_episode_length",)),
                ],
            ),
            (
                "train_termination.png",
                "Train Fall/Timeout Rate",
                [
                    ("fall_rate", ("fall_rate",)),
                    ("time_out_rate", ("time_out_rate", "timeout_rate")),
                ],
            ),
            (
                "train_tracking_reward.png",
                "Train Velocity Tracking Reward",
                [
                    ("track_lin_vel_reward", ("track_lin_vel_reward",)),
                    ("track_ang_vel_reward", ("track_ang_vel_reward",)),
                ],
            ),
            (
                "train_losses.png",
                "Train PPO Losses",
                [
                    ("policy_loss", ("policy_loss",)),
                    ("value_loss", ("value_loss",)),
                    ("entropy_loss", ("entropy_loss",)),
                ],
            ),
            (
                "train_ppo_diagnostics.png",
                "Train PPO Diagnostics",
                [
                    ("kl", ("kl",)),
                    ("learning_rate", ("learning_rate",)),
                    ("mean_action_std", ("mean_action_std",)),
                ],
            ),
        ]
        for file_name, title, curves in train_specs:
            out_path = os.path.join(out_dir, file_name)
            if _plot_group(
                train_df,
                title=title,
                x=x_train,
                x_label=x_label_train,
                curves=curves,
                smooth=max(1, args.smooth),
                out_path=out_path,
            ):
                generated_paths.append(out_path)
    elif train_df is not None and len(train_df) == 0:
        _warn(f"Train metrics file is empty: {train_csv}")

    if eval_df is not None and len(eval_df) > 0:
        x_eval, x_label_eval = _resolve_x(eval_df)
        eval_specs = [
            (
                "eval_reward_length.png",
                "Eval Reward / Episode Length",
                [
                    ("mean_episode_reward", ("mean_episode_reward",)),
                    ("mean_episode_length", ("mean_episode_length",)),
                ],
            ),
            (
                "eval_termination.png",
                "Eval Fall/Timeout Rate",
                [
                    ("fall_rate", ("fall_rate",)),
                    ("timeout_rate", ("timeout_rate", "time_out_rate")),
                ],
            ),
            (
                "eval_tracking_error.png",
                "Eval Velocity Tracking Error",
                [
                    ("mean_lin_vel_tracking_error", ("mean_lin_vel_tracking_error",)),
                    ("mean_yaw_vel_tracking_error", ("mean_yaw_vel_tracking_error",)),
                ],
            ),
        ]
        for file_name, title, curves in eval_specs:
            out_path = os.path.join(out_dir, file_name)
            if _plot_group(
                eval_df,
                title=title,
                x=x_eval,
                x_label=x_label_eval,
                curves=curves,
                smooth=max(1, args.smooth),
                out_path=out_path,
            ):
                generated_paths.append(out_path)
    elif eval_df is not None and len(eval_df) == 0:
        _warn(f"Eval metrics file is empty: {eval_csv}")

    if args.show:
        # In headless Docker this may no-op; keep optional for local interactive use.
        plt.show()

    if generated_paths:
        _info("Generated plot files:")
        for path in generated_paths:
            print(path)
    else:
        _warn("No plots were generated. Check available metrics files and columns.")


if __name__ == "__main__":
    main()
