# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
from dataclasses import dataclass


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


@dataclass
class BenchmarkOutputLayout:
    run_dir: str

    @classmethod
    def create(cls, run_dir: str) -> "BenchmarkOutputLayout":
        run_dir = os.path.abspath(run_dir)
        layout = cls(run_dir=run_dir)
        ensure_dir(layout.config_snapshot_dir)
        ensure_dir(layout.raw_metrics_dir)
        ensure_dir(layout.summary_metrics_dir)
        ensure_dir(layout.videos_dir)
        ensure_dir(layout.plots_dir)
        ensure_dir(layout.traces_dir)
        ensure_dir(layout.reports_dir)
        return layout

    @property
    def config_snapshot_dir(self) -> str:
        return os.path.join(self.run_dir, "config_snapshot")

    @property
    def raw_metrics_dir(self) -> str:
        return os.path.join(self.run_dir, "raw_metrics")

    @property
    def summary_metrics_dir(self) -> str:
        return os.path.join(self.run_dir, "summary_metrics")

    @property
    def videos_dir(self) -> str:
        return os.path.join(self.run_dir, "videos")

    @property
    def plots_dir(self) -> str:
        return os.path.join(self.run_dir, "plots")

    @property
    def traces_dir(self) -> str:
        return os.path.join(self.run_dir, "traces")

    @property
    def reports_dir(self) -> str:
        return os.path.join(self.run_dir, "reports")

    def case_raw_metrics_dir(self, suite_name: str, case_name: str) -> str:
        return ensure_dir(os.path.join(self.raw_metrics_dir, suite_name, case_name))

    def case_videos_dir(self, suite_name: str, case_name: str) -> str:
        return ensure_dir(os.path.join(self.videos_dir, suite_name, case_name))

    def case_traces_dir(self, suite_name: str, case_name: str) -> str:
        return ensure_dir(os.path.join(self.traces_dir, suite_name, case_name))
