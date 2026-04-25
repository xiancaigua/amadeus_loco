# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os

from isaaclab.managers.recorder_manager import DatasetExportMode
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.locomotion.velocity.config.anymal_c.rough_env_cfg import (
    AnymalCRoughEnvCfg,
    AnymalCRoughEnvCfg_PLAY,
)

from .recorders import AmadeusOfflineRecorderManagerCfg

LOCAL_ASSETS_ROOT_ENV_VAR = "AMADEUS_ISAACLAB_ASSETS_ROOT"
DEFAULT_LOCAL_ASSETS_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../../../..", "assets", "isaaclab_data")
)


def _resolve_local_assets_root() -> str | None:
    """Resolve local assets root from env var or default project cache path."""
    env_root = os.getenv(LOCAL_ASSETS_ROOT_ENV_VAR)
    if env_root:
        return os.path.abspath(env_root)
    return DEFAULT_LOCAL_ASSETS_ROOT if os.path.isdir(DEFAULT_LOCAL_ASSETS_ROOT) else None


def _maybe_override_anymal_assets_with_local_paths(env_cfg) -> None:
    """Override ANYmal asset paths with local files when a local assets root is available.

    Expected directory layout under the resolved root:
    - Robots/ANYbotics/ANYmal-C/anymal_c.usd
    - ActuatorNets/ANYbotics/anydrive_3_lstm_jit.pt
    """
    assets_root = _resolve_local_assets_root()
    if not assets_root:
        return

    local_robot_usd = os.path.join(assets_root, "Robots", "ANYbotics", "ANYmal-C", "anymal_c.usd")
    local_actuator_net = os.path.join(assets_root, "ActuatorNets", "ANYbotics", "anydrive_3_lstm_jit.pt")

    missing_paths = [path for path in (local_robot_usd, local_actuator_net) if not os.path.isfile(path)]
    if missing_paths:
        if os.getenv(LOCAL_ASSETS_ROOT_ENV_VAR):
            # Explicit override should fail fast to avoid silently falling back to remote assets.
            raise FileNotFoundError(
                "Local asset override was requested via "
                f"{LOCAL_ASSETS_ROOT_ENV_VAR}={assets_root}, but required files are missing: {missing_paths}"
            )
        # Default cache path is optional; if not fully available, keep upstream behavior.
        return

    print(
        "[INFO] Using local ANYmal assets from: "
        f"{assets_root} (set {LOCAL_ASSETS_ROOT_ENV_VAR} to override explicitly)."
    )
    env_cfg.scene.robot.spawn.usd_path = local_robot_usd
    legs_actuator = env_cfg.scene.robot.actuators.get("legs", None)
    if legs_actuator is not None and hasattr(legs_actuator, "network_file"):
        legs_actuator.network_file = local_actuator_net


@configclass
class AmadeusQuadrupedRoughEnvCfg(AnymalCRoughEnvCfg):
    """External-project baseline built on top of the official ANYmal-C rough locomotion task."""

    recorders: AmadeusOfflineRecorderManagerCfg = AmadeusOfflineRecorderManagerCfg()

    def __post_init__(self):
        super().__post_init__()
        _maybe_override_anymal_assets_with_local_paths(self)
        # Disable command debug markers (arrow_x.usd) to avoid remote UI asset dependency in headless server runs.
        if hasattr(self, "commands") and hasattr(self.commands, "base_velocity"):
            self.commands.base_velocity.debug_vis = False
        self.env_name = "Template-Amadeus-Quadruped-Rough-v0"
        self.recorders.dataset_filename = "train_dataset"
        self.recorders.dataset_export_mode = DatasetExportMode.EXPORT_ALL


@configclass
class AmadeusQuadrupedRoughEnvCfg_PLAY(AnymalCRoughEnvCfg_PLAY):
    """Play/evaluation configuration for the external quadruped baseline."""

    recorders: AmadeusOfflineRecorderManagerCfg = AmadeusOfflineRecorderManagerCfg()

    def __post_init__(self):
        super().__post_init__()
        _maybe_override_anymal_assets_with_local_paths(self)
        if hasattr(self, "commands") and hasattr(self.commands, "base_velocity"):
            self.commands.base_velocity.debug_vis = False
        self.env_name = "Template-Amadeus-Quadruped-Rough-Play-v0"
        self.recorders.dataset_filename = "eval_dataset"
        self.recorders.export_in_close = True
