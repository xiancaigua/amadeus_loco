# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers.recorder_manager import DatasetExportMode
from isaaclab.utils import configclass
from isaaclab_assets.robots.anymal import ANYMAL_C_CFG

from .config import (
    AmadeusQuadrupedRoughActionsCfg,
    AmadeusQuadrupedRoughCommandsCfg,
    AmadeusQuadrupedRoughCurriculumCfg,
    AmadeusQuadrupedRoughEventCfg,
    AmadeusQuadrupedRoughObservationsCfg,
    AmadeusQuadrupedRoughRewardsCfg,
    AmadeusQuadrupedRoughSceneCfg,
    AmadeusQuadrupedRoughTerminationsCfg,
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
class AmadeusQuadrupedRoughEnvCfg(ManagerBasedRLEnvCfg):
    """Explicit manager-based rough-terrain ANYmal-C baseline config for Amadeus."""

    scene: AmadeusQuadrupedRoughSceneCfg = AmadeusQuadrupedRoughSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: AmadeusQuadrupedRoughObservationsCfg = AmadeusQuadrupedRoughObservationsCfg()
    actions: AmadeusQuadrupedRoughActionsCfg = AmadeusQuadrupedRoughActionsCfg()
    commands: AmadeusQuadrupedRoughCommandsCfg = AmadeusQuadrupedRoughCommandsCfg()
    rewards: AmadeusQuadrupedRoughRewardsCfg = AmadeusQuadrupedRoughRewardsCfg()
    terminations: AmadeusQuadrupedRoughTerminationsCfg = AmadeusQuadrupedRoughTerminationsCfg()
    events: AmadeusQuadrupedRoughEventCfg = AmadeusQuadrupedRoughEventCfg()
    curriculum: AmadeusQuadrupedRoughCurriculumCfg = AmadeusQuadrupedRoughCurriculumCfg()
    recorders: AmadeusOfflineRecorderManagerCfg = AmadeusOfflineRecorderManagerCfg()

    def __post_init__(self):
        # Scene/robot setup
        self.scene.robot = ANYMAL_C_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Environment timing
        self.decimation = 4
        self.episode_length_s = 20.0

        # Simulation setup
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        # Sensor update rates
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        # Terrain curriculum setup
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False

        # Local asset override for offline/headless clusters
        _maybe_override_anymal_assets_with_local_paths(self)

        # Disable command debug markers (arrow_x.usd) to avoid remote UI asset dependency in headless runs.
        if hasattr(self.commands, "base_velocity"):
            self.commands.base_velocity.debug_vis = False

        self.env_name = "Template-Amadeus-Quadruped-Rough-v0"
        self.recorders.dataset_filename = "train_dataset"
        self.recorders.dataset_export_mode = DatasetExportMode.EXPORT_ALL


@configclass
class AmadeusQuadrupedRoughEnvCfg_PLAY(AmadeusQuadrupedRoughEnvCfg):
    """Play/evaluation configuration for the external quadruped baseline."""

    def __post_init__(self):
        super().__post_init__()

        # Smaller play scene for evaluation
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        # Play mode: deterministic obs and no stochastic perturbation pushes
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None

        self.env_name = "Template-Amadeus-Quadruped-Rough-Play-v0"
        self.recorders.dataset_filename = "eval_dataset"
        self.recorders.export_in_close = True
