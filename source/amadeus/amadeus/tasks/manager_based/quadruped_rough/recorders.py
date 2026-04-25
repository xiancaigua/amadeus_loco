# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import json
import os
from collections.abc import Sequence

import h5py
import numpy as np
import torch

from isaaclab.managers import RecorderManagerBaseCfg, RecorderTerm, RecorderTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.datasets import EpisodeData
from isaaclab.utils.datasets.dataset_file_handler_base import DatasetFileHandlerBase


class ChunkedHDF5DatasetFileHandler(DatasetFileHandlerBase):
    """HDF5 dataset writer that rotates files to avoid a single unbounded dataset file."""

    max_episodes_per_file = 128

    def __init__(self):
        self._hdf5_file_stream = None
        self._hdf5_data_group = None
        self._file_path_stem = None
        self._env_name = ""
        self._env_args = {}
        self._chunk_index = 0
        self._chunk_episode_count = 0
        self._total_episode_count = 0

    def open(self, file_path: str, mode: str = "r"):
        if self._hdf5_file_stream is not None:
            raise RuntimeError("HDF5 dataset file stream is already in use")
        self._hdf5_file_stream = h5py.File(file_path, mode)
        self._hdf5_data_group = self._hdf5_file_stream["data"]
        self._chunk_episode_count = len(self._hdf5_data_group)
        self._total_episode_count = self._chunk_episode_count
        self._file_path_stem = file_path.removesuffix(".hdf5")
        env_args = self._hdf5_data_group.attrs.get("env_args")
        if env_args:
            self._env_args = json.loads(env_args)
            self._env_name = self._env_args.get("env_name", "")

    def create(self, file_path: str, env_name: str = None):
        if self._hdf5_file_stream is not None:
            raise RuntimeError("HDF5 dataset file stream is already in use")
        self._file_path_stem = file_path.removesuffix(".hdf5")
        self._env_name = env_name or ""
        self._env_args = {"env_name": self._env_name, "type": 2}
        self._chunk_index = 0
        self._chunk_episode_count = 0
        self._total_episode_count = 0
        self._open_new_chunk()

    def get_env_name(self) -> str | None:
        return self._env_name

    def add_env_args(self, env_args: dict) -> None:
        self._raise_if_not_initialized()
        self._env_args.update(env_args)
        self._hdf5_data_group.attrs["env_args"] = json.dumps(self._env_args)

    def set_env_name(self, env_name: str) -> None:
        self._raise_if_not_initialized()
        self._env_name = env_name
        self.add_env_args({"env_name": env_name})

    def load_episode(self, episode_name: str) -> EpisodeData | None:
        raise NotImplementedError("Chunked dataset export is write-oriented; use the postprocess script for reading.")

    def get_num_episodes(self) -> int:
        return self._total_episode_count

    def write_episode(self, episode: EpisodeData, demo_id: int | None = None):
        self._raise_if_not_initialized()
        if episode.is_empty():
            return
        if self._chunk_episode_count >= self.max_episodes_per_file:
            self._open_new_chunk()

        if demo_id is None:
            episode_group_name = f"demo_{self._total_episode_count}"
        else:
            episode_group_name = f"demo_{demo_id}"
        h5_episode_group = self._hdf5_data_group.create_group(episode_group_name)

        if "actions" in episode.data:
            h5_episode_group.attrs["num_samples"] = len(episode.data["actions"])
        else:
            h5_episode_group.attrs["num_samples"] = 0

        if episode.seed is not None:
            h5_episode_group.attrs["seed"] = episode.seed
        if episode.success is not None:
            h5_episode_group.attrs["success"] = episode.success

        def create_dataset_helper(group, key, value):
            if isinstance(value, dict):
                key_group = group.create_group(key)
                for sub_key, sub_value in value.items():
                    create_dataset_helper(key_group, sub_key, sub_value)
            else:
                group.create_dataset(key, data=value.cpu().numpy(), compression="gzip")

        for key, value in episode.data.items():
            create_dataset_helper(h5_episode_group, key, value)

        self._hdf5_data_group.attrs["total"] += h5_episode_group.attrs["num_samples"]
        self._chunk_episode_count += 1
        self._total_episode_count += 1

    def flush(self):
        self._raise_if_not_initialized()
        self._hdf5_file_stream.flush()

    def close(self):
        if self._hdf5_file_stream is not None:
            self._hdf5_file_stream.close()
            self._hdf5_file_stream = None
            self._hdf5_data_group = None

    def _open_new_chunk(self):
        self.close()
        chunk_path = f"{self._file_path_stem}_{self._chunk_index:04d}.hdf5"
        dir_path = os.path.dirname(chunk_path)
        if dir_path and not os.path.isdir(dir_path):
            os.makedirs(dir_path, exist_ok=True)
        self._hdf5_file_stream = h5py.File(chunk_path, "w")
        self._hdf5_data_group = self._hdf5_file_stream.create_group("data")
        self._hdf5_data_group.attrs["total"] = 0
        self._hdf5_data_group.attrs["env_args"] = json.dumps(self._env_args)
        self._chunk_episode_count = 0
        self._chunk_index += 1

    def _raise_if_not_initialized(self):
        if self._hdf5_file_stream is None:
            raise RuntimeError("HDF5 dataset file stream is not initialized")


class QuadrupedTransitionRecorder(RecorderTerm):
    """Record transition tuples with optional contact/terrain side channels."""

    def __init__(self, cfg: RecorderTermCfg, env):
        super().__init__(cfg, env)
        self._env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long).unsqueeze(-1)
        self._episode_ids = torch.full((env.num_envs, 1), -1, device=env.device, dtype=torch.long)

    def record_post_reset(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            self._episode_ids[:] += 1
        elif len(env_ids) > 0:
            self._episode_ids[env_ids] += 1
        return None, None

    def record_pre_step(self):
        payload = {
            "obs": self._env.obs_buf["policy"].clone(),
            "actions": self._env.action_manager.action.clone(),
            "episode_id": self._episode_ids.clone(),
            "env_id": self._env_ids.clone(),
            "timestep": self._env.episode_length_buf.unsqueeze(-1).clone(),
        }
        if getattr(self.cfg, "command_name", None):
            payload["command"] = self._env.command_manager.get_command(self.cfg.command_name).clone()
        if getattr(self.cfg, "record_critic_obs", False) and "critic" in self._env.obs_buf:
            payload["critic_obs"] = self._env.obs_buf["critic"].clone()
        return "transition", payload

    def record_post_step(self):
        payload = {
            "next_obs": self._env.obs_buf["policy"].clone(),
            "reward": self._env.reward_buf.unsqueeze(-1).clone(),
            "done": self._env.reset_buf.unsqueeze(-1).to(torch.uint8),
            "terminated": self._env.reset_terminated.unsqueeze(-1).to(torch.uint8),
            "time_out": self._env.reset_time_outs.unsqueeze(-1).to(torch.uint8),
        }

        if getattr(self.cfg, "command_name", None):
            command = self._env.command_manager.get_command(self.cfg.command_name)
            robot = self._env.scene["robot"]
            lin_vel_error = torch.linalg.norm(command[:, :2] - robot.data.root_lin_vel_b[:, :2], dim=1, keepdim=True)
            yaw_vel_error = torch.abs(command[:, 2:3] - robot.data.root_ang_vel_b[:, 2:3])
            payload["lin_vel_tracking_error"] = lin_vel_error
            payload["yaw_vel_tracking_error"] = yaw_vel_error

        if getattr(self.cfg, "record_contact_forces", False) and hasattr(self._env.scene, "sensors"):
            sensor = self._env.scene.sensors.get("contact_forces")
            if sensor is not None:
                payload["contact_forces"] = sensor.data.net_forces_w_history.clone()

        if getattr(self.cfg, "record_height_scan", False) and hasattr(self._env.scene, "sensors"):
            sensor = self._env.scene.sensors.get("height_scanner")
            if sensor is not None and hasattr(sensor.data, "ray_hits_w"):
                payload["terrain_rays"] = sensor.data.ray_hits_w[..., 2].clone()

        return "transition", payload


@configclass
class QuadrupedTransitionRecorderCfg(RecorderTermCfg):
    """Config for transition-level offline data collection."""

    class_type: type[RecorderTerm] = QuadrupedTransitionRecorder
    command_name: str = "base_velocity"
    record_contact_forces: bool = False
    record_height_scan: bool = False
    record_critic_obs: bool = False


@configclass
class AmadeusOfflineRecorderManagerCfg(RecorderManagerBaseCfg):
    """Recorder configuration used by the quadruped baseline."""

    dataset_file_handler_class_type: type = ChunkedHDF5DatasetFileHandler
    dataset_export_dir_path: str = "outputs/quadruped_rough_baseline/datasets/train"
    dataset_filename: str = "train_dataset"
    export_in_record_pre_reset: bool = True
    export_in_close: bool = True

    transition = QuadrupedTransitionRecorderCfg()
