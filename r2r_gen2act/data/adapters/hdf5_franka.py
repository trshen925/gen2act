from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from r2r_gen2act.data.action.stats import load_action_stats
from r2r_gen2act.data.episode_index import build_windows
from r2r_gen2act.data.transforms import image_to_tensor
from r2r_gen2act.data.types import EpisodeRecord


class HDF5FrankaDataset(Dataset):
    def __init__(self, cfg: dict, split: str = "train") -> None:
        self.cfg = cfg
        self.data_cfg = cfg["data"]
        self.split = split
        self.hdf5_path = Path(self.data_cfg["hdf5_path"])
        self.source_camera = str(self.data_cfg.get("source_camera", "table_cam"))
        self.target_camera = str(self.data_cfg.get("target_camera", "wrist_cam"))
        self.action_key = str(self.data_cfg.get("action_key", "actions"))
        self.source_len = int(self.data_cfg["source_len"])
        self.target_history_len = int(self.data_cfg["target_history_len"])
        self.target_offset = int(self.data_cfg.get("target_offset", 0))
        self.image_size = int(self.data_cfg["image_size"])
        self.action_stride = int(self.data_cfg.get("action_stride", 1))
        self.terminate_positive_window = int(self.data_cfg.get("terminate_positive_window", 5))
        self._file: h5py.File | None = None
        self._episodes = self._load_episodes()
        max_windows = self.data_cfg.get("max_windows")
        max_windows = None if max_windows in (None, "") else int(max_windows)
        self._samples = build_windows(self._episodes, self.source_len, self.target_history_len, self.target_offset, self.action_stride, max_windows)
        self._episode_by_id = {e.episode_id: e for e in self._episodes}

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_file"] = None
        return state

    def _open(self) -> h5py.File:
        if self._file is None:
            self._file = h5py.File(self.hdf5_path, "r")
        return self._file

    def _load_episodes(self) -> list[EpisodeRecord]:
        stats_path = self.data_cfg.get("action_stats_path")
        train_ids = val_ids = None
        if stats_path:
            stats = load_action_stats(stats_path)
            train_ids = set(stats.get("train_demos", []))
            val_ids = set(stats.get("val_demos", []))
        episodes: list[EpisodeRecord] = []
        with h5py.File(self.hdf5_path, "r") as f:
            for name in sorted(f["data"].keys()):
                if train_ids is not None and val_ids is not None:
                    split = "train" if name in train_ids else "val" if name in val_ids else None
                    if self.split in ("train", "val") and split != self.split:
                        continue
                else:
                    split = self.split
                n = int(f[f"data/{name}/{self.action_key}"].shape[0])
                episodes.append(EpisodeRecord(name, n, None, None, None, split))
        max_episodes = self.data_cfg.get("max_episodes")
        if max_episodes not in (None, ""):
            episodes = episodes[: int(max_episodes)]
        return episodes

    @property
    def episodes(self):
        return list(self._episodes)

    @property
    def samples(self):
        return list(self._samples)

    def __len__(self) -> int:
        return len(self._samples)

    def _read_video(self, demo: str, camera: str, indices: list[int]) -> torch.Tensor:
        f = self._open()
        arr = f[f"data/{demo}/obs/{camera}"]
        frames = [image_to_tensor(np.asarray(arr[i]), self.image_size) for i in indices]
        return torch.stack(frames, dim=0)

    def _read_source_video(self, demo: str, start: int, num_steps: int) -> torch.Tensor:
        mode = str(self.data_cfg.get("source_sampling", "linspace"))
        if mode == "window":
            s = min(start, max(0, num_steps - self.source_len))
            indices = list(range(s, s + self.source_len))
        else:
            indices = [int(round(x)) for x in np.linspace(0, num_steps - 1, self.source_len)]
        return self._read_video(demo, self.source_camera, indices)

    def _read_target_history(self, demo: str, start: int) -> torch.Tensor:
        return self._read_video(demo, self.target_camera, list(range(start, start + self.target_history_len)))

    def _action_at(self, demo: str, step: int) -> np.ndarray:
        f = self._open()
        arr = np.asarray(f[f"data/{demo}/{self.action_key}"][step], dtype=np.float32)
        if arr.shape[0] < 7:
            raise ValueError(f"{demo}/{self.action_key}[{step}] has shape {arr.shape}, expected at least 7")
        return arr[:7]

    def sample_window(self, episode_id: str, start_index: int) -> dict:
        episode = self._episode_by_id[episode_id]
        target_step = start_index + self.target_history_len - 1 + self.target_offset
        action = self._action_at(episode_id, target_step)
        terminate = int(target_step >= max(0, episode.num_steps - self.terminate_positive_window))
        grip_thr = float(self.data_cfg.get("gripper_threshold", 0.0))
        return {
            "episode_id": episode_id,
            "start_index": int(start_index),
            "target_step": int(target_step),
            "source_video": self._read_source_video(episode_id, start_index, episode.num_steps),
            "target_history": self._read_target_history(episode_id, start_index),
            "action": torch.as_tensor(action, dtype=torch.float32),
            "gripper": torch.tensor(int(float(action[-1]) > grip_thr), dtype=torch.long),
            "terminate": torch.tensor(terminate, dtype=torch.long),
            "metadata": {"hdf5_path": str(self.hdf5_path), "source_camera": self.source_camera, "target_camera": self.target_camera},
        }

    def __getitem__(self, index: int) -> dict:
        episode_id, start_index = self._samples[index]
        return self.sample_window(episode_id, start_index)
