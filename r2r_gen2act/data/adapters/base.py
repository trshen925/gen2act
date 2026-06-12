from __future__ import annotations

from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
import torch
from torch.utils.data import Dataset

from r2r_gen2act.data.episode_index import build_windows
from r2r_gen2act.data.transforms import apply_image_augmentation, image_to_tensor
from r2r_gen2act.data.types import EpisodeRecord


class WindowedRobotDataset(Dataset):
    def __init__(self, cfg: dict, split: str = "train") -> None:
        self.cfg = cfg
        self.data_cfg = cfg["data"]
        self.split = split
        self.source_len = int(self.data_cfg["source_len"])
        self.target_history_len = int(self.data_cfg["target_history_len"])
        self.target_offset = int(self.data_cfg.get("target_offset", 0))
        self.future_horizon = int(self.data_cfg.get("future_horizon", 0))
        self.image_size = int(self.data_cfg["image_size"])
        self.action_stride = int(self.data_cfg.get("action_stride", 1))
        self.terminate_positive_window = int(self.data_cfg.get("terminate_positive_window", 5))
        self.proprioception_cfg = self.data_cfg.get("proprioception", {})
        self.proprioception_enabled = bool(self.proprioception_cfg.get("enabled", False))
        self.load_videos = bool(self.data_cfg.get("load_videos", True))
        self.window_jitter_cfg = self.data_cfg.get("window_jitter", {})
        self.augmentation_cfg = self.data_cfg.get("augmentation", {})
        self._episodes = self._load_episodes()
        max_windows = self.data_cfg.get("max_windows")
        max_windows = None if max_windows in (None, "") else int(max_windows)
        self._samples = build_windows(self._episodes, self.source_len, self.target_history_len, self.target_offset, self.action_stride, max_windows, self.future_horizon)
        self._episode_by_id = {e.episode_id: e for e in self._episodes}
        self._video_cache: dict[Path, Any] = {}

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_video_cache"] = {}
        return state

    def _load_episodes(self) -> list[EpisodeRecord]:
        raise NotImplementedError

    def _read_action_payload(self, episode: EpisodeRecord) -> Any:
        raise NotImplementedError

    def _action_at(self, payload: Any, step: int) -> np.ndarray:
        raise NotImplementedError

    def _terminate_at(self, payload: Any, step: int, num_steps: int) -> int:
        raise NotImplementedError

    def _proprioception_at(self, payload: Any, start_index: int, target_step: int) -> np.ndarray:
        raise NotImplementedError

    def _gripper_at(self, action: np.ndarray) -> int:
        return int(float(action[-1]) > float(self.data_cfg.get("gripper_threshold", 0.0)))

    @property
    def episodes(self) -> list[EpisodeRecord]:
        return list(self._episodes)

    @property
    def samples(self) -> list[tuple[str, int]]:
        return list(self._samples)

    def __len__(self) -> int:
        return len(self._samples)

    def _reader(self, path: Path):
        reader = self._video_cache.get(path)
        if reader is None:
            reader = imageio.get_reader(str(path))
            self._video_cache[path] = reader
        return reader

    def _video_length(self, reader) -> int:
        try:
            length = int(reader.count_frames())
            if length > 0:
                return length
        except Exception:
            pass
        try:
            length = int(reader.get_length())
            if length > 0 and length < 10**9:
                return length
        except Exception:
            pass
        return 0

    def _read_video_indices(self, path: Path, indices: list[int]) -> torch.Tensor:
        reader = self._reader(path)
        length = self._video_length(reader)
        if length > 0:
            indices = [min(max(0, int(idx)), length - 1) for idx in indices]
        frames = [image_to_tensor(reader.get_data(int(idx)), self.image_size) for idx in indices]
        return torch.stack(frames, dim=0)

    def _read_source_video(self, episode: EpisodeRecord, start_index: int) -> torch.Tensor:
        if episode.source_video_path is None:
            raise ValueError(f"Episode {episode.episode_id} has no source video")
        mode = str(self.data_cfg.get("source_sampling", "linspace"))
        reader = self._reader(episode.source_video_path)
        source_length = self._video_length(reader) or episode.num_steps
        if mode == "window":
            start = min(start_index, max(0, source_length - self.source_len))
            indices = list(range(start, start + self.source_len))
        else:
            indices = [int(round(x)) for x in np.linspace(0, source_length - 1, self.source_len)]
        return self._read_video_indices(episode.source_video_path, indices)

    def _read_target_history(self, episode: EpisodeRecord, start_index: int) -> torch.Tensor:
        if episode.target_video_path is None:
            raise ValueError(f"Episode {episode.episode_id} has no target video")
        return self._read_video_indices(episode.target_video_path, list(range(start_index, start_index + self.target_history_len)))

    def _jitter_start_index(self, episode: EpisodeRecord, start_index: int) -> int:
        if self.split != "train" or not bool(self.window_jitter_cfg.get("enabled", False)):
            return start_index
        max_offset = int(self.window_jitter_cfg.get("max_offset", 0))
        if max_offset <= 0:
            return start_index
        last_start = episode.num_steps - self.target_history_len - self.target_offset - max(0, self.future_horizon)
        low = max(0, int(start_index) - max_offset)
        high = min(max(0, last_start), int(start_index) + max_offset)
        if high <= low:
            return int(start_index)
        return int(torch.randint(low, high + 1, ()).item())

    def sample_window(self, episode_id: str, start_index: int) -> dict:
        episode = self._episode_by_id[episode_id]
        start_index = self._jitter_start_index(episode, start_index)
        target_step = start_index + self.target_history_len - 1 + self.target_offset
        payload = self._read_action_payload(episode)
        action = self._action_at(payload, target_step).astype(np.float32)
        terminate = self._terminate_at(payload, target_step, episode.num_steps)
        sample = {
            "episode_id": episode.episode_id,
            "start_index": int(start_index),
            "target_step": int(target_step),
            "action": torch.as_tensor(action, dtype=torch.float32),
            "gripper": torch.tensor(self._gripper_at(action), dtype=torch.long),
            "terminate": torch.tensor(int(terminate), dtype=torch.long),
            "metadata": {"source_video_path": str(episode.source_video_path), "target_video_path": str(episode.target_video_path)},
        }
        if self.load_videos:
            source_video = self._read_source_video(episode, start_index)
            target_history = self._read_target_history(episode, start_index)
            if self.split == "train":
                source_video = apply_image_augmentation(source_video, self.augmentation_cfg)
                target_history = apply_image_augmentation(target_history, self.augmentation_cfg)
            sample["source_video"] = source_video
            sample["target_history"] = target_history
        if self.proprioception_enabled:
            sample["proprioception"] = torch.as_tensor(self._proprioception_at(payload, start_index, target_step), dtype=torch.float32)
        return sample

    def __getitem__(self, index: int) -> dict:
        episode_id, start_index = self._samples[index]
        return self.sample_window(episode_id, start_index)
