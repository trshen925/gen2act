from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from r2r_gen2act.data.action.mappings import droid_action, terminate_from_payload
from r2r_gen2act.data.adapters.base import WindowedRobotDataset
from r2r_gen2act.data.split import split_episode_ids
from r2r_gen2act.data.types import EpisodeRecord


class OpenXDroidDataset(WindowedRobotDataset):
    def _load_episodes(self) -> list[EpisodeRecord]:
        root = Path(self.data_cfg["root"])
        source_name = str(self.data_cfg.get("source_video_name", "generated.mp4"))
        target_name = str(self.data_cfg.get("target_video_name", "groundtruth.mp4"))
        metadata_name = str(self.data_cfg.get("metadata_name", "data.json"))
        pattern = str(self.data_cfg.get("episode_glob", "episode_*"))
        dirs = sorted([p for p in root.glob(pattern) if p.is_dir() and (p / metadata_name).exists()])
        max_episodes = self.data_cfg.get("max_episodes")
        if max_episodes not in (None, ""):
            dirs = dirs[: int(max_episodes)]
        ids = [d.name for d in dirs]
        _, val_ids = split_episode_ids(ids, float(self.data_cfg.get("val_ratio", 0.2)), int(self.data_cfg.get("split_seed", 42)))
        episodes = []
        for d in dirs:
            split = "val" if d.name in val_ids else "train"
            if self.split in ("train", "val") and split != self.split:
                continue
            meta = d / metadata_name
            src = d / source_name
            tgt = d / target_name
            if not meta.exists() or not src.exists() or not tgt.exists():
                continue
            with meta.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            episodes.append(EpisodeRecord(d.name, int(payload["num_steps"]), src, tgt, meta, split))
        mapping = self.cfg.get("action", {}).get("mapping", {}).get("type", "droid_actions_first6_plus_gripper")
        print(f"[OpenXDroidDataset] action_mapping={mapping} split={self.split} episodes={len(episodes)}")
        return episodes

    def _read_action_payload(self, episode: EpisodeRecord) -> dict:
        assert episode.metadata_path is not None
        with episode.metadata_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _action_at(self, payload: dict, step: int) -> np.ndarray:
        mapping = self.cfg.get("action", {}).get("mapping", {}).get("type", "droid_actions_first6_plus_gripper")
        return droid_action(payload, step, mapping, self.future_horizon)

    def _terminate_at(self, payload: dict, step: int, num_steps: int) -> int:
        return terminate_from_payload(payload, step, num_steps, self.terminate_positive_window)

    def _proprioception_at(self, payload: dict, start_index: int, target_step: int) -> np.ndarray:
        prop_cfg = self.proprioception_cfg
        source = str(prop_cfg.get("source", "observations"))
        key = str(prop_cfg.get("key", "cartesian_position"))
        dims = int(prop_cfg.get("dims", 6))
        step_mode = str(prop_cfg.get("step", "target"))
        step = start_index + self.target_history_len - 1 if step_mode == "history_last" else target_step
        values = payload[source][key]
        step = min(max(0, int(step)), len(values) - 1)
        return np.asarray(values[step], dtype=np.float32).reshape(-1)[:dims]
