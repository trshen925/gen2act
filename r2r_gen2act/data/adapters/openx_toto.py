from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from r2r_gen2act.data.action.mappings import terminate_from_payload, toto_action
from r2r_gen2act.data.adapters.base import WindowedRobotDataset
from r2r_gen2act.data.split import split_episode_ids
from r2r_gen2act.data.types import EpisodeRecord


class OpenXTotoDataset(WindowedRobotDataset):
    def _load_episodes(self) -> list[EpisodeRecord]:
        robot_root = Path(self.data_cfg["robot_root"])
        generated_root = Path(self.data_cfg.get("generated_root", ""))
        source_name = str(self.data_cfg.get("source_video_name", "generated.mp4"))
        target_name = str(self.data_cfg.get("target_video_name", "image.mp4"))
        metadata_name = str(self.data_cfg.get("metadata_name", "data.json"))
        robot_dirs = sorted([p for p in robot_root.glob("episode_*") if p.is_dir()])
        max_episodes = self.data_cfg.get("max_episodes")
        if max_episodes not in (None, ""):
            robot_dirs = robot_dirs[: int(max_episodes)]
        ids = [d.name for d in robot_dirs]
        _, val_ids = split_episode_ids(ids, float(self.data_cfg.get("val_ratio", 0.2)), int(self.data_cfg.get("split_seed", 42)))
        episodes = []
        for robot_dir in robot_dirs:
            split = "val" if robot_dir.name in val_ids else "train"
            if self.split in ("train", "val") and split != self.split:
                continue
            generated_dir = generated_root / robot_dir.name
            src = generated_dir / source_name if generated_root else robot_dir / target_name
            tgt = robot_dir / target_name
            meta = robot_dir / metadata_name
            if not src.exists() or not tgt.exists() or not meta.exists():
                continue
            with meta.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            episodes.append(EpisodeRecord(robot_dir.name, int(payload["num_steps"]), src, tgt, meta, split))
        print(f"[OpenXTotoDataset] action_mapping=toto_world_rotation_gripper split={self.split} episodes={len(episodes)}")
        return episodes

    def _read_action_payload(self, episode: EpisodeRecord) -> dict:
        assert episode.metadata_path is not None
        with episode.metadata_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _action_at(self, payload: dict, step: int) -> np.ndarray:
        return toto_action(payload, step)

    def _terminate_at(self, payload: dict, step: int, num_steps: int) -> int:
        return terminate_from_payload(payload, step, num_steps, self.terminate_positive_window)
