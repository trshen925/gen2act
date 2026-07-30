"""Adapter for locally decompressed DROID 1.0.1 episodes.

Expected layout:
  <root>/episode_XXXXXX/episode.parquet
  <root>/episode_XXXXXX/metadata.json
  <root>/episode_XXXXXX/steps_observation_exterior_image_{1,2}_left.mp4
  <root>/episode_XXXXXX/steps_observation_wrist_image_left.mp4

The offline manifest is mandatory. It limits the dataset to successful, validated
episodes and stores the pi0.5-DROID non-idle ranges without decoding any videos.
"""
from __future__ import annotations

import bisect
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from r2r_gen2act.data.adapters.droid_ex_out import _json_col
from r2r_gen2act.data.adapters.openx_droid import OpenXDroidDataset
from r2r_gen2act.data.split import split_episode_ids
from r2r_gen2act.data.types import EpisodeRecord


_OBS_JOINT = "steps/observation/joint_position"
_OBS_GRIP = "steps/observation/gripper_position"
_ACT_JOINT_VEL = "steps/action_dict/joint_velocity"
_ACT_GRIP = "steps/action_dict/gripper_position"
_IS_LAST = "steps/is_last"
_IS_TERM = "steps/is_terminal"


class RawDroidDataset(OpenXDroidDataset):
    def __init__(self, cfg: dict, split: str = "train") -> None:
        manifest_path = Path(str(cfg["data"].get("filter_manifest", "")))
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"raw_droid requires data.filter_manifest; not found: {manifest_path}. "
                "Run scripts/build_raw_droid_pi05_manifest.py first."
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self._manifest_path = manifest_path
        self._manifest_version = int(manifest.get("version", 0))
        # build_raw_droid_pi05_manifest.py emits version 2, which only adds
        # episode-level provenance/language fields on top of version 1. Every
        # field this adapter reads (episode_id, num_steps, keep_ranges,
        # close_events, release_events) is unchanged, so both load.
        if self._manifest_version not in (1, 2):
            raise ValueError(f"Unsupported raw DROID manifest version {self._manifest_version}")
        configured_root = Path(str(cfg["data"]["root"])).resolve()
        manifest_root = Path(str(manifest.get("root", ""))).resolve()
        if configured_root != manifest_root:
            raise ValueError(
                f"Raw DROID manifest was built for {manifest_root}, but data.root is {configured_root}"
            )
        self._manifest_by_id = {str(item["episode_id"]): item for item in manifest.get("episodes", [])}
        self._range_starts_by_id = {
            episode_id: [int(keep_range[0]) for keep_range in item.get("keep_ranges", [])]
            for episode_id, item in self._manifest_by_id.items()
        }
        self._payload_cache: dict[str, dict] = {}
        super().__init__(cfg, split=split)

    def __getstate__(self):
        state = super().__getstate__()
        state["_payload_cache"] = {}
        return state

    def _camera_video(self, episode_dir: Path, episode_id: str) -> Path | None:
        selection = str(self.data_cfg.get("camera_selection", "deterministic_random"))
        if selection == "exterior_1":
            choices = [1]
        elif selection == "exterior_2":
            choices = [2]
        elif selection == "deterministic_random":
            seed = int(self.data_cfg.get("camera_seed", self.data_cfg.get("split_seed", 42)))
            digest = hashlib.sha256(f"{seed}:{episode_id}".encode("utf-8")).digest()
            first = 1 + digest[0] % 2
            choices = [first, 3 - first]
        else:
            raise ValueError(
                f"Unknown camera_selection={selection!r}; expected exterior_1, exterior_2, "
                "or deterministic_random"
            )
        if not bool(self.data_cfg.get("validate_manifest_paths", False)):
            return episode_dir / f"steps_observation_exterior_image_{choices[0]}_left.mp4"
        for camera in choices:
            path = episode_dir / f"steps_observation_exterior_image_{camera}_left.mp4"
            if path.is_file() and path.stat().st_size > 0:
                return path
        return None

    def _load_episodes(self) -> list[EpisodeRecord]:
        root = Path(self.data_cfg["root"])
        items = sorted(self._manifest_by_id.values(), key=lambda item: str(item["episode_id"]))
        max_episodes = self.data_cfg.get("max_episodes")
        if max_episodes not in (None, ""):
            items = items[: int(max_episodes)]
        raw_ids = [str(item["episode_id"]) for item in items]
        val_count = self.data_cfg.get("val_count")
        val_count = None if val_count in (None, "") else int(val_count)
        _, val_ids = split_episode_ids(
            raw_ids,
            float(self.data_cfg.get("val_ratio", 0.2)),
            int(self.data_cfg.get("split_seed", 42)),
            val_count,
        )

        episodes = []
        validate_paths = bool(self.data_cfg.get("validate_manifest_paths", False))
        for item in items:
            episode_id = str(item["episode_id"])
            episode_split = "val" if episode_id in val_ids else "train"
            if self.split in ("train", "val") and episode_split != self.split:
                continue
            episode_dir = root / episode_id
            parquet_path = episode_dir / "episode.parquet"
            metadata_path = episode_dir / "metadata.json"
            wrist_path = episode_dir / "steps_observation_wrist_image_left.mp4"
            front_path = self._camera_video(episode_dir, episode_id)
            if front_path is None:
                continue
            if validate_paths:
                if not parquet_path.is_file() or not metadata_path.is_file():
                    continue
                if self.wrist_current_enabled and (not wrist_path.is_file() or wrist_path.stat().st_size <= 0):
                    continue
            episodes.append(EpisodeRecord(
                episode_id=episode_id,
                num_steps=int(item["num_steps"]),
                source_video_path=front_path,
                target_video_path=front_path,
                metadata_path=metadata_path,
                split=episode_split,
                extra={
                    "parquet_path": str(parquet_path),
                    "wrist_video_path": str(wrist_path),
                    "source_frame_start": 0,
                },
            ))
        print(
            f"[RawDroidDataset] split={self.split} episodes={len(episodes)} "
            f"manifest={self._manifest_path}"
        )
        return episodes

    def _manifest_ranges(self, episode_id: str) -> list[list[int]]:
        return self._manifest_by_id[episode_id].get("keep_ranges", [])

    def _target_is_kept(self, episode_id: str, target: int) -> bool:
        ranges = self._manifest_ranges(episode_id)
        idx = bisect.bisect_right(self._range_starts_by_id[episode_id], int(target)) - 1
        return idx >= 0 and int(target) < int(ranges[idx][1])

    def _filter_pi05_ranges(self, samples: list[tuple[str, int]]) -> list[tuple[str, int]]:
        filtered = []
        for episode_id, start in samples:
            target = int(start) + self.target_history_len - 1 + self.target_offset
            if self._target_is_kept(episode_id, target):
                filtered.append((episode_id, int(start)))
        print(f"[pi05_nonidle_filter] split={self.split} raw={len(samples)} kept={len(filtered)}")
        return filtered

    def _apply_native_action_sampling(self, samples: list[tuple[str, int]]) -> list[tuple[str, int]]:
        return super()._apply_native_action_sampling(self._filter_pi05_ranges(samples))

    def _jitter_start_index(self, episode: EpisodeRecord, start_index: int) -> int:
        candidate = super()._jitter_start_index(episode, start_index)
        target = candidate + self.target_history_len - 1 + self.target_offset
        return candidate if self._target_is_kept(episode.episode_id, target) else int(start_index)

    def _build_native_action_sample_index(self, samples: list[tuple[str, int]]) -> list[tuple[str, int]]:
        """Build C39 event-balanced pools from manifest events, without rescanning parquet."""
        cfg = self.data_cfg.get("native_action_sampling", {}) or {}
        before = int(cfg.get("event_before", 8))
        after = int(cfg.get("event_after", 16))
        normal_pool = []
        close_pool = []
        release_pool = []
        for episode_id, start in samples:
            item = self._manifest_by_id[episode_id]
            target = int(start) + self.target_history_len - 1 + self.target_offset
            closes = item.get("close_events", [])
            releases = item.get("release_events", [])
            close_cover = any(target - before <= int(event) <= target + after for event in closes)
            release_cover = any(target - before <= int(event) <= target + after for event in releases)
            window = (episode_id, int(start))
            if close_cover:
                close_pool.append(window)
            if release_cover:
                release_pool.append(window)
            if not close_cover and not release_cover:
                normal_pool.append(window)

        ratios = np.asarray([
            float(cfg.get("normal_ratio", 0.60)),
            float(cfg.get("close_ratio", 0.22)),
            float(cfg.get("release_ratio", 0.18)),
        ], dtype=np.float64)
        if np.any(ratios < 0) or ratios.sum() <= 0:
            raise ValueError("native_action_sampling ratios must be non-negative with positive sum")
        ratios /= ratios.sum()
        pools = [normal_pool, close_pool, release_pool]
        names = ["normal", "close", "release"]
        missing = [name for name, pool, ratio in zip(names, pools, ratios) if ratio > 0 and not pool]
        if missing:
            raise RuntimeError(f"raw DROID sampling pools are empty: {missing}")
        total = int(cfg.get("num_samples", len(samples)))
        counts = np.floor(ratios * total).astype(int)
        counts[0] += total - int(counts.sum())
        rng = np.random.default_rng(int(cfg.get("seed", self.cfg.get("train", {}).get("seed", 42))))
        selected = []
        for pool, count in zip(pools, counts):
            choices = rng.integers(0, len(pool), size=int(count))
            selected.extend(pool[int(index)] for index in choices)
        rng.shuffle(selected)
        print(
            "[raw_droid_event_sampling] "
            f"input={len(samples)} pools(normal={len(normal_pool)},close={len(close_pool)},"
            f"release={len(release_pool)}) selected={len(selected)}"
        )
        return selected

    def _native_sampling_fingerprint(self) -> dict:
        fingerprint = super()._native_sampling_fingerprint()
        fingerprint["filter_manifest"] = str(self._manifest_path)
        fingerprint["filter_manifest_mtime_ns"] = self._manifest_path.stat().st_mtime_ns
        fingerprint["raw_droid_manifest_version"] = self._manifest_version
        return fingerprint

    def _read_action_payload(self, episode: EpisodeRecord) -> dict:
        cached = self._payload_cache.get(episode.episode_id)
        if cached is not None:
            return cached
        table = pq.read_table(
            episode.extra["parquet_path"],
            columns=[_OBS_JOINT, _OBS_GRIP, _ACT_JOINT_VEL, _ACT_GRIP, _IS_LAST, _IS_TERM],
        )
        joint = np.asarray(_json_col(table, _OBS_JOINT), dtype=np.float32)
        velocity = np.asarray(_json_col(table, _ACT_JOINT_VEL), dtype=np.float32)
        observation_gripper = np.asarray(table.column(_OBS_GRIP).to_pylist(), dtype=np.float32).reshape(-1, 1)
        action_gripper = np.asarray(table.column(_ACT_GRIP).to_pylist(), dtype=np.float32).reshape(-1, 1)
        payload = {
            "num_steps": len(joint),
            "observations": {
                "joint_position": joint,
                "gripper_position": observation_gripper,
            },
            "action_dict": {
                "joint_velocity": velocity,
                "gripper_position": action_gripper,
            },
            "is_last": [int(value) for value in table.column(_IS_LAST).to_pylist()],
            "is_terminal": [int(value) for value in table.column(_IS_TERM).to_pylist()],
        }
        cache_limit = int(self.data_cfg.get("parquet_payload_cache", 256))
        if cache_limit > 0:
            if len(self._payload_cache) >= cache_limit:
                self._payload_cache.pop(next(iter(self._payload_cache)))
            self._payload_cache[episode.episode_id] = payload
        return payload

    def _read_native_action_payload(self, episode: EpisodeRecord) -> dict:
        return self._read_action_payload(episode)
