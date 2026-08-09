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
        clip_mapping_value = str(cfg["data"].get("clip_mapping", "") or "")
        self._clip_mapping_path = Path(clip_mapping_value) if clip_mapping_value else None
        self._clip_mapping_by_id: dict[str, dict] = {}
        self._mapped_train_samples: list[tuple[str, int]] = []
        self._mapped_train_raw_ids: set[str] = set()
        if self._clip_mapping_path is not None:
            if not self._clip_mapping_path.is_file():
                raise FileNotFoundError(f"Raw DROID clip mapping not found: {self._clip_mapping_path}")
            clip_mapping = json.loads(self._clip_mapping_path.read_text(encoding="utf-8"))
            if int(clip_mapping.get("version", 0)) != 1:
                raise ValueError(
                    f"Unsupported raw DROID clip mapping version {clip_mapping.get('version')}")
            self._clip_mapping_by_id = {
                str(item["clip_id"]): item for item in clip_mapping.get("clips", [])
            }
            self._mapped_train_samples = [
                (str(clip_id), int(start))
                for clip_id, start in clip_mapping.get("train_samples", [])
            ]
            self._mapped_train_raw_ids = {
                str(item["raw_episode_id"])
                for item in clip_mapping.get("clips", [])
                if str(item.get("split")) == "train"
            }
        manifest_path = Path(str(cfg["data"].get("filter_manifest", "")))
        configured_root = Path(str(cfg["data"]["root"])).resolve()
        if not manifest_path.is_file() and not self._clip_mapping_by_id:
            raise FileNotFoundError(
                f"raw_droid requires data.filter_manifest; not found: {manifest_path}. "
                "Run scripts/build_raw_droid_pi05_manifest.py first."
            )
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        else:
            # A C42 mapping already contains the physical raw episode IDs and
            # lengths, so a C40 filtering manifest is optional on a new cluster.
            manifest = {"version": 1, "root": str(configured_root), "episodes": []}
            print(f"[RawDroidDataset] mapped mode without filter manifest: {manifest_path}")
        self._manifest_path = manifest_path
        self._manifest_version = int(manifest.get("version", 0))
        # build_raw_droid_pi05_manifest.py emits version 2, which only adds
        # episode-level provenance/language fields on top of version 1. Every
        # field this adapter reads (episode_id, num_steps, keep_ranges,
        # close_events, release_events) is unchanged, so both load.
        if self._manifest_version not in (1, 2):
            raise ValueError(f"Unsupported raw DROID manifest version {self._manifest_version}")
        manifest_root = Path(str(manifest.get("root", ""))).resolve()
        if configured_root != manifest_root and not self._clip_mapping_by_id:
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

        # Mapped clips share full raw MP4s. The cache must clamp against the raw
        # video length, not the logical clip length.
        if self._clip_mapping_by_id:
            for episode in self._episodes:
                raw_length = int((episode.extra or {})["video_num_steps"])
                for path in (
                    episode.source_video_path,
                    episode.target_video_path,
                    Path(str((episode.extra or {})["wrist_video_path"])),
                ):
                    if path is not None:
                        self._known_video_lengths[Path(path)] = raw_length

    def __getstate__(self):
        state = super().__getstate__()
        state["_payload_cache"] = {}
        return state

    def _camera_video(self, episode_dir: Path, episode_id: str) -> Path | None:
        mapped = self._clip_mapping_by_id.get(episode_id)
        if mapped is not None:
            camera = int(mapped["camera"])
            if camera not in (1, 2):
                raise ValueError(f"Mapped clip {episode_id} has invalid exterior camera {camera}")
            return episode_dir / f"steps_observation_exterior_image_{camera}_left.mp4"
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
        if self._clip_mapping_by_id:
            episodes = []
            excluded_val_overlap = 0
            validate_paths = bool(self.data_cfg.get("validate_manifest_paths", False))
            exclude_val_overlap = bool(
                self.data_cfg.get("exclude_mapped_val_raw_overlap", False)
            )
            for clip_id in sorted(self._clip_mapping_by_id):
                item = self._clip_mapping_by_id[clip_id]
                episode_split = str(item["split"])
                if self.split in ("train", "val") and episode_split != self.split:
                    continue
                raw_episode_id = str(item["raw_episode_id"])
                if (
                    self.split == "val"
                    and exclude_val_overlap
                    and raw_episode_id in self._mapped_train_raw_ids
                ):
                    excluded_val_overlap += 1
                    continue
                episode_dir = root / raw_episode_id
                parquet_path = episode_dir / "episode.parquet"
                metadata_path = episode_dir / "metadata.json"
                wrist_path = episode_dir / "steps_observation_wrist_image_left.mp4"
                front_path = self._camera_video(episode_dir, clip_id)
                if front_path is None:
                    continue
                if validate_paths:
                    required = (parquet_path, metadata_path, wrist_path, front_path)
                    if any(not path.is_file() or path.stat().st_size <= 0 for path in required):
                        continue
                start, end = [int(value) for value in item["frame_range"]]
                if start < 0 or end <= start:
                    raise ValueError(f"Mapped clip {clip_id} has invalid frame range [{start}, {end})")
                episodes.append(EpisodeRecord(
                    episode_id=clip_id,
                    num_steps=end - start,
                    source_video_path=front_path,
                    target_video_path=front_path,
                    metadata_path=metadata_path,
                    split=episode_split,
                    extra={
                        "raw_episode_id": raw_episode_id,
                        "parquet_path": str(parquet_path),
                        "wrist_video_path": str(wrist_path),
                        "source_frame_start": start,
                        "wrist_cache_frame_start": start,
                        "front_frame_start": start,
                        "video_num_steps": int(item["raw_num_steps"]),
                    },
                ))
            print(
                f"[RawDroidDataset] split={self.split} mapped_clips={len(episodes)} "
                f"excluded_val_raw_overlap={excluded_val_overlap} "
                f"mapping={self._clip_mapping_path}"
            )
            return episodes

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
        if self._clip_mapping_by_id:
            # C39's cached windows already encode its velocity/event selection.
            return samples
        filtered = []
        for episode_id, start in samples:
            target = int(start) + self.target_history_len - 1 + self.target_offset
            if self._target_is_kept(episode_id, target):
                filtered.append((episode_id, int(start)))
        print(f"[pi05_nonidle_filter] split={self.split} raw={len(samples)} kept={len(filtered)}")
        return filtered

    def _apply_native_action_sampling(self, samples: list[tuple[str, int]]) -> list[tuple[str, int]]:
        if self._clip_mapping_by_id:
            if self.split != "train" or not bool(
                self.data_cfg.get("use_mapped_train_samples", True)
            ):
                return samples
            available = {episode_id for episode_id, _ in samples}
            selected = [
                (episode_id, start)
                for episode_id, start in self._mapped_train_samples
                if episode_id in available
            ]
            if len(selected) != len(self._mapped_train_samples):
                missing = len(self._mapped_train_samples) - len(selected)
                raise RuntimeError(
                    f"C39 clip mapping has {missing} training samples whose clips are unavailable"
                )
            print(
                f"[c39_mapped_sampling] clips={len(available)} selected={len(selected)}"
            )
            return selected
        return super()._apply_native_action_sampling(self._filter_pi05_ranges(samples))

    def _jitter_start_index(self, episode: EpisodeRecord, start_index: int) -> int:
        if self._clip_mapping_by_id:
            return OpenXDroidDataset._jitter_start_index(self, episode, start_index)
        candidate = super()._jitter_start_index(episode, start_index)
        target = candidate + self.target_history_len - 1 + self.target_offset
        return candidate if self._target_is_kept(episode.episode_id, target) else int(start_index)

    def _build_native_action_sample_index(self, samples: list[tuple[str, int]]) -> list[tuple[str, int]]:
        """Build C39 event-balanced pools from manifest events, without rescanning parquet."""
        cfg = self.data_cfg.get("native_action_sampling", {}) or {}
        if str(cfg.get("sampling_mode", "")) == "keep_range_uniform":
            # Match openpi: every timestep in every manifest keep range is an
            # equally likely training target. Sample by cumulative range length
            # instead of materializing the full ~21M-timestep pool.
            range_episode: list[str] = []
            range_start: list[int] = []
            cumulative = [0]
            for episode in self._episodes:
                for start, end in self._manifest_ranges(episode.episode_id):
                    start = int(start)
                    end = min(int(end), int(episode.num_steps) - max(0, self.effective_future_horizon))
                    if end <= start:
                        continue
                    range_episode.append(episode.episode_id)
                    range_start.append(start)
                    cumulative.append(cumulative[-1] + end - start)
            if not cumulative[-1]:
                raise RuntimeError("raw DROID keep-range uniform pool is empty")
            total = int(cfg.get("num_samples", len(samples)))
            rng = np.random.default_rng(int(cfg.get("seed", self.cfg.get("train", {}).get("seed", 42))))
            offsets = rng.integers(0, cumulative[-1], size=total, dtype=np.int64)
            range_idx = np.searchsorted(np.asarray(cumulative[1:], dtype=np.int64), offsets, side="right")
            selected = [
                (range_episode[int(index)], range_start[int(index)] + int(offset - cumulative[int(index)]))
                for index, offset in zip(range_idx, offsets)
            ]
            rng.shuffle(selected)
            print(
                "[raw_droid_keep_range_uniform] "
                f"ranges={len(range_episode)} kept_targets={cumulative[-1]} selected={len(selected)}"
            )
            return selected
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
        if self._clip_mapping_by_id:
            start = int((episode.extra or {})["source_frame_start"])
            end = start + int(episode.num_steps)
            payload = {
                "num_steps": int(episode.num_steps),
                "observations": {
                    key: value[start:end] for key, value in payload["observations"].items()
                },
                "action_dict": {
                    key: value[start:end] for key, value in payload["action_dict"].items()
                },
                "is_last": payload["is_last"][start:end],
                "is_terminal": payload["is_terminal"][start:end],
            }
        cache_limit = int(self.data_cfg.get("parquet_payload_cache", 256))
        if cache_limit > 0:
            if len(self._payload_cache) >= cache_limit:
                self._payload_cache.pop(next(iter(self._payload_cache)))
            self._payload_cache[episode.episode_id] = payload
        return payload

    def _read_native_action_payload(self, episode: EpisodeRecord) -> dict:
        return self._read_action_payload(episode)
