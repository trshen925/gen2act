from __future__ import annotations

import json
import os
from collections import OrderedDict
from pathlib import Path
import time
from typing import Any

import imageio.v2 as imageio
import numpy as np
import torch
from torch.utils.data import Dataset

from r2r_gen2act.data.episode_index import build_windows
from r2r_gen2act.data.overlay import draw_trajectory, raw_to_display_px
from r2r_gen2act.data.transforms import (
    apply_image_augmentation, apply_sampled_image_augmentation,
    apply_structural_augmentation, image_to_letterbox_tensor, image_to_tensor,
    sample_image_augmentation, translate_image_reflect,
)
from r2r_gen2act.data.types import EpisodeRecord


class WindowedRobotDataset(Dataset):
    def __init__(self, cfg: dict, split: str = "train") -> None:
        self.cfg = cfg
        self.data_cfg = cfg["data"]
        self.split = split
        self.source_len = int(self.data_cfg["source_len"])
        source_micro_clip_cfg = self.data_cfg.get("source_micro_clip", {}) or {}
        self.source_micro_clip_enabled = bool(source_micro_clip_cfg.get("enabled", False))
        self.source_micro_clip_frames = int(source_micro_clip_cfg.get("frames", 1))
        self.source_micro_clip_stride = int(source_micro_clip_cfg.get("stride", 1))
        self.source_micro_clip_alignment = str(
            source_micro_clip_cfg.get("alignment", "causal"))
        self.target_history_len = int(self.data_cfg["target_history_len"])
        self.target_offset = int(self.data_cfg.get("target_offset", 0))
        self.future_horizon = int(self.data_cfg.get("future_horizon", 0))
        self.chunk_size = int(cfg.get("action", {}).get("chunk_size", 1))
        mapping_cfg = cfg.get("action", {}).get("mapping", {}) or {}
        self.chunk_start_offset = int(mapping_cfg.get("chunk_start_offset", self.future_horizon))
        self.chunk_stride = max(1, int(mapping_cfg.get("chunk_stride", max(1, self.future_horizon))))
        # Preserve legacy future-pose semantics by default (+stride ... +H*stride),
        # while native controls can start at action[t] (offset=0).
        self.effective_future_horizon = self.chunk_start_offset + self.chunk_stride * max(0, self.chunk_size - 1)
        self.image_size = int(self.data_cfg["image_size"])
        self.action_stride = int(self.data_cfg.get("action_stride", 1))
        self.terminate_positive_window = int(self.data_cfg.get("terminate_positive_window", 5))
        self.proprioception_cfg = self.data_cfg.get("proprioception", {})
        self.proprioception_enabled = bool(self.proprioception_cfg.get("enabled", False))
        # append normalized task progress (target_step/num_steps) as an extra proprioception dim
        # (a coarse localization cue for the demo: "how far into the task am I").
        self.proprioception_append_progress = bool(self.proprioception_cfg.get("append_progress", False))
        self.proprioception_append_gripper = bool(
            self.proprioception_cfg.get("append_current_gripper", False))
        self.proprioception_gripper_continuous = bool(
            self.proprioception_cfg.get("current_gripper_continuous", False))
        self.proprioception_normalization = self.proprioception_cfg.get("normalization", {}) or {}
        # Step 7: per-episode tracked EE-neighborhood points (preproc/cotracker_ee_points.py).
        self.point_tracking_cfg = self.data_cfg.get("point_tracking", {})
        self.point_tracking_enabled = bool(self.point_tracking_cfg.get("enabled", False))
        self.point_num_points = int(self.point_tracking_cfg.get("num_points", 10))
        self.point_num_time = int(self.point_tracking_cfg.get("num_time", 60))
        # C5: "track_points" (cotracker 10 pts, default) or "ee_projection" (the EE-projection path as a
        # single-point value sequence — the SAME signal the overlay draws, but fed as values not pixels).
        self.point_track_source = str(self.point_tracking_cfg.get("source", "track_points"))
        # per-window slice [lo, hi] relative to target_step (None = whole-episode global trajectory)
        pw = self.point_tracking_cfg.get("window")
        self.point_window = list(pw) if isinstance(pw, (list, tuple)) and len(pw) == 2 else None
        # C1: optional SECOND causal track [lo, hi] (e.g. [-24, 0] = recent past motion up to now).
        # Emitted as `point_track_causal` IN ADDITION to the global `point_track` (does not replace it):
        # global track = demo semantic intent (whole episode), causal track = current motion momentum.
        cw = self.point_tracking_cfg.get("causal_window")
        self.point_causal_window = list(cw) if isinstance(cw, (list, tuple)) and len(cw) == 2 else None
        self.point_causal_num_time = int(self.point_tracking_cfg.get("causal_num_time", self.point_num_time))
        # HAMSTER-style trajectory overlay: render the whole-episode demo EE 2D path onto the current
        # frame so the backbone can ground it (localize + read future direction, not just momentum).
        self.overlay_cfg = self.data_cfg.get("trajectory_overlay", {})
        self.overlay_enabled = bool(self.overlay_cfg.get("enabled", False))
        self.overlay_radius = float(self.overlay_cfg.get("radius", 1.6))
        self.overlay_noise_std = float(self.overlay_cfg.get("coord_noise_std", 0.0))  # px noise on path (train)
        self.overlay_mark_current = bool(self.overlay_cfg.get("mark_current", False))  # green dot @ target_step (viz)
        self._ee_path_cache: dict = {}
        # C7 video-to-trajectory: input = current + N future frames (uniform over a future span + jitter);
        # target = each future frame's ABSOLUTE camera-frame EE pose (read FROM the given frames, not
        # predicted from the past -> well-determined). The N future frames go in the source_video slot.
        self.future_traj_cfg = self.data_cfg.get("future_traj", {})
        self.future_traj_enabled = bool(self.future_traj_cfg.get("enabled", False))
        self.future_traj_num = int(self.future_traj_cfg.get("num_frames", 8))
        self.future_traj_span = int(self.future_traj_cfg.get("span", 0))   # 0 = sample to episode end
        self.future_traj_jitter = int(self.future_traj_cfg.get("jitter", 0))
        # C15: auxiliary abs-EE-pose loss on source video frames (regularizer for underdetermined delta BC)
        aux_traj_cfg = self.data_cfg.get("aux_traj", {})
        self.aux_traj_enabled = bool(aux_traj_cfg.get("enabled", False))
        # C18: auxiliary temporal-progress loss — predict current step's normalized progress in the demo
        # (target_step / (num_steps-1)) ∈ [0,1]. A weak demo-current alignment signal.
        aux_progress_cfg = self.data_cfg.get("aux_progress", {})
        self.aux_progress_enabled = bool(aux_progress_cfg.get("enabled", False))
        # C18: source-frame sampling jitter — instead of always linspace over the WHOLE clip, randomly
        # float the start/end by up to `float_frac` of the clip length (train only) for demo diversity.
        source_float_cfg = self.data_cfg.get("source_float", {})
        self.source_float_enabled = bool(source_float_cfg.get("enabled", False))
        self.source_float_frac = float(source_float_cfg.get("float_frac", 0.2))
        # C34: asymmetric demo cropping. Keep the historical `float_frac` as
        # the fallback for older experiments, while allowing a front-only crop
        # so the source always retains the late grasp/release portion.
        self.source_float_front_frac = float(
            source_float_cfg.get("front_max_frac", self.source_float_frac))
        self.source_float_back_frac = float(
            source_float_cfg.get("back_max_frac", self.source_float_frac))
        if self.source_float_front_frac < 0.0 or self.source_float_back_frac < 0.0:
            raise ValueError("source_float front/back fractions must be non-negative")
        source_time_crop_cfg = self.data_cfg.get("source_time_crop", {}) or {}
        self.source_time_crop_enabled = bool(source_time_crop_cfg.get("enabled", False))
        self.source_time_crop_min_seconds = float(source_time_crop_cfg.get("min_seconds", 10.0))
        self.source_time_crop_max_seconds = float(source_time_crop_cfg.get("max_seconds", 30.0))
        if self.source_time_crop_min_seconds <= 0.0:
            raise ValueError("source_time_crop.min_seconds must be positive")
        if self.source_time_crop_max_seconds < self.source_time_crop_min_seconds:
            raise ValueError("source_time_crop.max_seconds must be >= min_seconds")
        # C20: Δt time-conditioning — emit per-frame real seconds-since-previous-sampled-frame so the
        # model knows the demo's pacing (fixed 8 frames, but a 3s clip vs 30s clip → very different Δt).
        self.dt_time_enabled = bool(self.data_cfg.get("dt_time_embed", {}).get("enabled", False))
        self.fps = float(self.data_cfg.get("fps", 15))
        # C24: dynamic source-frame count for (near-)constant real Δt across clips. Sample every
        # `stride` frames → k = clamp(round(num_frames/stride), min, max); linspace(0,n-1,k). At 15fps,
        # stride=15 → Δt target 1.0s. k depends only on num_steps (→ bucketable).
        dyn_cfg = self.data_cfg.get("dynamic_source", {})
        self.dynamic_source_enabled = bool(dyn_cfg.get("enabled", False))
        self.dynamic_source_stride = float(dyn_cfg.get("stride", 15))
        self.dynamic_source_min = int(dyn_cfg.get("min", 4))
        self.dynamic_source_max = int(dyn_cfg.get("max", 16))
        self.load_videos = bool(self.data_cfg.get("load_videos", True))
        # Optional pre-decoded frames: read <video>.parent/<frames_subdir>/<idx:06d>.<ext> instead of
        # randomly seeking the mp4 (see scripts/extract_frames.py). Much faster; empty = use mp4.
        self.frames_subdir = str(self.data_cfg.get("frames_subdir", "") or "")
        self.frames_ext = str(self.data_cfg.get("frames_ext", "jpg"))
        self.front_letterbox = bool(self.data_cfg.get("front_letterbox", False))
        wrist_cfg = self.data_cfg.get("wrist_current", {}) or {}
        self.wrist_current_enabled = bool(wrist_cfg.get("enabled", False))
        self.wrist_frames_subdir = str(wrist_cfg.get("frames_subdir", "wrist_frames"))
        self.wrist_frames_ext = str(wrist_cfg.get("frames_ext", "jpg"))
        self.wrist_allow_video_fallback = bool(wrist_cfg.get("allow_video_fallback", False))
        self.wrist_letterbox = bool(wrist_cfg.get("letterbox", False))
        # C33: fixed recent observations in clip-frame units. Keep target_history_len=1
        # so the action target remains at the current frame rather than shifting it.
        history_offsets = self.data_cfg.get("current_history_offsets", [0])
        if not isinstance(history_offsets, (list, tuple)) or not history_offsets:
            raise ValueError("data.current_history_offsets must be a non-empty list of frame offsets")
        self.current_history_offsets = [int(offset) for offset in history_offsets]
        if self.current_history_offsets[-1] != 0 or any(
            later <= earlier for earlier, later in zip(self.current_history_offsets, self.current_history_offsets[1:])
        ):
            raise ValueError("data.current_history_offsets must be strictly increasing and end with 0")
        self._frame_count_cache: dict[Path, int] = {}
        self.window_jitter_cfg = self.data_cfg.get("window_jitter", {})
        # C11: depth 3D lifting — load per-frame depth alongside RGB frames.
        depth_cfg = self.data_cfg.get("depth", {})
        self.depth_enabled = bool(depth_cfg.get("enabled", False))
        self.depth_representation = str(depth_cfg.get("representation", "frames"))
        self.depth_num_patches = int(depth_cfg.get("num_patches", 256))
        self.depth_frames_subdir = str(depth_cfg.get("frames_subdir", "depth_frames"))
        self.depth_ext = str(depth_cfg.get("frames_ext", "png"))
        self.source_jitter_cfg = self.data_cfg.get("source_jitter", {})
        self.augmentation_cfg = self.data_cfg.get("augmentation", {})
        self.front_translation_cfg = self.data_cfg.get("front_translation", {})
        self.front_translation_enabled = bool(self.front_translation_cfg.get("enabled", False))
        # C10: EE-targeted structural augmentation to simulate generated-video gripper gap.
        self.struct_aug_cfg = self.data_cfg.get("structural_augmentation", {})
        self.struct_aug_enabled = bool(self.struct_aug_cfg.get("enabled", False))
        self._episodes = self._load_episodes()
        max_windows = self.data_cfg.get("max_windows")
        max_windows = None if max_windows in (None, "") else int(max_windows)
        self._episode_by_id = {e.episode_id: e for e in self._episodes}
        raw_samples = build_windows(
            self._episodes, self.source_len, self.target_history_len, self.target_offset,
            self.action_stride, max_windows, self.effective_future_horizon)
        self._samples = self._apply_native_action_sampling(raw_samples)
        self._known_video_lengths = {
            Path(path): int((episode.extra or {}).get("video_num_steps", episode.num_steps))
            for episode in self._episodes
            for path in (episode.source_video_path, episode.target_video_path)
            if path is not None
        }
        # Raw-DROID wrist videos share the episode timeline. Clipped adapters
        # use a non-zero source_frame_start and are deliberately excluded.
        for episode in self._episodes:
            wrist_path = str(episode.extra.get("wrist_video_path", "") or "")
            if wrist_path and int(episode.extra.get("source_frame_start", 0)) == 0:
                self._known_video_lengths[Path(wrist_path)] = int(episode.num_steps)
        self._video_cache: dict[Path, Any] = {}
        # Optional per-worker raw-frame LRU. Keep raw frames so stochastic
        # translation/augmentation still runs independently for each window.
        self.decoded_frame_cache_mb = float(self.data_cfg.get("decoded_frame_cache_mb", 0.0))
        self.decoded_frame_cache_block_frames = int(self.data_cfg.get("decoded_frame_cache_block_frames", 120))
        self._decoded_frame_cache: OrderedDict[tuple[Path, int], tuple[np.ndarray, int]] = OrderedDict()
        self._decoded_frame_cache_bytes = 0
        self._incomplete_frames_dirs: set[Path] = set()
        # C24: per-window source-frame count k (depends only on episode num_steps) — for bucket sampling.
        if self.dynamic_source_enabled:
            k_by_ep = {e.episode_id: self._dynamic_source_len(e.num_steps) for e in self._episodes}
            self._window_k = [k_by_ep[eid] for eid, _ in self._samples]
        else:
            self._window_k = None

    def window_k(self) -> list[int] | None:
        """Per-window source-frame count k (for KBucketBatchSampler). None if not dynamic."""
        return self._window_k

    def _native_sampling_fingerprint(self) -> dict:
        cfg = self.data_cfg.get("native_action_sampling", {}) or {}
        return {
            "episode_ids": [e.episode_id for e in self._episodes],
            "target_history_len": self.target_history_len,
            "target_offset": self.target_offset,
            "effective_future_horizon": self.effective_future_horizon,
            "action_stride": self.action_stride,
            "chunk_size": self.chunk_size,
            "sampling": cfg,
        }

    @staticmethod
    def _max_true_run(mask: np.ndarray) -> int:
        best = run = 0
        for value in np.asarray(mask, dtype=bool):
            run = run + 1 if bool(value) else 0
            best = max(best, run)
        return best

    def _build_native_action_sample_index(self, samples: list[tuple[str, int]]) -> list[tuple[str, int]]:
        cfg = self.data_cfg.get("native_action_sampling", {}) or {}
        by_episode: dict[str, list[int]] = {}
        for episode_id, start in samples:
            by_episode.setdefault(episode_id, []).append(int(start))

        close_pool: list[tuple[str, int]] = []
        release_pool: list[tuple[str, int]] = []
        normal_pool: list[tuple[str, int]] = []
        velocity_threshold = float(cfg.get("velocity_idle_threshold", 1e-3))
        max_idle_run = int(cfg.get("max_idle_run", 7))
        event_before = int(cfg.get("event_before", 8))
        event_after = int(cfg.get("event_after", 16))
        gripper_threshold = float(self.data_cfg.get("gripper_threshold", 0.5))

        for episode_id, starts in by_episode.items():
            payload = self._read_native_action_payload(self._episode_by_id[episode_id])
            action_dict = payload.get("action_dict", {})
            velocity = np.asarray(action_dict.get("joint_velocity"), dtype=np.float32)
            gripper = np.asarray(action_dict.get("gripper_position"), dtype=np.float32).reshape(-1)
            if velocity.ndim != 2 or velocity.shape[1] != 7 or len(gripper) != len(velocity):
                raise ValueError(f"native action sampling requires [T,7] joint_velocity for {episode_id}")
            is_open = gripper > gripper_threshold
            transitions = np.diff(is_open.astype(np.int8))
            close_events = np.flatnonzero(transitions == -1) + 1
            release_events = np.flatnonzero(transitions == 1) + 1
            idle = np.max(np.abs(velocity), axis=1) <= velocity_threshold

            for start in starts:
                target = int(start) + self.target_history_len - 1 + self.target_offset
                lo = target + self.chunk_start_offset
                hi = min(len(velocity), lo + self.chunk_stride * max(0, self.chunk_size - 1) + 1)
                chunk_idle = idle[lo:hi:self.chunk_stride]
                close_cover = bool(np.any((close_events >= target - event_before) & (close_events <= target + event_after)))
                release_cover = bool(np.any((release_events >= target - event_before) & (release_events <= target + event_after)))
                item = (episode_id, int(start))
                if close_cover:
                    close_pool.append(item)
                if release_cover:
                    release_pool.append(item)
                if not close_cover and not release_cover and self._max_true_run(chunk_idle) <= max_idle_run:
                    normal_pool.append(item)

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
            raise RuntimeError(f"native action sampling pools are empty: {missing}")

        total = int(cfg.get("num_samples", len(samples)))
        counts = np.floor(ratios * total).astype(int)
        counts[0] += total - int(counts.sum())
        rng = np.random.default_rng(int(cfg.get("seed", self.cfg.get("train", {}).get("seed", 42))))
        selected: list[tuple[str, int]] = []
        for pool, count in zip(pools, counts):
            if count <= 0:
                continue
            choices = rng.integers(0, len(pool), size=int(count))
            selected.extend(pool[int(i)] for i in choices)
        rng.shuffle(selected)
        print(
            "[native_action_sampling] "
            f"raw={len(samples)} pools(normal={len(normal_pool)},close={len(close_pool)},release={len(release_pool)}) "
            f"selected={len(selected)} ratios={ratios.round(3).tolist()}"
        )
        return selected

    def _apply_native_action_sampling(self, samples: list[tuple[str, int]]) -> list[tuple[str, int]]:
        cfg = self.data_cfg.get("native_action_sampling", {}) or {}
        if self.split != "train" or not bool(cfg.get("enabled", False)):
            return samples
        cache_value = str(cfg.get("cache_path", "") or "")
        if not cache_value:
            return self._build_native_action_sample_index(samples)
        cache_path = Path(cache_value)
        fingerprint = self._native_sampling_fingerprint()

        def load_cache() -> list[tuple[str, int]] | None:
            if not cache_path.exists():
                return None
            data = json.loads(cache_path.read_text(encoding="utf-8"))
            if data.get("fingerprint") != fingerprint:
                return None
            return [(str(e), int(s)) for e, s in data["samples"]]

        cached = load_cache()
        if cached is not None:
            print(f"[native_action_sampling] loaded {len(cached)} windows from {cache_path}")
            return cached

        # Under torchrun, rank 0 builds once and all other ranks wait for the
        # atomically-renamed cache file instead of scanning every parquet.
        rank = int(os.environ.get("RANK", "0"))
        if rank != 0:
            deadline = time.monotonic() + float(cfg.get("cache_wait_seconds", 3600))
            while time.monotonic() < deadline:
                cached = load_cache()
                if cached is not None:
                    return cached
                time.sleep(2.0)
            raise TimeoutError(f"Timed out waiting for native sampling cache {cache_path}")

        selected = self._build_native_action_sample_index(samples)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_suffix(cache_path.suffix + f".{os.getpid()}.tmp")
        temporary.write_text(json.dumps({"fingerprint": fingerprint, "samples": selected}), encoding="utf-8")
        temporary.replace(cache_path)
        return selected

    def _dynamic_source_len(self, num_steps: int) -> int:
        k = round(int(num_steps) / max(1e-6, self.dynamic_source_stride))
        return int(min(max(k, self.dynamic_source_min), self.dynamic_source_max))

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_video_cache"] = {}
        state["_incomplete_frames_dirs"] = set()
        state["_ee_path_cache"] = {}
        return state

    def _load_episodes(self) -> list[EpisodeRecord]:
        raise NotImplementedError

    def _read_action_payload(self, episode: EpisodeRecord) -> Any:
        raise NotImplementedError

    def _read_native_action_payload(self, episode: EpisodeRecord) -> Any:
        """Read fields needed by native-action indexing; adapters may provide a cheaper path."""
        return self._read_action_payload(episode)

    def _action_at(self, payload: Any, step: int) -> np.ndarray:
        raise NotImplementedError

    def _terminate_at(self, payload: Any, step: int, num_steps: int) -> int:
        raise NotImplementedError

    def _proprioception_at(self, payload: Any, start_index: int, target_step: int) -> np.ndarray:
        raise NotImplementedError

    def _ee_path_pixels(self, episode: EpisodeRecord):
        """Whole-episode EE 2D path in original frame pixels for the overlay -> (path[N,2], W, H) or None.
        Default None: datasets without a projection just skip the overlay."""
        return None

    def _camera_abs_pose_at(self, payload: Any, step: int) -> np.ndarray:
        """C7: absolute camera-frame EE pose at step = [cam_pos(3), 6D-rot(6), gripper(1)]. Adapter-specific."""
        raise NotImplementedError

    def _future_traj_ee_image_fracs(self, payload: Any, cam_positions: np.ndarray) -> np.ndarray | None:
        """C10 structural aug: project cam_positions [F, 3] to normalised 224×224 coords [F, 2] ∈ [0,1].
        Default None (no intrinsics); openx_droid overrides this."""
        return None

    def _read_depth_at(self, episode: Any, indices: list[int]) -> torch.Tensor | None:
        """C11 depth 3D lifting: load depth frames [T, H_d, W_d] uint16 (mm). Default None."""
        return None

    def _get_camera_K_224(self, episode: Any, payload: Any) -> np.ndarray | None:
        """C11: camera intrinsics (fx, fy, cx, cy) scaled to the 224×224 image. Default None."""
        return None

    def _read_front_geometry_at(self, episode: Any, indices: list[int]) -> torch.Tensor | None:
        """Read precomputed [X,Y,Z,valid_ratio] geometry for current front RGB patches."""
        return None

    def _future_traj_indices(self, num_steps: int, target_step: int) -> list[int]:
        """N future-frame indices uniformly over [target_step+1, span end] (+train jitter)."""
        lo = min(int(target_step) + 1, max(1, num_steps - 1))
        hi = (num_steps - 1) if self.future_traj_span <= 0 else min(int(target_step) + self.future_traj_span, num_steps - 1)
        hi = max(hi, lo)
        idx = np.linspace(lo, hi, self.future_traj_num).round().astype(int)
        if self.split == "train" and self.future_traj_jitter > 0:
            j = torch.randint(-self.future_traj_jitter, self.future_traj_jitter + 1, (len(idx),)).numpy()
            idx = idx + j
        return np.clip(idx, lo, num_steps - 1).astype(int).tolist()

    def _future_traj_action(self, payload: Any, future_idx: list[int]) -> np.ndarray:
        """[F, 10] = absolute camera-frame pose + gripper for each future frame."""
        return np.stack([self._camera_abs_pose_at(payload, int(i)) for i in future_idx]).astype(np.float32)

    def _overlay_current_frame(self, frame_chw: torch.Tensor, episode_id: str, target_step: int) -> torch.Tensor:
        cached = self._ee_path_cache.get(episode_id, "__miss__")
        if cached == "__miss__":
            cached = self._ee_path_pixels(self._episode_by_id[episode_id])
            self._ee_path_cache[episode_id] = cached
        if cached is None:
            return frame_chw
        path_px, w, h = cached
        disp = raw_to_display_px(path_px, w, h, self.image_size)
        if self.split == "train" and self.overlay_noise_std > 0:
            disp = disp + torch.randn(disp.shape[0], disp.shape[1]).numpy() * self.overlay_noise_std
        hwc = frame_chw.permute(1, 2, 0).contiguous().numpy()
        mark = int(min(max(0, target_step), len(disp) - 1)) if self.overlay_mark_current else None
        draw_trajectory(hwc, disp, radius=self.overlay_radius, mark_idx=mark)
        return torch.from_numpy(hwc).permute(2, 0, 1).contiguous()

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
            # Force single-threaded ffmpeg: the multithreaded rawvideo encoder calls
            # pthread_create per reader, and with many DataLoader workers x ranks that hits the
            # container's process/thread limit (EAGAIN -> "Could not load meta information").
            threads = str(max(1, int(self.data_cfg.get("ffmpeg_threads", 1))))
            reader = imageio.get_reader(str(path), format="ffmpeg", input_params=["-threads", threads], output_params=["-threads", threads])
            # Bound the cache: each cached reader is a live ffmpeg subprocess. Without a cap a
            # worker leaks one process per distinct video it visits and exhausts the pid limit.
            max_cache = int(self.data_cfg.get("video_reader_cache", 8))
            if max_cache > 0 and len(self._video_cache) >= max_cache:
                old_path = next(iter(self._video_cache))
                old_reader = self._video_cache.pop(old_path)
                try:
                    old_reader.close()
                except Exception:
                    pass
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

    def _frames_dir(self, path: Path) -> Path | None:
        if not self.frames_subdir:
            return None
        d = path.parent / self.frames_subdir
        if not d.is_dir():
            return None
        # Only use frames dir if it has at least one frame file (not empty/incomplete)
        if not any(d.glob(f"*.{self.frames_ext}")):
            return None
        return d

    def _clip_length(self, path: Path) -> int:
        """Frame count for a clip: from the pre-decoded frames dir if present, else the mp4 reader."""
        known = getattr(self, "_known_video_lengths", {}).get(Path(path))
        if known is not None and known > 0:
            return int(known)
        fd = self._frames_dir(path)
        if fd is not None:
            n = self._frame_count_cache.get(fd)
            if n is None:
                n = len(list(fd.glob(f"*.{self.frames_ext}")))
                self._frame_count_cache[fd] = n
            return n
        return self._video_length(self._reader(path))

    def _read_video_indices(self, path: Path, indices: list[int]) -> torch.Tensor:
        transform = image_to_letterbox_tensor if self.front_letterbox else image_to_tensor
        cached_paths = self._cached_frame_paths(path, indices)
        if cached_paths is not None:
            frames = [transform(imageio.imread(str(frame_path)), self.image_size) for frame_path in cached_paths]
            return torch.stack(frames, dim=0)
        decoded = self._cached_raw_frames(path, indices)
        if decoded is not None:
            frames = [transform(frame, self.image_size) for frame in decoded]
            return torch.stack(frames, dim=0)
        reader = self._reader(path)
        # Raw DROID manifests already provide num_steps. Recounting an MP4 here
        # launches a second ffmpeg scan for every sample, which is often slower
        # than decoding the requested frames themselves.
        length = self._clip_length(path)
        if length > 0:
            indices = [min(max(0, int(idx)), length - 1) for idx in indices]
        frames = [transform(reader.get_data(int(idx)), self.image_size) for idx in indices]
        return torch.stack(frames, dim=0)

    def _cached_raw_frames(self, path: Path, indices: list[int]) -> list[np.ndarray] | None:
        """Decode/cache bounded contiguous blocks to avoid repeated random MP4 seeks."""
        budget = int(max(0.0, getattr(self, "decoded_frame_cache_mb", 0.0)) * 1024 * 1024)
        block_size = int(getattr(self, "decoded_frame_cache_block_frames", 120))
        if budget <= 0 or block_size <= 0 or not indices:
            return None
        length = int(self._known_video_lengths.get(Path(path), 0))
        if length <= 0:
            return None
        normalized = [min(max(0, int(index)), length - 1) for index in indices]
        blocks: dict[int, np.ndarray] = {}
        for index in sorted(set(normalized)):
            block_id = index // block_size
            key = (path, block_id)
            cached = self._decoded_frame_cache.get(key)
            if cached is None:
                start = block_id * block_size
                end = min(length, start + block_size)
                reader = self._reader(path)
                try:
                    frames = np.stack([reader.get_data(i) for i in range(start, end)], axis=0)
                except Exception:
                    return None
                size = int(frames.nbytes)
                if size > budget:
                    return None
                while self._decoded_frame_cache and self._decoded_frame_cache_bytes + size > budget:
                    _, (_, old_size) = self._decoded_frame_cache.popitem(last=False)
                    self._decoded_frame_cache_bytes -= old_size
                self._decoded_frame_cache[key] = (frames, size)
                self._decoded_frame_cache_bytes += size
            else:
                frames = cached[0]
                self._decoded_frame_cache.move_to_end(key)
            blocks[block_id] = frames
        return [blocks[index // block_size][index % block_size] for index in normalized]

    def _cached_frame_paths(self, path: Path, indices: list[int]) -> list[Path] | None:
        """Return cached frame paths only when this request is completely available.

        Some legacy extraction jobs left partially populated ``frames/`` folders.
        A single missing image marks that directory as incomplete for this worker,
        so all later requests safely use the source video instead of crashing.
        """
        frames_dir = self._frames_dir(path)
        if frames_dir is None or frames_dir in self._incomplete_frames_dirs:
            return None
        length = self._clip_length(path)
        normalized = [
            min(max(0, int(index)), length - 1) if length > 0 else int(index)
            for index in indices
        ]
        paths = [frames_dir / f"{index:06d}.{self.frames_ext}" for index in normalized]
        if all(frame_path.is_file() for frame_path in paths):
            return paths
        self._incomplete_frames_dirs.add(frames_dir)
        return None

    def _compute_source_indices(self, episode: EpisodeRecord, start_index: int) -> list[int]:
        """Compute the source video frame indices (with train jitter) for a given window."""
        if episode.source_video_path is None:
            raise ValueError(f"Episode {episode.episode_id} has no source video")
        mode = str(self.data_cfg.get("source_sampling", "linspace"))
        if "front_frame_start" in (episode.extra or {}):
            source_length = int(episode.num_steps)
        else:
            source_length = self._clip_length(episode.source_video_path) or episode.num_steps
        if mode == "window":
            start = min(start_index, max(0, source_length - self.source_len))
            indices = list(range(start, start + self.source_len))
        elif mode in ("future_window", "future_linspace"):
            target_step = start_index + self.target_history_len - 1 + self.target_offset
            lo = max(0, min(int(target_step), source_length - 1))
            hi = max(lo, min(int(target_step) + max(1, self.effective_future_horizon), source_length - 1))
            indices = [int(round(x)) for x in np.linspace(lo, hi, self.source_len)]
        elif mode == "future_offsets":
            target_step = start_index + self.target_history_len - 1 + self.target_offset
            offsets = self.data_cfg.get("source_future_offsets")
            if not offsets:
                offsets = [(i + 1) * 3 for i in range(self.source_len)]
            offsets = [int(x) for x in offsets]
            if len(offsets) != self.source_len:
                raise ValueError(f"data.source_future_offsets length ({len(offsets)}) must match source_len ({self.source_len})")
            indices = [int(target_step) + off for off in offsets]
        else:
            # C24: dynamic frame count for constant Δt (k = clamp(round(n/stride), min, max)).
            k = self._dynamic_source_len(episode.num_steps) if self.dynamic_source_enabled else self.source_len
            lo, hi = 0, source_length - 1
            target_step = start_index + self.target_history_len - 1 + self.target_offset
            if self.source_time_crop_enabled:
                lo, hi = self._source_time_crop_bounds(source_length, target_step)
            # C18: float the linspace window start/end by up to float_frac of the clip (train only),
            # so the sampled frames cover a different span each epoch → demo diversity.
            elif self.source_float_enabled and self.split == "train" and source_length > 2:
                front_span = self.source_float_front_frac * (source_length - 1)
                back_span = self.source_float_back_frac * (source_length - 1)
                # Keep the randomly cropped demonstration anchored no later
                # than the current target window. Otherwise, an early target
                # state may be paired with a source video that begins later.
                front_span = min(front_span, max(0, int(target_step)))
                lo = int(round(float(torch.rand(()).item()) * front_span))
                hi = int(round((source_length - 1) - float(torch.rand(()).item()) * back_span))
                if hi - lo < k:
                    lo, hi = 0, source_length - 1
            indices = [int(round(x)) for x in np.linspace(lo, hi, k)]
        return self._jitter_source_indices(indices, source_length)

    def _source_time_crop_bounds(self, source_length: int, target_step: int) -> tuple[int, int]:
        """Choose a bounded-duration source interval that always contains the current frame."""
        last = max(0, int(source_length) - 1)
        target = min(max(0, int(target_step)), last)
        max_span = max(1, int(round(self.source_time_crop_max_seconds * self.fps)))
        if last <= max_span:
            return 0, last

        min_span = max(1, int(round(self.source_time_crop_min_seconds * self.fps)))
        min_span = min(min_span, max_span)
        if self.split == "train" and max_span > min_span:
            span = int(torch.randint(min_span, max_span + 1, ()).item())
        else:
            span = max_span

        first_lo = max(0, target - span)
        last_lo = min(target, last - span)
        if self.split == "train" and last_lo > first_lo:
            lo = int(torch.randint(first_lo, last_lo + 1, ()).item())
        else:
            lo = min(max(0, target - span // 2), last - span)
        hi = lo + span
        if not lo <= target <= hi:
            raise RuntimeError(f"source crop [{lo}, {hi}] does not contain target frame {target}")
        return lo, hi

    def _read_source_video(self, episode: EpisodeRecord, start_index: int) -> torch.Tensor:
        indices = self._compute_source_indices(episode, start_index)
        return self._read_source_indices(episode, indices)

    def _source_frame_groups(
        self, episode: EpisodeRecord, anchor_indices: list[int]
    ) -> list[list[int]]:
        """Expand macro source anchors into causal local micro-clips.

        The normal path returns one frame per anchor.  When ``source_micro_clip``
        is enabled, anchor ``a`` represents the consecutive causal clip ending
        at ``a``.  Boundary frames are repeated so every anchor has a fixed
        number of frames and the batch remains collatable.
        """
        if not bool(getattr(self, "source_micro_clip_enabled", False)):
            return [[int(index)] for index in anchor_indices]
        alignment = str(getattr(self, "source_micro_clip_alignment", "causal"))
        if alignment != "causal":
            raise ValueError(
                f"Unsupported source_micro_clip.alignment={alignment!r}")
        last = max(0, int(episode.num_steps) - 1)
        frames = int(getattr(self, "source_micro_clip_frames", 1))
        stride = int(getattr(self, "source_micro_clip_stride", 1))
        offsets = [-(frames - 1 - position) * stride for position in range(frames)]
        return [
            [min(max(0, int(anchor) + offset), last) for offset in offsets]
            for anchor in anchor_indices
        ]

    def _source_frame_plan(
        self, episode: EpisodeRecord, anchor_indices: list[int]
    ) -> tuple[list[int], list[int], int]:
        """Return one sorted read plan for all source micro-clip frames.

        ``restore`` maps the sorted read result back to anchor-major order.  For
        the Wan ``8 x T=5`` path this sorts all 40 logical frame indices before
        the video reader is called, so extraction happens in one forward-ordered
        request instead of eight independent five-frame requests.
        """
        groups = self._source_frame_groups(episode, anchor_indices)
        clip_frames = len(groups[0]) if groups else 1
        flat = [index for group in groups for index in group]
        order = sorted(range(len(flat)), key=lambda position: (flat[position], position))
        sorted_indices = [flat[position] for position in order]
        restore = [0] * len(order)
        for sorted_position, original_position in enumerate(order):
            restore[original_position] = sorted_position
        return sorted_indices, restore, clip_frames

    @staticmethod
    def _restore_source_frame_plan(
        sorted_frames: torch.Tensor,
        restore: list[int],
        num_anchors: int,
        clip_frames: int,
    ) -> torch.Tensor:
        if restore:
            restored = sorted_frames[torch.as_tensor(
                restore, dtype=torch.long, device=sorted_frames.device)]
        else:
            restored = sorted_frames
        restored = restored.reshape(num_anchors, clip_frames, *sorted_frames.shape[1:])
        # Preserve the historical [K,C,H,W] dataset interface outside micro-clip mode.
        return restored[:, 0] if clip_frames == 1 else restored

    def _read_source_indices(
        self, episode: EpisodeRecord, anchor_indices: list[int]
    ) -> torch.Tensor:
        if episode.source_video_path is None:
            raise ValueError(f"Episode {episode.episode_id} has no source video")
        sorted_indices, restore, clip_frames = self._source_frame_plan(
            episode, anchor_indices)
        sorted_frames = self._read_video_indices(
            episode.source_video_path,
            self._front_video_indices(episode, sorted_indices),
        )
        return self._restore_source_frame_plan(
            sorted_frames, restore, len(anchor_indices), clip_frames)

    @staticmethod
    def _front_video_indices(episode: EpisodeRecord, indices: list[int]) -> list[int]:
        """Map logical clip-local indices onto a shared raw front video."""
        offset = int((episode.extra or {}).get("front_frame_start", 0))
        return [offset + int(index) for index in indices]

    def _jitter_source_indices(self, indices: list[int], source_length: int) -> list[int]:
        """Train-only per-frame wobble around the sampled source frames.

        Each sampled index is independently shifted by a random offset in
        [-max_offset, +max_offset] and clamped to the valid range, so the uniform
        8-frame grid fluctuates a little between epochs without changing geometry.
        """
        if self.split != "train" or not bool(self.source_jitter_cfg.get("enabled", False)):
            return indices
        max_offset = int(self.source_jitter_cfg.get("max_offset", 0))
        if max_offset <= 0 or source_length <= 1:
            return indices
        out = []
        for idx in indices:
            offset = int(torch.randint(-max_offset, max_offset + 1, ()).item())
            out.append(min(max(0, int(idx) + offset), source_length - 1))
        return out

    def _read_point_track(self, episode: EpisodeRecord, target_step: int = 0,
                          window: Any = "__default__", num_time: int | None = None) -> torch.Tensor:
        """Load tracked points for the episode -> [num_points, num_time, 2] normalized to [-1,1].

        track_points.npy is [T, N, 2] in pixels; we uniformly resample T -> num_time and normalize
        by the frame size from track_points.json. Missing preprocessing -> zeros (still trainable).

        `window` = [lo, hi] slices the trajectory to [target_step+lo, target_step+hi] BEFORE resampling
        (point motion local to this window, not a per-episode-global constant). The default
        ("__default__") uses self.point_window; pass an explicit [lo, hi] (e.g. causal [-24, 0]) or None
        (= whole episode) to override. `num_time` overrides the resample length."""
        win = self.point_window if (isinstance(window, str) and window == "__default__") else window
        nt = self.point_num_time if num_time is None else int(num_time)
        if self.point_track_source == "ee_projection":
            return self._read_ee_projection_track(episode, target_step, win, nt)
        ep_dir = Path(episode.source_video_path).parent
        npy = ep_dir / "track_points.npy"
        if not npy.exists():
            return torch.zeros(self.point_num_points, nt, 2, dtype=torch.float32)
        tracks = np.load(npy).astype(np.float32)  # [T, N, 2]
        meta_p = ep_dir / "track_points.json"
        info = json.loads(meta_p.read_text(encoding="utf-8")) if meta_p.exists() else {}
        w = float(info.get("W", 320)); h = float(info.get("H", 180))
        t = tracks.shape[0]
        if win is not None:
            lo = min(max(0, int(target_step) + int(win[0])), max(0, t - 1))
            hi = min(max(lo + 1, int(target_step) + int(win[1])), t)
            idx = np.linspace(lo, max(lo, hi - 1), nt).round().astype(int)
        else:
            idx = np.linspace(0, max(0, t - 1), nt).round().astype(int)
        sampled = tracks[idx].transpose(1, 0, 2)  # [N, S, 2]
        sampled[..., 0] = 2.0 * sampled[..., 0] / max(1.0, w - 1.0) - 1.0
        sampled[..., 1] = 2.0 * sampled[..., 1] / max(1.0, h - 1.0) - 1.0
        n = self.point_num_points
        if sampled.shape[0] < n:  # pad by repeating points if fewer than expected
            reps = [sampled[i % sampled.shape[0]] for i in range(n)]
            sampled = np.stack(reps)
        return torch.as_tensor(sampled[:n], dtype=torch.float32)

    def _read_ee_projection_track(self, episode: EpisodeRecord, target_step: int, win, nt: int) -> torch.Tensor:
        """C5: the EE-projection path (same signal the overlay draws) as a single-point value sequence
        [1, nt, 2] normalized to [-1,1]. Window slicing matches _read_point_track."""
        cached = self._ee_path_cache.get(episode.episode_id, "__miss__")
        if cached == "__miss__":
            cached = self._ee_path_pixels(episode)
            self._ee_path_cache[episode.episode_id] = cached
        if cached is None:
            return torch.zeros(1, nt, 2, dtype=torch.float32)
        path_px, w, h = cached
        arr = np.asarray(path_px, dtype=np.float32).copy()
        arr[:, 0] = 2.0 * arr[:, 0] / max(1.0, w - 1.0) - 1.0
        arr[:, 1] = 2.0 * arr[:, 1] / max(1.0, h - 1.0) - 1.0
        t = arr.shape[0]
        if win is not None:
            lo = min(max(0, int(target_step) + int(win[0])), max(0, t - 1))
            hi = min(max(lo + 1, int(target_step) + int(win[1])), t)
            idx = np.linspace(lo, max(lo, hi - 1), nt).round().astype(int)
        else:
            idx = np.linspace(0, max(0, t - 1), nt).round().astype(int)
        return torch.as_tensor(arr[idx], dtype=torch.float32).reshape(1, nt, 2)

    def _read_target_history(self, episode: EpisodeRecord, start_index: int) -> torch.Tensor:
        if episode.target_video_path is None:
            raise ValueError(f"Episode {episode.episode_id} has no target video")
        if self.current_history_offsets != [0]:
            target_step = start_index + self.target_history_len - 1 + self.target_offset
            indices = [int(target_step) + offset for offset in self.current_history_offsets]
            return self._read_video_indices(
                episode.target_video_path, self._front_video_indices(episode, indices))
        indices = list(range(start_index, start_index + self.target_history_len))
        return self._read_video_indices(
            episode.target_video_path, self._front_video_indices(episode, indices))

    def _read_wrist_current(self, episode: EpisodeRecord, target_step: int) -> torch.Tensor:
        """Read the wrist frame synchronized with the external current observation."""
        target_step = min(max(0, int(target_step)), max(0, int(episode.num_steps) - 1))
        cache_dir = Path(episode.metadata_path).parent / self.wrist_frames_subdir
        cache_index = int((episode.extra or {}).get("wrist_cache_frame_start", 0)) + target_step
        cached = cache_dir / f"{cache_index:06d}.{self.wrist_frames_ext}"
        if cached.exists():
            image = imageio.imread(str(cached))
            return (image_to_letterbox_tensor(image, self.image_size)
                    if self.wrist_letterbox else image_to_tensor(image, self.image_size))
        if not self.wrist_allow_video_fallback:
            raise FileNotFoundError(
                f"Missing wrist cache frame {cached}; run scripts/preprocess_wrist_frames.py first")
        wrist_video = Path(str(episode.extra.get("wrist_video_path", "")))
        if not wrist_video.exists():
            raise FileNotFoundError(f"Missing raw wrist video for {episode.episode_id}: {wrist_video}")
        raw_idx = int(episode.extra.get("source_frame_start", 0)) + target_step
        decoded = self._cached_raw_frames(wrist_video, [raw_idx])
        if decoded is not None:
            image = decoded[0]
            return (image_to_letterbox_tensor(image, self.image_size)
                    if self.wrist_letterbox else image_to_tensor(image, self.image_size))
        reader = self._reader(wrist_video)
        image = reader.get_data(raw_idx)
        return (image_to_letterbox_tensor(image, self.image_size)
                if self.wrist_letterbox else image_to_tensor(image, self.image_size))

    def _read_front_with_translation(self, episode: EpisodeRecord, indices: list[int], dx_frac: float, dy_frac: float) -> torch.Tensor:
        """Read native front frames, apply one shared translation, then standard C37 preprocessing."""
        path = episode.target_video_path
        assert path is not None
        indices = self._front_video_indices(episode, indices)
        transform = image_to_letterbox_tensor if self.front_letterbox else image_to_tensor
        cached_paths = self._cached_frame_paths(path, indices)
        if cached_paths is not None:
            frames = []
            for frame_path in cached_paths:
                image = imageio.imread(str(frame_path))
                frames.append(transform(
                    translate_image_reflect(image, dx_frac, dy_frac), self.image_size))
            return torch.stack(frames, dim=0)
        decoded = self._cached_raw_frames(path, indices)
        if decoded is not None:
            frames = []
            for image in decoded:
                frames.append(transform(
                    translate_image_reflect(image, dx_frac, dy_frac), self.image_size))
            return torch.stack(frames, dim=0)
        reader = self._reader(path)
        length = self._clip_length(path)
        frames = []
        for index in indices:
            index = min(max(0, int(index)), length - 1) if length > 0 else int(index)
            frames.append(transform(translate_image_reflect(reader.get_data(index), dx_frac, dy_frac), self.image_size))
        return torch.stack(frames, dim=0)

    def _read_wrist_history(self, episode: EpisodeRecord, target_step: int) -> torch.Tensor:
        """Read wrist observations aligned to ``current_history_offsets``."""
        frames = [self._read_wrist_current(episode, int(target_step) + offset)
                  for offset in self.current_history_offsets]
        return torch.stack(frames, dim=0)

    def _jitter_start_index(self, episode: EpisodeRecord, start_index: int) -> int:
        if self.split != "train" or not bool(self.window_jitter_cfg.get("enabled", False)):
            return start_index
        max_offset = int(self.window_jitter_cfg.get("max_offset", 0))
        if max_offset <= 0:
            return start_index
        last_start = episode.num_steps - self.target_history_len - self.target_offset - max(0, self.effective_future_horizon)
        low = max(0, int(start_index) - max_offset)
        high = min(max(0, last_start), int(start_index) + max_offset)
        if high <= low:
            return int(start_index)
        return int(torch.randint(low, high + 1, ()).item())

    def _translate_model_projection(self, projection: np.ndarray, payload: dict, dx_frac: float, dy_frac: float) -> np.ndarray:
        """Move model-input EE coordinates by a native-frame translation."""
        if (
            not self.front_translation_enabled
            or str(self.proprioception_cfg.get("source", "")) != "camera_projection"
            or str(self.proprioception_cfg.get("projection_image_space", "original")) != "model_input"
        ):
            return projection
        raw_h, raw_w = [int(x) for x in payload["image_shape"][:2]]
        scale = self.image_size / float(min(raw_h, raw_w))
        dx_model = float(dx_frac) * raw_w * scale
        dy_model = float(dy_frac) * raw_h * scale
        out = np.asarray(projection, dtype=np.float32).copy()
        out[0] += 2.0 * dx_model / max(1.0, self.image_size - 1.0)
        out[1] += 2.0 * dy_model / max(1.0, self.image_size - 1.0)
        return out

    def sample_window(self, episode_id: str, start_index: int) -> dict:
        episode = self._episode_by_id[episode_id]
        start_index = self._jitter_start_index(episode, start_index)
        target_step = start_index + self.target_history_len - 1 + self.target_offset
        payload = self._read_action_payload(episode)
        future_idx = self._future_traj_indices(episode.num_steps, target_step) if self.future_traj_enabled else None
        if future_idx is not None:
            action = self._future_traj_action(payload, future_idx)  # [F, cam-abs pose(9) + gripper]
        else:
            action = self._action_at(payload, target_step).astype(np.float32)
        thr = float(self.data_cfg.get("gripper_threshold", 0.0))
        if action.ndim == 2:
            # Action chunk [N, dim]: keep both the continuous gripper command and
            # its thresholded event label.
            n = action.shape[0]
            gripper_value = action[:, -1].astype(np.float32)
            gripper = (action[:, -1] > thr).astype(np.int64)
            term_steps = future_idx if future_idx is not None else [
                min(target_step + self.chunk_start_offset + k * self.chunk_stride, episode.num_steps - 1)
                for k in range(n)]
            terminate = np.asarray([self._terminate_at(payload, int(s), episode.num_steps) for s in term_steps], dtype=np.int64)
            gripper_t = torch.as_tensor(gripper, dtype=torch.long)
            gripper_value_t = torch.as_tensor(gripper_value, dtype=torch.float32)
            terminate_t = torch.as_tensor(terminate, dtype=torch.long)
        else:
            gripper_t = torch.tensor(self._gripper_at(action), dtype=torch.long)
            gripper_value_t = torch.tensor(float(action[-1]), dtype=torch.float32)
            terminate_t = torch.tensor(int(self._terminate_at(payload, target_step, episode.num_steps)), dtype=torch.long)
        sample = {
            "episode_id": episode.episode_id,
            "start_index": int(start_index),
            "target_step": int(target_step),
            "action": torch.as_tensor(action, dtype=torch.float32),
            "gripper": gripper_t,
            "gripper_value": gripper_value_t,
            "terminate": terminate_t,
            "metadata": {"source_video_path": str(episode.source_video_path), "target_video_path": str(episode.target_video_path)},
        }
        front_dx = front_dy = 0.0
        if self.load_videos:
            _src_idx = None
            if future_idx is not None:
                _src_idx = list(future_idx)
            elif self.aux_traj_enabled:
                # compute indices once so both the video and traj_target use the same frames
                _src_idx = self._compute_source_indices(episode, start_index)
                traj_target = np.stack([self._camera_abs_pose_at(payload, int(i)) for i in _src_idx]).astype(np.float32)
                sample["traj_target"] = torch.as_tensor(traj_target, dtype=torch.float32)  # [source_len, 10]
            else:
                _src_idx = self._compute_source_indices(episode, start_index)
            if not self.front_translation_enabled:
                source_video = self._read_source_indices(episode, _src_idx)
            # C20: per-frame Δt (real seconds since previous sampled frame; first frame = 0). Lets the
            # model tell a fast short demo from a slow long one (same 8 frames, different pacing).
            if self.dt_time_enabled and _src_idx is not None:
                idx_arr = np.asarray(_src_idx, dtype=np.float32)
                dt = np.zeros_like(idx_arr)
                dt[1:] = np.diff(idx_arr) / max(1.0, float(self.fps))
                sample["source_dt"] = torch.as_tensor(dt, dtype=torch.float32)  # [source_len]
            if self.split == "train" and self.front_translation_enabled:
                max_x = float(self.front_translation_cfg.get("max_x_frac", 0.0))
                max_y = float(self.front_translation_cfg.get("max_y_frac", 0.0))
                front_dx = float((torch.rand(()) * 2.0 - 1.0) * max_x)
                front_dy = float((torch.rand(()) * 2.0 - 1.0) * max_y)
            if self.front_translation_enabled:
                target_indices = [int(target_step) + offset for offset in self.current_history_offsets]
                source_sorted_indices, source_restore, source_clip_frames = self._source_frame_plan(
                    episode, _src_idx)
                # Source and current use the same front MP4. Decode their union in
                # ascending order so the ffmpeg reader never seeks backwards
                # between two independent requests for the same sample.  The full
                # 40-frame micro-clip plan is sorted before this single extraction.
                combined_indices = sorted(set(target_indices + source_sorted_indices))
                combined = self._read_front_with_translation(
                    episode, combined_indices, front_dx, front_dy)
                positions = {index: position for position, index in enumerate(combined_indices)}
                target_history = torch.stack(
                    [combined[positions[index]] for index in target_indices], dim=0)
                source_sorted_frames = torch.stack(
                    [combined[positions[index]] for index in source_sorted_indices], dim=0)
                source_video = self._restore_source_frame_plan(
                    source_sorted_frames, source_restore, len(_src_idx), source_clip_frames)
            else:
                target_history = self._read_target_history(episode, start_index)
            if self.split == "train":
                source_aug = sample_image_augmentation(self.augmentation_cfg)
                current_aug = sample_image_augmentation(self.augmentation_cfg)
                source_video = apply_sampled_image_augmentation(source_video, source_aug)
                target_history = apply_sampled_image_augmentation(target_history, current_aug)
                if self.struct_aug_enabled:
                    # project future EE positions to 224×224 image coords for EE-targeted aug
                    cam_pos = action[:, :3] if action.ndim == 2 else None
                    ee_fracs = (self._future_traj_ee_image_fracs(payload, cam_pos)
                                if cam_pos is not None else None)
                    if source_video.ndim == 5:
                        anchors, clip_frames = source_video.shape[:2]
                        flat_source = source_video.reshape(
                            anchors * clip_frames, *source_video.shape[2:])
                        flat_ee_fracs = (
                            np.repeat(ee_fracs, clip_frames, axis=0)
                            if ee_fracs is not None else None
                        )
                        source_video = apply_structural_augmentation(
                            flat_source, self.struct_aug_cfg, flat_ee_fracs).reshape_as(source_video)
                    else:
                        source_video = apply_structural_augmentation(
                            source_video, self.struct_aug_cfg, ee_fracs)
            sample["source_video"] = source_video
            sample["target_history"] = target_history
            if self.wrist_current_enabled:
                wrist_current = self._read_wrist_history(episode, target_step)
                if self.split == "train":
                    wrist_current = apply_sampled_image_augmentation(wrist_current, current_aug)
                sample["wrist_current"] = wrist_current
            if self.overlay_enabled:
                # draw the demo EE path onto the current (last) frame AFTER augmentation (crisp path)
                target_history[-1] = self._overlay_current_frame(target_history[-1], episode.episode_id, target_step)
        if self.proprioception_enabled:
            prop = np.asarray(self._proprioception_at(payload, start_index, target_step), dtype=np.float32).reshape(-1)
            if self.split == "train" and self.front_translation_enabled:
                prop = self._translate_model_projection(prop, payload, front_dx, front_dy)
            if self.proprioception_append_progress:
                # normalized task progress (target_step / (num_steps-1)) in [0,1]: a coarse
                # localization anchor telling the model "how far into the demo am I".
                denom = max(1, int(episode.num_steps) - 1)
                progress = float(min(1.0, max(0.0, target_step / denom)))
                prop = np.concatenate([prop, np.asarray([progress], dtype=np.float32)])
            if self.proprioception_append_gripper:
                grip_seq = payload.get("observations", {}).get("gripper_position")
                if grip_seq is None:
                    grip_seq = payload.get("action_dict", {}).get("gripper_position")
                if grip_seq is None:
                    raise KeyError("append_current_gripper requires observations.gripper_position")
                grip_values = np.asarray(grip_seq, dtype=np.float32).reshape(-1)
                grip_idx = min(max(0, int(target_step)), len(grip_values) - 1)
                if self.proprioception_gripper_continuous:
                    grip_state = float(grip_values[grip_idx])
                else:
                    grip_state = float(
                        grip_values[grip_idx] > float(self.data_cfg.get("gripper_threshold", 0.0)))
                prop = np.concatenate([prop, np.asarray([grip_state], dtype=np.float32)])
            if bool(self.proprioception_normalization.get("enabled", False)):
                q01 = np.asarray(self.proprioception_normalization.get("q01", []), dtype=np.float32)
                q99 = np.asarray(self.proprioception_normalization.get("q99", []), dtype=np.float32)
                if q01.shape != prop.shape or q99.shape != prop.shape:
                    raise ValueError(
                        f"proprioception q01/q99 must match shape {prop.shape}, got {q01.shape}/{q99.shape}")
                prop = (2.0 * (prop - q01) / np.maximum(q99 - q01, 1e-6) - 1.0)
                if bool(self.proprioception_normalization.get("clip", True)):
                    prop = np.clip(prop, -1.0, 1.0)
            sample["proprioception"] = torch.as_tensor(prop, dtype=torch.float32)
        if self.depth_enabled:
            depth_idx = future_idx if future_idx is not None else [target_step]
            if self.depth_representation == "patch_geometry":
                geometry = self._read_front_geometry_at(episode, depth_idx)
                if geometry is None:
                    geometry = torch.zeros(
                        len(depth_idx), self.depth_num_patches, 4, dtype=torch.float32)
                sample["front_geometry"] = geometry.to(torch.float32)
            else:
                depth = self._read_depth_at(episode, depth_idx)
                if depth is not None:
                    sample["depth_video"] = depth          # [T, H_d, W_d] uint16
                K = self._get_camera_K_224(episode, payload)
                if K is not None:
                    sample["camera_K"] = torch.as_tensor(
                        np.asarray(K, dtype=np.float32), dtype=torch.float32)  # [4]
        if self.point_tracking_enabled:
            sample["point_track"] = self._read_point_track(episode, target_step)
            if self.point_causal_window is not None:
                sample["point_track_causal"] = self._read_point_track(
                    episode, target_step, window=self.point_causal_window, num_time=self.point_causal_num_time)
        if self.aux_progress_enabled:
            # normalized position of current step within the demo ∈ [0,1] (demo-current alignment target)
            denom = max(1, int(episode.num_steps) - 1)
            progress = float(min(1.0, max(0.0, target_step / denom)))
            sample["progress_target"] = torch.as_tensor([progress], dtype=torch.float32)  # [1]
        return sample

    def __getitem__(self, index: int) -> dict:
        episode_id, start_index = self._samples[index]
        return self.sample_window(episode_id, start_index)
