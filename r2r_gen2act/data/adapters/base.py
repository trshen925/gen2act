from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
import torch
from torch.utils.data import Dataset

from r2r_gen2act.data.episode_index import build_windows
from r2r_gen2act.data.overlay import draw_trajectory, raw_to_display_px
from r2r_gen2act.data.transforms import apply_image_augmentation, apply_structural_augmentation, image_to_tensor
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
        self.chunk_size = int(cfg.get("action", {}).get("chunk_size", 1))
        # Furthest future step a window must reach: chunk step N is at +N*future_horizon.
        self.effective_future_horizon = self.future_horizon * max(1, self.chunk_size)
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
        wrist_cfg = self.data_cfg.get("wrist_current", {}) or {}
        self.wrist_current_enabled = bool(wrist_cfg.get("enabled", False))
        self.wrist_frames_subdir = str(wrist_cfg.get("frames_subdir", "wrist_frames"))
        self.wrist_frames_ext = str(wrist_cfg.get("frames_ext", "jpg"))
        self.wrist_allow_video_fallback = bool(wrist_cfg.get("allow_video_fallback", False))
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
        # C10: EE-targeted structural augmentation to simulate generated-video gripper gap.
        self.struct_aug_cfg = self.data_cfg.get("structural_augmentation", {})
        self.struct_aug_enabled = bool(self.struct_aug_cfg.get("enabled", False))
        self._episodes = self._load_episodes()
        max_windows = self.data_cfg.get("max_windows")
        max_windows = None if max_windows in (None, "") else int(max_windows)
        self._samples = build_windows(self._episodes, self.source_len, self.target_history_len, self.target_offset, self.action_stride, max_windows, self.effective_future_horizon)
        self._episode_by_id = {e.episode_id: e for e in self._episodes}
        self._video_cache: dict[Path, Any] = {}
        # C24: per-window source-frame count k (depends only on episode num_steps) — for bucket sampling.
        if self.dynamic_source_enabled:
            k_by_ep = {e.episode_id: self._dynamic_source_len(e.num_steps) for e in self._episodes}
            self._window_k = [k_by_ep[eid] for eid, _ in self._samples]
        else:
            self._window_k = None

    def window_k(self) -> list[int] | None:
        """Per-window source-frame count k (for KBucketBatchSampler). None if not dynamic."""
        return self._window_k

    def _dynamic_source_len(self, num_steps: int) -> int:
        k = round(int(num_steps) / max(1e-6, self.dynamic_source_stride))
        return int(min(max(k, self.dynamic_source_min), self.dynamic_source_max))

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_video_cache"] = {}
        state["_ee_path_cache"] = {}
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
        fd = self._frames_dir(path)
        if fd is not None:
            n = self._frame_count_cache.get(fd)
            if n is None:
                n = len(list(fd.glob(f"*.{self.frames_ext}")))
                self._frame_count_cache[fd] = n
            return n
        return self._video_length(self._reader(path))

    def _read_video_indices(self, path: Path, indices: list[int]) -> torch.Tensor:
        fd = self._frames_dir(path)
        if fd is not None:
            length = self._frame_count_cache.get(fd) or self._clip_length(path)
            if length > 0:
                indices = [min(max(0, int(idx)), length - 1) for idx in indices]
            frames = [image_to_tensor(imageio.imread(str(fd / f"{int(idx):06d}.{self.frames_ext}")), self.image_size) for idx in indices]
            return torch.stack(frames, dim=0)
        reader = self._reader(path)
        length = self._video_length(reader)
        if length > 0:
            indices = [min(max(0, int(idx)), length - 1) for idx in indices]
        frames = [image_to_tensor(reader.get_data(int(idx)), self.image_size) for idx in indices]
        return torch.stack(frames, dim=0)

    def _compute_source_indices(self, episode: EpisodeRecord, start_index: int) -> list[int]:
        """Compute the source video frame indices (with train jitter) for a given window."""
        if episode.source_video_path is None:
            raise ValueError(f"Episode {episode.episode_id} has no source video")
        mode = str(self.data_cfg.get("source_sampling", "linspace"))
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
            # C18: float the linspace window start/end by up to float_frac of the clip (train only),
            # so the sampled frames cover a different span each epoch → demo diversity.
            if self.source_float_enabled and self.split == "train" and source_length > 2:
                front_span = self.source_float_front_frac * (source_length - 1)
                back_span = self.source_float_back_frac * (source_length - 1)
                lo = int(round(float(torch.rand(()).item()) * front_span))
                hi = int(round((source_length - 1) - float(torch.rand(()).item()) * back_span))
                if hi - lo < k:
                    lo, hi = 0, source_length - 1
            indices = [int(round(x)) for x in np.linspace(lo, hi, k)]
        return self._jitter_source_indices(indices, source_length)

    def _read_source_video(self, episode: EpisodeRecord, start_index: int) -> torch.Tensor:
        indices = self._compute_source_indices(episode, start_index)
        return self._read_video_indices(episode.source_video_path, indices)

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
            return self._read_video_indices(episode.target_video_path, indices)
        return self._read_video_indices(episode.target_video_path, list(range(start_index, start_index + self.target_history_len)))

    def _read_wrist_current(self, episode: EpisodeRecord, target_step: int) -> torch.Tensor:
        """Read the wrist frame synchronized with the external current observation."""
        target_step = min(max(0, int(target_step)), max(0, int(episode.num_steps) - 1))
        cache_dir = Path(episode.metadata_path).parent / self.wrist_frames_subdir
        cached = cache_dir / f"{target_step:06d}.{self.wrist_frames_ext}"
        if cached.exists():
            return image_to_tensor(imageio.imread(str(cached)), self.image_size)
        if not self.wrist_allow_video_fallback:
            raise FileNotFoundError(
                f"Missing wrist cache frame {cached}; run scripts/preprocess_wrist_frames.py first")
        wrist_video = Path(str(episode.extra.get("wrist_video_path", "")))
        if not wrist_video.exists():
            raise FileNotFoundError(f"Missing raw wrist video for {episode.episode_id}: {wrist_video}")
        raw_idx = int(episode.extra.get("source_frame_start", 0)) + target_step
        return self._read_video_indices(wrist_video, [raw_idx])[0]

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
            # Action chunk [N, dim]: per-step gripper and terminate at +(k+1)*future_horizon.
            n = action.shape[0]
            gripper = (action[:, -1] > thr).astype(np.int64)
            term_steps = future_idx if future_idx is not None else [
                min(target_step + (k + 1) * max(1, self.future_horizon), episode.num_steps - 1) for k in range(n)]
            terminate = np.asarray([self._terminate_at(payload, int(s), episode.num_steps) for s in term_steps], dtype=np.int64)
            gripper_t = torch.as_tensor(gripper, dtype=torch.long)
            terminate_t = torch.as_tensor(terminate, dtype=torch.long)
        else:
            gripper_t = torch.tensor(self._gripper_at(action), dtype=torch.long)
            terminate_t = torch.tensor(int(self._terminate_at(payload, target_step, episode.num_steps)), dtype=torch.long)
        sample = {
            "episode_id": episode.episode_id,
            "start_index": int(start_index),
            "target_step": int(target_step),
            "action": torch.as_tensor(action, dtype=torch.float32),
            "gripper": gripper_t,
            "terminate": terminate_t,
            "metadata": {"source_video_path": str(episode.source_video_path), "target_video_path": str(episode.target_video_path)},
        }
        if self.load_videos:
            _src_idx = None
            if future_idx is not None:
                source_video = self._read_video_indices(episode.source_video_path, future_idx)
                _src_idx = list(future_idx)
            elif self.aux_traj_enabled:
                # compute indices once so both the video and traj_target use the same frames
                _src_idx = self._compute_source_indices(episode, start_index)
                source_video = self._read_video_indices(episode.source_video_path, _src_idx)
                traj_target = np.stack([self._camera_abs_pose_at(payload, int(i)) for i in _src_idx]).astype(np.float32)
                sample["traj_target"] = torch.as_tensor(traj_target, dtype=torch.float32)  # [source_len, 10]
            else:
                _src_idx = self._compute_source_indices(episode, start_index)
                source_video = self._read_video_indices(episode.source_video_path, _src_idx)
            # C20: per-frame Δt (real seconds since previous sampled frame; first frame = 0). Lets the
            # model tell a fast short demo from a slow long one (same 8 frames, different pacing).
            if self.dt_time_enabled and _src_idx is not None:
                idx_arr = np.asarray(_src_idx, dtype=np.float32)
                dt = np.zeros_like(idx_arr)
                dt[1:] = np.diff(idx_arr) / max(1.0, float(self.fps))
                sample["source_dt"] = torch.as_tensor(dt, dtype=torch.float32)  # [source_len]
            target_history = self._read_target_history(episode, start_index)
            if self.split == "train":
                source_video = apply_image_augmentation(source_video, self.augmentation_cfg)
                target_history = apply_image_augmentation(target_history, self.augmentation_cfg)
                if self.struct_aug_enabled:
                    # project future EE positions to 224×224 image coords for EE-targeted aug
                    cam_pos = action[:, :3] if action.ndim == 2 else None
                    ee_fracs = (self._future_traj_ee_image_fracs(payload, cam_pos)
                                if cam_pos is not None else None)
                    source_video = apply_structural_augmentation(source_video, self.struct_aug_cfg, ee_fracs)
            sample["source_video"] = source_video
            sample["target_history"] = target_history
            if self.wrist_current_enabled:
                wrist_current = self._read_wrist_history(episode, target_step)
                if self.split == "train":
                    wrist_current = apply_image_augmentation(wrist_current, self.augmentation_cfg)
                sample["wrist_current"] = wrist_current
            if self.overlay_enabled:
                # draw the demo EE path onto the current (last) frame AFTER augmentation (crisp path)
                target_history[-1] = self._overlay_current_frame(target_history[-1], episode.episode_id, target_step)
        if self.proprioception_enabled:
            prop = np.asarray(self._proprioception_at(payload, start_index, target_step), dtype=np.float32).reshape(-1)
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
                grip_state = float(
                    grip_values[grip_idx] > float(self.data_cfg.get("gripper_threshold", 0.0)))
                prop = np.concatenate([prop, np.asarray([grip_state], dtype=np.float32)])
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
