"""PointWorld-DROID dataset adapter.

Reads preprocessed episodes from /path/to/pointworld-droid-3000/{uuid}/:
  gt.mp4 (or frames/)         — RGB video (symlink to DROID-1.0.1 exterior_image_1)
  depth_frames/{i:06d}.png    — uint16 grayscale PNG depth in mm (180×320)
  data.json                   — robot state + camera calibration (droid-ex compatible format)

The data.json format is intentionally identical to the droid-ex dataset so that the
existing OpenXDroidDataset machinery (cartesian_position, intrinsics, extrinsics,
_camera_abs_pose_at for C7) can be reused wholesale.

Extra capabilities vs OpenXDroidDataset:
  * `_read_depth_at(episode, indices)` — loads depth PNG frames for C11 depth 3D lifting
  * `_get_camera_K_224(episode, payload)` — scales intrinsics to 224×224 crop geometry
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from r2r_gen2act.data.adapters.openx_droid import OpenXDroidDataset
from r2r_gen2act.data.split import split_episode_ids
from r2r_gen2act.data.types import EpisodeRecord


class PointWorldDroidDataset(OpenXDroidDataset):
    """PointWorld-DROID preprocessed dataset adapter.

    Directory layout expected under `data.root`:
        {uuid}/
            gt.mp4              RGB video (exterior camera 1)
            frames/             pre-decoded JPEG frames (optional, fast loading)
            depth_frames/       uint16 PNG depth frames in mm
            depth_frames_wrist/ uint16 PNG wrist-camera depth (optional)
            data.json           robot state + calibration (droid-ex format)
    """

    def _load_episodes(self) -> list[EpisodeRecord]:
        root = Path(self.data_cfg["root"])
        source_name = str(self.data_cfg.get("source_video_name", "gt.mp4"))
        target_name = str(self.data_cfg.get("target_video_name", "gt.mp4"))
        metadata_name = str(self.data_cfg.get("metadata_name", "data.json"))

        # Detect episode directories: any directory with a data.json
        dirs = sorted([p for p in root.iterdir()
                       if p.is_dir() and (p / metadata_name).exists()])
        max_episodes = self.data_cfg.get("max_episodes")
        if max_episodes not in (None, ""):
            dirs = dirs[: int(max_episodes)]

        ids = [d.name for d in dirs]
        val_count = self.data_cfg.get("val_count")
        val_count = None if val_count in (None, "") else int(val_count)
        _, val_ids = split_episode_ids(
            ids,
            float(self.data_cfg.get("val_ratio", 0.2)),
            int(self.data_cfg.get("split_seed", 42)),
            val_count,
        )

        prop_cfg = self.proprioception_cfg

        episodes: list[EpisodeRecord] = []
        for d in dirs:
            split = "val" if d.name in val_ids else "train"
            if self.split in ("train", "val") and split != self.split:
                continue

            meta_path = d / metadata_name
            src = d / source_name
            tgt = d / target_name

            if not meta_path.exists():
                continue
            # If source/target mp4 doesn't exist, check frames/ subdir
            if not src.exists():
                frames_d = d / self.frames_subdir if self.frames_subdir else None
                if frames_d and frames_d.is_dir():
                    # No mp4, but frames exist — still valid for training
                    src = d / source_name   # keep path (will be handled by _read_video_indices)
                else:
                    continue
            if not tgt.exists():
                tgt = src  # fallback: both point to same video

            with meta_path.open("r", encoding="utf-8") as f:
                payload = json.load(f)

            # Filter out episodes with bad camera projection (same as openx_droid)
            if (self.proprioception_enabled
                    and str(prop_cfg.get("source", "")) == "camera_projection"):
                if (not self._has_camera_projection_calibration(payload)
                        or not self._camera_projection_quality_ok(payload)):
                    continue

            episodes.append(EpisodeRecord(
                d.name,
                int(payload["num_steps"]),
                src,
                tgt,
                meta_path,
                split,
            ))

        mapping = (self.cfg.get("action", {}).get("mapping", {})
                   .get("type", "droid_observation_cartesian_future_delta_pose6d_camera"))
        print(f"[PointWorldDroidDataset] action_mapping={mapping} split={self.split} "
              f"episodes={len(episodes)}")
        return episodes

    def _read_video_indices(self, path: "Path", indices: list[int]) -> "torch.Tensor":
        """Override to handle 1-indexed frames (000001.jpg…) in the PointWorld frames dir."""
        from r2r_gen2act.data.transforms import image_to_tensor
        import imageio.v2 as _imageio
        fd = self._frames_dir(path)
        if fd is not None:
            length = self._frame_count_cache.get(fd) or self._clip_length(path)
            # frames are 1-indexed: 000001.jpg … 000{N}.jpg  → add 1 to the 0-based index
            clamped = [min(max(0, int(idx)), length - 1) for idx in indices]
            frames = [image_to_tensor(
                _imageio.imread(str(fd / f"{int(i) + 1:06d}.{self.frames_ext}")),
                self.image_size) for i in clamped]
            return torch.stack(frames, dim=0)
        # fallback: mp4 path (base class)
        return super()._read_video_indices(path, indices)

    def _camera_serial(self, payload: dict, key: str = "") -> str:
        """Use ext1_cam_serial from data.json (PointWorld has 3 serials, base class errors)."""
        if key:
            return key
        serial = payload.get("ext1_cam_serial", "")
        if serial:
            return serial
        # fallback: base class logic
        return super()._camera_serial(payload, key)

    def _action_at(self, payload: dict, step: int) -> np.ndarray:
        """Override to inject ext1_cam_serial — PointWorld has 3 camera serials and droid_action
        would fail on _camera_extrinsic_pose without an explicit key."""
        from r2r_gen2act.data.action.mappings import droid_action
        mapping_cfg = dict(self.cfg.get("action", {}).get("mapping", {}))
        if not mapping_cfg.get("camera_extrinsics_key"):
            mapping_cfg["camera_extrinsics_key"] = payload.get("ext1_cam_serial", "")
        mapping = mapping_cfg.get("type", "droid_observation_cartesian_future_delta_pose6d_camera")
        return droid_action(payload, step, mapping, self.future_horizon, self.chunk_size, mapping_cfg)

    def _camera_abs_pose_at(self, payload: dict, step: int) -> np.ndarray:
        """Override to inject ext1_cam_serial as camera_extrinsics_key before calling parent."""
        from r2r_gen2act.data.action.mappings import _base_to_camera_rotation, _camera_extrinsic_pose
        from r2r_gen2act.data.action.rotation import euler_to_matrix, matrix_to_6d
        # Build a mapping_cfg that explicitly names ext1 serial → avoids multi-serial ambiguity
        mapping_cfg = dict(self.cfg.get("action", {}).get("mapping", {}))
        mapping_cfg["camera_extrinsics_key"] = payload.get("ext1_cam_serial", "")
        extrinsic = _camera_extrinsic_pose(payload, mapping_cfg)
        r_cam_base = _base_to_camera_rotation(payload, mapping_cfg)
        obs = payload["observations"]["cartesian_position"]
        step = min(max(0, int(step)), len(obs) - 1)
        cart = np.asarray(obs[step], dtype=np.float64)
        pos_base = cart[:3]
        convention = str(mapping_cfg.get("extrinsics_convention", "camera_pose_in_base"))
        if convention == "camera_pose_in_base":
            pos_cam = r_cam_base @ (pos_base - extrinsic[:3])
        else:
            pos_cam = r_cam_base @ pos_base + extrinsic[:3]
        r_cam = r_cam_base @ euler_to_matrix(cart[3:6])
        sixd = matrix_to_6d(r_cam)
        grip_seq = payload.get("observations", {}).get("gripper_position")
        if grip_seq is None:
            grip_seq = payload.get("action_dict", {}).get("gripper_position", [[0.0]])
        grip = float(np.asarray(grip_seq, dtype=np.float32).reshape(len(obs), -1)[step, 0])
        return np.concatenate([pos_cam.astype(np.float32), sixd.astype(np.float32),
                                np.asarray([grip], np.float32)])

    # ── depth loading ──────────────────────────────────────────────────────────

    def _read_depth_at(self, episode: EpisodeRecord, indices: list[int]) -> torch.Tensor | None:
        """Load depth frames [T, H_d, W_d] uint16 mm from pre-decoded PNG files."""
        if not self.depth_enabled:
            return None
        ep_dir = Path(episode.source_video_path).parent
        depth_dir = ep_dir / self.depth_frames_subdir
        if not depth_dir.is_dir():
            return None

        # Figure out frame count (for clamping)
        pngs = sorted(depth_dir.glob(f"*.{self.depth_ext}"))
        if not pngs:
            return None
        n_frames = len(pngs)

        # Infer frame shape from first readable PNG
        _h, _w = 180, 320
        for _p in pngs[:3]:
            try:
                _sample = np.array(Image.open(str(_p)))
                _h, _w = _sample.shape[:2]
                break
            except Exception:
                pass

        frames = []
        for idx in indices:
            idx_c = int(np.clip(idx, 0, n_frames - 1))
            path = depth_dir / f"{idx_c:06d}.{self.depth_ext}"
            try:
                arr = np.array(Image.open(str(path))).astype(np.uint16)
            except Exception:
                arr = np.zeros((_h, _w), dtype=np.uint16)
            frames.append(torch.from_numpy(arr))

        return torch.stack(frames, dim=0)     # [T, H_d, W_d] uint16

    # ── camera intrinsics scaled to 224×224 ───────────────────────────────────

    def _get_camera_K_224(self, episode: EpisodeRecord, payload: dict) -> np.ndarray | None:
        """Return (fx, fy, cx, cy) scaled to the 224×224 image after resize_center_crop.

        data.json stores intrinsics at the original 320×180 sensor resolution.
        After resize_center_crop (landscape: new_h=224, new_w≈398, crop_left=87):
            scale = 224 / H_orig   (for landscape H_orig < W_orig)
            fx_224 = fx * new_W / W_orig  = fx * scale  (isotropic)
            cx_224 = cx * scale - crop_left
            fy_224 = fy * scale
            cy_224 = cy * scale
        """
        try:
            serial = payload.get("ext1_cam_serial", "")
            intr_map = payload.get("calibration", {}).get("intrinsics", {})
            if serial not in intr_map:
                # Try first available
                if not intr_map:
                    return None
                serial = next(iter(intr_map))
            intr = intr_map[serial]
            fx, cx, fy, cy = [float(v) for v in intr["cameraMatrix"][:4]]
            W_orig = float(intr.get("width", 320))
            H_orig = float(intr.get("height", 180))
            image_h, image_w = [int(x) for x in payload.get("image_shape", [H_orig, W_orig])[:2]]

            # Scale from sensor resolution → actual video resolution
            fx *= image_w / W_orig
            cx *= image_w / W_orig
            fy *= image_h / H_orig
            cy *= image_h / H_orig

            # Apply resize_center_crop geometry to 224×224
            img_size = self.image_size  # 224
            if image_h < image_w:
                new_h = img_size
                new_w = int(round(image_w * img_size / image_h))
            else:
                new_w = img_size
                new_h = int(round(image_h * img_size / image_w))
            scale_x = new_w / image_w
            scale_y = new_h / image_h
            crop_top  = max(0, (new_h - img_size) // 2)
            crop_left = max(0, (new_w - img_size) // 2)

            fx_224 = fx * scale_x
            fy_224 = fy * scale_y
            cx_224 = cx * scale_x - crop_left
            cy_224 = cy * scale_y - crop_top

            return np.array([fx_224, fy_224, cx_224, cy_224], dtype=np.float32)
        except Exception:
            return None

