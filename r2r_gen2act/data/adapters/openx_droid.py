from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from r2r_gen2act.data.action.mappings import droid_action, terminate_from_payload
from r2r_gen2act.data.action.rotation import EULER_CONVENTION
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
        val_count = self.data_cfg.get("val_count")
        val_count = None if val_count in (None, "") else int(val_count)
        _, val_ids = split_episode_ids(ids, float(self.data_cfg.get("val_ratio", 0.2)), int(self.data_cfg.get("split_seed", 42)), val_count)
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
            if self.proprioception_enabled and str(self.proprioception_cfg.get("source", "")) == "camera_projection":
                if not self._has_camera_projection_calibration(payload) or not self._camera_projection_quality_ok(payload):
                    continue
            episodes.append(EpisodeRecord(d.name, int(payload["num_steps"]), src, tgt, meta, split))
        mapping = self.cfg.get("action", {}).get("mapping", {}).get("type", "droid_actions_first6_plus_gripper")
        print(f"[OpenXDroidDataset] action_mapping={mapping} split={self.split} episodes={len(episodes)}")
        return episodes

    def _read_action_payload(self, episode: EpisodeRecord) -> dict:
        assert episode.metadata_path is not None
        with episode.metadata_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _action_at(self, payload: dict, step: int) -> np.ndarray:
        mapping_cfg = self.cfg.get("action", {}).get("mapping", {})
        mapping = mapping_cfg.get("type", "droid_actions_first6_plus_gripper")
        return droid_action(payload, step, mapping, self.future_horizon, self.chunk_size, mapping_cfg)

    def _terminate_at(self, payload: dict, step: int, num_steps: int) -> int:
        return terminate_from_payload(payload, step, num_steps, self.terminate_positive_window)

    def _has_camera_projection_calibration(self, payload: dict) -> bool:
        try:
            serial = self._camera_serial(payload, str(self.proprioception_cfg.get("camera_extrinsics_key", self.cfg.get("action", {}).get("mapping", {}).get("camera_extrinsics_key", "")) or ""))
        except ValueError:
            return False
        intrinsics = payload.get("calibration", {}).get("intrinsics")
        if not isinstance(intrinsics, dict) or not isinstance(intrinsics.get(serial), dict):
            return False
        intr = intrinsics[serial]
        matrix = intr.get("cameraMatrix")
        return (
            isinstance(matrix, list)
            and len(matrix) >= 4
            and float(intr.get("width", 0) or 0) > 0
            and float(intr.get("height", 0) or 0) > 0
        )

    def _camera_projection_quality_ok(self, payload: dict) -> bool:
        prop_cfg = self.proprioception_cfg
        max_abs = float(prop_cfg.get("projection_max_abs", 3.0))
        max_outside_frac = float(prop_cfg.get("projection_max_outside_frac", 0.05))
        min_depth = float(prop_cfg.get("projection_min_depth", 1e-6))
        mapping_cfg = self.cfg.get("action", {}).get("mapping", {})
        serial = self._camera_serial(payload, str(prop_cfg.get("camera_extrinsics_key", mapping_cfg.get("camera_extrinsics_key", "")) or ""))
        calibration = payload["calibration"]
        extrinsic = np.asarray(calibration["extrinsics"][serial], dtype=np.float64)
        rot = Rotation.from_euler(EULER_CONVENTION, extrinsic[3:6]).as_matrix()
        points_base = np.asarray(payload["observations"]["cartesian_position"], dtype=np.float64)[:, :3]
        convention = str(prop_cfg.get("extrinsics_convention", mapping_cfg.get("extrinsics_convention", "camera_pose_in_base")))
        if convention == "camera_pose_in_base":
            points_camera = (rot.T @ (points_base - extrinsic[:3]).T).T
        elif convention == "base_to_camera":
            points_camera = (rot @ points_base.T).T + extrinsic[:3]
        else:
            raise ValueError(f"Unknown extrinsics_convention={convention!r}")
        z = points_camera[:, 2]
        if np.any(z <= min_depth):
            return False
        intr = calibration["intrinsics"][serial]
        fx, cx, fy, cy = [float(x) for x in intr["cameraMatrix"][:4]]
        image_h, image_w = [int(x) for x in payload["image_shape"][:2]]
        fx *= image_w / float(intr["width"])
        cx *= image_w / float(intr["width"])
        fy *= image_h / float(intr["height"])
        cy *= image_h / float(intr["height"])
        u = fx * points_camera[:, 0] / z + cx
        v = fy * points_camera[:, 1] / z + cy
        norm = np.stack([2.0 * u / max(1.0, image_w - 1.0) - 1.0, 2.0 * v / max(1.0, image_h - 1.0) - 1.0], axis=1)
        return bool(np.mean(np.abs(norm) > max_abs) <= max_outside_frac)

    def _camera_serial(self, payload: dict, key: str = "") -> str:
        extrinsics = payload.get("calibration", {}).get("extrinsics", {})
        if key:
            if key not in extrinsics:
                raise ValueError(f"camera_extrinsics_key={key!r} not found in calibration.extrinsics")
            return key
        candidates = [k for k, v in extrinsics.items() if isinstance(v, list) and len(v) == 6]
        if len(candidates) != 1:
            raise ValueError("Expected exactly one 6D camera extrinsic; set camera_extrinsics_key")
        return candidates[0]

    def _project_ee_to_normalized_image(self, payload: dict, step: int, prop_cfg: dict) -> np.ndarray:
        mapping_cfg = self.cfg.get("action", {}).get("mapping", {})
        serial = self._camera_serial(payload, str(prop_cfg.get("camera_extrinsics_key", mapping_cfg.get("camera_extrinsics_key", "")) or ""))
        calibration = payload["calibration"]
        extrinsic = np.asarray(calibration["extrinsics"][serial], dtype=np.float64)
        rot = Rotation.from_euler(EULER_CONVENTION, extrinsic[3:6]).as_matrix()
        pos_base = np.asarray(payload["observations"]["cartesian_position"][step], dtype=np.float64)[:3]
        convention = str(prop_cfg.get("extrinsics_convention", mapping_cfg.get("extrinsics_convention", "camera_pose_in_base")))
        if convention == "camera_pose_in_base":
            pos_camera = rot.T @ (pos_base - extrinsic[:3])
        elif convention == "base_to_camera":
            pos_camera = rot @ pos_base + extrinsic[:3]
        else:
            raise ValueError(f"Unknown extrinsics_convention={convention!r}")

        intr = calibration["intrinsics"][serial]
        fx, cx, fy, cy = [float(x) for x in intr["cameraMatrix"]]
        image_h, image_w = [int(x) for x in payload["image_shape"][:2]]
        fx *= image_w / float(intr["width"])
        cx *= image_w / float(intr["width"])
        fy *= image_h / float(intr["height"])
        cy *= image_h / float(intr["height"])
        z = max(float(pos_camera[2]), 1e-6)
        u = fx * float(pos_camera[0]) / z + cx
        v = fy * float(pos_camera[1]) / z + cy
        # Normalize pixel coordinates to [-1, 1] at image borders. Do not clamp: out-of-frame
        # end-effector locations intentionally become values outside [-1, 1].
        u_norm = (2.0 * u / max(1.0, image_w - 1.0)) - 1.0
        v_norm = (2.0 * v / max(1.0, image_h - 1.0)) - 1.0
        if int(prop_cfg.get("dims", 2)) >= 3:
            return np.asarray([u_norm, v_norm, float(pos_camera[2])], dtype=np.float32)[: int(prop_cfg.get("dims", 2))]
        return np.asarray([u_norm, v_norm], dtype=np.float32)

    def _camera_abs_pose_at(self, payload: dict, step: int) -> np.ndarray:
        """C7 video-to-trajectory: ABSOLUTE camera-frame EE pose at `step` = [cam_pos(3), 6D-rot(6)] + gripper.
        Consistent with the camera projection (position) and the camera-frame action mapping (rotation)."""
        from r2r_gen2act.data.action.mappings import _base_to_camera_rotation, _camera_extrinsic_pose
        from r2r_gen2act.data.action.rotation import euler_to_matrix, matrix_to_6d
        mapping_cfg = self.cfg.get("action", {}).get("mapping", {})
        extrinsic = _camera_extrinsic_pose(payload, mapping_cfg)
        r_cam_base = _base_to_camera_rotation(payload, mapping_cfg)  # R that maps base-frame vectors -> camera frame
        obs = payload["observations"]["cartesian_position"]
        step = min(max(0, int(step)), len(obs) - 1)
        cart = np.asarray(obs[step], dtype=np.float64)
        pos_base = cart[:3]
        convention = str(mapping_cfg.get("extrinsics_convention", "camera_pose_in_base"))
        if convention == "camera_pose_in_base":
            pos_cam = r_cam_base @ (pos_base - extrinsic[:3])
        else:  # base_to_camera
            pos_cam = r_cam_base @ pos_base + extrinsic[:3]
        r_cam = r_cam_base @ euler_to_matrix(cart[3:6])
        sixd = matrix_to_6d(r_cam)
        grip_seq = payload.get("observations", {}).get("gripper_position")
        if grip_seq is None:
            grip_seq = payload.get("action_dict", {}).get("gripper_position", [[0.0]])
        grip = float(np.asarray(grip_seq, dtype=np.float32).reshape(len(obs), -1)[step, 0])
        return np.concatenate([pos_cam.astype(np.float32), sixd.astype(np.float32), np.asarray([grip], np.float32)])

    def _future_traj_ee_image_fracs(self, payload: dict, cam_positions: np.ndarray) -> np.ndarray | None:
        """C10 structural aug: project camera-frame positions [F, 3] to normalised 224×224 coords [F, 2] ∈ [0,1].
        cam_positions are already in camera frame (from _camera_abs_pose_at pos_cam); no extrinsic needed.
        Returns None if intrinsics are unavailable or any z ≤ 0."""
        try:
            mapping_cfg = self.cfg.get("action", {}).get("mapping", {})
            prop_cfg = self.proprioception_cfg
            serial = self._camera_serial(
                payload,
                str(prop_cfg.get("camera_extrinsics_key", mapping_cfg.get("camera_extrinsics_key", "")) or ""),
            )
            calibration = payload.get("calibration", {})
            intr = calibration.get("intrinsics", {}).get(serial)
            if intr is None:
                return None
            image_h, image_w = [int(x) for x in payload["image_shape"][:2]]
            fx, cx, fy, cy = [float(v) for v in intr["cameraMatrix"][:4]]
            # scale intrinsics from sensor resolution to actual video frame resolution
            fx *= image_w / float(intr["width"])
            cx *= image_w / float(intr["width"])
            fy *= image_h / float(intr["height"])
            cy *= image_h / float(intr["height"])
            # project each camera-frame position to pixel coords
            xs, ys, zs = cam_positions[:, 0], cam_positions[:, 1], cam_positions[:, 2]
            if np.any(zs <= 0):
                return None
            u_px = fx * xs / zs + cx   # pixel x in (image_w × image_h) frame
            v_px = fy * ys / zs + cy   # pixel y in (image_h × image_h) frame
            # map from (image_w × image_h) through resize_center_crop to 224×224
            # resize: scale so the short side becomes 224
            image_size = self.image_size
            if image_h < image_w:
                new_h = image_size
                new_w = int(round(image_w * image_size / image_h))
            else:
                new_w = image_size
                new_h = int(round(image_h * image_size / image_w))
            scale_x = new_w / image_w
            scale_y = new_h / image_h
            u_rs = u_px * scale_x
            v_rs = v_px * scale_y
            # center crop: top/left offset
            crop_top  = max(0, (new_h - image_size) // 2)
            crop_left = max(0, (new_w - image_size) // 2)
            u_224 = u_rs - crop_left
            v_224 = v_rs - crop_top
            # normalise to [0, 1] (clamped)
            u_frac = np.clip(u_224 / max(1, image_size - 1), 0.0, 1.0)
            v_frac = np.clip(v_224 / max(1, image_size - 1), 0.0, 1.0)
            return np.stack([u_frac, v_frac], axis=1).astype(np.float32)  # [F, 2]
        except Exception:
            return None

    def _ee_path_pixels(self, episode: EpisodeRecord):
        """Whole-episode EE 2D path in ORIGINAL frame pixels for the trajectory overlay.
        Returns (path_px[N,2] float32 (x,y), W, H) or None if projection is unavailable."""
        try:
            payload = self._read_action_payload(episode)
            if "image_shape" not in payload or "calibration" not in payload:
                return None
            h, w = int(payload["image_shape"][0]), int(payload["image_shape"][1])
            prop_cfg = self.proprioception_cfg
            positions = payload["observations"]["cartesian_position"]
            n = min(int(episode.num_steps), len(positions))
            pts = []
            for step in range(n):
                uv = self._project_ee_to_normalized_image(payload, step, prop_cfg)
                u_px = (float(uv[0]) + 1.0) / 2.0 * (w - 1.0)
                v_px = (float(uv[1]) + 1.0) / 2.0 * (h - 1.0)
                pts.append([u_px, v_px])
            if len(pts) < 2:
                return None
            return np.asarray(pts, dtype=np.float32), w, h
        except Exception:
            return None

    def _proprioception_at(self, payload: dict, start_index: int, target_step: int) -> np.ndarray:
        prop_cfg = self.proprioception_cfg
        source = str(prop_cfg.get("source", "observations"))
        key = str(prop_cfg.get("key", "cartesian_position"))
        dims = int(prop_cfg.get("dims", 6))
        step_mode = str(prop_cfg.get("step", "target"))
        step = start_index + self.target_history_len - 1 if step_mode == "history_last" else target_step
        if source == "camera_projection":
            values = payload["observations"]["cartesian_position"]
            step = min(max(0, int(step)), len(values) - 1)
            return self._project_ee_to_normalized_image(payload, step, prop_cfg)[:dims]
        values = payload[source][key]
        step = min(max(0, int(step)), len(values) - 1)
        return np.asarray(values[step], dtype=np.float32).reshape(-1)[:dims]
