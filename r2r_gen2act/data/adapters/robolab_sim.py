"""Adapter for RoboLab simulation clips (banana_in_bowl_*, and future RoboLab tasks).

Format is nearly identical to droid-ex-3000-out (parquet + meta.json + rgb.mp4 + extrinsics.json +
frames/), so this subclasses DroidExOutDataset and only overrides payload assembly to handle the
DIFFERENT coordinate conventions of the Isaac-sim data:

  1. EE `cartesian_position` is in the GLOBAL WORLD frame (each parallel env sits at a different world
     grid offset, x~15/-14/5, y~±10). We subtract the per-clip env origin (= mean of EE x,y; z=0) to
     get env-local BASE-frame EE — matching droid's base-frame convention.
  2. The camera extrinsic uses a different projection convention than droid:
       Rcb = Rot('xyz', euler_cam) @ diag(-1,-1,1);   p_cam = Rcb @ (p_base - t)
     droid instead does  p_cam = Rot('xyz', euler')^T @ (p_base - t)  (extrinsics_convention=
     camera_pose_in_base). So we store an EQUIVALENT euler' = matrix_to_euler(Rcb.T) with t unchanged,
     after which ALL downstream droid camera machinery (_camera_abs_pose_at, projection, delta mapping)
     works unmodified.
  3. Intrinsics come from the clip's own intrinsics.json / extrinsics.json intrinsic_matrix
     (fx=fy=524, cx=640, cy=360 for a 1280x720 sensor), not the shared KarlP file.

Validated: projecting EE with this convention lands on the gripper in all 10 banana clips
(gen2act/viz_ee_check/banana_all/). The 6D extrinsic format in banana extrinsics.json is
`camera.cam2base_extrinsics_6d = [tx,ty,tz, rx,ry,rz]` (Euler xyz radians).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from r2r_gen2act.data.adapters.droid_ex_out import DroidExOutDataset, _json_col, _OBS_CART, _OBS_GRIP, _IS_LAST, _IS_TERM
from r2r_gen2act.data.split import split_episode_ids
from r2r_gen2act.data.types import EpisodeRecord
import pyarrow.parquet as pq

_AXIS_CORRECTION = np.diag([-1.0, -1.0, 1.0])   # banana OpenGL-style camera axis flip


class RobolabSimDataset(DroidExOutDataset):
    def _load_episodes(self):
        root = Path(self.data_cfg["root"])
        pattern = str(self.data_cfg.get("episode_glob", "*"))
        video_name = str(self.data_cfg.get("source_video_name", "rgb.mp4"))
        metadata_name = str(self.data_cfg.get("metadata_name", "meta.json"))
        dirs = sorted([p for p in root.glob(pattern)
                       if p.is_dir() and (p / metadata_name).exists() and (p / "data.parquet").exists()
                       and (p / "extrinsics.json").exists()])
        max_episodes = self.data_cfg.get("max_episodes")
        if max_episodes not in (None, ""):
            dirs = dirs[: int(max_episodes)]
        ids = [d.name for d in dirs]
        val_count = self.data_cfg.get("val_count")
        val_count = None if val_count in (None, "") else int(val_count)
        _, val_ids = split_episode_ids(
            ids, float(self.data_cfg.get("val_ratio", 0.2)),
            int(self.data_cfg.get("split_seed", 42)), val_count)

        episodes: list[EpisodeRecord] = []
        for d in dirs:
            split = "val" if d.name in val_ids else "train"
            if self.split in ("train", "val") and split != self.split:
                continue
            with (d / metadata_name).open("r", encoding="utf-8") as f:
                meta = json.load(f)
            video = d / video_name
            if not video.exists():
                continue
            fsub = str(self.data_cfg.get("frames_subdir", "") or "")
            if fsub:
                fdir = d / fsub
                if not fdir.is_dir() or not any(fdir.iterdir()):
                    continue
            num_steps = int(meta["num_frames"])
            rec = EpisodeRecord(d.name, num_steps, video, video, d / metadata_name, split,
                                extra={"extrinsics_path": str(d / "extrinsics.json")})
            if self.proprioception_enabled and str(self.proprioception_cfg.get("source", "")) == "camera_projection":
                try:
                    payload = self._read_action_payload(rec)
                except (ValueError, KeyError):
                    continue
                if not self._has_camera_projection_calibration(payload) or not self._camera_projection_quality_ok(payload):
                    continue
            episodes.append(rec)
        mapping = self.cfg.get("action", {}).get("mapping", {}).get("type", "?")
        print(f"[RobolabSimDataset] action_mapping={mapping} split={self.split} episodes={len(episodes)}")
        return episodes

    def _read_action_payload(self, episode) -> dict:
        cached = self._payload_cache.get(episode.episode_id)
        if cached is not None:
            return cached
        meta_dir = episode.metadata_path.parent
        table = pq.read_table(str(meta_dir / "data.parquet"))
        cart = np.asarray(_json_col(table, _OBS_CART), dtype=np.float64)          # [T,6] world-frame pose
        grip = np.asarray(table.column(_OBS_GRIP).to_pylist(), dtype=np.float64).reshape(-1, 1)
        is_last = [int(x) for x in table.column(_IS_LAST).to_pylist()] if _IS_LAST in table.column_names else [0] * len(cart)
        is_term = [int(x) for x in table.column(_IS_TERM).to_pylist()] if _IS_TERM in table.column_names else [0] * len(cart)

        # --- 1. world -> env-local base: subtract env origin (mean EE x,y; z=0) ---
        env_origin = np.array([cart[:, 0].mean(), cart[:, 1].mean(), 0.0])
        cart_base = cart.copy()
        cart_base[:, :3] = cart[:, :3] - env_origin   # positions to base frame; orientation (3:6) unchanged

        # --- 2. camera extrinsic: banana convention -> droid camera_pose_in_base equivalent ---
        ext_path = episode.metadata_path.parent / "extrinsics.json"
        with open(ext_path, "r", encoding="utf-8") as f:
            ext = json.load(f)
        cam = ext["camera"]
        e6 = np.asarray(cam["cam2base_extrinsics_6d"], dtype=np.float64)
        t_cam = e6[:3]
        Rcb = Rotation.from_euler("xyz", e6[3:6]).as_matrix() @ _AXIS_CORRECTION   # base->camera rotation
        # droid does p_cam = Rot(euler')^T (p-t); we want Rot(euler')^T = Rcb -> euler' = eulerOf(Rcb.T)
        euler_equiv = Rotation.from_matrix(Rcb.T).as_euler("xyz")
        extrinsic_6d = [float(t_cam[0]), float(t_cam[1]), float(t_cam[2]),
                        float(euler_equiv[0]), float(euler_equiv[1]), float(euler_equiv[2])]

        # --- 3. intrinsics from the clip's own files ---
        im = cam.get("intrinsic_matrix")
        if im is not None:
            fx, cx, fy, cy = float(im[0][0]), float(im[0][2]), float(im[1][1]), float(im[1][2])
            W, H = 1280, 720
            intr_path = episode.metadata_path.parent / "intrinsics.json"
            if intr_path.exists():
                ij = json.load(open(intr_path)).get(episode.episode_id, {}).get("rgb", {})
                W = int(ij.get("width", W)); H = int(ij.get("height", H))
        else:
            fx = fy = 524.0; cx, cy = 640.0, 360.0; W, H = 1280, 720

        serial = "rgb"
        calibration = {
            "extrinsics": {serial: extrinsic_6d},
            "intrinsics": {serial: {"cameraMatrix": [fx, cx, fy, cy], "width": W, "height": H}},
        }
        payload = {
            "num_steps": int(len(cart_base)),
            "observations": {"cartesian_position": cart_base, "gripper_position": grip},
            "calibration": calibration,
            "image_shape": [H, W, 3],
            "is_last": is_last,
            "is_terminal": is_term,
            "_serial": serial,
            "_env_origin": env_origin.tolist(),
        }
        if len(self._payload_cache) < 256:
            self._payload_cache[episode.episode_id] = payload
        return payload
