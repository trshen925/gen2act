"""Adapter for the droid-ex-3000-out dataset (VLM+motion filtered DROID 1.0.1 clips).

Format differs from the legacy droid-ex (`data.json` + `frames/`):
  - <clip>/data.parquet   — DROID fields; array cells are JSON strings
  - <clip>/meta.json      — num_frames, camera, task, provenance.episode_dir
  - <clip>/rgb.mp4        — RGB video (num_frames frames)
  - <clip>/depth.npz      — precomputed depth (unused by C19)
Extrinsics live in the SOURCE episode dir (meta.provenance.episode_dir/extrinsics.json),
intrinsics in a shared KarlP-droid/intrinsics.json indexed episode_id -> serial.

This adapter assembles the legacy `payload` dict on the fly so all the camera-frame
action mappings / projection / quality filters in OpenXDroidDataset work unchanged.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from r2r_gen2act.data.adapters.openx_droid import OpenXDroidDataset
from r2r_gen2act.data.split import split_episode_ids
from r2r_gen2act.data.types import EpisodeRecord

# columns we need from the parquet (array fields are stored as JSON strings)
_OBS_CART = "steps/observation/cartesian_position"
_OBS_GRIP = "steps/observation/gripper_position"
_IS_LAST = "steps/is_last"
_IS_TERM = "steps/is_terminal"

_CAMERA_KEY_BY_VIDEO = {
    "exterior_image_1_left": "exterior_1",
    "exterior_image_2_left": "exterior_2",
}


def _json_col(table, name: str) -> list:
    """Decode a parquet column whose cells may be JSON-string arrays."""
    out = []
    for x in table.column(name).to_pylist():
        out.append(json.loads(x) if isinstance(x, str) else x)
    return out


class DroidExOutDataset(OpenXDroidDataset):
    def __init__(self, cfg: dict, split: str = "train") -> None:
        # shared intrinsics table (episode_id -> serial -> {cameraMatrix, width, height})
        self._intrinsics_path = str(cfg["data"].get(
            "intrinsics_json", "/mnt/pfs/data/shentingrui/KarlP-droid/intrinsics.json"))
        self._intrinsics_all: dict | None = None
        self._payload_cache: dict[str, dict] = {}
        wrist_cfg = cfg["data"].get("wrist_current", {}) or {}
        self._wrist_raw_root = Path(str(wrist_cfg.get(
            "raw_root", "/mnt/pfs/data/fenghaoran/droid/decompressed/1.0.1")))
        self._wrist_video_name = str(wrist_cfg.get(
            "video_name", "steps_observation_wrist_image_left.mp4"))
        super().__init__(cfg, split=split)

    # ── intrinsics loaded lazily & cached (125 MB json, load once) ──────────
    def _intrinsics(self) -> dict:
        if self._intrinsics_all is None:
            with open(self._intrinsics_path, "r", encoding="utf-8") as f:
                self._intrinsics_all = json.load(f)
        return self._intrinsics_all

    def __getstate__(self):
        # don't pickle the big intrinsics dict / caches into dataloader workers
        state = self.__dict__.copy()
        state["_intrinsics_all"] = None
        state["_payload_cache"] = {}
        return state

    def _load_episodes(self) -> list[EpisodeRecord]:
        root = Path(self.data_cfg["root"])
        pattern = str(self.data_cfg.get("episode_glob", "[0-9][0-9][0-9][0-9][0-9]"))
        video_name = str(self.data_cfg.get("source_video_name", "rgb.mp4"))
        metadata_name = str(self.data_cfg.get("metadata_name", "meta.json"))
        dirs = sorted([p for p in root.glob(pattern)
                       if p.is_dir() and (p / metadata_name).exists() and (p / "data.parquet").exists()])
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
            src_ep_dir = meta.get("provenance", {}).get("episode_dir", "")
            ext_path = Path(src_ep_dir) / "extrinsics.json" if src_ep_dir else None
            if ext_path is None or not ext_path.exists():
                continue
            video = d / video_name
            if not video.exists():
                continue
            # skip clips whose pre-extracted frames dir is empty (corrupt mp4 → 0 frames)
            fsub = str(self.data_cfg.get("frames_subdir", "") or "")
            if fsub:
                fdir = d / fsub
                if not fdir.is_dir() or not any(fdir.iterdir()):
                    continue
            num_steps = int(meta["num_frames"])
            raw_episode_id = str(meta.get("episode_id", ""))
            wrist_video = self._wrist_raw_root / raw_episode_id / self._wrist_video_name
            source_range = meta.get("source_frame_range", [0, num_steps])
            if self.wrist_current_enabled and not wrist_video.exists():
                continue
            rec = EpisodeRecord(d.name, num_steps, video, video, d / metadata_name, split,
                                extra={
                                    "extrinsics_path": str(ext_path),
                                    "camera_name": str(meta.get("camera", "")),
                                    "source_frame_start": int(source_range[0]),
                                    "wrist_video_path": str(wrist_video),
                                })
            # camera_projection calibration + quality filters (inherited) read the assembled payload.
            # A bad extrinsics.json (missing exterior_1, ~2/35k) raises → skip that clip robustly.
            if self.proprioception_enabled and str(self.proprioception_cfg.get("source", "")) == "camera_projection":
                try:
                    payload = self._read_action_payload(rec)
                except (ValueError, KeyError):
                    continue
                if not self._has_camera_projection_calibration(payload) or not self._camera_projection_quality_ok(payload):
                    continue
            episodes.append(rec)
        mapping = self.cfg.get("action", {}).get("mapping", {}).get("type", "?")
        print(f"[DroidExOutDataset] action_mapping={mapping} split={self.split} episodes={len(episodes)}")
        return episodes

    def _read_action_payload(self, episode: EpisodeRecord) -> dict:
        cached = self._payload_cache.get(episode.episode_id)
        if cached is not None:
            return cached
        with open(episode.extra["extrinsics_path"], "r", encoding="utf-8") as f:
            ext = json.load(f)
        episode_id = ext.get("episode_id", "")
        cams = ext.get("cameras", {})
        selection = str(self.data_cfg.get("camera_selection", "legacy_exterior_1"))
        camera_name = str(episode.extra.get("camera_name", ""))
        if selection == "legacy_exterior_1":
            camera_key = "exterior_1"
        elif selection == "from_meta":
            camera_key = _CAMERA_KEY_BY_VIDEO.get(camera_name, "")
            if not camera_key:
                raise ValueError(
                    f"Unsupported meta.camera={camera_name!r} for clip {episode.episode_id}; "
                    f"known values={sorted(_CAMERA_KEY_BY_VIDEO)}"
                )
        else:
            raise ValueError(
                f"Unknown data.camera_selection={selection!r}; expected "
                "'legacy_exterior_1' or 'from_meta'"
            )
        if camera_key not in cams:
            raise ValueError(
                f"extrinsics.json for {episode.episode_id} has no {camera_key!r} camera "
                f"selected from meta.camera={camera_name!r} (keys={list(cams.keys())})"
            )
        cam = cams[camera_key]
        allowed_sources = self.data_cfg.get("allowed_camera_calibration_sources", [])
        if isinstance(allowed_sources, str):
            allowed_sources = [allowed_sources]
        allowed_sources = {str(v) for v in (allowed_sources or [])}
        calibration_source = str(cam.get("source", ""))
        if allowed_sources and calibration_source not in allowed_sources:
            raise ValueError(
                f"Camera calibration source {calibration_source!r} for clip {episode.episode_id} "
                f"is not in allowed_camera_calibration_sources={sorted(allowed_sources)}"
            )
        serial = str(cam["serial"])
        extrinsic_6d = [float(v) for v in cam["cam2base_extrinsics_6d"]]

        # Read the larger parquet only after camera/calibration validation, so rejected
        # predicted calibrations do not add avoidable shared-filesystem I/O.
        meta_dir = episode.metadata_path.parent
        table = pq.read_table(str(meta_dir / "data.parquet"))
        cart = np.asarray(_json_col(table, _OBS_CART), dtype=np.float64)          # [T,6]
        grip = np.asarray(table.column(_OBS_GRIP).to_pylist(), dtype=np.float64).reshape(-1, 1)  # [T,1]
        is_last = [int(x) for x in table.column(_IS_LAST).to_pylist()] if _IS_LAST in table.column_names else [0] * len(cart)
        is_term = [int(x) for x in table.column(_IS_TERM).to_pylist()] if _IS_TERM in table.column_names else [0] * len(cart)

        intr_ep = self._intrinsics().get(episode_id, {})
        intr = intr_ep.get(serial)
        if not isinstance(intr, dict) or "cameraMatrix" not in intr:
            # signal "no calibration" -> filtered out upstream
            calibration = {"extrinsics": {serial: extrinsic_6d}, "intrinsics": {}}
        else:
            calibration = {
                "extrinsics": {serial: extrinsic_6d},
                "intrinsics": {serial: {
                    "cameraMatrix": [float(x) for x in intr["cameraMatrix"]],
                    "width": int(intr.get("width", 1280)),
                    "height": int(intr.get("height", 720)),
                }},
            }
        H = int(intr.get("height", 720)) if isinstance(intr, dict) else 720
        W = int(intr.get("width", 1280)) if isinstance(intr, dict) else 1280

        payload = {
            "num_steps": int(len(cart)),
            "observations": {
                "cartesian_position": cart,
                "gripper_position": grip,
            },
            "calibration": calibration,
            "image_shape": [H, W, 3],
            "is_last": is_last,
            "is_terminal": is_term,
            "_serial": serial,
            "_camera_name": camera_name,
            "_camera_key": camera_key,
            "_camera_calibration_source": calibration_source,
            "_camera_calibration_metric_type": str(cam.get("metric_type", "")),
            "_camera_calibration_quality": cam.get("quality_metric"),
        }
        # keep cache small (dataloader workers each build their own)
        if len(self._payload_cache) < 256:
            self._payload_cache[episode.episode_id] = payload
        return payload
