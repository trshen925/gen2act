"""RoboLab parquet adapter for the C39 native joint-velocity task.

This is deliberately separate from ``robolab_sim``: the latter is kept
backwards-compatible with the older Cartesian RoboLab experiments.  RoboLab
parquets contain the same native fields as DROID, but use per-clip camera
calibration and local-coordinate conversion handled by the parent adapter.
"""
from __future__ import annotations

import json

import numpy as np
import pyarrow.parquet as pq

from r2r_gen2act.data.adapters.droid_ex_out import (
    _ACT_GRIP,
    _ACT_JOINT_POS,
    _ACT_JOINT_VEL,
    _OBS_JOINT,
)
from r2r_gen2act.data.adapters.robolab_sim import RobolabSimDataset


def _decode(values):
    return [json.loads(value) if isinstance(value, str) else value for value in values]


class RobolabC39Dataset(RobolabSimDataset):
    """RoboLab data with C39-compatible joint observations and targets."""

    def _load_episodes(self):
        episodes = super()._load_episodes()
        # The common RoboLab adapter only needs the external RGB stream. C39
        # additionally consumes the synchronized wrist stream.
        for episode in episodes:
            wrist = episode.metadata_path.parent / str(
                self.data_cfg.get("wrist_video_name", "wrist.mp4"))
            episode.extra["wrist_video_path"] = str(wrist)
            episode.extra["source_frame_start"] = 0
        return episodes

    def _read_action_payload(self, episode):
        payload = super()._read_action_payload(episode)
        table = pq.read_table(
            str(episode.metadata_path.parent / "data.parquet"),
            columns=[_OBS_JOINT, _ACT_JOINT_POS, _ACT_JOINT_VEL, _ACT_GRIP],
        )
        payload["observations"]["joint_position"] = np.asarray(
            _decode(table.column(_OBS_JOINT).to_pylist()), dtype=np.float64)
        payload["action_dict"] = {
            "joint_position": np.asarray(
                _decode(table.column(_ACT_JOINT_POS).to_pylist()), dtype=np.float64),
            "joint_velocity": np.asarray(
                _decode(table.column(_ACT_JOINT_VEL).to_pylist()), dtype=np.float64),
            "gripper_position": np.asarray(
                table.column(_ACT_GRIP).to_pylist(), dtype=np.float64).reshape(-1, 1),
        }
        return payload

    def _read_native_action_payload(self, episode):
        table = pq.read_table(
            str(episode.metadata_path.parent / "data.parquet"),
            columns=[_ACT_JOINT_VEL, _ACT_GRIP],
        )
        return {
            "action_dict": {
                "joint_velocity": np.asarray(
                    _decode(table.column(_ACT_JOINT_VEL).to_pylist()), dtype=np.float32),
                "gripper_position": np.asarray(
                    table.column(_ACT_GRIP).to_pylist(), dtype=np.float32).reshape(-1, 1),
            }
        }
