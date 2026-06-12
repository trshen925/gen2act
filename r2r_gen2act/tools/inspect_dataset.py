from __future__ import annotations

import json

import torch

from r2r_gen2act.data.factories import build_dataset, build_action_codec


def inspect_dataset(cfg: dict, split: str = "train", load_sample: bool = True) -> dict:
    ds = build_dataset(cfg, split)
    episode_ids = [e.episode_id for e in ds.episodes]
    report = {
        "split": split,
        "num_episodes": len(episode_ids),
        "num_windows": len(ds),
        "first_episodes": episode_ids[:5],
    }
    positives = 0
    checked = min(len(ds), int(cfg["data"].get("inspect_terminate_windows", 200)))
    for i in range(checked):
        sample = ds[i]
        positives += int(sample["terminate"].item())
    report["terminate_positive_in_checked_windows"] = positives
    if checked < len(ds) and positives == 0:
        tail_start = max(0, len(ds) - min(len(ds), 200))
        tail_positives = 0
        for i in range(tail_start, len(ds)):
            sample = ds[i]
            tail_positives += int(sample["terminate"].item())
        report["terminate_positive_in_tail_windows"] = tail_positives
    if load_sample and len(ds) > 0:
        sample = ds[0]
        codec = build_action_codec(cfg)
        bins = codec.discretize(sample["action"][: codec.pose_dims].unsqueeze(0))
        report["sample"] = {
            "episode_id": sample["episode_id"],
            "start_index": sample["start_index"],
            "target_step": sample["target_step"],
            "source_video_shape": list(sample["source_video"].shape),
            "target_history_shape": list(sample["target_history"].shape),
            "action_shape": list(sample["action"].shape),
            "action": sample["action"].tolist(),
            "action_bins": bins[0].tolist(),
            "gripper": int(sample["gripper"].item()),
            "terminate": int(sample["terminate"].item()),
        }
    return report


def print_report(report: dict) -> None:
    print(json.dumps(report, indent=2, ensure_ascii=False))
