#!/usr/bin/env python3
"""Event-aware held-out evaluation for diffused-gripper policies.

An event is read from the raw DROID gripper observation at its original 15 Hz
resolution: close is open (>0.5) -> closed (<=0.5); release is the inverse.
The eight predicted action states are at current_step + [5, 10, ..., 40].
"""
from __future__ import annotations

import argparse
import gc
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from r2r_gen2act.config.load import load_config
from r2r_gen2act.data.factories import build_action_codec, build_dataset
from r2r_gen2act.modeling.factory import build_policy
from r2r_gen2act.training.checkpoint import load_checkpoint


class IndexedDataset(Dataset):
    def __init__(self, dataset: Dataset, indices: list[int]) -> None:
        self.dataset, self.indices = dataset, indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict:
        sample = self.dataset[self.indices[index]]
        sample["event_eval_index"] = index
        return sample


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _events(dataset) -> dict[str, list[dict]]:
    result: dict[str, list[dict]] = {}
    for episode in dataset._episodes:
        payload = dataset._read_action_payload(episode)
        values = np.asarray(payload["observations"]["gripper_position"], dtype=np.float32).reshape(-1)
        opened = values > 0.5
        transitions = np.flatnonzero(opened[1:] != opened[:-1]) + 1
        result[episode.episode_id] = [
            {"step": int(step), "kind": "release" if bool(opened[step]) else "close"}
            for step in transitions
        ]
    return result


def _select_event_windows(dataset, horizon: int) -> tuple[list[int], list[dict], dict]:
    """Select one-horizon pre, covering, and post windows around unambiguous events."""
    events_by_id = _events(dataset)
    selected: list[tuple[int, dict]] = []
    counts = defaultdict(int)
    for dataset_index, (episode_id, start) in enumerate(dataset._samples):
        current = int(start) + dataset.target_history_len - 1 + dataset.target_offset
        nearby = [event for event in events_by_id[episode_id] if abs(event["step"] - current) <= 2 * horizon]
        if not nearby:
            continue
        covering = [event for event in nearby if current < event["step"] <= current + horizon]
        if len(covering) == 1:
            event, category = covering[0], "cover"
        elif len(covering) > 1:
            continue
        else:
            event = min(nearby, key=lambda item: abs(item["step"] - current))
            if current + horizon < event["step"] <= current + 2 * horizon:
                category = "pre"
            elif event["step"] <= current <= event["step"] + horizon:
                category = "post"
            else:
                continue
        record = {
            "dataset_index": dataset_index,
            "episode_id": episode_id,
            "current_step": current,
            "event_step": event["step"],
            "event_kind": event["kind"],
            "category": category,
        }
        selected.append((dataset_index, record))
        counts[f"{event['kind']}_{category}"] += 1

    # A timing value should be counted once per event, from the latest valid
    # current frame before it. This produces the shortest prediction lead time.
    canonical: dict[tuple[str, int], int] = {}
    for position, (_, record) in enumerate(selected):
        if record["category"] != "cover":
            continue
        key = (record["episode_id"], record["event_step"])
        if key not in canonical or record["current_step"] > selected[canonical[key]][1]["current_step"]:
            canonical[key] = position
    for position in canonical.values():
        selected[position][1]["timing_window"] = True
    for _, record in selected:
        record.setdefault("timing_window", False)

    return [item[0] for item in selected], [item[1] for item in selected], dict(counts)


def _predict(model, batch: dict, codec, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    source = batch["source_video"].to(device, non_blocking=True)
    history = batch["target_history"].to(device, non_blocking=True)
    prop = batch.get("proprioception")
    prop = prop.to(device, non_blocking=True) if torch.is_tensor(prop) else None
    point_track = batch.get("point_track")
    point_track = point_track.to(device, non_blocking=True) if torch.is_tensor(point_track) else None
    kwargs = {}
    for key in ("point_track_causal", "source_dt", "wrist_current", "front_geometry"):
        value = batch.get(key)
        if torch.is_tensor(value):
            kwargs[key] = value.to(device, non_blocking=True)
    output = model(source, history, prop, None, point_track, **kwargs)
    action = output["action_pred"]
    pose = codec.unnormalize(action[..., :codec.pose_dims])
    gripper_open = ((action[..., codec.pose_dims] + 1.0) * 0.5).clamp(0.0, 1.0)
    return pose.float().cpu().numpy(), gripper_open.float().cpu().numpy()


def _binary_metrics(pred: np.ndarray, truth: np.ndarray) -> dict:
    tp = int(np.logical_and(pred, truth).sum())
    fp = int(np.logical_and(pred, ~truth).sum())
    fn = int(np.logical_and(~pred, truth).sum())
    tn = int(np.logical_and(~pred, ~truth).sum())
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    return {"n": int(len(pred)), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "recall": recall,
            "accuracy": (tp + tn) / len(pred) if len(pred) else None}


def _summarize(rows: list[dict]) -> dict:
    result: dict = {}
    for kind in ("close", "release"):
        kind_rows = [row for row in rows if row["event_kind"] == kind]
        group = {}
        for category in ("pre", "cover", "post"):
            part = [row for row in kind_rows if row["category"] == category]
            pred = np.concatenate([row["pred_open"] for row in part]) if part else np.empty(0, bool)
            truth = np.concatenate([row["true_open"] for row in part]) if part else np.empty(0, bool)
            # For a close, the destination state is closed; for a release, open.
            destination_pred = ~pred if kind == "close" else pred
            destination_truth = ~truth if kind == "close" else truth
            xyz = np.concatenate([row["xyz_error_cm"] for row in part]) if part else np.empty((0, 3))
            value = {
                "windows": len(part),
                "destination_state": "closed" if kind == "close" else "open",
                "destination_precision_recall": _binary_metrics(destination_pred, destination_truth),
                "xyz_mae_cm": xyz.mean(axis=0).tolist() if len(xyz) else None,
                "xyz_mae_cm_mean": float(xyz.mean()) if len(xyz) else None,
            }
            if category == "cover" and part:
                before = np.concatenate([row["future_steps"] < row["event_step"] for row in part])
                value["xyz_mae_before_event_cm_mean"] = float(xyz[before].mean()) if before.any() else None
                value["xyz_mae_after_event_cm_mean"] = float(xyz[~before].mean()) if (~before).any() else None
            group[category] = value

        timing_rows = [row for row in kind_rows if row["timing_window"]]
        timing_error = [row["timing_error_frames"] for row in timing_rows if row["timing_error_frames"] is not None]
        group["timing"] = {
            "events": len(timing_rows),
            "detected": len(timing_error),
            "miss_rate": 1.0 - len(timing_error) / len(timing_rows) if timing_rows else None,
            "mean_error_frames": float(np.mean(timing_error)) if timing_error else None,
            "mean_abs_error_frames": float(np.mean(np.abs(timing_error))) if timing_error else None,
            "median_abs_error_frames": float(np.median(np.abs(timing_error))) if timing_error else None,
            "mean_abs_error_ms": float(np.mean(np.abs(timing_error)) / 15.0 * 1000.0) if timing_error else None,
        }
        result[kind] = group
    return result


def _evaluate(config_path: Path, checkpoint: Path, args) -> dict:
    cfg = load_config(config_path)
    dataset = build_dataset(cfg, "val")
    horizon = int(cfg["data"]["future_horizon"]) * int(cfg["action"]["chunk_size"])
    indices, records, selected_counts = _select_event_windows(dataset, horizon)
    if args.max_windows and len(indices) > args.max_windows:
        indices, records = indices[:args.max_windows], records[:args.max_windows]
    if not indices:
        raise RuntimeError("No event-adjacent validation windows selected")
    print(f"[{config_path.stem}] selected={len(indices)} horizon={horizon}f ({horizon / 15:.2f}s) counts={selected_counts}", flush=True)

    loader = DataLoader(IndexedDataset(dataset, indices), batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=args.device.startswith("cuda"))
    device = torch.device(args.device)
    codec = build_action_codec(cfg)
    model = build_policy(cfg).to(device)
    load_checkpoint(checkpoint, model, device, strict=False)
    model.eval()
    rows: list[dict | None] = [None] * len(records)
    with torch.no_grad():
        for batch in loader:
            pose, open_prob = _predict(model, batch, codec, device)
            positions = batch["event_eval_index"].tolist()
            for batch_pos, record_pos in enumerate(positions):
                record = dict(records[int(record_pos)])
                future_steps = record["current_step"] + np.arange(1, pose.shape[1] + 1) * int(cfg["data"]["future_horizon"])
                episode = dataset._episode_by_id[record["episode_id"]]
                payload = dataset._read_action_payload(episode)
                raw_open = np.asarray(payload["observations"]["gripper_position"], dtype=np.float32).reshape(-1) > 0.5
                true_open = raw_open[np.clip(future_steps, 0, len(raw_open) - 1)]
                pred_open = open_prob[batch_pos] > 0.5
                target_xyz = batch["action"][batch_pos, :, :3].numpy()
                record.update({
                    "future_steps": future_steps,
                    "pred_open": pred_open,
                    "true_open": true_open,
                    "xyz_error_cm": np.abs(pose[batch_pos, :, :3] - target_xyz) * 100.0,
                    "timing_error_frames": None,
                })
                if record["timing_window"]:
                    destination_open = record["event_kind"] == "release"
                    hit = np.flatnonzero(pred_open == destination_open)
                    if len(hit):
                        record["timing_error_frames"] = int(future_steps[int(hit[0])] - record["event_step"])
                rows[int(record_pos)] = record
    assert all(row is not None for row in rows)
    rows = [row for row in rows if row is not None]
    metrics = _summarize(rows)
    del model, loader, dataset
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {"config": str(config_path), "checkpoint": str(checkpoint), "horizon_frames": horizon,
            "horizon_seconds": horizon / 15.0, "selected_windows": len(rows),
            "selected_counts": selected_counts, "metrics": metrics}


def _format(value) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _write_markdown(path: Path, report: dict) -> None:
    lines = ["# Event-Aware Gripper Validation", "",
             "Events use raw 15 Hz gripper state: close=open to closed, release=closed to open.",
             "A cover window has the event in `(current, current+horizon]`; timing uses one latest-pre-event window per event.", ""]
    for name, result in report["models"].items():
        lines += [f"## {name}", "", f"Horizon: {result['horizon_frames']} frames ({result['horizon_seconds']:.2f} s)",
                  f"Selected event-adjacent windows: {result['selected_windows']}", "",
                  "| event | subset | windows | destination precision | destination recall | XYZ MAE (cm) |", "|---|---:|---:|---:|---:|---:|"]
        for kind, values in result["metrics"].items():
            for category in ("pre", "cover", "post"):
                value = values[category]
                pr = value["destination_precision_recall"]
                lines.append(f"| {kind} | {category} | {value['windows']} | {_format(pr['precision'])} | {_format(pr['recall'])} | {_format(value['xyz_mae_cm_mean'])} |")
            timing = values["timing"]
            cover = values["cover"]
            lines += ["", f"{kind} timing: events={timing['events']}, detected={timing['detected']}, "
                      f"miss={_format(timing['miss_rate'])}, MAE={_format(timing['mean_abs_error_frames'])} frames "
                      f"({_format(timing['mean_abs_error_ms'])} ms), signed mean={_format(timing['mean_error_frames'])} frames.",
                      f"{kind} cover-window XYZ: before event={_format(cover.get('xyz_mae_before_event_cm_mean'))} cm, "
                      f"after event={_format(cover.get('xyz_mae_after_event_cm_mean'))} cm.", ""]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--c35-config", type=Path, default=ROOT / "configs/droidexFULL_C35_diffuse_gripper_eval32.yaml")
    parser.add_argument("--c35-checkpoint", type=Path, default=ROOT / "outputs/droidexFULL_C35_diffuse_gripper_fulltrain/latest.pt")
    parser.add_argument("--c36-config", type=Path, default=ROOT / "configs/droidexFULL_C36_no_frontdepth_diffuse_gripper_eval32.yaml")
    parser.add_argument("--c36-checkpoint", type=Path, default=ROOT / "outputs/droidexFULL_C36_no_frontdepth_diffuse_gripper_fulltrain/latest.pt")
    parser.add_argument("--model-name", default="", help="Evaluate one model under this display name.")
    parser.add_argument("--config", type=Path, default=None, help="Single-model evaluation config.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Single-model evaluation checkpoint.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--max-windows", type=int, default=0, help="0 means all selected event windows.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/event_aware_c35_c36_eval32_seed0")
    args = parser.parse_args()
    _seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {"seed": args.seed, "definition": {"threshold": 0.5, "fps": 15,
              "close": ">0.5 to <=0.5", "release": "<=0.5 to >0.5"}, "models": {}}
    if (args.config is None) != (args.checkpoint is None):
        raise ValueError("--config and --checkpoint must be provided together")
    models = ((args.model_name or "model", args.config, args.checkpoint),) if args.config else (
        ("C35", args.c35_config, args.c35_checkpoint), ("C36", args.c36_config, args.c36_checkpoint))
    for name, config, checkpoint in models:
        _seed(args.seed)
        report["models"][name] = _evaluate(config, checkpoint, args)
    json_path = args.output_dir / "event_aware_report.json"
    md_path = args.output_dir / "event_aware_report.md"
    json_path.write_text(json.dumps(report, indent=2, default=lambda value: value.item() if isinstance(value, np.generic) else str(value)) + "\n")
    _write_markdown(md_path, report)
    print(f"json={json_path}")
    print(f"markdown={md_path}")


if __name__ == "__main__":
    main()
