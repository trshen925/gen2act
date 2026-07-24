#!/usr/bin/env python3
"""Evaluate action error against source-video duration on held-out windows."""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from r2r_gen2act.config.load import load_config
from r2r_gen2act.data.factories import build_action_codec, build_dataset
from r2r_gen2act.modeling.factory import build_policy
from r2r_gen2act.training.checkpoint import load_checkpoint


def _source_span_seconds(dataset, episode_ids: list[str], fps: float) -> np.ndarray:
    lengths = []
    for episode_id in episode_ids:
        episode = dataset._episode_by_id[str(episode_id)]
        # Validation has no source_float crop: the 8 linspace frames span this full source video.
        frame_count = dataset._clip_length(episode.source_video_path) or episode.num_steps
        lengths.append(max(0, frame_count - 1) / fps)
    return np.asarray(lengths, dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--max-windows", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cfg = load_config(args.config)
    codec = build_action_codec(cfg)
    pose_dims = codec.pose_dims
    device = torch.device("cuda")
    dataset = build_dataset(cfg, "val")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)
    model = build_policy(cfg).to(device)
    load_checkpoint(args.checkpoint, model, device, strict=False)
    model.eval()

    errors, spans = [], []
    window_count = 0
    with torch.no_grad():
        for batch in loader:
            source = batch["source_video"].to(device)
            target = batch["target_history"].to(device)
            prop = batch.get("proprioception")
            prop = prop.to(device) if torch.is_tensor(prop) else None
            kwargs = {}
            for key in ("source_dt", "wrist_current", "front_geometry", "point_track_causal"):
                value = batch.get(key)
                if torch.is_tensor(value):
                    kwargs[key] = value.to(device)
            point_track = batch.get("point_track")
            point_track = point_track.to(device) if torch.is_tensor(point_track) else None
            pred = model(source, target, prop, None, point_track, **kwargs)["action_pred"]
            pose = codec.unnormalize(pred[..., :pose_dims])
            truth = batch["action"][..., :pose_dims]
            # One XYZ MAE per window, averaged across its eight action-chunk steps and axes.
            errors.append((pose[..., :3].cpu() - truth[..., :3]).abs().mean(dim=(1, 2)).numpy() * 100.0)
            spans.append(_source_span_seconds(dataset, [str(x) for x in batch["episode_id"]], float(cfg["data"]["fps"])))
            window_count += source.shape[0]
            if window_count >= args.max_windows:
                break

    error = np.concatenate(errors)[:args.max_windows]
    span = np.concatenate(spans)[:len(error)]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(args.output_dir / "error_vs_source_length.npz", xyz_mae_cm=error, source_span_sec=span)

    # Quantile bins keep each boxplot similarly populated despite a long-tailed duration distribution.
    edges = np.unique(np.quantile(span, np.linspace(0.0, 1.0, 6)))
    groups, labels, summary = [], [], []
    for index, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (span >= left) & ((span < right) if index < len(edges) - 2 else (span <= right))
        values = error[mask]
        groups.append(values)
        labels.append(f"{left:.1f}-{right:.1f}s\\n(n={len(values)})")
        summary.append((left, right, len(values), values.mean(), np.median(values)))

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), constrained_layout=True)
    axes[0].scatter(span, error, s=9, alpha=0.35, edgecolors="none")
    axes[0].set_xlabel("source video / sampled-span duration (s)")
    axes[0].set_ylabel("per-window XYZ MAE (cm)")
    axes[0].set_title(f"C35 ep10: error vs source duration (n={len(error)})")
    axes[0].grid(alpha=0.25)
    axes[1].boxplot(groups, tick_labels=labels, showfliers=False)
    axes[1].set_xlabel("duration quantile bin")
    axes[1].set_ylabel("per-window XYZ MAE (cm)")
    axes[1].set_title("Equal-count duration bins; whiskers exclude outliers")
    axes[1].grid(axis="y", alpha=0.25)
    figure_path = args.output_dir / "error_vs_source_length.png"
    fig.savefig(figure_path, dpi=180)
    print(f"overall_xyz_mae_cm={error.mean():.4f}")
    print("duration_bin_sec,n,mean_xyz_mae_cm,median_xyz_mae_cm")
    for left, right, count, mean, median in summary:
        print(f"{left:.3f}-{right:.3f},{count},{mean:.4f},{median:.4f}")
    print(f"wrote={figure_path}")


if __name__ == "__main__":
    main()
