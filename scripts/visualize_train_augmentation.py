#!/usr/bin/env python3
"""Visualize the photometric augmentations used by a training configuration."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from r2r_gen2act.config.load import load_config
from r2r_gen2act.data.factories import build_dataset
from r2r_gen2act.data.transforms import apply_image_augmentation


def _to_hwc(frame: torch.Tensor) -> np.ndarray:
    return frame.detach().cpu().permute(1, 2, 0).numpy().clip(0.0, 1.0)


def _save_stream(name: str, original: torch.Tensor, augmented: torch.Tensor, output: Path, diff_gain: float) -> float:
    """Write original, augmented, and amplified absolute pixel difference for each frame."""
    frames = original.shape[0]
    fig, axes = plt.subplots(3, frames, figsize=(3.0 * frames, 9.0), squeeze=False)
    abs_diff = (augmented - original).abs()
    for index in range(frames):
        axes[0, index].imshow(_to_hwc(original[index]))
        axes[1, index].imshow(_to_hwc(augmented[index]))
        axes[2, index].imshow(_to_hwc(abs_diff[index] * diff_gain))
        axes[0, index].set_title(f"frame {index}")
        for row in range(3):
            axes[row, index].axis("off")
    axes[0, 0].set_ylabel("original", fontsize=12)
    axes[1, 0].set_ylabel("train aug", fontsize=12)
    axes[2, 0].set_ylabel(f"abs diff x{diff_gain:g}", fontsize=12)
    mean_abs = float(abs_diff.mean())
    fig.suptitle(f"{name}: mean |pixel delta| = {mean_abs:.4f}", fontsize=14)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return mean_abs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=ROOT / "configs/droidexFULL_C35_diffuse_gripper_fulltrain.yaml")
    parser.add_argument("--episode-id", default="00000", help="Five-digit clip ID.")
    parser.add_argument("--start-index", type=int, default=30, help="Target-window start index.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducible augmentations.")
    parser.add_argument("--diff-gain", type=float, default=5.0, help="Display multiplier for absolute difference.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/train_augmentation")
    args = parser.parse_args()

    cfg = load_config(args.config)
    # The validation split disables augmentation in sample_window while retaining the configured data layout.
    dataset = build_dataset(cfg, "val")
    sample = dataset.sample_window(args.episode_id, args.start_index)

    source = sample["source_video"]
    front = sample["target_history"]
    wrist = sample.get("wrist_current")
    torch.manual_seed(args.seed)
    # Mirror the exact C34/C35 training call order in WindowedRobotDataset.sample_window.
    source_aug = apply_image_augmentation(source.clone(), cfg["data"]["augmentation"])
    front_aug = apply_image_augmentation(front.clone(), cfg["data"]["augmentation"])
    wrist_aug = apply_image_augmentation(wrist.clone(), cfg["data"]["augmentation"]) if wrist is not None else None

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.episode_id}_start{args.start_index:05d}_seed{args.seed}"
    results = {
        "source": _save_stream("source demo (8 frames)", source, source_aug,
                               args.output_dir / f"{stem}_source.png", args.diff_gain),
        "front_history": _save_stream("front observation history", front, front_aug,
                                      args.output_dir / f"{stem}_front.png", args.diff_gain),
    }
    if wrist is not None and wrist_aug is not None:
        results["wrist_history"] = _save_stream("wrist observation history", wrist, wrist_aug,
                                                 args.output_dir / f"{stem}_wrist.png", args.diff_gain)
    print(f"episode={args.episode_id} start_index={sample['start_index']} target_step={sample['target_step']}")
    print("augmentation=", cfg["data"]["augmentation"])
    for stream, mean_abs in results.items():
        print(f"{stream}: mean_abs_pixel_delta={mean_abs:.6f}")
    print(f"wrote={args.output_dir}")


if __name__ == "__main__":
    main()
