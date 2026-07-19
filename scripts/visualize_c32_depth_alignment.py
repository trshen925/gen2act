#!/usr/bin/env python3
"""Save a C32 front RGB/depth/patch-geometry alignment diagnostic."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from r2r_gen2act.config.load import load_config
from r2r_gen2act.data.factories import build_dataset
from r2r_gen2act.data.transforms import image_to_tensor
from r2r_gen2act.modeling.depth_lifting import DepthTo3DPatchGeometry
from scripts.preprocess_c32_patch_geometry import model_input_intrinsics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path,
                        default=ROOT / "configs/droidexFULL_C32_wrist_frontdepth_cont12_lr3e5.yaml")
    parser.add_argument("--clip-id", default="00000")
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "artifacts/c32_depth_alignment/00000_frame000.png")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg["data"]["max_episodes"] = int(args.clip_id) + 1
    cfg["data"]["augmentation"]["enabled"] = False
    cfg["data"]["window_jitter"]["enabled"] = False
    dataset = build_dataset(cfg, "all")
    episode = next(ep for ep in dataset._episodes if ep.episode_id == args.clip_id)
    frame = int(np.clip(args.frame_index, 0, episode.num_steps - 1))

    clip_dir = Path(cfg["data"]["root"]) / args.clip_id
    rgb_path = clip_dir / cfg["data"].get("frames_subdir", "frames") / f"{frame:06d}.jpg"
    rgb = image_to_tensor(imageio.imread(rgb_path), int(cfg["data"]["image_size"])).permute(1, 2, 0).numpy()

    depth_dir = Path(cfg["data"]["depth"]["root"]) / args.clip_id
    with np.load(depth_dir / "depth.npz", allow_pickle=False) as archive:
        depth_mm = np.asarray(archive["depth_mm"][frame], dtype=np.uint16)
        source_frame = int(archive["frame_indices"][frame])
    meta = json.loads((depth_dir / "depth_meta.json").read_text())
    if source_frame != int(meta["source_frame_range"][0]) + frame:
        raise ValueError("Depth frame mapping does not match clip-local RGB frame")
    payload = dataset._read_action_payload(episode)
    serial = str(meta["camera_serial"])
    intr = payload["calibration"]["intrinsics"][serial]
    K = model_input_intrinsics(intr, meta, int(cfg["data"]["image_size"]))

    lifter = DepthTo3DPatchGeometry(14, int(cfg["data"]["image_size"]), 20.0)
    depth_m = torch.from_numpy(depth_mm.astype(np.float32)).unsqueeze(0) / 1000.0
    depth_crop, valid_crop = lifter._resize_center_crop_valid(depth_m)
    depth_crop = depth_crop[0, 0].numpy()
    valid_crop = valid_crop[0, 0].numpy()
    geometry = np.load(depth_dir / cfg["data"]["depth"]["geometry_name"], mmap_mode="r")[frame].astype(np.float32)
    patch_z = geometry[:, 2].reshape(16, 16)
    patch_valid = geometry[:, 3].reshape(16, 16)

    ee = dataset._camera_abs_pose_at(payload, frame)[:3]
    ee_u = K[0] * ee[0] / ee[2] + K[2]
    ee_v = K[1] * ee[1] / ee[2] + K[3]

    positive = depth_crop[valid_crop > 0]
    vmax = float(np.quantile(positive, 0.98)) if len(positive) else 2.0
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    axes[0, 0].imshow(rgb)
    axes[0, 0].scatter([ee_u], [ee_v], s=90, facecolors="none", edgecolors="lime", linewidths=2)
    axes[0, 0].set_title(f"front RGB + EE, local={frame}, source={source_frame}")
    axes[0, 1].imshow(depth_crop, cmap="turbo", vmin=0.2, vmax=vmax)
    axes[0, 1].scatter([ee_u], [ee_v], s=90, facecolors="none", edgecolors="lime", linewidths=2)
    axes[0, 1].set_title("valid-aware depth after RGB crop (m)")
    axes[0, 2].imshow(valid_crop, cmap="gray", vmin=0, vmax=1)
    axes[0, 2].set_title("resized valid mask")
    axes[1, 0].imshow(patch_z, cmap="turbo", vmin=0.2, vmax=vmax)
    axes[1, 0].set_title("16x16 lifted patch Z (m)")
    axes[1, 1].imshow(patch_valid, cmap="gray", vmin=0, vmax=1)
    axes[1, 1].set_title("16x16 patch valid ratio")
    axes[1, 2].imshow(rgb)
    gy, gx = np.gradient(depth_crop)
    edge = np.hypot(gx, gy)
    threshold = np.quantile(edge[valid_crop > 0], 0.97)
    axes[1, 2].imshow(edge > threshold, cmap="Reds", alpha=0.35)
    axes[1, 2].set_title("depth discontinuities over RGB")
    for ax in axes.flat:
        ax.axis("off")
    fig.suptitle(
        f"C32 {args.clip_id}: EE camera XYZ={np.round(ee, 3).tolist()}, K224={np.round(K, 2).tolist()}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    print(args.output)


if __name__ == "__main__":
    main()
