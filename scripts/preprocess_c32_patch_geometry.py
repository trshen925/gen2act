#!/usr/bin/env python3
"""Convert FoundationStereo depth videos to compact DINO patch geometry sidecars."""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from r2r_gen2act.modeling.depth_lifting import DepthTo3DPatchGeometry


DEFAULT_DEPTH_ROOT = Path("/mnt/pfs/data/shentingrui/droid-ex-3000-foundation-depth")
DEFAULT_INTRINSICS = Path("/mnt/pfs/data/shentingrui/KarlP-droid/intrinsics.json")


def model_input_intrinsics(intr: dict, meta: dict, image_size: int) -> np.ndarray:
    fx, cx, fy, cy = [float(v) for v in intr["cameraMatrix"][:4]]
    # The depth generator records the exact rectified focal length it used.
    fx = float(meta.get("focal_px_fullres", fx))
    width = float(intr.get("width", 1280))
    height = float(intr.get("height", 720))
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid intrinsic resolution {width}x{height}")
    if height < width:
        new_h = image_size
        new_w = int(round(width * image_size / height))
    else:
        new_w = image_size
        new_h = int(round(height * image_size / width))
    sx, sy = new_w / width, new_h / height
    crop_left = max(0, (new_w - image_size) // 2)
    crop_top = max(0, (new_h - image_size) // 2)
    return np.asarray(
        [fx * sx, fy * sy, cx * sx - crop_left, cy * sy - crop_top],
        dtype=np.float32,
    )


def build_tasks(depth_root: Path, intrinsics_path: Path, geometry_name: str,
                overwrite: bool, max_clips: int | None, image_size: int) -> tuple[list[tuple], list[dict]]:
    with intrinsics_path.open("r", encoding="utf-8") as handle:
        intrinsics = json.load(handle)
    tasks = []
    skipped = []
    for depth_path in sorted(depth_root.glob("[0-9][0-9][0-9][0-9][0-9]/depth.npz")):
        out_path = depth_path.parent / geometry_name
        if out_path.exists() and not overwrite:
            continue
        try:
            meta = json.loads((depth_path.parent / "depth_meta.json").read_text())
            episode_id = str(meta["episode_id_karlp"])
            serial = str(meta["camera_serial"])
            intr = intrinsics[episode_id][serial]
            K = model_input_intrinsics(intr, meta, image_size)
        except (OSError, KeyError, TypeError, ValueError, ZeroDivisionError, json.JSONDecodeError) as exc:
            skipped.append({"path": str(depth_path), "reason": f"{type(exc).__name__}: {exc}"})
            continue
        tasks.append((str(depth_path), str(out_path), K.tolist()))
        if max_clips is not None and len(tasks) >= max_clips:
            break
    return tasks, skipped


def process_one(task: tuple, patch_size: int, image_size: int,
                max_depth_m: float, batch_size: int) -> tuple[str, str]:
    depth_path, out_path, K_values = task
    try:
        torch.set_num_threads(1)
        with np.load(depth_path, allow_pickle=False) as archive:
            depth = np.asarray(archive["depth_mm"], dtype=np.uint16)
            frame_indices = np.asarray(archive["frame_indices"], dtype=np.int64)
        if depth.ndim != 3 or len(frame_indices) != len(depth):
            raise ValueError(f"depth/frame shape mismatch: {depth.shape}, {frame_indices.shape}")
        if len(frame_indices) > 1 and not np.all(np.diff(frame_indices) == 1):
            raise ValueError("frame_indices are not contiguous")
        lifter = DepthTo3DPatchGeometry(patch_size, image_size, max_depth_m).eval()
        K = torch.tensor(K_values, dtype=torch.float32).unsqueeze(0)
        chunks = []
        with torch.inference_mode():
            for start in range(0, len(depth), batch_size):
                d = torch.from_numpy(depth[start:start + batch_size])
                geometry = lifter(d, K.expand(len(d), -1))
                chunks.append(geometry.cpu().numpy().astype(np.float16))
        result = np.concatenate(chunks, axis=0)
        destination = Path(out_path)
        tmp = destination.with_name(f".{destination.name}.{os.getpid()}.tmp.npy")
        np.save(tmp, result, allow_pickle=False)
        os.replace(tmp, destination)
        return depth_path, "ok"
    except Exception as exc:
        return depth_path, f"error: {type(exc).__name__}: {exc}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth-root", type=Path, default=DEFAULT_DEPTH_ROOT)
    parser.add_argument("--intrinsics-json", type=Path, default=DEFAULT_INTRINSICS)
    parser.add_argument("--geometry-name", default="patch_geometry_v1.npy")
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--patch-size", type=int, default=14)
    parser.add_argument("--max-depth-m", type=float, default=20.0)
    parser.add_argument("--max-clips", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    tasks, skipped = build_tasks(
        args.depth_root, args.intrinsics_json, args.geometry_name,
        args.overwrite, args.max_clips, args.image_size)
    log_dir = args.depth_root / "_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "patch_geometry_skipped.json").write_text(
        json.dumps(skipped, indent=2) + "\n")
    print(f"geometry tasks={len(tasks)} skipped_no_geometry={len(skipped)} workers={args.workers}", flush=True)
    failures = []
    worker_args = (args.patch_size, args.image_size, args.max_depth_m, args.batch_size)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(process_one, task, *worker_args) for task in tasks]
        for index, future in enumerate(futures, 1):
            path, status = future.result()
            if status != "ok":
                failures.append({"path": path, "status": status})
            if index % 100 == 0 or index == len(futures):
                print(f"processed={index}/{len(futures)} failures={len(failures)}", flush=True)
    failure_path = log_dir / "patch_geometry_failures.json"
    failure_path.write_text(json.dumps(failures, indent=2) + "\n")
    print(f"done failures={len(failures)} log={failure_path}")


if __name__ == "__main__":
    main()
