#!/usr/bin/env python3
"""Materialize the "decompressed DROID 1.0.1" layout that raw_droid expects.

``r2r_gen2act/data/adapters/raw_droid.py`` and
``scripts/build_raw_droid_pi05_manifest.py`` read this layout::

    <root>/episode_NNNNNN/episode.parquet
    <root>/episode_NNNNNN/metadata.json
    <root>/episode_NNNNNN/steps_observation_exterior_image_1_left.mp4
    <root>/episode_NNNNNN/steps_observation_exterior_image_2_left.mp4
    <root>/episode_NNNNNN/steps_observation_wrist_image_left.mp4

The original was produced from the RLDS/TFDS release on another cluster. This
machine only has the raw DROID release
(``<lab>/<success|failure>/<date>/<timestamp>/`` with ``trajectory.h5`` and
``recordings/MP4/<serial>.mp4``), so this script rebuilds the same layout from
raw episodes instead. It writes only what the manifest builder and the C40
adapter actually read:

  * six parquet columns (joint position/velocity, both gripper channels,
    is_last/is_terminal) -- C40 conditions on joint state, so no camera
    calibration or cartesian pose is needed;
  * a metadata.json whose ``context`` carries the canonical
    ``r2d2-data-full/<lab>/success/...`` paths the manifest parses back into an
    episode identity, plus ``num_steps`` matching the parquet row count;
  * the three videos rescaled to 320x180, the resolution the RLDS-derived
    dataset used and the one C40's letterbox assumes.

Episode numbering is assigned from the sorted raw relative path, so the ids are
stable across runs and shards. Already-finished episodes are skipped, making the
job resumable.

    python scripts/build_decompressed_droid_from_raw.py --out <ROOT> --workers 64
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

RAW_ROOT = Path("/mnt/project/public/public_datasets/droid_raw/droid_raw/1.0.1")
# The canonical prefix the manifest builder looks for when recovering identity.
GS_PREFIX = "gs://xembodiment_data/r2d2/r2d2-data-full"
FRAME_W, FRAME_H = 320, 180

VIDEO_TARGETS = {
    "ext1_cam_serial": "steps_observation_exterior_image_1_left.mp4",
    "ext2_cam_serial": "steps_observation_exterior_image_2_left.mp4",
    "wrist_cam_serial": "steps_observation_wrist_image_left.mp4",
}


def _ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def _ffprobe_exe() -> str:
    exe = _ffmpeg_exe()
    candidate = Path(exe).with_name("ffprobe")
    return str(candidate) if candidate.exists() else "ffprobe"


def _count_frames(video: Path) -> int:
    """Frame count without decoding pixels; falls back to a full packet count."""
    # Container metadata first: these DROID MP4s carry a correct nb_frames, and
    # -count_frames would decode every video a second time.
    for args in (
        ["-show_entries", "stream=nb_frames"],
        ["-count_frames", "-show_entries", "stream=nb_read_frames"],
    ):
        cmd = [_ffprobe_exe(), "-v", "error", "-select_streams", "v:0", *args,
               "-of", "default=nw=1:nk=1", str(video)]
        try:
            out = subprocess.run(cmd, capture_output=True, timeout=600, check=False)
        except subprocess.TimeoutExpired:
            return -1
        text = out.stdout.decode().strip().splitlines()
        if text and text[0].isdigit():
            return int(text[0])
    return -1


def _transcode(src: Path, dst: Path) -> bool:
    cmd = [
        _ffmpeg_exe(), "-nostdin", "-loglevel", "error", "-y", "-i", str(src),
        "-vf", f"scale={FRAME_W}:{FRAME_H}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-pix_fmt", "yuv420p", "-an", str(dst),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=1800, check=False)
    except subprocess.TimeoutExpired:
        return False
    return proc.returncode == 0 and dst.exists() and dst.stat().st_size > 0


def _read_h5(h5_path: Path) -> dict | None:
    import h5py

    try:
        with h5py.File(str(h5_path), "r") as f:
            obs, act = f["observation/robot_state"], f["action"]
            return {
                "joint_position": np.asarray(obs["joint_positions"], dtype=np.float32),
                "gripper_position": np.asarray(obs["gripper_position"], dtype=np.float32).reshape(-1),
                "action_joint_velocity": np.asarray(act["joint_velocity"], dtype=np.float32),
                "action_gripper_position": np.asarray(act["gripper_position"], dtype=np.float32).reshape(-1),
            }
    except (OSError, KeyError):
        return None


def _write_parquet(arrays: dict, n: int, path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    is_last = [0] * n
    is_last[-1] = 1
    table = pa.table({
        "steps/observation/joint_position": [row.tolist() for row in arrays["joint_position"][:n]],
        "steps/observation/gripper_position": arrays["gripper_position"][:n].tolist(),
        "steps/action_dict/joint_velocity": [row.tolist() for row in arrays["action_joint_velocity"][:n]],
        "steps/action_dict/gripper_position": arrays["action_gripper_position"][:n].tolist(),
        "steps/is_last": is_last,
        "steps/is_terminal": list(is_last),
    })
    pq.write_table(table, str(path), compression="zstd")


def _build_one(job: tuple) -> tuple[str, str]:
    episode_name, rel, out_root_s = job
    out_root = Path(out_root_s)
    out_dir = out_root / episode_name
    if (out_dir / "metadata.json").is_file():
        return episode_name, "skipped"

    raw_dir = RAW_ROOT / rel
    metas = list(raw_dir.glob("metadata_*.json"))
    if not metas:
        return episode_name, "no_raw_metadata"
    try:
        raw_meta = json.loads(metas[0].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return episode_name, "bad_raw_metadata"

    arrays = _read_h5(raw_dir / "trajectory.h5")
    if arrays is None:
        return episode_name, "bad_h5"

    sources: dict[str, Path] = {}
    for serial_key, target_name in VIDEO_TARGETS.items():
        serial = str(raw_meta.get(serial_key, "") or "")
        if not serial:
            return episode_name, f"missing_serial:{serial_key}"
        src = raw_dir / "recordings" / "MP4" / f"{serial}.mp4"
        if not src.is_file() or src.stat().st_size <= 0:
            return episode_name, f"missing_video:{serial_key}"
        sources[target_name] = src

    # Trajectory rows and every video must agree, so window indices never run
    # past the end of a stream.
    n = int(len(arrays["joint_position"]))
    for src in sources.values():
        count = _count_frames(src)
        if count <= 0:
            return episode_name, "unreadable_video"
        n = min(n, count)
    if n < 16:
        return episode_name, "too_short"

    tmp_dir = out_root / f".tmp_{episode_name}"
    shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    try:
        for target_name, src in sources.items():
            if not _transcode(src, tmp_dir / target_name):
                shutil.rmtree(tmp_dir, ignore_errors=True)
                return episode_name, "transcode_failed"
        _write_parquet(arrays, n, tmp_dir / "episode.parquet")
        base = f"{GS_PREFIX}/{rel}"
        metadata = {
            "num_steps": n,
            "episode_uuid_hint": raw_meta.get("uuid", ""),
            "context": {
                # The manifest builder recovers episode identity from these and
                # requires "/success/" plus the r2d2-data-full marker.
                "episode_metadata/file_path": f"{base}/trajectory.h5",
                "episode_metadata/recording_folderpath": f"{base}/recordings/MP4",
            },
            "provenance": {
                "raw_episode_dir": str(raw_dir),
                "relative_path": rel,
                "rebuilt_from": "droid_raw",
                "frame_size": [FRAME_H, FRAME_W],
            },
        }
        (tmp_dir / "metadata.json").write_text(json.dumps(metadata, indent=1), encoding="utf-8")
        shutil.rmtree(out_dir, ignore_errors=True)
        os.replace(tmp_dir, out_dir)
    except Exception as exc:  # noqa: BLE001
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return episode_name, f"error:{type(exc).__name__}:{exc}"
    return episode_name, "built"


def _enumerate_success(limit: int = 0) -> list[str]:
    """Sorted relative paths of every raw success episode holding a trajectory."""
    rels: list[str] = []
    for lab_dir in sorted(p for p in RAW_ROOT.iterdir() if p.is_dir()):
        success_dir = lab_dir / "success"
        if not success_dir.is_dir():
            continue
        for date_dir in sorted(p for p in success_dir.iterdir() if p.is_dir()):
            for ep_dir in sorted(p for p in date_dir.iterdir() if p.is_dir()):
                if (ep_dir / "trajectory.h5").is_file():
                    rels.append(str(ep_dir.relative_to(RAW_ROOT)))
        if limit and len(rels) >= limit:
            break
    return sorted(rels)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="destination root for episode_NNNNNN dirs")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0, help="only build the first N episodes")
    ap.add_argument("--plan-only", action="store_true")
    args = ap.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    plan_path = out_root / "_raw_plan.json"

    if plan_path.is_file():
        rels = json.loads(plan_path.read_text(encoding="utf-8"))["episodes"]
        print(f"[plan] reusing {plan_path}: {len(rels)} success episodes")
    else:
        t0 = time.time()
        rels = _enumerate_success()
        plan_path.write_text(json.dumps({"root": str(RAW_ROOT), "episodes": rels}), encoding="utf-8")
        print(f"[plan] enumerated {len(rels)} success episodes in {time.time() - t0:.0f}s -> {plan_path}")

    jobs = [(f"episode_{index:06d}", rel, str(out_root)) for index, rel in enumerate(rels)]
    if args.limit:
        jobs = jobs[: args.limit]
    if args.plan_only:
        for name, rel, _ in jobs[:5]:
            print(f"  {name}  <-  {rel}")
        return

    print(f"[build] {len(jobs)} episodes with {args.workers} workers -> {out_root}")
    t0 = time.time()
    counts: dict[str, int] = {}
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_build_one, job) for job in jobs]
        for fut in as_completed(futures):
            name, status = fut.result()
            key = status.split(":")[0]
            counts[key] = counts.get(key, 0) + 1
            done += 1
            if key not in ("built", "skipped") and counts[key] <= 5:
                print(f"[build]   {name}: {status}", flush=True)
            if done % 200 == 0 or done == len(jobs):
                rate = done / max(1e-9, time.time() - t0)
                eta = (len(jobs) - done) / max(1e-9, rate)
                print(f"[build] {done}/{len(jobs)} | {counts} | {rate:.2f} ep/s | "
                      f"ETA {eta / 3600:.1f} h", flush=True)
    print(f"[build] done in {(time.time() - t0) / 60:.1f} min: {counts}")
    if not counts.get("built") and not counts.get("skipped"):
        sys.exit("no episodes were built")


if __name__ == "__main__":
    main()
