#!/usr/bin/env python3
"""Extract clip-aligned current wrist RGB frames from the original DROID videos."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

try:
    import imageio_ffmpeg

    FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
except ImportError:
    FFMPEG = "ffmpeg"


def _complete(directory: Path, count: int, ext: str) -> bool:
    if count <= 0 or not directory.is_dir():
        return False
    files = list(directory.glob(f"*.{ext}"))
    return (len(files) == count and (directory / f"{0:06d}.{ext}").exists()
            and (directory / f"{count - 1:06d}.{ext}").exists())


def _extract_one(task: tuple[str, str, str, str, str, int, bool]) -> tuple[str, str, int]:
    clip_str, raw_root_str, video_name, out_subdir, ext, quality, overwrite = task
    clip = Path(clip_str)
    meta = json.loads((clip / "meta.json").read_text(encoding="utf-8"))
    count = int(meta["num_frames"])
    out = clip / out_subdir
    if not overwrite and _complete(out, count, ext):
        return clip.name, "skip", count

    f0, f1 = (int(v) for v in meta["source_frame_range"])
    if f1 - f0 != count:
        return clip.name, f"bad-meta: range={f0}:{f1} count={count}", 0
    raw_video = Path(raw_root_str) / str(meta["episode_id"]) / video_name
    if not raw_video.exists():
        return clip.name, f"missing-video: {raw_video}", 0

    tmp = clip / f".{out_subdir}.tmp.{os.getpid()}"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    vf = f"trim=start_frame={f0}:end_frame={f1},setpts=PTS-STARTPTS"
    cmd = [FFMPEG, "-nostdin", "-y", "-loglevel", "error", "-threads", "1",
           "-i", str(raw_video), "-vf", vf, "-vsync", "0"]
    if ext in ("jpg", "jpeg"):
        cmd += ["-qscale:v", str(quality)]
    cmd += ["-start_number", "0", str(tmp / f"%06d.{ext}")]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as exc:
        shutil.rmtree(tmp, ignore_errors=True)
        return clip.name, f"ffmpeg-error: {exc.stderr.decode(errors='replace')[:160]}", 0
    if not _complete(tmp, count, ext):
        actual = len(list(tmp.glob(f"*.{ext}")))
        shutil.rmtree(tmp, ignore_errors=True)
        return clip.name, f"frame-count: expected={count} actual={actual}", actual
    shutil.rmtree(out, ignore_errors=True)
    tmp.rename(out)
    return clip.name, "ok", count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True, help="Root containing numeric clip dirs")
    parser.add_argument("--raw-root", type=Path, required=True, help="Original DROID episode root")
    parser.add_argument("--video-name", default="steps_observation_wrist_image_left.mp4")
    parser.add_argument("--out-subdir", default="wrist_frames")
    parser.add_argument("--ext", choices=("jpg", "jpeg", "png"), default="jpg")
    parser.add_argument("--quality", type=int, default=2)
    parser.add_argument("--start-id", type=int, default=0, help="Inclusive numeric clip id")
    parser.add_argument("--end-id", type=int, default=35696, help="Exclusive numeric clip id")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    clips = [args.root / f"{i:05d}" for i in range(args.start_id, args.end_id)]
    clips = [p for p in clips if (p / "meta.json").exists()]
    print(f"ffmpeg={FFMPEG}")
    print(f"clips={len(clips)} range=[{args.start_id},{args.end_id}) workers={args.workers}")
    tasks = [(str(p), str(args.raw_root), args.video_name, args.out_subdir,
              args.ext, args.quality, args.overwrite) for p in clips]
    counts = {"ok": 0, "skip": 0, "error": 0}
    frames = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_extract_one, task) for task in tasks]
        for done, future in enumerate(as_completed(futures), 1):
            name, status, count = future.result()
            frames += count
            if status in ("ok", "skip"):
                counts[status] += 1
            else:
                counts["error"] += 1
                print(f"[{name}] {status}")
            if done % 200 == 0 or done == len(tasks):
                print(f"progress={done}/{len(tasks)} {counts} frames={frames}", flush=True)
    print(f"DONE {counts} frames={frames}")
    if counts["error"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
