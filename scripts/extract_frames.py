from __future__ import annotations

# Pre-decode every episode video into per-frame image files, so training reads pre-decoded
# images instead of randomly seeking + decoding mp4 every epoch (which starves the GPU).
#
# Sequential ffmpeg decode (one pass per video) is far faster than random-access get_data(idx),
# and you pay it ONCE instead of every epoch. Output layout:
#   <episode>/<out_subdir>/000000.jpg, 000001.jpg, ...   (0-based, matches frame index)
#
# Usage:
#   python scripts/extract_frames.py --root /mnt/pfs/share/shentingrui/dataset/droid-2000-new \
#       --video gt.mp4 --glob '[0-9][0-9][0-9][0-9]' --workers 32
#   (add --overwrite to redo; --limit N to test on the first N episodes)

import argparse
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

try:
    import imageio_ffmpeg
    FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    FFMPEG = "ffmpeg"


def _count_frames(video: Path) -> int:
    try:
        import imageio.v2 as imageio
        r = imageio.get_reader(str(video), format="ffmpeg")
        n = int(r.count_frames())
        r.close()
        return n
    except Exception:
        return -1


def extract_one(args: tuple) -> tuple:
    episode_dir, video_name, out_subdir, ext, quality, overwrite = args
    episode_dir = Path(episode_dir)
    video = episode_dir / video_name
    out_dir = episode_dir / out_subdir
    if not video.exists():
        return (episode_dir.name, "no-video", 0)
    existing = sorted(out_dir.glob(f"*.{ext}")) if out_dir.exists() else []
    if existing and not overwrite:
        return (episode_dir.name, "skip", len(existing))
    out_dir.mkdir(parents=True, exist_ok=True)
    # qscale:v 2 = near-lossless JPEG (lower = better). PNG ignores quality.
    cmd = [FFMPEG, "-nostdin", "-y", "-loglevel", "error", "-i", str(video)]
    if ext in ("jpg", "jpeg"):
        cmd += ["-qscale:v", str(quality)]
    cmd += ["-start_number", "0", str(out_dir / f"%06d.{ext}")]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        return (episode_dir.name, f"ffmpeg-error: {e.stderr.decode()[:120]}", 0)
    n = len(list(out_dir.glob(f"*.{ext}")))
    return (episode_dir.name, "ok", n)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True, help="dataset root containing episode dirs")
    ap.add_argument("--video", type=str, default="gt.mp4")
    ap.add_argument("--glob", type=str, default="[0-9][0-9][0-9][0-9]")
    ap.add_argument("--out-subdir", type=str, default="frames")
    ap.add_argument("--ext", type=str, default="jpg", choices=["jpg", "jpeg", "png"])
    ap.add_argument("--quality", type=int, default=2, help="JPEG qscale (2=best..31=worst)")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="only process first N episodes (test)")
    ap.add_argument("--verify", action="store_true", help="check frame count matches the video")
    args = ap.parse_args()

    eps = sorted(p for p in args.root.glob(args.glob) if p.is_dir() and (p / args.video).exists())
    if args.limit:
        eps = eps[: args.limit]
    print(f"ffmpeg={FFMPEG}")
    print(f"episodes with {args.video}: {len(eps)}  -> writing {args.out_subdir}/*.{args.ext}  workers={args.workers}")

    tasks = [(str(e), args.video, args.out_subdir, args.ext, args.quality, args.overwrite) for e in eps]
    ok = skip = err = total_frames = 0
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(extract_one, t) for t in tasks]
        for fut in as_completed(futs):
            name, status, n = fut.result()
            done += 1
            if status == "ok":
                ok += 1
                total_frames += n
            elif status == "skip":
                skip += 1
                total_frames += n
            else:
                err += 1
                print(f"  [{name}] {status}")
            if done % 200 == 0 or done == len(tasks):
                print(f"  progress {done}/{len(tasks)}  ok={ok} skip={skip} err={err} frames={total_frames}")

    print(f"\nDONE: ok={ok} skip={skip} err={err}  total_frames={total_frames}")

    if args.verify and eps:
        import random
        sample = random.sample(eps, min(5, len(eps)))
        print("verify (frame files vs video frame count):")
        for e in sample:
            nf = len(list((e / args.out_subdir).glob(f"*.{args.ext}")))
            nv = _count_frames(e / args.video)
            flag = "OK" if nf == nv else "MISMATCH"
            print(f"  {e.name}: files={nf} video={nv} {flag}")


if __name__ == "__main__":
    main()
