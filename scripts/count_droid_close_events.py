#!/usr/bin/env python3
"""Count raw DROID gripper-close events and their droid-ex-3000-out coverage."""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import json
import argparse
import time

import pyarrow.parquet as pq


RAW_ROOT = Path("/mnt/pfs/data/fenghaoran/droid/decompressed/1.0.1")
OUT_ROOT = Path("/mnt/pfs/data/shentingrui/droid-ex-3000-out")
CAMERAS = {
    "exterior_image_1_left": "steps_observation_exterior_image_1_left.mp4",
    "exterior_image_2_left": "steps_observation_exterior_image_2_left.mp4",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    args = parser.parse_args()
    # Index raw-frame intervals retained for each source episode and camera view.
    coverage: dict[tuple[str, str], list[tuple[int, int]]] = defaultdict(list)
    clips = Counter()
    for clip in OUT_ROOT.glob("[0-9][0-9][0-9][0-9][0-9]"):
        try:
            meta = json.loads((clip / "meta.json").read_text())
            episode = Path(meta["provenance"]["episode_dir"]).name
            camera = str(meta["camera"])
            start, end = map(int, meta["source_frame_range"])
            if camera in CAMERAS and end > start:
                coverage[(episode, camera)].append((start, end))
                clips[camera] += 1
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
            continue
    for intervals in coverage.values():
        intervals.sort()

    stats = Counter()
    raw_events = Counter()
    retained_events = Counter()
    omitted_events = Counter()
    failures = Counter()
    all_episodes = sorted(RAW_ROOT.glob("episode_*"))
    episodes = all_episodes[args.start:args.end]
    started = time.time()

    for index, episode_dir in enumerate(episodes, 1):
        views = [camera for camera, filename in CAMERAS.items()
                 if (episode_dir / filename).is_file()]
        if not views:
            failures["no_front_video"] += 1
            continue
        try:
            values = pq.read_table(
                episode_dir / "episode.parquet",
                columns=["steps/observation/gripper_position"],
            ).column(0).to_numpy(zero_copy_only=False)
        except Exception as exc:  # Record malformed episodes without stopping the census.
            failures[type(exc).__name__] += 1
            continue

        # Project convention: open is >0.5, so this rising closed-state edge is a close.
        close_frames = ((values[:-1] > 0.5) & ~(values[1:] > 0.5)).nonzero()[0] + 1
        stats["episodes"] += 1
        stats["views"] += len(views)
        for camera in views:
            raw_events[camera] += len(close_frames)
            intervals = coverage.get((episode_dir.name, camera), ())
            for frame in close_frames:
                if any(start <= frame < end for start, end in intervals):
                    retained_events[camera] += 1
                else:
                    omitted_events[camera] += 1
        if index % 1000 == 0:
            print(f"progress={index}/{len(episodes)} elapsed_sec={time.time() - started:.1f}", flush=True)

    print(json.dumps({
        "definition": "close is gripper_position > 0.5 to <= 0.5; each exterior view is counted independently",
        "range_semantics": "out clip source_frame_range is [start, end), indexed in raw episode frames",
        "raw_episode_directories": len(all_episodes),
        "slice": [args.start, args.end],
        "episodes_processed": stats["episodes"],
        "raw_front_videos_counted": stats["views"],
        "out_clips_by_view": dict(clips),
        "out_clips_total": sum(clips.values()),
        "raw_close_events_by_view": dict(raw_events),
        "raw_close_events_total": sum(raw_events.values()),
        "out_retained_close_events_by_view": dict(retained_events),
        "out_retained_close_events_total": sum(retained_events.values()),
        "out_omitted_close_events_by_view": dict(omitted_events),
        "out_omitted_close_events_total": sum(omitted_events.values()),
        "failures": dict(failures),
        "elapsed_seconds": time.time() - started,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
