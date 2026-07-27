#!/usr/bin/env python3
"""Summarize gripper state and transitions in droid-ex-3000-out clips.

For DROID, gripper_position > 0.5 denotes a physically closed gripper.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import json
import time

import pyarrow.parquet as pq


ROOT = Path("/mnt/pfs/data/shentingrui/droid-ex-3000-out")


def main() -> None:
    stats = Counter()
    clips = sorted(ROOT.glob("[0-9][0-9][0-9][0-9][0-9]"))
    start = time.time()
    for index, clip in enumerate(clips, 1):
        try:
            camera = json.loads((clip / "meta.json").read_text()).get("camera", "unknown")
            values = pq.read_table(
                clip / "data.parquet",
                columns=["steps/observation/gripper_position"],
            ).column(0).to_numpy(zero_copy_only=False)
            closed = values > 0.5
            closes = int((~closed[:-1] & closed[1:]).sum())
            releases = int((closed[:-1] & ~closed[1:]).sum())
        except Exception as exc:
            stats[f"error:{type(exc).__name__}"] += 1
            continue
        stats["clips"] += 1
        stats[f"clips:{camera}"] += 1
        stats["close_events"] += closes
        stats["release_events"] += releases
        stats[f"close_events:{camera}"] += closes
        stats[f"release_events:{camera}"] += releases
        stats["clips_with_close"] += int(closes > 0)
        stats[f"clips_with_close:{camera}"] += int(closes > 0)
        stats["clips_with_release"] += int(releases > 0)
        stats[f"clips_with_release:{camera}"] += int(releases > 0)
        stats["clips_ever_closed"] += int(closed.any())
        stats[f"clips_ever_closed:{camera}"] += int(closed.any())
        stats["clips_ever_open"] += int((~closed).any())
        stats[f"clips_ever_open:{camera}"] += int((~closed).any())
        stats["clips_start_closed"] += int(bool(closed[0]))
        stats[f"clips_start_closed:{camera}"] += int(bool(closed[0]))
        stats["clips_end_closed"] += int(bool(closed[-1]))
        stats[f"clips_end_closed:{camera}"] += int(bool(closed[-1]))
        if index % 1000 == 0:
            print(f"progress={index}/{len(clips)} elapsed_sec={time.time() - start:.1f}", flush=True)
    print(json.dumps(dict(stats), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
