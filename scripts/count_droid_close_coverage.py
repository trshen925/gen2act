#!/usr/bin/env python3
"""Compare DROID close events with droid-ex-3000-out using source identities.

The DROID gripper signal is open at <= 0.5 and closed at > 0.5. A close event
is therefore an upward threshold crossing. External camera views are separate
data items; output intervals are unioned so an event covered by overlapping
clips is counted once.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import json
import time

import pyarrow.parquet as pq


RAW_ROOT = Path("/mnt/pfs/data/fenghaoran/droid/decompressed/1.0.1")
OUT_ROOT = Path("/mnt/pfs/data/shentingrui/droid-ex-3000-out")
VIEWS = {
    "exterior_image_1_left": "steps_observation_exterior_image_1_left.mp4",
    "exterior_image_2_left": "steps_observation_exterior_image_2_left.mp4",
}
KARLP_PATHS = Path("/mnt/pfs/data/shentingrui/KarlP-droid/episode_id_to_path.json")
KARLP_SERIALS = Path("/mnt/pfs/data/shentingrui/KarlP-droid/camera_serials.json")
KARLP_EXTRINSICS = Path("/mnt/pfs/data/shentingrui/KarlP-droid/cam2base_extrinsics.json")
KARLP_EXTRINSICS_SUPERSET = Path("/mnt/pfs/data/shentingrui/KarlP-droid/cam2base_extrinsic_superset.json")
RAW_PATH_PREFIX = "gs://xembodiment_data/r2d2/r2d2-data-full/"


def _raw_identity(metadata: dict) -> str:
    path = str(metadata["context"]["episode_metadata/file_path"])
    if not path.startswith(RAW_PATH_PREFIX) or not path.endswith("/trajectory.h5"):
        raise ValueError(f"Unexpected DROID file_path={path!r}")
    return path[len(RAW_PATH_PREFIX):-len("/trajectory.h5")]


def _calibrated_serials(records: dict) -> dict[str, set[str]]:
    """Extract the camera serial keys whose values are 6D cam-to-base poses."""
    result: dict[str, set[str]] = {}
    for episode_id, record in records.items():
        serials = {
            str(key) for key, value in record.items()
            if isinstance(value, list) and len(value) == 6
        }
        if serials:
            result[str(episode_id)] = serials
    return result


def main() -> None:
    started = time.time()
    with KARLP_PATHS.open("r", encoding="utf-8") as handle:
        episode_paths = json.load(handle)
    with KARLP_SERIALS.open("r", encoding="utf-8") as handle:
        camera_serials = json.load(handle)
    with KARLP_EXTRINSICS.open("r", encoding="utf-8") as handle:
        cam2base = _calibrated_serials(json.load(handle))
    with KARLP_EXTRINSICS_SUPERSET.open("r", encoding="utf-8") as handle:
        superset = _calibrated_serials(json.load(handle))
    # A camera may occur in either annotation file. The union avoids counting
    # duplicated records twice while accepting every calibrated exterior view.
    for episode_id, serials in superset.items():
        cam2base.setdefault(episode_id, set()).update(serials)
    path_to_episode_id = {str(path): episode_id for episode_id, path in episode_paths.items()}
    # Unique DROID episode identity + camera -> source-frame intervals retained by output clips.
    retained: dict[tuple[str, str], list[tuple[int, int]]] = defaultdict(list)
    identity_cache: dict[str, str] = {}
    output = Counter()
    for index, clip in enumerate(sorted(OUT_ROOT.glob("[0-9][0-9][0-9][0-9][0-9]")), 1):
        try:
            meta = json.loads((clip / "meta.json").read_text())
            camera = str(meta["camera"])
            if camera not in VIEWS:
                continue
            source_dir = str(meta["provenance"]["episode_dir"])
            identity = identity_cache.get(source_dir)
            if identity is None:
                extrinsics = json.loads((Path(source_dir) / "extrinsics.json").read_text())
                identity = str(episode_paths[extrinsics["episode_id"]])
                identity_cache[source_dir] = identity
            start, end = map(int, meta["source_frame_range"])
            retained[(identity, camera)].append((start, end))
            output[f"clips:{camera}"] += 1
        except Exception as exc:
            output[f"error:{type(exc).__name__}"] += 1
        if index % 5000 == 0:
            print(f"output_index_progress={index} elapsed_sec={time.time() - started:.1f}", flush=True)
    for intervals in retained.values():
        intervals.sort()

    raw = Counter()
    for index, episode in enumerate(sorted(RAW_ROOT.glob("episode_*")), 1):
        try:
            with (episode / "metadata.json").open("r", encoding="utf-8") as handle:
                identity = _raw_identity(json.load(handle))
            episode_id = path_to_episode_id.get(identity)
            if episode_id is None or "/success/" not in f"/{identity}/":
                raw["not_karlp_success"] += 1
                continue
            serials = camera_serials.get(episode_id, {})
            calibration = cam2base.get(episode_id, {})
            views = [
                view for view, filename in VIEWS.items()
                if (episode / filename).is_file()
                and str(serials.get("ext1_cam_serial" if view.endswith("1_left") else "ext2_cam_serial", "")) in calibration
            ]
            if not views:
                raw["no_karlp_calibrated_front_view"] += 1
                continue
            table = pq.read_table(
                episode / "episode.parquet",
                columns=["t", "steps/observation/gripper_position"],
            )
            frames = table.column("t").to_numpy(zero_copy_only=False)
            values = table.column("steps/observation/gripper_position").to_numpy(zero_copy_only=False)
        except Exception as exc:
            raw[f"error:{type(exc).__name__}"] += 1
            continue
        close_rows = ((values[:-1] <= 0.5) & (values[1:] > 0.5)).nonzero()[0] + 1
        raw["episodes"] += 1
        for view in views:
            raw[f"events:{view}"] += len(close_rows)
            intervals = retained.get((identity, view), ())
            for row in close_rows:
                frame = int(frames[row])
                if any(start <= frame < end for start, end in intervals):
                    raw[f"covered:{view}"] += 1
                else:
                    raw[f"missing:{view}"] += 1
        if index % 5000 == 0:
            print(f"raw_index_progress={index}/95658 elapsed_sec={time.time() - started:.1f}", flush=True)

    result = {
        "definition": "close = gripper_position <= 0.5 to > 0.5",
        "deduplication": "output coverage is unioned by KarlP episode identity, view, and raw t frame",
        "raw_scope": "KarlP success episodes with an ext1/ext2 serial in the deduplicated union of cam2base_extrinsics.json and cam2base_extrinsic_superset.json",
        "output_clips": dict(output),
        "raw": dict(raw),
        "elapsed_seconds": time.time() - started,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
