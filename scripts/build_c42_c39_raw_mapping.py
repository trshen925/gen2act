"""Map C39's logical clips and cached training windows onto full raw DROID."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


CAMERA_NUMBER = {
    "exterior_image_1_left": 1,
    "exterior_image_2_left": 2,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--c39-root", type=Path,
        default=Path("/mnt/pfs/data/shentingrui/droid-ex-3000-out"),
    )
    parser.add_argument(
        "--raw-root", type=Path,
        default=Path("/mnt/pfs/data/fenghaoran/droid/decompressed/1.0.1"),
    )
    parser.add_argument(
        "--c39-index", type=Path,
        default=Path("artifacts/c39_jointvelocity_window_index.json"),
    )
    parser.add_argument("--max-clips", type=int, default=43415)
    parser.add_argument(
        "--output", type=Path,
        default=Path("metadata/c42_c39_to_raw_droid_mapping.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    c39_index = json.loads(args.c39_index.read_text(encoding="utf-8"))
    train_ids = {str(value) for value in c39_index["fingerprint"]["episode_ids"]}
    train_samples = [
        [str(clip_id), int(start)] for clip_id, start in c39_index["samples"]
    ]

    candidates = sorted(
        path for path in args.c39_root.glob("[0-9][0-9][0-9][0-9][0-9]")
        if path.is_dir() and (path / "meta.json").is_file()
        and (path / "data.parquet").is_file()
    )[: args.max_clips]
    clips = []
    skipped: dict[str, int] = {}
    for clip_dir in candidates:
        clip_id = clip_dir.name
        meta = json.loads((clip_dir / "meta.json").read_text(encoding="utf-8"))
        provenance_dir = Path(str(meta.get("provenance", {}).get("episode_dir", "")))
        required_c39 = [
            provenance_dir / "extrinsics.json",
            clip_dir / "rgb.mp4",
        ]
        frames_dir = clip_dir / "frames"
        if any(not path.is_file() for path in required_c39):
            reason = "missing_c39_source"
        elif not frames_dir.is_dir() or not any(frames_dir.iterdir()):
            reason = "missing_c39_frames"
        else:
            reason = ""
        if reason:
            skipped[reason] = skipped.get(reason, 0) + 1
            continue

        raw_episode_id = str(meta["episode_id"])
        raw_dir = args.raw_root / raw_episode_id
        camera_name = str(meta["camera"])
        if camera_name not in CAMERA_NUMBER:
            raise ValueError(f"Unsupported camera {camera_name!r} in clip {clip_id}")
        camera = CAMERA_NUMBER[camera_name]
        required_raw = [
            raw_dir / "metadata.json",
            raw_dir / "episode.parquet",
            raw_dir / f"steps_observation_exterior_image_{camera}_left.mp4",
            raw_dir / "steps_observation_wrist_image_left.mp4",
        ]
        missing = [str(path) for path in required_raw if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                f"C39 clip {clip_id} cannot be mapped to raw DROID; missing {missing}"
            )
        raw_meta = json.loads((raw_dir / "metadata.json").read_text(encoding="utf-8"))
        raw_num_steps = int(raw_meta["num_steps"])
        start, end = [int(value) for value in meta["source_frame_range"]]
        num_frames = int(meta["num_frames"])
        if end - start != num_frames or not 0 <= start < end <= raw_num_steps:
            raise ValueError(
                f"Invalid mapping for clip {clip_id}: range={[start, end]} "
                f"num_frames={num_frames} raw_num_steps={raw_num_steps}"
            )
        clips.append({
            "clip_id": clip_id,
            "raw_episode_id": raw_episode_id,
            "frame_range": [start, end],
            "camera": camera,
            "raw_num_steps": raw_num_steps,
            "split": "train" if clip_id in train_ids else "val",
        })

    mapped_ids = {item["clip_id"] for item in clips}
    missing_train_ids = sorted(train_ids - mapped_ids)
    if missing_train_ids:
        raise RuntimeError(
            f"Mapping is missing {len(missing_train_ids)} C39 training clips; "
            f"first={missing_train_ids[:10]}"
        )
    invalid_samples = [sample for sample in train_samples if sample[0] not in mapped_ids]
    if invalid_samples:
        raise RuntimeError(
            f"Mapping is missing clips for {len(invalid_samples)} C39 cached samples"
        )
    train_clips = sum(item["split"] == "train" for item in clips)
    val_clips = len(clips) - train_clips
    if train_clips != len(train_ids):
        raise RuntimeError(
            f"Expected {len(train_ids)} train clips, mapped {train_clips}"
        )

    output = {
        "version": 1,
        "description": (
            "C39 logical clips mapped to raw DROID 1.0.1. Frame ranges are "
            "half-open and train_samples are the exact C39 event-balanced index."
        ),
        "stats": {
            "candidate_clips": len(candidates),
            "mapped_clips": len(clips),
            "train_clips": train_clips,
            "val_clips": val_clips,
            "train_samples": len(train_samples),
            "skipped": skipped,
        },
        "clips": clips,
        "train_samples": train_samples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(output, separators=(",", ":")) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(output["stats"], indent=2))
    print(f"mapping: {args.output}")


if __name__ == "__main__":
    main()
