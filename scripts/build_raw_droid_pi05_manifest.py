#!/usr/bin/env python3
"""Build a validated pi0.5-style non-idle manifest for decompressed DROID."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
import os
from pathlib import Path
import re
import tempfile

import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm


REQUIRED_COLUMNS = [
    "steps/action_dict/joint_velocity",
    "steps/action_dict/gripper_position",
    "steps/observation/joint_position",
    "steps/observation/gripper_position",
]
REQUIRED_VIDEOS = [
    "steps_observation_exterior_image_1_left.mp4",
    "steps_observation_exterior_image_2_left.mp4",
    "steps_observation_wrist_image_left.mp4",
]
RAW_PATH_MARKER = "r2d2-data-full/"
_PATH_TO_EPISODE_ID: dict[str, str] = {}
_LANGUAGE_ANNOTATIONS: dict[str, dict[str, str]] = {}
_IDENTITY_TO_EPISODE_ID: dict[tuple[str, str, str, str, str], str] = {}
UUID_PATTERN = re.compile(
    r"^(?P<lab>[^+]+)\+[^+]+\+(?P<date>\d{4}-\d{2}-\d{2})-"
    r"(?P<hour>\d{2})h-(?P<minute>\d{2})m-(?P<second>\d{2})s$"
)
RAW_TIMESTAMP_PATTERN = re.compile(
    r"^[A-Za-z]{3}_(?P<month>[A-Za-z]{3})_+(?P<day>\d{1,2})_"
    r"(?P<hour>\d{2})[:_](?P<minute>\d{2})[:_](?P<second>\d{2})_(?P<year>\d{4})$"
)
MONTH_NUMBERS = {
    month: f"{index:02d}"
    for index, month in enumerate(
        ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"), 1
    )
}


def _init_worker(
    path_to_episode_id: dict[str, str],
    language_annotations: dict[str, dict[str, str]],
    identity_to_episode_id: dict[tuple[str, str, str, str, str], str],
) -> None:
    global _PATH_TO_EPISODE_ID, _LANGUAGE_ANNOTATIONS, _IDENTITY_TO_EPISODE_ID
    _PATH_TO_EPISODE_ID = path_to_episode_id
    _LANGUAGE_ANNOTATIONS = language_annotations
    _IDENTITY_TO_EPISODE_ID = identity_to_episode_id


def _raw_episode_path(metadata: dict) -> str:
    """Extract the KarlP ``LAB/result/date/timestamp`` episode identity."""
    context = metadata.get("context", {})
    for key in ("episode_metadata/recording_folderpath", "episode_metadata/file_path"):
        value = str(context.get(key, ""))
        if RAW_PATH_MARKER not in value:
            continue
        relative = value.split(RAW_PATH_MARKER, 1)[1]
        relative = relative.split("/recordings", 1)[0]
        relative = relative.removesuffix("/trajectory.h5")
        return relative.strip("/")
    raise ValueError("DROID metadata does not contain a canonical raw episode path")


def _raw_path_identity(raw_episode_path: str) -> tuple[str, str, str, str, str]:
    parts = raw_episode_path.split("/")
    if len(parts) != 4:
        raise ValueError(f"Unexpected DROID raw path: {raw_episode_path!r}")
    match = RAW_TIMESTAMP_PATTERN.fullmatch(parts[-1])
    if match is None or match["month"] not in MONTH_NUMBERS:
        raise ValueError(f"Unexpected DROID timestamp: {parts[-1]!r}")
    date = f"{match['year']}-{MONTH_NUMBERS[match['month']]}-{int(match['day']):02d}"
    return parts[0], date, match["hour"], match["minute"], match["second"]


def _uuid_identity(episode_uuid: str) -> tuple[str, str, str, str, str]:
    match = UUID_PATTERN.fullmatch(episode_uuid)
    if match is None:
        raise ValueError(f"Unexpected KarlP DROID UUID: {episode_uuid!r}")
    return match["lab"], match["date"], match["hour"], match["minute"], match["second"]


def _array_column(table, name: str) -> np.ndarray:
    values = []
    for value in table.column(name).to_pylist():
        values.append(json.loads(value) if isinstance(value, str) else value)
    return np.asarray(values, dtype=np.float32)


def _true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    padded = np.concatenate(([False], np.asarray(mask, dtype=bool), [False]))
    transitions = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(transitions == 1)
    ends = np.flatnonzero(transitions == -1)
    return [(int(start), int(end)) for start, end in zip(starts, ends, strict=True)]


def pi05_keep_ranges(
    joint_velocity: np.ndarray,
    idle_delta_threshold: float,
    min_idle_len: int,
    min_non_idle_len: int,
    trim_range_end: int,
) -> list[list[int]]:
    """Exact range logic from openpi/examples/droid/compute_droid_nonidle_ranges.py."""
    is_idle = np.concatenate(
        (
            np.asarray([False]),
            np.all(np.abs(np.diff(joint_velocity, axis=0)) < idle_delta_threshold, axis=1),
        )
    )
    keep = np.ones(len(joint_velocity), dtype=bool)
    for start, end in _true_runs(is_idle):
        if end - start >= min_idle_len:
            keep[start:end] = False

    ranges = []
    for start, end in _true_runs(keep):
        if end - start < min_non_idle_len:
            continue
        trimmed_end = end - trim_range_end
        if trimmed_end > start:
            ranges.append([int(start), int(trimmed_end)])
    return ranges


def _process_episode(args: tuple) -> tuple[str, dict | None, str]:
    episode_dir, idle_threshold, min_idle_len, min_non_idle_len, trim_range_end, gripper_threshold = args
    episode_dir = Path(episode_dir)
    try:
        metadata_path = episode_dir / "metadata.json"
        parquet_path = episode_dir / "episode.parquet"
        if not metadata_path.is_file() or not parquet_path.is_file():
            return episode_dir.name, None, "missing_metadata_or_parquet"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        context = metadata.get("context", {})
        source_path = " ".join(
            (
                str(context.get("episode_metadata/file_path", "")),
                str(context.get("episode_metadata/recording_folderpath", "")),
            )
        )
        if "/success/" not in source_path:
            return episode_dir.name, None, "not_success"

        raw_episode_path = _raw_episode_path(metadata)
        episode_uuid = _PATH_TO_EPISODE_ID.get(raw_episode_path)
        if episode_uuid is None:
            episode_uuid = _IDENTITY_TO_EPISODE_ID.get(_raw_path_identity(raw_episode_path))
            episode_id_source = "language_timestamp"
        else:
            episode_id_source = "episode_id_to_path"
        if episode_uuid is None:
            return episode_dir.name, None, "missing_or_ambiguous_episode_id_mapping"
        language_record = _LANGUAGE_ANNOTATIONS.get(episode_uuid)
        if language_record is None:
            return episode_dir.name, None, "missing_language_annotation"
        language_instructions = [
            str(language_record[key]).strip() for key in sorted(language_record) if str(language_record[key]).strip()
        ]
        if not language_instructions:
            return episode_dir.name, None, "empty_language_annotation"
        for video_name in REQUIRED_VIDEOS:
            video_path = episode_dir / video_name
            if not video_path.is_file() or video_path.stat().st_size <= 0:
                return episode_dir.name, None, f"missing_video:{video_name}"

        table = pq.read_table(parquet_path, columns=REQUIRED_COLUMNS)
        velocity = _array_column(table, "steps/action_dict/joint_velocity")
        joint = _array_column(table, "steps/observation/joint_position")
        action_gripper = np.asarray(
            table.column("steps/action_dict/gripper_position").to_pylist(), dtype=np.float32
        ).reshape(-1)
        observation_gripper = np.asarray(
            table.column("steps/observation/gripper_position").to_pylist(), dtype=np.float32
        ).reshape(-1)
        num_steps = int(table.num_rows)
        if velocity.shape != (num_steps, 7) or joint.shape != (num_steps, 7):
            return episode_dir.name, None, "bad_joint_shape"
        if len(action_gripper) != num_steps or len(observation_gripper) != num_steps:
            return episode_dir.name, None, "bad_gripper_shape"
        if int(metadata.get("num_steps", num_steps)) != num_steps:
            return episode_dir.name, None, "metadata_length_mismatch"
        if num_steps < min_non_idle_len:
            return episode_dir.name, None, "too_short"
        if not all(np.isfinite(value).all() for value in (velocity, joint, action_gripper, observation_gripper)):
            return episode_dir.name, None, "non_finite"

        keep_ranges = pi05_keep_ranges(velocity, idle_threshold, min_idle_len, min_non_idle_len, trim_range_end)
        if not keep_ranges:
            return episode_dir.name, None, "no_nonidle_range"
        is_open = action_gripper > gripper_threshold
        transitions = np.diff(is_open.astype(np.int8))
        close_events = (np.flatnonzero(transitions == -1) + 1).astype(int).tolist()
        release_events = (np.flatnonzero(transitions == 1) + 1).astype(int).tolist()
        return (
            episode_dir.name,
            {
                "episode_id": episode_dir.name,
                "episode_uuid": episode_uuid,
                "episode_id_source": episode_id_source,
                "raw_episode_path": raw_episode_path,
                "language_instructions": language_instructions,
                "num_steps": num_steps,
                "keep_ranges": keep_ranges,
                "close_events": close_events,
                "release_events": release_events,
            },
            "kept",
        )
    except Exception as exc:
        return episode_dir.name, None, f"error:{type(exc).__name__}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/mnt/pfs/data/fenghaoran/droid/decompressed/1.0.1"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/raw_droid_1_0_1_pi05_manifest.json"),
    )
    karlp_root = Path(__file__).resolve().parents[2] / "KarlP-droid"
    parser.add_argument(
        "--episode-id-to-path",
        type=Path,
        default=karlp_root / "episode_id_to_path.json",
    )
    parser.add_argument(
        "--language-annotations",
        type=Path,
        default=karlp_root / "droid_language_annotations.json",
    )
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--idle-delta-threshold", type=float, default=1e-3)
    parser.add_argument("--min-idle-len", type=int, default=7)
    parser.add_argument("--min-non-idle-len", type=int, default=16)
    parser.add_argument("--trim-range-end", type=int, default=10)
    parser.add_argument("--gripper-threshold", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    episode_id_to_path = json.loads(args.episode_id_to_path.read_text(encoding="utf-8"))
    path_to_episode_id = {str(path).strip("/"): str(episode_id) for episode_id, path in episode_id_to_path.items()}
    language_annotations = json.loads(args.language_annotations.read_text(encoding="utf-8"))
    identities: dict[tuple[str, str, str, str, str], list[str]] = {}
    for episode_uuid in language_annotations:
        identities.setdefault(_uuid_identity(episode_uuid), []).append(episode_uuid)
    identity_to_episode_id = {
        identity: episode_ids[0] for identity, episode_ids in identities.items() if len(episode_ids) == 1
    }
    episode_dirs = sorted(path for path in args.root.glob("episode_*") if path.is_dir())
    if args.max_episodes is not None:
        episode_dirs = episode_dirs[: args.max_episodes]
    tasks = [
        (
            str(path),
            args.idle_delta_threshold,
            args.min_idle_len,
            args.min_non_idle_len,
            args.trim_range_end,
            args.gripper_threshold,
        )
        for path in episode_dirs
    ]

    episodes = []
    rejection_counts = Counter()
    # Spawn is deliberate: forking after importing PyArrow can hang during pool
    # shutdown on shared compute nodes. Spawn keeps parquet/JSON parsing truly
    # parallel without inheriting Arrow's native thread state.
    with ProcessPoolExecutor(
        max_workers=max(1, args.workers),
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_init_worker,
        initargs=(path_to_episode_id, language_annotations, identity_to_episode_id),
    ) as executor:
        results = executor.map(_process_episode, tasks, chunksize=32)
        for _, episode, reason in tqdm(results, total=len(tasks), desc="DROID episodes"):
            rejection_counts[reason] += 1
            if episode is not None:
                episodes.append(episode)
    episodes.sort(key=lambda item: item["episode_id"])
    payload = {
        "version": 2,
        "root": str(args.root.resolve()),
        "filter": {
            "success_only": True,
            "require_karlp_episode_id": True,
            "require_language_annotation": True,
            "idle_definition": "all(abs(joint_velocity[t]-joint_velocity[t-1]) < threshold)",
            "idle_delta_threshold": args.idle_delta_threshold,
            "min_idle_len": args.min_idle_len,
            "min_non_idle_len": args.min_non_idle_len,
            "trim_range_end": args.trim_range_end,
            "gripper_threshold": args.gripper_threshold,
        },
        "stats": {
            "scanned_episodes": len(tasks),
            "kept_episodes": len(episodes),
            "kept_steps": int(sum(end - start for item in episodes for start, end in item["keep_ranges"])),
            "outcomes": dict(sorted(rejection_counts.items())),
        },
        "episodes": episodes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=args.output.parent,
        prefix=args.output.name + ".",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(payload, handle, separators=(",", ":"))
        temporary_path = Path(handle.name)
    temporary_path.replace(args.output)
    print(json.dumps(payload["stats"], indent=2))
    print(f"manifest: {args.output}")


if __name__ == "__main__":
    main()
