#!/usr/bin/env python3
"""Convert official DROID100 TFRecords into the RawDroid smoke-test layout.

The existing DROID100 LeRobot conversion retains all three MP4 streams but
replaces DROID's commanded joint velocities with a derived Franka action. This
script therefore reads state and action labels from the original TFRecords and
links the already encoded LeRobot videos into the layout expected by
``RawDroidDataset``.

No TensorFlow installation is required. TFRecord framing and the small
``tf.train.Example`` protobuf schema are decoded directly.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import struct
import tempfile
from typing import Iterator

from google.protobuf import descriptor_pb2, descriptor_pool, message_factory
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from build_raw_droid_pi05_manifest import pi05_keep_ranges


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
DEFAULT_TFRECORD_ROOT = (
    WORKSPACE_ROOT / "GR00T-Dreams/dataset/droid/droid_100/1.0.0"
)
DEFAULT_LEROBOT_ROOT = (
    WORKSPACE_ROOT / "GR00T-Dreams/IDM_dump/data/droid100_official_franka"
)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "artifacts/droid100_raw_smoke"
DEFAULT_MANIFEST = PROJECT_ROOT / "artifacts/droid100_raw_smoke_pi05_manifest.json"

VIDEO_KEYS = {
    "steps_observation_exterior_image_1_left.mp4": (
        "observation.images.exterior_image_1_left_pad_res256_freq15"
    ),
    "steps_observation_exterior_image_2_left.mp4": (
        "observation.images.exterior_image_2_left_pad_res256_freq15"
    ),
    "steps_observation_wrist_image_left.mp4": (
        "observation.images.wrist_image_left_pad_res256_freq15"
    ),
}


def _add_field(
    message,
    name: str,
    number: int,
    field_type: int,
    *,
    label: int = descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL,
    type_name: str = "",
    oneof_index: int | None = None,
) -> None:
    field = message.field.add(
        name=name,
        number=number,
        type=field_type,
        label=label,
    )
    if type_name:
        field.type_name = type_name
    if oneof_index is not None:
        field.oneof_index = oneof_index


def _tf_example_class():
    """Construct the subset of TensorFlow's Example schema used by DROID."""
    file_descriptor = descriptor_pb2.FileDescriptorProto(
        name="tf_example_minimal.proto",
        package="tensorflow",
        syntax="proto3",
    )

    def add_message(name: str):
        message = file_descriptor.message_type.add()
        message.name = name
        return message

    bytes_list = add_message("BytesList")
    _add_field(
        bytes_list,
        "value",
        1,
        descriptor_pb2.FieldDescriptorProto.TYPE_BYTES,
        label=descriptor_pb2.FieldDescriptorProto.LABEL_REPEATED,
    )
    float_list = add_message("FloatList")
    _add_field(
        float_list,
        "value",
        1,
        descriptor_pb2.FieldDescriptorProto.TYPE_FLOAT,
        label=descriptor_pb2.FieldDescriptorProto.LABEL_REPEATED,
    )
    float_list.field[0].options.packed = True
    int64_list = add_message("Int64List")
    _add_field(
        int64_list,
        "value",
        1,
        descriptor_pb2.FieldDescriptorProto.TYPE_INT64,
        label=descriptor_pb2.FieldDescriptorProto.LABEL_REPEATED,
    )
    int64_list.field[0].options.packed = True

    feature = add_message("Feature")
    feature.oneof_decl.add().name = "kind"
    _add_field(
        feature,
        "bytes_list",
        1,
        descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE,
        type_name=".tensorflow.BytesList",
        oneof_index=0,
    )
    _add_field(
        feature,
        "float_list",
        2,
        descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE,
        type_name=".tensorflow.FloatList",
        oneof_index=0,
    )
    _add_field(
        feature,
        "int64_list",
        3,
        descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE,
        type_name=".tensorflow.Int64List",
        oneof_index=0,
    )

    features = add_message("Features")
    entry = features.nested_type.add()
    entry.name = "FeatureEntry"
    entry.options.map_entry = True
    _add_field(entry, "key", 1, descriptor_pb2.FieldDescriptorProto.TYPE_STRING)
    _add_field(
        entry,
        "value",
        2,
        descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE,
        type_name=".tensorflow.Feature",
    )
    _add_field(
        features,
        "feature",
        1,
        descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE,
        label=descriptor_pb2.FieldDescriptorProto.LABEL_REPEATED,
        type_name=".tensorflow.Features.FeatureEntry",
    )

    example = add_message("Example")
    _add_field(
        example,
        "features",
        1,
        descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE,
        type_name=".tensorflow.Features",
    )
    pool = descriptor_pool.DescriptorPool()
    pool.Add(file_descriptor)
    return message_factory.GetMessageClass(
        pool.FindMessageTypeByName("tensorflow.Example")
    )


TF_EXAMPLE = _tf_example_class()


def iter_tfrecord(path: Path) -> Iterator[bytes]:
    """Yield serialized records from one TFRecord file."""
    with path.open("rb") as handle:
        while True:
            length_bytes = handle.read(8)
            if not length_bytes:
                return
            if len(length_bytes) != 8:
                raise ValueError(f"Truncated TFRecord length in {path}")
            length = struct.unpack("<Q", length_bytes)[0]
            if len(handle.read(4)) != 4:
                raise ValueError(f"Truncated TFRecord length CRC in {path}")
            payload = handle.read(length)
            if len(payload) != length:
                raise ValueError(f"Truncated TFRecord payload in {path}")
            if len(handle.read(4)) != 4:
                raise ValueError(f"Truncated TFRecord data CRC in {path}")
            yield payload


def decode_scalar_text(feature) -> str:
    values = feature.bytes_list.value
    if not values:
        return ""
    return bytes(values[0]).decode("utf-8", errors="replace").strip()


def decode_language_instructions(features) -> list[str]:
    instructions: list[str] = []
    for key in (
        "steps/language_instruction",
        "steps/language_instruction_2",
        "steps/language_instruction_3",
    ):
        values = features[key].bytes_list.value
        text = bytes(values[0]).decode("utf-8", errors="replace").strip() if values else ""
        if text and text not in instructions:
            instructions.append(text)
    return instructions


def float_matrix(features, key: str, num_steps: int, width: int) -> np.ndarray:
    values = np.asarray(features[key].float_list.value, dtype=np.float32)
    expected = num_steps * width
    if values.size != expected:
        raise ValueError(f"{key} has {values.size} values; expected {expected}")
    return values.reshape(num_steps, width)


def int_vector(features, key: str, num_steps: int) -> np.ndarray:
    values = np.asarray(features[key].int64_list.value, dtype=np.int8)
    if values.size != num_steps:
        raise ValueError(f"{key} has {values.size} values; expected {num_steps}")
    return values


def load_lerobot_episode_map(root: Path) -> dict[int, dict]:
    path = root / "meta/episodes.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"LeRobot episode metadata not found: {path}")
    mapping: dict[int, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        source_index = int(item["source_episode_index"])
        if source_index in mapping:
            raise ValueError(f"Duplicate source_episode_index={source_index} in {path}")
        mapping[source_index] = item
    return mapping


def video_paths(lerobot_root: Path, episode: dict, num_steps: int) -> dict[str, Path]:
    episode_index = int(episode["episode_index"])
    if int(episode["length"]) != num_steps:
        raise ValueError(
            f"LeRobot episode {episode_index} has {episode['length']} frames; "
            f"TFRecord has {num_steps}"
        )
    chunk = episode_index // 1000
    paths = {}
    for output_name, video_key in VIDEO_KEYS.items():
        path = (
            lerobot_root
            / f"videos/chunk-{chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
        )
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(f"LeRobot video not found: {path}")
        paths[output_name] = path.resolve()
    return paths


def write_episode_parquet(path: Path, arrays: dict[str, np.ndarray]) -> None:
    table = pa.table(
        {
            "steps/observation/joint_position": pa.array(
                arrays["observation_joint"].tolist(), type=pa.list_(pa.float32(), 7)
            ),
            "steps/observation/gripper_position": pa.array(
                arrays["observation_gripper"][:, 0], type=pa.float32()
            ),
            "steps/action_dict/joint_velocity": pa.array(
                arrays["action_joint_velocity"].tolist(), type=pa.list_(pa.float32(), 7)
            ),
            "steps/action_dict/gripper_position": pa.array(
                arrays["action_gripper"][:, 0], type=pa.float32()
            ),
            "steps/is_last": pa.array(arrays["is_last"].astype(bool), type=pa.bool_()),
            "steps/is_terminal": pa.array(
                arrays["is_terminal"].astype(bool), type=pa.bool_()
            ),
        }
    )
    pq.write_table(table, path, compression="zstd")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tfrecord-root", type=Path, default=DEFAULT_TFRECORD_ROOT)
    parser.add_argument("--lerobot-root", type=Path, default=DEFAULT_LEROBOT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--max-episodes", type=int, default=16)
    parser.add_argument("--idle-delta-threshold", type=float, default=1e-3)
    parser.add_argument("--min-idle-len", type=int, default=7)
    parser.add_argument("--min-non-idle-len", type=int, default=16)
    parser.add_argument("--trim-range-end", type=int, default=10)
    parser.add_argument("--gripper-threshold", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_episodes <= 0:
        raise ValueError("--max-episodes must be positive")
    shard_paths = sorted(args.tfrecord_root.glob("*.tfrecord-*"))
    if not shard_paths:
        raise FileNotFoundError(f"No TFRecord shards found under {args.tfrecord_root}")
    lerobot_map = load_lerobot_episode_map(args.lerobot_root)

    output_root = args.output_root.resolve()
    manifest_path = args.manifest.resolve()
    if output_root.exists():
        raise FileExistsError(f"Output root already exists: {output_root}")
    if manifest_path.exists():
        raise FileExistsError(f"Manifest already exists: {manifest_path}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(
        tempfile.mkdtemp(prefix=output_root.name + ".", dir=output_root.parent)
    )

    manifest_episodes = []
    scanned = 0
    skipped_no_ranges = 0
    source_index = 0
    try:
        for shard_path in shard_paths:
            for serialized in iter_tfrecord(shard_path):
                if len(manifest_episodes) >= args.max_episodes:
                    break
                example = TF_EXAMPLE.FromString(serialized)
                features = example.features.feature
                num_steps = len(features["steps/is_last"].int64_list.value)
                scanned += 1
                if source_index not in lerobot_map:
                    raise KeyError(f"No LeRobot video mapping for source episode {source_index}")

                arrays = {
                    "observation_joint": float_matrix(
                        features, "steps/observation/joint_position", num_steps, 7
                    ),
                    "observation_gripper": float_matrix(
                        features, "steps/observation/gripper_position", num_steps, 1
                    ),
                    "action_joint_velocity": float_matrix(
                        features, "steps/action_dict/joint_velocity", num_steps, 7
                    ),
                    "action_gripper": float_matrix(
                        features, "steps/action_dict/gripper_position", num_steps, 1
                    ),
                    "is_last": int_vector(features, "steps/is_last", num_steps),
                    "is_terminal": int_vector(features, "steps/is_terminal", num_steps),
                }
                if not all(np.isfinite(value).all() for value in arrays.values()):
                    raise ValueError(f"Non-finite values in source episode {source_index}")
                keep_ranges = pi05_keep_ranges(
                    arrays["action_joint_velocity"],
                    args.idle_delta_threshold,
                    args.min_idle_len,
                    args.min_non_idle_len,
                    args.trim_range_end,
                )
                if not keep_ranges:
                    skipped_no_ranges += 1
                    source_index += 1
                    example.Clear()
                    del serialized
                    continue

                episode_id = f"episode_{source_index:06d}"
                episode_dir = staging_root / episode_id
                episode_dir.mkdir()
                write_episode_parquet(episode_dir / "episode.parquet", arrays)
                for output_name, source_path in video_paths(
                    args.lerobot_root, lerobot_map[source_index], num_steps
                ).items():
                    os.symlink(source_path, episode_dir / output_name)

                file_path = decode_scalar_text(features["episode_metadata/file_path"])
                recording_path = decode_scalar_text(
                    features["episode_metadata/recording_folderpath"]
                )
                instructions = decode_language_instructions(features)
                metadata = {
                    "num_steps": num_steps,
                    "context": {
                        "episode_metadata/file_path": file_path,
                        "episode_metadata/recording_folderpath": recording_path,
                    },
                    "language_instructions": instructions,
                    "source": {
                        "dataset": "official_droid100_rlds",
                        "source_episode_index": source_index,
                        "tfrecord_shard": shard_path.name,
                    },
                }
                (episode_dir / "metadata.json").write_text(
                    json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
                )

                gripper = arrays["action_gripper"][:, 0]
                transitions = np.diff((gripper > args.gripper_threshold).astype(np.int8))
                manifest_episodes.append(
                    {
                        "episode_id": episode_id,
                        "episode_uuid": episode_id,
                        "episode_id_source": "droid100_tfrecord_index",
                        "raw_episode_path": recording_path or file_path,
                        "language_instructions": instructions,
                        "num_steps": num_steps,
                        "keep_ranges": keep_ranges,
                        "close_events": (np.flatnonzero(transitions == -1) + 1).tolist(),
                        "release_events": (np.flatnonzero(transitions == 1) + 1).tolist(),
                    }
                )
                print(
                    f"converted source={source_index} id={episode_id} "
                    f"steps={num_steps} keep_ranges={len(keep_ranges)}"
                )
                source_index += 1
                example.Clear()
                del serialized
            if len(manifest_episodes) >= args.max_episodes:
                break
        if len(manifest_episodes) < args.max_episodes:
            raise RuntimeError(
                f"Only converted {len(manifest_episodes)} episodes; "
                f"requested {args.max_episodes}"
            )
        staging_root.replace(output_root)
    except BaseException:
        shutil.rmtree(staging_root, ignore_errors=True)
        raise

    manifest = {
        "version": 1,
        "root": str(output_root),
        "filter": {
            "source": "official_droid100_rlds",
            "purpose": "wan_vae_smoke_test",
            "idle_definition": (
                "all(abs(joint_velocity[t]-joint_velocity[t-1]) < threshold)"
            ),
            "idle_delta_threshold": args.idle_delta_threshold,
            "min_idle_len": args.min_idle_len,
            "min_non_idle_len": args.min_non_idle_len,
            "trim_range_end": args.trim_range_end,
            "gripper_threshold": args.gripper_threshold,
        },
        "stats": {
            "scanned_episodes": scanned,
            "kept_episodes": len(manifest_episodes),
            "skipped_no_nonidle_range": skipped_no_ranges,
            "kept_steps": int(
                sum(
                    end - start
                    for episode in manifest_episodes
                    for start, end in episode["keep_ranges"]
                )
            ),
        },
        "episodes": manifest_episodes,
    }
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=manifest_path.parent,
        prefix=manifest_path.name + ".",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(manifest, handle, separators=(",", ":"))
        temporary_manifest = Path(handle.name)
    temporary_manifest.replace(manifest_path)
    print(json.dumps(manifest["stats"], indent=2))
    print(f"data root: {output_root}")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
