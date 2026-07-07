from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def default_config() -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[2]
    data_root = project_root.parents[0]
    return {
        "experiment": {
            "name": "robot2robot_gen2act",
            "output_dir": str(project_root / "outputs" / "robot2robot_gen2act"),
        },
        "data": {
            "dataset_type": "openx_droid",
            "root": str(data_root / "open-x" / "droid_100-gen"),
            "source_video_name": "generated.mp4",
            "target_video_name": "groundtruth.mp4",
            "metadata_name": "data.json",
            "source_len": 16,
            "target_history_len": 8,
            "target_offset": 0,
            "image_size": 224,
            "action_stride": 1,
            "max_episodes": None,
            "max_windows": None,
            "terminate_positive_window": 5,
            "val_ratio": 0.2,
            "val_count": None,
            "split_seed": 42,
            "split": "train",
            "gripper_threshold": 0.0,
            "load_videos": True,
            "ffmpeg_threads": 1,
            "video_reader_cache": 8,
            "frames_subdir": "",
            "frames_ext": "jpg",
            "future_horizon": 0,
            "window_jitter": {"enabled": False, "max_offset": 0},
            "source_jitter": {"enabled": False, "max_offset": 0},
            "augmentation": {"enabled": False, "p": 1.0, "brightness": 0.0, "contrast": 0.0, "saturation": 0.0, "noise_std": 0.0},
            "proprioception": {"enabled": False, "source": "observations", "key": "cartesian_position", "step": "target", "dims": 6},
        },
        "action": {
            "dim": 7,
            "pose_dims": 6,
            "mode": "classification",
            "regression_normalize": False,
            "chunk_size": 1,
            "num_bins": 256,
            "bounds_source": "config",
            "bounds_low": [-1.0] * 6,
            "bounds_high": [1.0] * 6,
            "mapping": {"type": "droid_actions_first6_plus_gripper"},
        },
        "model": {
            "backbone": {
                "name": "dinov2_vitb14",
                "pretrained": True,
                "freeze": False,
                "unfreeze_last_n_blocks": 0,
                "local_checkpoint": "",
                "allow_random_init": False,
            },
            "image_size": 224,
            "source_len": 16,
            "target_history_len": 8,
            "latent_tokens": 64,
            "hidden_dim": 768,
            "resampler_layers": 2,
            "resampler_heads": 8,
            "fusion_heads": 8,
            "fusion_layers": 1,
            "decoder_layers": 1,
            "num_queries": 1,
            "num_bins": 256,
            "pose_action_dims": 6,
            "proprioception_dim": 0,
            "point_tracking": {"enabled": False},
            "flow_dit": {"hidden_dim": 1024, "num_layers": 8, "heads": 16, "num_inference_steps": 16, "dropout": 0.1, "num_eval_samples": 1, "time_sampling": "beta", "noise_beta_alpha": 1.5, "noise_beta_beta": 1.0, "noise_s": 0.999, "vl_mixer_layers": 4, "interleave_self_attention": True},
        },
        "train": {
            "seed": 42,
            "batch_size": 8,
            "epochs": 50,
            "num_workers": 4,
            "amp": True,
            "amp_dtype": "bfloat16",
            "grad_clip_norm": 1.0,
            "shuffle": True,
            "optimizer": {
                "name": "adamw",
                "lr": 1.0e-4,
                "backbone_lr_multiplier": 0.1,
                "weight_decay": 0.05,
                "betas": [0.9, 0.95],
            },
            "scheduler": {"name": "none", "warmup_ratio": 0.0, "min_lr_ratio": 0.0},
            "losses": {"action_weight": 1.0, "gripper_weight": 0.2, "terminate_weight": 0.1},
            "logging": {"log_every": 50},
            "eval_every_epochs": 1,
            "debug": {"anomaly_detection": False, "debug_isfinite": False, "debug_norms": False},
            "checkpoint": {"save_every_epochs": 1, "strict_load": True},
        },
        "infer": {"split": "val", "episode_id": "", "start_index": 0, "save_path": ""},
    }


def _parse_simple_yaml(text: str) -> dict[str, Any]:
    # Fallback parser for the limited YAML used by this project. Prefer PyYAML when installed.
    lines = []
    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        lines.append(raw.rstrip())
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for line in lines:
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()
        if ":" not in stripped:
            continue
        key, val = stripped.split(":", 1)
        key = key.strip()
        val = val.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if val == "":
            new: dict[str, Any] = {}
            parent[key] = new
            stack.append((indent, new))
        else:
            parent[key] = _parse_scalar(val)
    return root


def _parse_scalar(val: str) -> Any:
    if val in ("null", "None", "~"):
        return None
    if val in ("true", "True"):
        return True
    if val in ("false", "False"):
        return False
    if val.startswith("[") and val.endswith("]"):
        return json.loads(val.replace("'", '"'))
    try:
        if any(c in val for c in ".eE"):
            return float(val)
        return int(val)
    except ValueError:
        return val.strip('"').strip("'")


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        raw = json.loads(text)
    else:
        try:
            import yaml  # type: ignore
        except ModuleNotFoundError:
            raw = _parse_simple_yaml(text)
        else:
            raw = yaml.safe_load(text)
    cfg = deep_update(default_config(), raw or {})
    cfg["_config_path"] = str(path)
    return cfg
