#!/usr/bin/env python3
"""Render representative C37 training tensors alongside the raw front crop window."""
from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from r2r_gen2act.config.load import load_config
from r2r_gen2act.data.factories import build_dataset


def _font(size: int) -> ImageFont.ImageFont:
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()


def _image(frame: torch.Tensor) -> Image.Image:
    array = frame.detach().cpu().permute(1, 2, 0).numpy()
    return Image.fromarray((np.clip(array, 0.0, 1.0) * 255.0).round().astype(np.uint8), "RGB")


def _row(title: str, frames: list[Image.Image]) -> Image.Image:
    width, height = frames[0].size
    label, gap = 180, 4
    out = Image.new("RGB", (label + len(frames) * (width + gap) - gap, height), "white")
    draw = ImageDraw.Draw(out)
    draw.multiline_text((8, height // 2 - 18), title, fill="black", font=_font(17), spacing=3)
    for index, frame in enumerate(frames):
        out.paste(frame, (label + index * (width + gap), 0))
    return out


def _raw_crop_panel(raw: np.ndarray, size: int, stream: str) -> tuple[Image.Image, dict]:
    h, w = raw.shape[:2]
    image = Image.fromarray(raw[..., :3]).convert("RGB")
    draw = ImageDraw.Draw(image)
    if h < w:
        scale = size / float(h)
        keep_w = size / scale
        left = (w - keep_w) / 2.0
        box = (left, 0.0, left + keep_w, float(h - 1))
    else:
        scale = size / float(w)
        keep_h = size / scale
        top = (h - keep_h) / 2.0
        box = (0.0, top, float(w - 1), top + keep_h)
    draw.rectangle(box, outline=(255, 230, 0), width=max(2, min(h, w) // 80))
    draw.text((6, 6), f"{stream} raw {w}x{h}; yellow = kept by resize-center-crop", fill="white", stroke_width=2, stroke_fill="black", font=_font(15))
    return image, {"raw_shape_hwc": [h, w, 3], "kept_raw_box_xyxy": [round(x, 2) for x in box]}


def _read_video_frame(path: Path, step: int) -> np.ndarray:
    reader = imageio.get_reader(str(path))
    raw = reader.get_data(step)
    reader.close()
    return raw


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/droidexFULL_C37_dense15_current_gripper_singleflow_fulltrain.yaml")
    parser.add_argument("--count", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/c37_train_model_inputs")
    args = parser.parse_args()
    cfg = load_config(args.config)
    torch.manual_seed(args.seed)
    # Building the complete train split scans tens of thousands of episodes and
    # validates every calibration. Use frozen validation episodes for quick
    # inspection, then deliberately enable the exact train-only transforms.
    dataset = build_dataset(cfg, "val")
    dataset.split = "train"
    count = min(args.count, len(dataset._samples))
    positions = np.linspace(0, len(dataset._samples) - 1, count, dtype=int)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {"config": str(args.config), "episodes": "frozen val episodes for fast inspection", "transforms": "exact train-time window jitter and photometric augmentation", "seed": args.seed, "note": "Yellow boxes are only an explanation of the pre-existing resize-center-crop; they are not extra augmentation.", "cases": []}
    links = []
    for ordinal, position in enumerate(positions, 1):
        episode_id, nominal_start = dataset._samples[int(position)]
        sample = dataset.sample_window(episode_id, nominal_start)
        episode = dataset._episode_by_id[episode_id]
        raw, geometry = _raw_crop_panel(
            _read_video_frame(episode.target_video_path, int(sample["target_step"])),
            int(cfg["data"]["image_size"]), "front")
        raw.thumbnail((640, 360), Image.Resampling.LANCZOS)
        offsets = ", ".join(str(x) for x in cfg["data"].get("current_history_offsets", [0]))
        rows = [_row("raw front\nyellow kept area", [raw]), _row(f"model front history\n[{offsets}]", [_image(x) for x in sample["target_history"]]), _row("model source\n8 demo frames", [_image(x) for x in sample["source_video"]])]
        if "wrist_current" in sample:
            wrist_step = max(0, int(sample["target_step"]))
            wrist_path = episode.metadata_path.parent / str(cfg["data"]["wrist_current"]["frames_subdir"]) / f"{wrist_step:06d}.{cfg['data']['wrist_current']['frames_ext']}"
            raw_wrist, wrist_geometry = _raw_crop_panel(
                imageio.imread(str(wrist_path)), int(cfg["data"]["image_size"]), "wrist")
            raw_wrist.thumbnail((640, 360), Image.Resampling.LANCZOS)
            rows.append(_row("raw wrist\nyellow kept area", [raw_wrist]))
            rows.append(_row(f"model wrist history\n[{offsets}]", [_image(x) for x in sample["wrist_current"]]))
        width = max(row.width for row in rows)
        height = 58 + sum(row.height + 5 for row in rows)
        canvas = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text((10, 10), f"Case {ordinal}: episode={episode_id}, nominal start={nominal_start}, actual train start={sample['start_index']}, target step={sample['target_step']}", fill="black", font=_font(17))
        y = 42
        for row in rows:
            canvas.paste(row, (0, y))
            y += row.height + 5
        name = f"case{ordinal:02d}_{episode_id}_start{int(sample['start_index']):05d}.png"
        canvas.save(args.output_dir / name)
        report["cases"].append({"file": name, "episode_id": episode_id, "nominal_start_index": int(nominal_start), "actual_start_index": int(sample["start_index"]), "target_step": int(sample["target_step"]), "front": geometry, "wrist": wrist_geometry if "wrist_current" in sample else None})
        links.append(name)
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    body = '<!doctype html><html><head><meta charset="utf-8"><title>C37 train inputs</title><style>body{font:16px sans-serif;margin:24px;background:#f5f5f5}img{max-width:100%;border:1px solid #bbb;background:white}h2{margin-top:36px}</style></head><body><h1>C37: real training inputs</h1><p>Rows named "model" are the exact 224x224 tensors delivered by the train dataset (after existing resize-center-crop and current photometric augmentation). Yellow rectangle is the raw region surviving the existing preprocessing.</p>'
    for name in links:
        safe = html.escape(name)
        body += f'<h2>{safe}</h2><a href="{safe}"><img src="{safe}"></a>'
    (args.output_dir / "index.html").write_text(body + "</body></html>", encoding="utf-8")
    print((args.output_dir / "index.html").resolve())


if __name__ == "__main__":
    main()
