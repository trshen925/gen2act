#!/usr/bin/env python3
"""Render the exact photometric augmentation used by a training config."""
from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from r2r_gen2act.config.load import load_config
from r2r_gen2act.data.factories import build_dataset
from r2r_gen2act.data.transforms import apply_image_augmentation


def _font(size: int) -> ImageFont.ImageFont:
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()


def _to_image(frame: torch.Tensor, gain: float = 1.0) -> Image.Image:
    array = frame.detach().cpu().permute(1, 2, 0).numpy()
    array = (np.clip(array * gain, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def _contact_sheet(rows: list[tuple[str, list[Image.Image]]], title: str) -> Image.Image:
    cell_w, cell_h = rows[0][1][0].size
    label_w, title_h, gap = 190, 52, 4
    columns = max(len(images) for _, images in rows)
    canvas = Image.new(
        "RGB",
        (label_w + columns * (cell_w + gap) - gap, title_h + len(rows) * (cell_h + gap) - gap),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 12), title, fill="black", font=_font(22))
    for row_index, (label, images) in enumerate(rows):
        y = title_h + row_index * (cell_h + gap)
        draw.text((10, y + cell_h // 2 - 10), label, fill="black", font=_font(18))
        for column, image in enumerate(images):
            canvas.paste(image, (label_w + column * (cell_w + gap), y))
    return canvas


def _metrics(original: torch.Tensor, augmented: torch.Tensor) -> dict[str, float]:
    delta = (augmented - original).abs()
    return {
        "mean_abs_pixel_delta": float(delta.mean()),
        "rms_pixel_delta": float(delta.square().mean().sqrt()),
        "max_abs_pixel_delta": float(delta.max()),
        "augmented_pixel_mean": float(augmented.mean()),
        "augmented_pixel_std": float(augmented.std()),
        "fraction_clipped_low": float((augmented <= 0.0).float().mean()),
        "fraction_clipped_high": float((augmented >= 1.0).float().mean()),
    }


def _save_outputs(
    stream: str,
    original: torch.Tensor,
    variants: dict[int, torch.Tensor],
    output_dir: Path,
    stem: str,
    diff_gain: float,
) -> tuple[list[Path], dict[str, dict[str, float]]]:
    written: list[Path] = []
    original_images = [_to_image(frame) for frame in original]
    overview_rows = [("original", original_images)]
    stream_metrics: dict[str, dict[str, float]] = {}

    for seed, augmented in variants.items():
        augmented_images = [_to_image(frame) for frame in augmented]
        diff_images = [_to_image((augmented - original).abs()[index], diff_gain) for index in range(len(original))]
        metrics = _metrics(original, augmented)
        stream_metrics[str(seed)] = metrics
        overview_rows.append((f"seed {seed}", augmented_images))
        detail = _contact_sheet(
            [("original", original_images), (f"train aug {seed}", augmented_images),
             (f"abs diff x{diff_gain:g}", diff_images)],
            f"{stream} | mean abs pixel delta {metrics['mean_abs_pixel_delta']:.5f}",
        )
        detail_path = output_dir / f"{stem}_{stream}_seed{seed}_detail.png"
        detail.save(detail_path)
        written.append(detail_path)

    overview = _contact_sheet(overview_rows, f"{stream} | all frames and random variants")
    overview_path = output_dir / f"{stem}_{stream}_overview.png"
    overview.save(overview_path)
    written.insert(0, overview_path)
    return written, stream_metrics


def _write_html(path: Path, title: str, metadata: dict, image_paths: list[Path]) -> None:
    metadata_text = html.escape(json.dumps(metadata, indent=2, ensure_ascii=True))
    sections = []
    for image_path in image_paths:
        name = html.escape(image_path.name)
        sections.append(f'<h2>{name}</h2><a href="{name}"><img src="{name}" loading="lazy"></a>')
    document = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{html.escape(title)}</title>
<style>body{{font:15px sans-serif;margin:24px;background:#f5f5f5;color:#111}}
img{{max-width:100%;height:auto;background:white;border:1px solid #bbb}}
pre{{background:white;padding:16px;overflow:auto;border:1px solid #ccc}} h2{{margin-top:32px}}</style>
</head><body><h1>{html.escape(title)}</h1><pre>{metadata_text}</pre>{''.join(sections)}</body></html>"""
    path.write_text(document, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/droidexFULL_C36_no_frontdepth_diffuse_gripper_fulltrain.yaml",
    )
    parser.add_argument("--episode-id", default=None, help="Clip ID; defaults to a valid val sample.")
    parser.add_argument("--start-index", type=int, default=30, help="Target-window start index.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--diff-gain", type=float, default=5.0)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/c36_train_augmentation")
    args = parser.parse_args()

    cfg = load_config(args.config)
    augmentation_cfg = cfg["data"].get("augmentation", {})
    if not augmentation_cfg.get("enabled", False):
        raise ValueError(f"Image augmentation is disabled in {args.config}")

    # Val gives us the unmodified tensors. We then replay the exact train-time
    # transform and call order below, without depending on dataset worker state.
    dataset = build_dataset(cfg, "val")
    if args.episode_id is None:
        episode_id, start_index = min(dataset._samples, key=lambda item: abs(item[1] - args.start_index))
    else:
        episode_id, start_index = args.episode_id, args.start_index
        if episode_id not in dataset._episode_by_id:
            examples = ", ".join(sorted(dataset._episode_by_id)[:5])
            raise ValueError(f"Episode {episode_id!r} is not in the val split; examples: {examples}")
    sample = dataset.sample_window(episode_id, start_index)
    streams = {
        "source": sample["source_video"],
        "front_history": sample["target_history"],
    }
    if "wrist_current" in sample:
        streams["wrist_history"] = sample["wrist_current"]

    variants = {name: {} for name in streams}
    for seed in args.seeds:
        torch.manual_seed(seed)
        for name, video in streams.items():
            variants[name][seed] = apply_image_augmentation(video.clone(), augmentation_cfg)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{episode_id}_start{int(sample['start_index']):05d}"
    metadata = {
        "config": str(args.config.resolve()),
        "episode_id": episode_id,
        "requested_start_index": args.start_index,
        "actual_start_index": int(sample["start_index"]),
        "target_step": int(sample["target_step"]),
        "seeds": args.seeds,
        "augmentation": augmentation_cfg,
        "structural_augmentation": cfg["data"].get("structural_augmentation"),
        "note": "Each stream uses the repository's real apply_image_augmentation function.",
        "streams": {},
    }
    image_paths: list[Path] = []
    for stream, original in streams.items():
        paths, values = _save_outputs(stream, original, variants[stream], args.output_dir, stem, args.diff_gain)
        image_paths.extend(paths)
        metadata["streams"][stream] = {
            "shape": list(original.shape),
            "original_pixel_mean": float(original.mean()),
            "original_pixel_std": float(original.std()),
            "variants": values,
        }

    report_json = args.output_dir / f"{stem}_report.json"
    report_json.write_text(json.dumps(metadata, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    report_html = args.output_dir / "index.html"
    _write_html(report_html, "C36 training image augmentation", metadata, image_paths)

    print(f"config={args.config.resolve()}")
    print(f"episode={episode_id} start_index={sample['start_index']} target_step={sample['target_step']}")
    print(f"augmentation={augmentation_cfg}")
    for stream, info in metadata["streams"].items():
        deltas = [variant["mean_abs_pixel_delta"] for variant in info["variants"].values()]
        print(f"{stream}: shape={info['shape']} mean_abs_delta_range=[{min(deltas):.6f}, {max(deltas):.6f}]")
    print(f"html={report_html.resolve()}")
    print(f"json={report_json.resolve()}")


if __name__ == "__main__":
    main()
