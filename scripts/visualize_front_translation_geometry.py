#!/usr/bin/env python3
"""Visualize a front-image translation and its synchronized 2D EE labels.

This is an inspection tool only. It does not alter a training dataset or config.
The action labels remain camera-frame 3D deltas; their projected current/future
EE points move with the same image translation as the front observation.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from r2r_gen2act.config.load import load_config
from r2r_gen2act.data.factories import build_dataset
from r2r_gen2act.data.transforms import image_to_tensor


def _font(size: int) -> ImageFont.ImageFont:
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()


def translate_reflect(video: torch.Tensor, dx: int, dy: int) -> torch.Tensor:
    """Translate [T,C,H,W] in pixels while retaining output size via reflection."""
    h, w = video.shape[-2:]
    pad_x, pad_y = abs(dx), abs(dy)
    padded = F.pad(video, (pad_x, pad_x, pad_y, pad_y), mode="reflect")
    left = pad_x - dx
    top = pad_y - dy
    return padded[..., top:top + h, left:left + w]


def _to_image(frame: torch.Tensor) -> Image.Image:
    array = frame.detach().cpu().permute(1, 2, 0).numpy()
    return Image.fromarray((np.clip(array, 0.0, 1.0) * 255.0).round().astype(np.uint8), "RGB")


def _normalized_to_pixel(uv: np.ndarray, size: int) -> np.ndarray:
    return (np.asarray(uv, dtype=np.float32) + 1.0) * 0.5 * float(size - 1)


def _raw_to_model_pixels(points: np.ndarray, raw_h: int, raw_w: int, size: int) -> np.ndarray:
    """Map raw pixels through the unchanged repository resize_center_crop geometry."""
    out = np.asarray(points, dtype=np.float32).copy()
    if raw_h < raw_w:
        scale = size / float(raw_h)
        left = max(0, (int(round(raw_w * scale)) - size) // 2)
        out[..., 0] = out[..., 0] * scale - left
        out[..., 1] = out[..., 1] * scale
    else:
        scale = size / float(raw_w)
        top = max(0, (int(round(raw_h * scale)) - size) // 2)
        out[..., 0] = out[..., 0] * scale
        out[..., 1] = out[..., 1] * scale - top
    return out


def _model_to_raw_pixels(points: np.ndarray, raw_h: int, raw_w: int, size: int) -> np.ndarray:
    """Inverse of _raw_to_model_pixels, used only to visualize GT projections."""
    out = np.asarray(points, dtype=np.float32).copy()
    if raw_h < raw_w:
        scale = size / float(raw_h)
        left = max(0, (int(round(raw_w * scale)) - size) // 2)
        out[..., 0] = (out[..., 0] + left) / scale
        out[..., 1] = out[..., 1] / scale
    else:
        scale = size / float(raw_w)
        top = max(0, (int(round(raw_h * scale)) - size) // 2)
        out[..., 0] = out[..., 0] / scale
        out[..., 1] = (out[..., 1] + top) / scale
    return out


def _draw_path(image: Image.Image, points: np.ndarray, current: np.ndarray, title: str) -> Image.Image:
    out = image.copy()
    draw = ImageDraw.Draw(out)
    points = np.asarray(points, dtype=np.float32)
    for i in range(len(points) - 1):
        p0, p1 = tuple(points[i]), tuple(points[i + 1])
        draw.line((p0, p1), fill=(0, 220, 255), width=3)
    for i, point in enumerate(points):
        x, y = map(float, point)
        radius = 4
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(255, 80, 0), outline="white", width=1)
        if i in (0, len(points) - 1):
            draw.text((x + 5, y + 4), f"t+{i + 1}", fill="white", stroke_width=2, stroke_fill="black", font=_font(14))
    x, y = map(float, current)
    draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill=(0, 255, 0), outline="black", width=1)
    draw.text((7, 7), title, fill="white", stroke_width=2, stroke_fill="black", font=_font(16))
    return out


def _raw_debug_panel(raw: torch.Tensor, dx: int, dy: int, size: int) -> Image.Image:
    """Show the native translated frame and the pre-existing C37 crop window."""
    image = _to_image(raw)
    draw = ImageDraw.Draw(image)
    raw_h, raw_w = raw.shape[-2:]
    if raw_h < raw_w:
        scale = size / float(raw_h)
        crop_w = size / scale
        left = (raw_w - crop_w) * 0.5
        draw.rectangle((left, 0, left + crop_w, raw_h - 1), outline=(255, 230, 0), width=3)
        note = "yellow = existing center-crop window after resize"
    else:
        scale = size / float(raw_w)
        crop_h = size / scale
        top = (raw_h - crop_h) * 0.5
        draw.rectangle((0, top, raw_w - 1, top + crop_h), outline=(255, 230, 0), width=3)
        note = "yellow = existing center-crop window after resize"
    draw.text((7, 7), f"native {raw_w}x{raw_h}, dx={dx:+d}, dy={dy:+d}", fill="white", stroke_width=2, stroke_fill="black", font=_font(15))
    draw.text((7, raw_h - 22), note, fill="white", stroke_width=2, stroke_fill="black", font=_font(13))
    return image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/droidexFULL_C37_dense15_current_gripper_singleflow_fulltrain.yaml")
    parser.add_argument("--episode-id", default=None)
    parser.add_argument("--start-index", type=int, default=30)
    parser.add_argument("--translation-frac", type=float, default=0.10)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/c37_front_translation_geometry")
    args = parser.parse_args()

    if not 0.0 < args.translation_frac < 0.5:
        raise ValueError("--translation-frac must be in (0, 0.5)")
    cfg = load_config(args.config)
    dataset = build_dataset(cfg, "val")
    if args.episode_id is None:
        episode_id, start_index = min(dataset._samples, key=lambda item: abs(item[1] - args.start_index))
    else:
        episode_id, start_index = args.episode_id, args.start_index
    sample = dataset.sample_window(episode_id, start_index)
    episode = dataset._episode_by_id[episode_id]
    payload = dataset._read_action_payload(episode)
    target_step = int(sample["target_step"])
    h, w = sample["target_history"].shape[-2:]
    reader = imageio.get_reader(str(episode.target_video_path))
    raw_array = reader.get_data(target_step)
    reader.close()
    raw_h, raw_w = raw_array.shape[:2]
    raw = torch.from_numpy(raw_array).permute(2, 0, 1).float().div(255.0)
    shift_x = int(round(args.translation_frac * raw_w))
    shift_y = int(round(args.translation_frac * raw_h))
    # Translate at the native rectangular resolution. There is no augmentation
    # crop: C37's existing image_to_tensor preprocessing happens afterwards.
    shifts = [(shift_x, shift_y), (shift_x, -shift_y), (-shift_x, shift_y), (-shift_x, -shift_y)]
    horizon = int(cfg["action"]["chunk_size"])
    future_steps = [min(target_step + k, episode.num_steps - 1) for k in range(1, horizon + 1)]
    current_uv = dataset._project_ee_to_normalized_image(payload, target_step, dataset.proprioception_cfg)[:2]
    future_uv = np.stack([dataset._project_ee_to_normalized_image(payload, step, dataset.proprioception_cfg)[:2] for step in future_steps])
    current_px = _normalized_to_pixel(current_uv, w)
    future_px = _normalized_to_pixel(future_uv, w)
    original = image_to_tensor(raw_array, h)
    raw_current = _model_to_raw_pixels(current_px, raw_h, raw_w, h)
    raw_future = _model_to_raw_pixels(future_px, raw_h, raw_w, h)
    images = []
    raw_panels = []
    for dx, dy in shifts:
        translated_raw = translate_reflect(raw.unsqueeze(0), dx, dy)[0]
        raw_panels.append(_raw_debug_panel(translated_raw, dx, dy, h))
        translated = image_to_tensor(translated_raw.permute(1, 2, 0).numpy(), h)
        shifted_current = _raw_to_model_pixels(raw_current + [dx, dy], raw_h, raw_w, h)
        shifted_future = _raw_to_model_pixels(raw_future + [dx, dy], raw_h, raw_w, h)
        title = f"native dx={dx:+d}px ({dx / raw_w:+.0%}), dy={dy:+d}px ({dy / raw_h:+.0%})"
        images.append(_draw_path(_to_image(translated), shifted_future, shifted_current, title))
    original_overlay = _draw_path(
        _to_image(original), future_px, current_px,
        "original | green=current, cyan/orange=future t+1..t+15",
    )
    cell_w, cell_h = original_overlay.size
    canvas = Image.new("RGB", (cell_w * 2 + 12, cell_h * 3 + 106), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 10), "C37 front-image translation: image and 2D EE supervision move together", fill="black", font=_font(23))
    draw.text((10, 43), "Translation uses reflection padding on the native rectangular frame; it does not crop. Then the unchanged C37 resize-center-crop makes the 224x224 input.", fill="black", font=_font(14))
    draw.text((10, 64), "3D action values remain unchanged. The current EE 2D condition and projected action endpoints receive the same native-frame translation.", fill="black", font=_font(14))
    all_images = [original_overlay] + images
    for index, image in enumerate(all_images):
        row, col = divmod(index, 2)
        canvas.paste(image, (col * (cell_w + 12), 106 + row * (cell_h + 4)))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / f"{episode_id}_start{int(sample['start_index']):05d}_translation_sweep.png"
    canvas.save(path)
    raw_canvas = Image.new("RGB", (raw_w * 2 + 12, raw_h * 2 + 56), "white")
    raw_draw = ImageDraw.Draw(raw_canvas)
    raw_draw.text((8, 8), "Native-frame translation (reflection padding), before unchanged C37 resize-center-crop", fill="black", font=_font(18))
    for index, panel in enumerate(raw_panels):
        row, col = divmod(index, 2)
        raw_canvas.paste(panel, (col * (raw_w + 12), 40 + row * (raw_h + 4)))
    raw_path = args.output_dir / f"{episode_id}_start{int(sample['start_index']):05d}_native_translation_and_existing_crop_window.png"
    raw_canvas.save(raw_path)
    report = {
        "config": "configs/droidexFULL_C37_dense15_current_gripper_singleflow_fulltrain.yaml",
        "episode_id": episode_id,
        "start_index": int(sample["start_index"]),
        "target_step": target_step,
        "translation_frac": args.translation_frac,
        "raw_frame_shape_hwc": [raw_h, raw_w, 3],
        "raw_pixel_shift_xy": [shift_x, shift_y],
        "augmentation_crop": "none; native-size translation with reflection padding",
        "model_preprocessing": "unchanged image_to_tensor -> resize_center_crop",
        "horizon": horizon,
        "future_steps": future_steps,
        "meaning": "The front image, current EE 2D proprioception, and projected 3D action endpoints use the identical (dx,dy) pixel transform. The underlying camera-frame 3D action values are intentionally unchanged.",
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(path.resolve())


if __name__ == "__main__":
    main()
