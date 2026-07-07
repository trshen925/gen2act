"""HAMSTER-style trajectory overlay: render the demo EE 2D path onto the current frame so the (frozen)
DINOv2 backbone can ground it to the scene — the model can visually localize "where am I on the path"
and read the future direction from the demo, instead of falling back to motion momentum.

Coordinates: the EE path comes from the projection in ORIGINAL frame pixels (W,H). The displayed frame
is resize_center_crop'd to a square image_size S. `raw_to_display_px` applies the SAME geometry so the
drawn path aligns with the cropped image. We draw on the final float [0,1] HWC frame (after augmentation)
so the path stays crisp.
"""
from __future__ import annotations

import numpy as np


def raw_to_display_px(path_px: np.ndarray, w: int, h: int, image_size: int) -> np.ndarray:
    """Map original-frame pixels [T,2]=(x,y) to the resize_center_crop'd image_size display pixels.
    Mirrors transforms.resize_center_crop exactly (shorter side -> image_size, center crop)."""
    s = int(image_size)
    out = np.asarray(path_px, dtype=np.float64).copy()
    if h < w:
        scale = s / float(h)
        new_w = int(round(w * scale))
        left = max(0, (new_w - s) // 2)
        top = 0
    else:
        scale = s / float(w)
        new_h = int(round(h * scale))
        top = max(0, (new_h - s) // 2)
        left = 0
    out[:, 0] = out[:, 0] * scale - left
    out[:, 1] = out[:, 1] * scale - top
    return out


def _blend_disk(img: np.ndarray, cx: float, cy: float, radius: float, color) -> None:
    """Paint a filled disk (color in [0,1]) onto float HWC image [0,1], in place."""
    h, w = img.shape[:2]
    x0, x1 = max(0, int(np.floor(cx - radius))), min(w, int(np.ceil(cx + radius + 1)))
    y0, y1 = max(0, int(np.floor(cy - radius))), min(h, int(np.ceil(cy + radius + 1)))
    if x0 >= x1 or y0 >= y1:
        return
    ys, xs = np.mgrid[y0:y1, x0:x1]
    mask = (xs - cx) ** 2 + (ys - cy) ** 2 <= radius ** 2
    if mask.any():
        img[y0:y1, x0:x1][mask] = np.asarray(color, dtype=img.dtype)


def draw_trajectory(frame_hwc: np.ndarray, path_disp_px: np.ndarray, radius: float = 1.6,
                    mark_idx: int | None = None) -> np.ndarray:
    """Draw a blue->red (time) polyline of path_disp_px [T,2] on a float HWC [0,1] frame, in place.
    Optionally mark one index (e.g. the current target_step) with a green dot for alignment checks."""
    out = frame_hwc
    t = len(path_disp_px)
    if t < 2:
        return out
    for i in range(t - 1):
        p0 = path_disp_px[i]
        p1 = path_disp_px[i + 1]
        f = i / (t - 1)
        color = (f, 0.0, 1.0 - f)  # start=blue (0,0,1) -> end=red (1,0,0)
        d = float(np.hypot(p1[0] - p0[0], p1[1] - p0[1]))
        steps = max(1, int(d / max(0.5, radius)) + 1)
        for s in range(steps + 1):
            tt = s / steps
            px = p0[0] * (1 - tt) + p1[0] * tt
            py = p0[1] * (1 - tt) + p1[1] * tt
            _blend_disk(out, px, py, radius, color)
    if mark_idx is not None and 0 <= mark_idx < t:
        _blend_disk(out, path_disp_px[mark_idx][0], path_disp_px[mark_idx][1], radius + 1.5, (0.0, 1.0, 0.0))
    return out
