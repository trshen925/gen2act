from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from r2r_gen2act.data.action.rotation import EULER_CONVENTION


def _camera_serial(calibration: dict, explicit: str = "") -> str:
    if explicit:
        return explicit
    candidates = [k for k, v in calibration.get("extrinsics", {}).items() if isinstance(v, list) and len(v) == 6]
    if len(candidates) != 1:
        raise ValueError("Expected one 6D extrinsic; pass --camera-serial to disambiguate")
    return candidates[0]


def _intrinsics_for_image(calibration: dict, serial: str, image_shape: tuple[int, int]) -> tuple[float, float, float, float]:
    intr = calibration["intrinsics"][serial]
    fx, cx, fy, cy = [float(x) for x in intr["cameraMatrix"]]
    h, w = image_shape
    sx = w / float(intr["width"])
    sy = h / float(intr["height"])
    return fx * sx, cx * sx, fy * sy, cy * sy


def project_points(payload: dict, serial: str, convention: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    image_shape = tuple(int(x) for x in payload["image_shape"][:2])
    cal = payload["calibration"]
    fx, cx, fy, cy = _intrinsics_for_image(cal, serial, image_shape)
    extrinsic = np.asarray(cal["extrinsics"][serial], dtype=np.float64)
    r = Rotation.from_euler(EULER_CONVENTION, extrinsic[3:6]).as_matrix()
    t = extrinsic[:3]
    points_base = np.asarray(payload["observations"]["cartesian_position"], dtype=np.float64)[:, :3]
    if convention == "camera_pose_in_base":
        points_camera = (r.T @ (points_base - t).T).T
    elif convention == "base_to_camera":
        points_camera = (r @ points_base.T).T + t
    else:
        raise ValueError(f"Unknown convention: {convention}")
    z = points_camera[:, 2]
    uv = np.empty((points_camera.shape[0], 2), dtype=np.float64)
    uv[:, 0] = fx * points_camera[:, 0] / z + cx
    uv[:, 1] = fy * points_camera[:, 1] / z + cy
    valid = np.isfinite(uv).all(axis=1) & (z > 1e-6)
    return uv, z, valid


def _draw_circle(img: np.ndarray, x: int, y: int, radius: int, color: tuple[int, int, int]) -> None:
    h, w = img.shape[:2]
    r2 = radius * radius
    for yy in range(max(0, y - radius), min(h, y + radius + 1)):
        for xx in range(max(0, x - radius), min(w, x + radius + 1)):
            if (xx - x) * (xx - x) + (yy - y) * (yy - y) <= r2:
                img[yy, xx, :3] = color


def _draw_line(img: np.ndarray, p0: tuple[int, int], p1: tuple[int, int], color: tuple[int, int, int]) -> None:
    x0, y0 = p0
    x1, y1 = p1
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    h, w = img.shape[:2]
    while True:
        if 0 <= x0 < w and 0 <= y0 < h:
            img[y0, x0, :3] = color
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy


def overlay_projection(frame: np.ndarray, uv: np.ndarray, valid: np.ndarray, step: int, trail: int) -> np.ndarray:
    img = np.asarray(frame).copy()
    if img.ndim == 2:
        img = np.repeat(img[..., None], 3, axis=-1)
    if img.shape[-1] == 4:
        img = img[..., :3]
    h, w = img.shape[:2]
    lo = max(0, step - trail)
    pts: list[tuple[int, int]] = []
    for i in range(lo, min(step + 1, len(uv))):
        if not valid[i]:
            continue
        x, y = int(round(uv[i, 0])), int(round(uv[i, 1]))
        if 0 <= x < w and 0 <= y < h:
            pts.append((x, y))
    for a, b in zip(pts[:-1], pts[1:]):
        _draw_line(img, a, b, (0, 255, 255))
    for p in pts[:-1]:
        _draw_circle(img, p[0], p[1], 2, (255, 180, 0))
    if pts:
        _draw_circle(img, pts[-1][0], pts[-1][1], 5, (255, 0, 0))
        _draw_circle(img, pts[-1][0], pts[-1][1], 2, (255, 255, 255))
    return img


def main() -> None:
    parser = argparse.ArgumentParser(description="Project DROID end-effector positions into gt.mp4 frames.")
    parser.add_argument("--root", type=Path, default=Path("/mnt/pfs/share/shentingrui/dataset/droid-ex/droid_2000_with_extrinsics_filtered"))
    parser.add_argument("--episode", type=str, required=True, help="Episode directory name, e.g. 0001 or 3021")
    parser.add_argument("--frame", type=int, default=0, help="Frame / observation step for single-image output")
    parser.add_argument("--output", type=Path, default=None, help="Output png/mp4 path")
    parser.add_argument("--all", action="store_true", help="Write an overlay video for all frames instead of one PNG")
    parser.add_argument("--trail", type=int, default=20, help="Number of previous projected points to draw")
    parser.add_argument("--camera-serial", type=str, default="")
    parser.add_argument("--convention", choices=["camera_pose_in_base", "base_to_camera"], default="camera_pose_in_base")
    args = parser.parse_args()

    episode_dir = args.root / args.episode
    payload = json.loads((episode_dir / "data.json").read_text(encoding="utf-8"))
    serial = _camera_serial(payload["calibration"], args.camera_serial)
    uv, z, valid = project_points(payload, serial, args.convention)
    video_path = episode_dir / "gt.mp4"
    reader = imageio.get_reader(str(video_path), format="ffmpeg", input_params=["-threads", "1"], output_params=["-threads", "1"])
    output = args.output
    if output is None:
        suffix = "mp4" if args.all else "png"
        output = ROOT / "outputs" / "projection_checks" / f"{args.episode}_ee_projection.{suffix}"
    output.parent.mkdir(parents=True, exist_ok=True)

    h, w = payload["image_shape"][:2]
    inside = valid & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    print(f"episode={args.episode} serial={serial} convention={args.convention}")
    print(f"projected_inside={inside.sum()}/{len(inside)} z_range=({z.min():.4f},{z.max():.4f}) output={output}")

    if args.all:
        frames = []
        n = int(payload["num_steps"])
        for step in range(n):
            try:
                frame = reader.get_data(step)
            except Exception:
                break
            frames.append(overlay_projection(frame, uv, valid, step, args.trail))
        imageio.mimsave(str(output), frames, fps=int(payload.get("fps", 15)), macro_block_size=1)
    else:
        step = min(max(0, int(args.frame)), int(payload["num_steps"]) - 1)
        frame = reader.get_data(step)
        imageio.imwrite(str(output), overlay_projection(frame, uv, valid, step, args.trail))

    try:
        reader.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
