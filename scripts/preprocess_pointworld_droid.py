#!/usr/bin/env python3
"""
Preprocess PointWorld-DROID + local DROID-1.0.1 into gen2act-compatible episode format.

Input:
  - PointWorld-DROID restored:  /mnt/pfs/data/fenghaoran/PointWorld-DROID_restored/droid/
      flows-fs-optimized/*_flows.h5  → robot state, camera intrinsics/extrinsics, clips
      depth_320x180/*_depth.h5       → per-frame uint16 depth (mm), all cameras
      cameras/*_cameras.json         → optimized 4×4 extrinsics
  - Local DROID 1.0.1:  /mnt/pfs/data/fenghaoran/droid/decompressed/1.0.1/episode_NNNNNN/
      episode.parquet                → robot state (cartesian_position, gripper_position)
      steps_observation_exterior_image_1_left.mp4
      steps_observation_exterior_image_2_left.mp4
      steps_observation_wrist_image_left.mp4

Output per episode (--output-dir/{uuid}/):
  gt.mp4             → symlink to exterior_image_1_left.mp4  (ext1 camera)
  depth_frames/
    {serial}_{i:06d}.png   → 16-bit grayscale PNG, depth in mm (uint16)
  data.json          → robot state + camera calibration (droid-ex compatible format)

Usage:
  python scripts/preprocess_pointworld_droid.py \
      --output-dir /mnt/pfs/share/shentingrui/dataset/pointworld-droid-3000 \
      --max-episodes 3000 \
      --workers 16
"""

import argparse
import ast
import json
import re
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import h5py
import imageio.v2 as imageio
import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation


# ── paths ──────────────────────────────────────────────────────────────────
PW_ROOT    = Path('/mnt/pfs/data/fenghaoran/PointWorld-DROID_restored/droid')
DROID_ROOT = Path('/mnt/pfs/data/fenghaoran/droid/decompressed/1.0.1')
EULER_CONV = 'xyz'   # must match existing gen2act code (EULER_CONVENTION)


# ── index building ─────────────────────────────────────────────────────────

def _local_key(ep_dir: Path) -> tuple[str, str] | tuple[None, None]:
    """Extract {Lab}/success/{date}/{timestamp} from local DROID episode metadata."""
    try:
        meta = json.loads((ep_dir / 'metadata.json').read_text())
        folder = meta['context'].get('episode_metadata/recording_folderpath', '')
        m = re.search(r'r2d2-data-full/(.+?)/recordings', folder)
        if m:
            return m.group(1), str(ep_dir)
    except Exception:
        pass
    return None, None


def build_local_index(workers: int = 32) -> dict[str, str]:
    """Build {lab_key -> local_ep_path} index in parallel."""
    eps = sorted(DROID_ROOT.iterdir())
    print(f'[index] scanning {len(eps)} local DROID episodes with {workers} workers...')
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(_local_key, eps))
    index = {k: v for k, v in results if k}
    print(f'[index] built: {len(index)} entries')
    return index


def pw_episode_key(flows_h5_path: Path) -> str | None:
    """Extract matching key from PointWorld flows.h5 scene_path attribute."""
    try:
        with h5py.File(flows_h5_path, 'r') as f:
            scene_path = str(f.attrs.get('scene_path', ''))
        m = re.search(r'droid_raw/1\.0\.1/(.+?)/?$', scene_path)
        return m.group(1) if m else None
    except Exception:
        return None


# ── per-episode processing ─────────────────────────────────────────────────

def mat4x4_to_6d(mat: np.ndarray) -> list[float]:
    """Convert 4×4 SE(3) to [x,y,z,roll,pitch,yaw] (camera pose in base, xyz convention)."""
    R = mat[:3, :3]
    t = mat[:3, 3]
    euler = Rotation.from_matrix(R).as_euler(EULER_CONV)
    return [float(t[0]), float(t[1]), float(t[2]),
            float(euler[0]), float(euler[1]), float(euler[2])]


def save_depth_frames(depth_arr: np.ndarray, out_dir: Path) -> None:
    """Save (F, H, W) uint16 depth array as individual 16-bit grayscale PNGs."""
    out_dir.mkdir(parents=True, exist_ok=True)
    F = depth_arr.shape[0]
    for i in range(F):
        imageio.imwrite(str(out_dir / f'{i:06d}.png'), depth_arr[i])


def extract_rgb_frames(mp4_path: Path, out_dir: Path, image_size: int = 224) -> None:
    """Decode mp4 into JPEG frames (same as scripts/extract_frames.py)."""
    import subprocess
    out_dir.mkdir(parents=True, exist_ok=True)
    # Use ffmpeg directly for speed; output as %06d.jpg
    cmd = [
        'ffmpeg', '-y', '-i', str(mp4_path),
        '-q:v', '2',          # JPEG quality (2=high)
        '-vf', f'scale={image_size}:{image_size}:force_original_aspect_ratio=increase,'
               f'crop={image_size}:{image_size}',
        str(out_dir / '%06d.jpg'),
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(f'ffmpeg failed: {result.stderr.decode()[:200]}')


def process_episode(uuid: str, pw_flows_h5: Path, pw_depth_h5: Path,
                    pw_cameras_json: Path, local_ep_dir: Path,
                    out_ep_dir: Path) -> str:
    """Process one episode. Returns uuid on success, raises on error."""

    out_ep_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Read PointWorld flows.h5: attrs + first clip for intrinsics/extrinsics ──
    with h5py.File(pw_flows_h5, 'r') as f:
        attrs = dict(f.attrs)
        ext1_serial = str(attrs.get('ext1_cam_serial', ''))
        ext2_serial = str(attrs.get('ext2_cam_serial', ''))
        wrist_serial = str(attrs.get('wrist_cam_serial', ''))
        # read intrinsics from first available clip's ext1 camera
        clip_keys = sorted([k for k in f.keys() if ':' in k],
                            key=lambda x: int(x.split(':')[0]))
        intr_matrix = None
        for ck in clip_keys:
            cam_key = f'camera_{ext1_serial}_ext'
            if cam_key in f[ck]:
                intr_matrix = f[ck][cam_key]['intrinsic'][:]  # (3,3)
                break
        if intr_matrix is None and clip_keys:
            # fallback: any ext camera
            for ck in clip_keys:
                for cam_key in f[ck].keys():
                    if cam_key.startswith('camera_') and 'intrinsic' in f[ck][cam_key]:
                        intr_matrix = f[ck][cam_key]['intrinsic'][:]
                        # extract serial from cam_key e.g. 'camera_22008760_ext'
                        parts = cam_key.split('_')
                        if len(parts) >= 2:
                            ext1_serial = parts[1]
                        break
                if intr_matrix is not None:
                    break

    # ── 2. Read optimised extrinsics from cameras.json ──
    cameras_data = json.loads(pw_cameras_json.read_text())
    ext1_extrinsic_6d = None
    ext2_extrinsic_6d = None
    wrist_extrinsic_6d = None
    if ext1_serial in cameras_data:
        ext1_extrinsic_6d = mat4x4_to_6d(
            np.array(cameras_data[ext1_serial]['optimized_extrinsics']))
    if ext2_serial in cameras_data:
        ext2_extrinsic_6d = mat4x4_to_6d(
            np.array(cameras_data[ext2_serial]['optimized_extrinsics']))
    # wrist camera not in cameras.json (only ext cameras are optimised)
    # use raw extrinsics from flows.h5 attrs for wrist
    wrist_ext_raw = attrs.get('wrist_cam_extrinsics')
    if wrist_ext_raw is not None:
        wrist_extrinsic_6d = [float(x) for x in np.asarray(wrist_ext_raw)]

    # ── 3. Read robot state from local DROID parquet ──
    table = pq.read_table(local_ep_dir / 'episode.parquet')
    cart_pos = np.array([ast.literal_eval(v.as_py())
                         for v in table['steps/observation/cartesian_position']])  # (T, 6)
    grip_pos = np.array([v.as_py()
                         for v in table['steps/observation/gripper_position']])    # (T,)
    lang = table['steps/language_instruction'][0].as_py() or ''
    num_steps = len(cart_pos)

    # ── 4. Symlink mp4 ──
    mp4_src = local_ep_dir / 'steps_observation_exterior_image_1_left.mp4'
    mp4_dst = out_ep_dir / 'gt.mp4'
    if not mp4_dst.exists():
        mp4_dst.symlink_to(mp4_src)

    # ── 5. Save depth frames (skip if already complete) ──
    depth_dir = out_ep_dir / 'depth_frames'
    if not depth_dir.exists() or len(list(depth_dir.glob('*.png'))) == 0:
        with h5py.File(pw_depth_h5, 'r') as f:
            ext1_key = f'{ext1_serial}+ext'
            if ext1_key in f:
                save_depth_frames(f[ext1_key]['depth'][:], depth_dir)
            wrist_key = f'{wrist_serial}+wrist'
            if wrist_key in f:
                save_depth_frames(f[wrist_key]['depth'][:],
                                  out_ep_dir / 'depth_frames_wrist')

    # ── 6. Extract RGB frames to JPEG (fast training access) ──
    frames_dir = out_ep_dir / 'frames'
    if not frames_dir.exists() or len(list(frames_dir.glob('*.jpg'))) == 0:
        extract_rgb_frames(mp4_src, frames_dir)

    # ── 6. Build data.json (droid-ex compatible format) ──
    # intrinsics: [fx, cx, fy, cy] at 320×180 resolution
    if intr_matrix is not None:
        fx = float(intr_matrix[0, 0])
        cx = float(intr_matrix[0, 2])
        fy = float(intr_matrix[1, 1])
        cy = float(intr_matrix[1, 2])
        intr_entry = {'cameraMatrix': [fx, cx, fy, cy], 'width': 320, 'height': 180}
    else:
        intr_entry = None

    calibration: dict = {'extrinsics': {}, 'intrinsics': {}}
    if ext1_extrinsic_6d is not None:
        calibration['extrinsics'][ext1_serial] = ext1_extrinsic_6d
    if ext2_extrinsic_6d is not None:
        calibration['extrinsics'][ext2_serial] = ext2_extrinsic_6d
    if wrist_extrinsic_6d is not None:
        calibration['extrinsics'][wrist_serial] = wrist_extrinsic_6d
    if intr_entry is not None:
        calibration['intrinsics'][ext1_serial] = intr_entry

    data = {
        'uuid': uuid,
        'num_steps': num_steps,
        'language_instruction': lang,
        'image_shape': [180, 320],          # H × W of the depth/image resolution
        'source_local_path': str(local_ep_dir),
        'ext1_cam_serial': ext1_serial,
        'ext2_cam_serial': ext2_serial,
        'wrist_cam_serial': wrist_serial,
        'calibration': calibration,
        'observations': {
            'cartesian_position': cart_pos.tolist(),    # (T, 6) xyz + euler
            'gripper_position': [[float(g)] for g in grip_pos],
        },
    }
    data_json_path = out_ep_dir / 'data.json'
    if not data_json_path.exists():
        data_json_path.write_text(json.dumps(data, separators=(',', ':')))

    return uuid


# ── worker wrapper (top-level for pickling) ────────────────────────────────

def _worker(args):
    uuid, flows_h5, depth_h5, cameras_json, local_ep_dir, out_ep_dir = args
    try:
        process_episode(uuid, flows_h5, depth_h5, cameras_json,
                        local_ep_dir, out_ep_dir)
        return uuid, None
    except Exception as e:
        return uuid, str(e)


# ── main ───────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output-dir', required=True,
                    help='Output dataset root directory')
    ap.add_argument('--max-episodes', type=int, default=3000,
                    help='Number of PointWorld episodes to process')
    ap.add_argument('--workers', type=int, default=16,
                    help='Number of parallel workers')
    ap.add_argument('--index-cache', default='/tmp/pw_droid_index.json',
                    help='Cache file for local DROID index')
    ap.add_argument('--skip-existing', action='store_true', default=True,
                    help='Skip already processed episodes (default True)')
    args = ap.parse_args()

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # ── Build or load local DROID index ──
    idx_cache = Path(args.index_cache)
    if idx_cache.exists():
        print(f'[index] loading cached index from {idx_cache}')
        index = json.loads(idx_cache.read_text())
    else:
        index = build_local_index(workers=args.workers)
        idx_cache.write_text(json.dumps(index))
        print(f'[index] cached to {idx_cache}')

    # ── Collect PointWorld episodes ──
    pw_flows_dir   = PW_ROOT / 'flows-fs-optimized'
    pw_depth_dir   = PW_ROOT / 'depth_320x180'
    pw_cameras_dir = PW_ROOT / 'cameras'

    all_flows = sorted(pw_flows_dir.glob('*_flows.h5'))
    print(f'[pw] found {len(all_flows)} PointWorld episodes')

    # Select first max_episodes that have matching local DROID episodes
    selected: list[tuple] = []
    missing = 0
    for flows_h5 in all_flows:
        if len(selected) >= args.max_episodes:
            break
        uuid = flows_h5.stem.replace('_flows', '')
        depth_h5     = pw_depth_dir   / f'{uuid}_depth.h5'
        cameras_json = pw_cameras_dir / f'{uuid}_cameras.json'

        if not depth_h5.exists() or not cameras_json.exists():
            missing += 1
            continue

        # look up local DROID match
        key = pw_episode_key(flows_h5)
        if key is None or key not in index:
            missing += 1
            continue

        local_ep_dir = Path(index[key])
        out_ep_dir   = out_root / uuid

        # skip if fully done: has data.json AND frames/ with JPEGs
        if args.skip_existing and (out_ep_dir / 'data.json').exists():
            frames_dir = out_ep_dir / 'frames'
            if frames_dir.exists() and len(list(frames_dir.glob('*.jpg'))) > 0:
                selected.append(None)  # fully done
                continue

        selected.append((uuid, flows_h5, depth_h5, cameras_json,
                         local_ep_dir, out_ep_dir))

    already_done = sum(1 for s in selected if s is None)
    to_process   = [s for s in selected if s is not None]
    print(f'[select] {len(selected)} episodes selected '
          f'({already_done} already done, {len(to_process)} to process, '
          f'{missing} skipped/missing)')

    if not to_process:
        print('[done] nothing to process')
        return

    # ── Process in parallel ──
    ok = 0
    fail = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(_worker, args_tuple): args_tuple[0]
                   for args_tuple in to_process}
        for i, fut in enumerate(as_completed(futures), 1):
            uuid_done, err = fut.result()
            if err:
                print(f'  [{i}/{len(to_process)}] FAIL {uuid_done}: {err}')
                fail += 1
            else:
                ok += 1
            if i % 50 == 0 or i == len(to_process):
                print(f'  [{i}/{len(to_process)}] ok={ok} fail={fail}')

    total_done = already_done + ok
    print(f'\n[done] {total_done}/{len(selected)} episodes ready '
          f'(new: {ok}, skip: {already_done}, fail: {fail})')
    print(f'Output: {out_root}')


if __name__ == '__main__':
    main()
