"""Permutation-importance over the conditioning streams of a fused_query_reg model.

For a trained checkpoint, run held-out eval; for each stream, permute it across the batch (roll by 1)
so it loses its correspondence to the action target while staying in-distribution, and measure how much
XYZ MAE rises (and how dx corr drops). The stream whose permutation hurts most is the dominant one.

    python scripts/stream_importance.py --config <cfg> --checkpoint <ckpt> [--max-windows 800]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from r2r_gen2act.config.load import load_config
from r2r_gen2act.data.factories import build_action_codec, build_dataset
from r2r_gen2act.modeling.factory import build_policy
from r2r_gen2act.training.checkpoint import load_checkpoint
from torch.utils.data import DataLoader


def roll(x):
    """In-batch derangement: shift along batch dim by 1 (decorrelates stream from target)."""
    return torch.roll(x, shifts=1, dims=0) if torch.is_tensor(x) else x


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--max-windows", type=int, default=800)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = torch.device(args.device)
    codec = build_action_codec(cfg)
    pose_dims = codec.pose_dims
    ds = build_dataset(cfg, args.split)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=8)
    model = build_policy(cfg).to(device)
    load_checkpoint(args.checkpoint, model, device, strict=False)
    model.eval()
    normalize = bool(cfg.get("action", {}).get("regression_normalize", False)) or str(cfg["action"]["mode"]) == "flow"
    has_causal = bool((cfg.get("data", {}).get("point_tracking", {}) or {}).get("causal_window"))

    # conditions: name -> which stream to permute
    conds = ["intact", "perm_source_video", "perm_current_frame", "perm_global_track",
             "perm_ee_progress"]
    if has_causal:
        conds += ["perm_causal_track", "perm_both_tracks"]

    preds = {c: [] for c in conds}
    tgts = []
    n = 0
    with torch.no_grad():
        for batch in loader:
            src = batch["source_video"].to(device)
            tgt_h = batch["target_history"].to(device)
            prop = batch.get("proprioception")
            prop = prop.to(device) if torch.is_tensor(prop) else None
            pt = batch.get("point_track")
            pt = pt.to(device) if torch.is_tensor(pt) else None
            ptc = batch.get("point_track_causal")
            ptc = ptc.to(device) if torch.is_tensor(ptc) else None
            wrist = batch.get("wrist_current")
            wrist = wrist.to(device) if torch.is_tensor(wrist) else None
            source_dt = batch.get("source_dt")
            source_dt = source_dt.to(device) if torch.is_tensor(source_dt) else None
            front_geometry = batch.get("front_geometry")
            front_geometry = front_geometry.to(device) if torch.is_tensor(front_geometry) else None

            def run(src_, tgt_, prop_, pt_, ptc_, geometry_=front_geometry):
                kw = {"point_track_causal": ptc_} if ptc_ is not None else {}
                if wrist is not None:
                    kw["wrist_current"] = wrist
                if source_dt is not None:
                    kw["source_dt"] = source_dt
                if geometry_ is not None:
                    kw["front_geometry"] = geometry_
                out = model(src_, tgt_, prop_, None, pt_, **kw)
                p = out["action_pred"]
                if normalize:
                    p = codec.unnormalize(p)
                return p.reshape(-1, pose_dims).float().cpu().numpy()

            preds["intact"].append(run(src, tgt_h, prop, pt, ptc))
            preds["perm_source_video"].append(run(roll(src), tgt_h, prop, pt, ptc))
            preds["perm_current_frame"].append(run(
                src, roll(tgt_h), prop, pt, ptc,
                roll(front_geometry) if front_geometry is not None else None))
            preds["perm_global_track"].append(run(src, tgt_h, prop, roll(pt), ptc))
            preds["perm_ee_progress"].append(run(src, tgt_h, roll(prop) if prop is not None else None, pt, ptc))
            if has_causal:
                preds["perm_causal_track"].append(run(src, tgt_h, prop, pt, roll(ptc)))
                preds["perm_both_tracks"].append(run(src, tgt_h, prop, roll(pt), roll(ptc)))

            tgts.append(batch["action"][..., :pose_dims].reshape(-1, pose_dims).numpy())
            n += src.shape[0]
            if n >= args.max_windows:
                break

    T = np.concatenate(tgts)[: args.max_windows * cfg["action"]["chunk_size"]]
    base_mae = np.abs(T[:, :3] - T[:, :3].mean(0)).mean() * 100
    print(f"\nwindows~{n}  ckpt={args.checkpoint.name}  predict-mean baseline XYZ MAE = {base_mae:.3f} cm")
    print(f"{'condition':>20} {'XYZ_MAE':>9} {'d_vs_intact':>12} {'dx_corr':>8} {'dy_corr':>8} {'dz_corr':>8}")
    intact_mae = None
    for c in conds:
        P = np.concatenate(preds[c])[: len(T)]
        xyz_mae = np.abs(P[:, :3] - T[:, :3]).mean() * 100
        if c == "intact":
            intact_mae = xyz_mae
        corrs = []
        for d in range(3):
            p, t = P[:, d], T[:, d]
            corrs.append(float(np.corrcoef(p, t)[0, 1]) if p.std() > 1e-8 and t.std() > 1e-8 else 0.0)
        delta = xyz_mae - intact_mae
        print(f"{c:>20} {xyz_mae:8.3f}c {delta:+11.3f} {corrs[0]:8.3f} {corrs[1]:8.3f} {corrs[2]:8.3f}")
    print("\n(larger +d_vs_intact = that stream matters more to the trained model)")


if __name__ == "__main__":
    main()
