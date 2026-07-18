"""Held-out action diagnostic: per-dim correlation / MAE / pred-std vs target-std, and the
'predict the mean' baseline. Works for flow (samples the DiT) and regression heads alike.

This is the real judge for the flow-matching head (training loss is only a velocity proxy):
run model.eval() over held-out windows, collect sampled action chunks, compare to ground truth
in RAW units (cm for xyz). Mirrors the metrics used in EXPERIMENTS.md.

    python scripts/diagnose_actions.py --config <cfg> --checkpoint <ckpt> [--max-windows 800]
"""
from __future__ import annotations

import argparse
import random
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--max-windows", type=int, default=800)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    cfg = load_config(args.config)
    device = torch.device(args.device)
    codec = build_action_codec(cfg)
    pose_dims = codec.pose_dims
    ds = build_dataset(cfg, args.split)
    # C24: variable source-frame count → bucket batches by k so source_video collates.
    if bool(cfg["data"].get("dynamic_source", {}).get("enabled", False)) and getattr(ds, "window_k", lambda: None)() is not None:
        from r2r_gen2act.data.bucket_sampler import KBucketBatchSampler
        bsampler = KBucketBatchSampler(ds.window_k(), args.batch_size, shuffle=False)
        loader = DataLoader(ds, batch_sampler=bsampler, num_workers=8)
    else:
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=8)
    model = build_policy(cfg).to(device)
    load_checkpoint(args.checkpoint, model, device, strict=False)
    model.eval()
    normalize = bool(cfg.get("action", {}).get("regression_normalize", False)) or str(cfg["action"]["mode"]) == "flow"

    preds, tgts = [], []
    n = 0
    with torch.no_grad():
        for batch in loader:
            src = batch["source_video"].to(device)
            tgt_h = batch["target_history"].to(device)
            prop = batch.get("proprioception")
            prop = prop.to(device) if torch.is_tensor(prop) else None
            point_track = batch.get("point_track")
            point_track = point_track.to(device) if torch.is_tensor(point_track) else None
            ptc = batch.get("point_track_causal")
            kw = {"point_track_causal": ptc.to(device)} if torch.is_tensor(ptc) else {}
            sdt = batch.get("source_dt")
            if torch.is_tensor(sdt):
                kw["source_dt"] = sdt.to(device)
            wrist = batch.get("wrist_current")
            if torch.is_tensor(wrist):
                kw["wrist_current"] = wrist.to(device)
            out = model(src, tgt_h, prop, None, point_track, **kw)
            pred = out["action_pred"]
            if normalize:
                pred = codec.unnormalize(pred)
            preds.append(pred.reshape(-1, pose_dims).float().cpu().numpy())
            tgts.append(batch["action"][..., :pose_dims].reshape(-1, pose_dims).numpy())
            n += src.shape[0]
            if n >= args.max_windows:
                break
    chunk_size = cfg["action"]["chunk_size"]
    future_horizon = cfg["data"].get("future_horizon", 1)
    fps = 15  # DROID recording frequency

    P = np.concatenate(preds)[: args.max_windows * chunk_size]
    T = np.concatenate(tgts)[: len(P)]
    names = ["dx", "dy", "dz"] + [f"r{i}" for i in range(pose_dims - 3)]
    print(f"\nwindows~{n}  samples={len(P)}  mode={cfg['action']['mode']}  ckpt={args.checkpoint.name}")
    print(f"{'dim':>5} {'corr':>7} {'MAE':>9} {'pred_std':>9} {'tgt_std':>9} {'std_ratio':>9}")
    for d in range(pose_dims):
        p, t = P[:, d], T[:, d]
        corr = float(np.corrcoef(p, t)[0, 1]) if p.std() > 1e-8 and t.std() > 1e-8 else 0.0
        mae = float(np.abs(p - t).mean())
        unit = "cm" if d < 3 else ""
        scale = 100.0 if d < 3 else 1.0
        print(f"{names[d]:>5} {corr:7.3f} {mae*scale:8.3f}{unit:>1} {p.std()*scale:8.3f} {t.std()*scale:8.3f} {p.std()/max(t.std(),1e-8):8.3f}")
    xyz_mae = np.abs(P[:, :3] - T[:, :3]).mean() * 100
    base_mae = np.abs(T[:, :3] - T[:, :3].mean(0)).mean() * 100
    print(f"\nXYZ MAE = {xyz_mae:.3f} cm   |  'predict-mean' baseline = {base_mae:.3f} cm   "
          f"({'BEATS' if xyz_mae < base_mae else 'WORSE THAN'} baseline by {abs(1-xyz_mae/base_mae)*100:.1f}%)")

    # ── per-chunk-step breakdown ──────────────────────────────────────────────
    n_win = len(P) // chunk_size
    P_s = P[: n_win * chunk_size].reshape(n_win, chunk_size, pose_dims)
    T_s = T[: n_win * chunk_size].reshape(n_win, chunk_size, pose_dims)
    print(f"\nPer-chunk-step XYZ MAE  (future_horizon={future_horizon}, fps={fps}):")
    print(f"  {'step':>4}  {'t_future':>9}  {'dx MAE':>8}  {'dy MAE':>8}  {'dz MAE':>8}  {'XYZ MAE':>8}  {'baseline':>8}")
    for k in range(chunk_size):
        t_ms = int((k + 1) * future_horizon / fps * 1000)
        p_k, t_k = P_s[:, k, :3] * 100, T_s[:, k, :3] * 100
        mae_k = np.abs(p_k - t_k).mean(0)
        xyz_k = mae_k.mean()
        base_k = np.abs(t_k - t_k.mean(0)).mean()
        print(f"  k={k}   +{t_ms:>5}ms   {mae_k[0]:>7.2f}cm  {mae_k[1]:>7.2f}cm  {mae_k[2]:>7.2f}cm  {xyz_k:>7.2f}cm  {base_k:>7.2f}cm")


if __name__ == "__main__":
    main()
