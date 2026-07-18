"""Geodesic rotation-angle error for a fused_query_reg model (6D rotation head).

The per-6D-component corr/MAE in diagnose_actions.py under-describes rotation. Here we reconstruct the
rotation matrix from the predicted/target 6D (Gram-Schmidt) and report the GEODESIC ANGLE error (deg),
plus two reference points:
  - "predict identity" (zero rotation): error = the GT rotation magnitude itself. If model error >= this,
    the model is no better than assuming no rotation.
  - "predict mean rotation": error vs the mean GT rotation (a constant-rotation baseline).

    python scripts/rotation_eval.py --config <cfg> --checkpoint <ckpt> [--max-windows 800]
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
from r2r_gen2act.data.action.rotation import sixd_to_matrix
from r2r_gen2act.data.factories import build_action_codec, build_dataset
from r2r_gen2act.modeling.factory import build_policy
from r2r_gen2act.training.checkpoint import load_checkpoint
from torch.utils.data import DataLoader


def angles_deg(R_a, R_b):
    """Geodesic angle (deg) between two batches of rotation matrices [...,3,3]."""
    Rrel = R_a @ R_b.transpose(-1, -2)
    tr = Rrel[..., 0, 0] + Rrel[..., 1, 1] + Rrel[..., 2, 2]
    cos = ((tr - 1.0) / 2.0).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.arccos(cos))


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

    pred6, tgt6 = [], []
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
            kw = {"point_track_causal": ptc.to(device)} if torch.is_tensor(ptc) else {}
            wrist = batch.get("wrist_current")
            if torch.is_tensor(wrist):
                kw["wrist_current"] = wrist.to(device)
            out = model(src, tgt_h, prop, None, pt, **kw)
            p = out["action_pred"]
            if normalize:
                p = codec.unnormalize(p)
            pred6.append(p.reshape(-1, pose_dims)[:, 3:9].float().cpu())
            tgt6.append(batch["action"][..., :pose_dims].reshape(-1, pose_dims)[:, 3:9].float())
            n += src.shape[0]
            if n >= args.max_windows:
                break

    P = torch.cat(pred6)[: args.max_windows * cfg["action"]["chunk_size"]]
    T = torch.cat(tgt6)[: len(P)]
    Rp = sixd_to_matrix(P)
    Rt = sixd_to_matrix(T)
    eye = torch.eye(3).expand_as(Rt)
    # mean GT rotation matrix (re-orthonormalized) as a constant-prediction baseline
    Rmean = sixd_to_matrix(T.mean(0, keepdim=True)).expand_as(Rt)

    err_model = angles_deg(Rp, Rt)
    err_identity = angles_deg(eye, Rt)        # = GT rotation magnitude
    err_mean = angles_deg(Rmean, Rt)

    def stat(x):
        return f"mean {x.mean():5.2f}  median {x.median():5.2f}  p90 {x.quantile(0.9):5.2f}"

    print(f"\nwindows~{n}  samples={len(P)}  ckpt={args.checkpoint.name}   (geodesic angle error, degrees)")
    print(f"  GT rotation magnitude (= 'predict identity' err): {stat(err_identity)}")
    print(f"  predict-mean-rotation baseline err              : {stat(err_mean)}")
    print(f"  MODEL prediction err                            : {stat(err_model)}")
    better_id = (err_model < err_identity).float().mean() * 100
    print(f"\n  model beats 'predict identity' on {better_id:.1f}% of samples; "
          f"mean err {err_model.mean():.2f} vs identity {err_identity.mean():.2f} vs mean-rot {err_mean.mean():.2f} deg")


if __name__ == "__main__":
    main()
