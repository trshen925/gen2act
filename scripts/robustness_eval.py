"""C7 robustness probe: the video-to-trajectory model reads EE pose from pixels. At deploy the frames are
GENERATED semantic video (gripper deformed / wrong size / blurry / occluded). We can't generate here, so we
PERTURB the held-out real frames and measure how much pose corr/MAE degrades — a proxy for the real->generated
transfer gap. Big drop = over-reliant on precise pixels (won't transfer); small drop = robust.

    python scripts/robustness_eval.py --config <C7 cfg> --checkpoint <latest.pt> [--max-windows 800]
"""
from __future__ import annotations

import argparse, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from r2r_gen2act.config.load import load_config
from r2r_gen2act.data.factories import build_action_codec, build_dataset
from r2r_gen2act.modeling.factory import build_policy
from r2r_gen2act.training.checkpoint import load_checkpoint
from torch.utils.data import DataLoader


def perturb(frames: torch.Tensor, kind: str) -> torch.Tensor:
    """frames [...,3,H,W] in [0,1]."""
    flat = frames.reshape(-1, *frames.shape[-3:])
    if kind == "clean":
        out = flat
    elif kind == "blur":                       # generation softness
        out = TF.gaussian_blur(flat, kernel_size=11, sigma=2.0)
    elif kind == "color":                      # wrong colors / lighting
        out = TF.adjust_brightness(flat, 1.3); out = TF.adjust_contrast(out, 0.7)
        out = TF.adjust_saturation(out, 1.6); out = TF.adjust_hue(out, 0.08)
    elif kind == "downscale":                  # low-detail generated frames
        d = F.interpolate(flat, scale_factor=0.4, mode="bilinear", align_corners=False)
        out = F.interpolate(d, size=flat.shape[-2:], mode="bilinear", align_corners=False)
    elif kind == "cutout":                     # partial occlusion / artifacts
        out = flat.clone(); n, _, h, w = out.shape
        ch, cw = h // 4, w // 4
        for i in range(n):
            for _ in range(3):
                y = int(torch.randint(0, h - ch, (1,))); x = int(torch.randint(0, w - cw, (1,)))
                out[i, :, y:y + ch, x:x + cw] = 0.5
    elif kind == "noise":
        out = (flat + torch.randn_like(flat) * 0.1)
    else:
        raise ValueError(kind)
    return out.clamp(0, 1).reshape(frames.shape)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--max-windows", type=int, default=800)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg = load_config(args.config); device = torch.device(args.device)
    codec = build_action_codec(cfg); pose_dims = codec.pose_dims
    ds = build_dataset(cfg, "val")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=8)
    model = build_policy(cfg).to(device); load_checkpoint(args.checkpoint, model, device, strict=False); model.eval()
    normalize = bool(cfg.get("action", {}).get("regression_normalize", False)) or str(cfg["action"]["mode"]) == "flow"
    kinds = ["clean", "blur", "color", "downscale", "cutout", "noise"]
    preds = {k: [] for k in kinds}; tgts = []
    n = 0
    with torch.no_grad():
        for batch in loader:
            src = batch["source_video"].to(device); tgt = batch["target_history"].to(device)
            prop = batch.get("proprioception"); prop = prop.to(device) if torch.is_tensor(prop) else None
            for k in kinds:
                out = model(perturb(src, k), perturb(tgt, k), prop, None, None)
                p = out["action_pred"]
                if normalize: p = codec.unnormalize(p)
                preds[k].append(p.reshape(-1, pose_dims).float().cpu().numpy())
            tgts.append(batch["action"][..., :pose_dims].reshape(-1, pose_dims).numpy())
            n += src.shape[0]
            if n >= args.max_windows: break
    T = np.concatenate(tgts)[: args.max_windows * cfg["action"]["chunk_size"]]
    print(f"\nckpt={args.checkpoint.name}  windows~{n}  (proxy for real->generated transfer gap)")
    print(f"{'perturb':>10} {'XYZ_MAE':>9} {'x_corr':>7} {'y_corr':>7} {'z_corr':>7}")
    for k in kinds:
        P = np.concatenate(preds[k])[: len(T)]
        mae = np.abs(P[:, :3] - T[:, :3]).mean() * 100
        c = [float(np.corrcoef(P[:, d], T[:, d])[0, 1]) if P[:, d].std() > 1e-8 else 0.0 for d in range(3)]
        print(f"{k:>10} {mae:8.3f}c {c[0]:7.3f} {c[1]:7.3f} {c[2]:7.3f}")


if __name__ == "__main__":
    main()
