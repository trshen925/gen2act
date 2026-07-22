"""Phase-transition eval: separate "momentum extrapolation" from "demo understanding".

A window is a TRANSITION frame if the future motion direction reverses vs the past (cos angle small),
or the gripper state changes — exactly where "repeat last velocity" (momentum) MUST fail. We compare the
model's prediction against the momentum predictor (= the previous window's GT +H delta) on three subsets:
all / transition / smooth. If the model only matches momentum on smooth frames but is NO better than
momentum on transition frames, it learned inertia, not the demo. If it beats momentum on transitions, it
genuinely read the demo intent ("time to change direction / grasp").

    python scripts/phase_transition_eval.py --config <cfg> --checkpoint <ckpt> [--cos-thresh 0.3]
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


def stats(pred, tgt):
    mae = np.abs(pred[:, :3] - tgt[:, :3]).mean() * 100
    corr = []
    for d in range(3):
        p, t = pred[:, d], tgt[:, d]
        corr.append(float(np.corrcoef(p, t)[0, 1]) if p.std() > 1e-8 and t.std() > 1e-8 else 0.0)
    return mae, corr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--max-episodes", type=int, default=93)
    ap.add_argument("--cos-thresh", type=float, default=0.3, help="future-vs-past direction cos below this = transition")
    ap.add_argument("--min-disp", type=float, default=0.01, help="ignore windows with future |disp| below this (m) as direction-noise")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = torch.device(args.device)
    codec = build_action_codec(cfg)
    pose_dims = codec.pose_dims
    ds = build_dataset(cfg, "val")
    model = build_policy(cfg).to(device)
    load_checkpoint(args.checkpoint, model, device, strict=False)
    model.eval()
    normalize = bool(cfg.get("action", {}).get("regression_normalize", False)) or str(cfg["action"]["mode"]) == "flow"
    stride = int(cfg["data"]["future_horizon"])
    eff = stride * int(cfg["action"]["chunk_size"])
    keys = ["source_video", "target_history", "wrist_current", "front_geometry", "proprioception", "point_track", "point_track_causal"]

    m_pred, mom_pred, gt, is_trans = [], [], [], []
    with torch.no_grad():
        for ep in ds.episodes[: args.max_episodes]:
            ts = list(range(0, max(1, ep.num_steps - eff), stride))
            if len(ts) < 3:
                continue
            samples = [ds.sample_window(ep.episode_id, t) for t in ts]
            batch = {k: torch.stack([s[k] for s in samples]).to(device) for k in keys
                     if k in samples[0] and torch.is_tensor(samples[0][k])}
            kw = {"point_track_causal": batch["point_track_causal"]} if "point_track_causal" in batch else {}
            if "wrist_current" in batch:
                kw["wrist_current"] = batch["wrist_current"]
            if "front_geometry" in batch:
                kw["front_geometry"] = batch["front_geometry"]
            out = model(batch["source_video"], batch["target_history"], batch.get("proprioception"), None,
                        batch.get("point_track"), **kw)
            pred = out["action_pred"]
            if normalize:
                pred = codec.unnormalize(pred[..., :pose_dims])
            else:
                pred = pred[..., :pose_dims]
            pred = pred[:, 0, :3].float().cpu().numpy()                         # model +H delta
            gtd = torch.stack([s["action"] for s in samples])[:, 0, :3].numpy()  # GT +H delta
            grip = torch.stack([s["gripper"] for s in samples])[:, 0].numpy()    # gripper at +H
            for i in range(1, len(gtd)):
                past, future = gtd[i - 1], gtd[i]                                # momentum = past delta
                if np.linalg.norm(future) < args.min_disp:
                    continue
                cos = float(np.dot(past, future) / (np.linalg.norm(past) * np.linalg.norm(future) + 1e-8))
                trans = (cos < args.cos_thresh) or (grip[i] != grip[i - 1])
                m_pred.append(pred[i]); mom_pred.append(past); gt.append(future); is_trans.append(trans)

    m_pred = np.array(m_pred); mom_pred = np.array(mom_pred); gt = np.array(gt); is_trans = np.array(is_trans)
    print(f"\nckpt={args.checkpoint.name}  windows={len(gt)}  transition={is_trans.sum()} ({100*is_trans.mean():.0f}%)  "
          f"cos<{args.cos_thresh} or gripper-change")
    for name, mask in [("ALL", np.ones(len(gt), bool)), ("TRANSITION", is_trans), ("SMOOTH", ~is_trans)]:
        if mask.sum() < 5:
            continue
        mm, mc = stats(m_pred[mask], gt[mask])
        om, oc = stats(mom_pred[mask], gt[mask])
        print(f"\n[{name}]  n={mask.sum()}")
        print(f"  {'':>8} {'XYZ_MAE':>9} {'dx':>7} {'dy':>7} {'dz':>7}")
        print(f"  {'MODEL':>8} {mm:8.3f}c {mc[0]:7.3f} {mc[1]:7.3f} {mc[2]:7.3f}")
        print(f"  {'momentum':>8} {om:8.3f}c {oc[0]:7.3f} {oc[1]:7.3f} {oc[2]:7.3f}")
        verdict = "MODEL beats momentum" if mm < om else "momentum beats/ties MODEL"
        print(f"  -> {verdict} (MAE {mm:.2f} vs {om:.2f} cm)")


if __name__ == "__main__":
    main()
