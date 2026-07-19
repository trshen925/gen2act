"""Full-episode trajectory rollout: chain the model's +5-step camera-frame displacement predictions
from start to end (open-loop position integration) and measure CUMULATIVE drift vs the ground-truth
trajectory.

Per held-out episode, at t = 0,5,10,...: take action[:,0,:3] = the +5-step camera-frame displacement
(GT and predicted). Reconstruct trajectory by cumulative sum; drift[k] = ||cumsum_pred - cumsum_gt||.
Observations are teacher-forced (GT frame/EE/tracks each step) — no simulator renders predicted states,
so this isolates the integration drift of the predicted deltas (true closed-loop would be worse).

Baselines: integrate the global MEAN +5 delta (dead-reckon at average velocity).

    python scripts/trajectory_rollout.py --config <cfg> --checkpoint <ckpt> [--max-episodes 30]
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--max-episodes", type=int, default=30)
    ap.add_argument("--stride", type=int, default=5, help="reconstruction stride (= future_horizon)")
    ap.add_argument("--num-eval-samples", type=int, default=0, help="flow head sampling count override (0=keep config); K=1 tests single-sample magnitude-preserving rollout")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = torch.device(args.device)
    codec = build_action_codec(cfg)
    pose_dims = codec.pose_dims
    ds = build_dataset(cfg, "val")
    model = build_policy(cfg).to(device)
    load_checkpoint(args.checkpoint, model, device, strict=False)
    if args.num_eval_samples > 0 and hasattr(model, "head") and hasattr(model.head, "num_eval_samples"):
        model.head.num_eval_samples = int(args.num_eval_samples)
        print(f"[rollout] flow head num_eval_samples overridden -> {args.num_eval_samples}")
    model.eval()
    normalize = bool(cfg.get("action", {}).get("regression_normalize", False)) or str(cfg["action"]["mode"]) == "flow"
    eff = int(cfg["data"]["future_horizon"]) * int(cfg["action"]["chunk_size"])

    keys = ["source_video", "target_history", "wrist_current", "front_geometry", "proprioception", "point_track", "point_track_causal"]
    eps = ds.episodes[: args.max_episodes]

    # first pass: collect GT +5 deltas to compute the global mean-delta baseline
    rows = []  # per-episode: (gt_deltas[K,3], pred_deltas[K,3])
    all_gt = []
    with torch.no_grad():
        for ep in eps:
            ts = list(range(0, max(1, ep.num_steps - eff), args.stride))
            if len(ts) < 3:
                continue
            samples = [ds.sample_window(ep.episode_id, t) for t in ts]
            batch = {}
            for k in keys:
                if k in samples[0] and torch.is_tensor(samples[0][k]):
                    batch[k] = torch.stack([s[k] for s in samples]).to(device)
            kw = {"point_track_causal": batch["point_track_causal"]} if "point_track_causal" in batch else {}
            if "wrist_current" in batch:
                kw["wrist_current"] = batch["wrist_current"]
            if "front_geometry" in batch:
                kw["front_geometry"] = batch["front_geometry"]
            out = model(batch["source_video"], batch["target_history"], batch.get("proprioception"), None,
                        batch.get("point_track"), **kw)
            pred = out["action_pred"]
            if normalize:
                pred = codec.unnormalize(pred)
            pred_d = pred[:, 0, :3].float().cpu().numpy()                       # +5 cam-frame delta
            gt_d = torch.stack([s["action"] for s in samples])[:, 0, :3].numpy()
            rows.append((gt_d, pred_d))
            all_gt.append(gt_d)

    mean_delta = np.concatenate(all_gt).mean(0)  # global mean +5 delta (dead-reckon baseline)

    # metrics
    end_drift, end_drift_mean_bl, rel_drift, path_len, step_mae = [], [], [], [], []
    axis_end_drift = []   # per-axis |cumsum_pred - cumsum_gt| at endpoint (cm)
    axis_step_mae = []     # per-axis +5 delta MAE (cm)
    axis_gt_span = []      # per-axis GT start->end displacement magnitude (cm)
    frac_drift = {0.25: [], 0.5: [], 0.75: [], 1.0: []}
    for gt_d, pred_d in rows:
        K = len(gt_d)
        traj_gt = np.cumsum(gt_d, 0)
        traj_pred = np.cumsum(pred_d, 0)
        traj_bl = np.cumsum(np.tile(mean_delta, (K, 1)), 0)
        drift = np.linalg.norm(traj_pred - traj_gt, axis=1) * 100   # cm, per step
        drift_bl = np.linalg.norm(traj_bl - traj_gt, axis=1) * 100
        plen = np.linalg.norm(gt_d, axis=1).sum() * 100             # total GT path length cm
        end_drift.append(drift[-1])
        end_drift_mean_bl.append(drift_bl[-1])
        path_len.append(plen)
        rel_drift.append(drift[-1] / max(plen, 1e-6) * 100)         # % of path length
        step_mae.append(np.abs(pred_d - gt_d).mean() * 100)
        axis_end_drift.append(np.abs(traj_pred[-1] - traj_gt[-1]) * 100)   # [3] cm
        axis_step_mae.append(np.abs(pred_d - gt_d).mean(0) * 100)          # [3] cm
        axis_gt_span.append(np.abs(traj_gt[-1]) * 100)                     # [3] cm
        for f in frac_drift:
            frac_drift[f].append(drift[min(K - 1, int(round(f * (K - 1))))])

    end_drift = np.array(end_drift); path_len = np.array(path_len)
    axis_end_drift = np.array(axis_end_drift); axis_step_mae = np.array(axis_step_mae); axis_gt_span = np.array(axis_gt_span)
    print(f"\nepisodes={len(rows)}  recon stride={args.stride}  ckpt={args.checkpoint.name}")
    print(f"  per-step +5 delta MAE (building block) : {np.mean(step_mae):.3f} cm")
    print(f"  GT trajectory path length              : mean {path_len.mean():.1f} cm  (start->end displacement varies)")
    print(f"  cumulative drift @ episode fraction (cm): " +
          "  ".join(f"{int(f*100)}%={np.mean(v):.2f}" for f, v in frac_drift.items()))
    print(f"  ENDPOINT drift (model)                 : mean {end_drift.mean():.2f} cm  median {np.median(end_drift):.2f}  p90 {np.quantile(end_drift,0.9):.2f}")
    print(f"  ENDPOINT drift (dead-reckon mean delta): mean {np.mean(end_drift_mean_bl):.2f} cm")
    print(f"  endpoint drift as % of GT path length  : mean {np.mean(rel_drift):.1f}%")
    print(f"\n  -> model beats dead-reckon-mean on endpoint drift: "
          f"{(end_drift < np.array(end_drift_mean_bl)).mean()*100:.0f}% of episodes")
    ax = ["x(img-horiz)", "y(img-vert)", "z(DEPTH)"]
    print(f"\n  PER-AXIS (camera frame), mean over episodes:")
    print(f"  {'axis':>12} {'step_MAE':>9} {'end_drift':>10} {'GT_span':>9} {'drift/span':>10}")
    for i in range(3):
        sm, ed, sp = axis_step_mae[:, i].mean(), axis_end_drift[:, i].mean(), axis_gt_span[:, i].mean()
        print(f"  {ax[i]:>12} {sm:8.3f}c {ed:9.2f}c {sp:8.2f}c {ed/max(sp,1e-6):9.2f}")
    tot = axis_end_drift.mean(0).sum()
    print(f"  endpoint drift share: " + "  ".join(f"{ax[i].split('(')[0]}={axis_end_drift[:,i].mean()/tot*100:.0f}%" for i in range(3)))


if __name__ == "__main__":
    main()
