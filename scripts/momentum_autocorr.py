"""Momentum ceiling: how well does PAST camera-frame EE motion predict FUTURE motion, purely from
data (no model)? If corr(past_delta, future_delta) ~= C1's dx corr 0.48, then C1's dx is explained by
momentum (motion autocorrelation), i.e. the causal track is a momentum shortcut. If it's much lower,
C1's dx must come from elsewhere (the global demo track) -> the causal track does real localization.

Walks each val episode at stride = future_horizon; the +H camera-frame delta (action[0,:3]) is the
'future motion' at t and the 'past motion' at t+stride. Reports corr(past, future) per axis for a
1-window-back momentum and a 4-window-back averaged momentum (~ the causal window span).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from r2r_gen2act.config.load import load_config
from r2r_gen2act.data.factories import build_dataset

cfg = load_config("configs/droidex2000_C1_dualtrack_causal.yaml")
ds = build_dataset(cfg, "val")
stride = int(cfg["data"]["future_horizon"])
eff = stride * int(cfg["action"]["chunk_size"])

past1, past4, fut = [], [], []
for ep in ds.episodes:
    ts = list(range(0, max(1, ep.num_steps - eff), stride))
    if len(ts) < 6:
        continue
    deltas = np.stack([ds.sample_window(ep.episode_id, t)["action"][0, :3].numpy() for t in ts])  # [K,3]
    for i in range(4, len(deltas)):
        fut.append(deltas[i])
        past1.append(deltas[i - 1])                 # motion over the immediately preceding +H window
        past4.append(deltas[i - 4:i].mean(0))       # avg of preceding 4 windows (~20 steps, ~causal span)

fut = np.array(fut); past1 = np.array(past1); past4 = np.array(past4)
names = ["dx", "dy", "dz"]
print(f"\nval windows={len(fut)}  (momentum ceiling = corr of PAST vs FUTURE camera-frame +{stride} delta)")
print(f"{'axis':>5} {'corr_1back':>11} {'corr_4back_avg':>15}   (compare C1 dx0.48/dy0.65/dz0.39)")
for d in range(3):
    c1 = np.corrcoef(past1[:, d], fut[:, d])[0, 1]
    c4 = np.corrcoef(past4[:, d], fut[:, d])[0, 1]
    print(f"{names[d]:>5} {c1:11.3f} {c4:15.3f}")

# XYZ MAE of momentum predictors vs the predict-mean baseline (cm), for the +H delta
mae_1 = np.abs(past1[:, :3] - fut[:, :3]).mean() * 100
mae_4 = np.abs(past4[:, :3] - fut[:, :3]).mean() * 100
mae_mean = np.abs(fut[:, :3] - fut[:, :3].mean(0)).mean() * 100
print(f"\nXYZ MAE on the +{stride} delta (cm):  momentum-1back={mae_1:.3f}  momentum-20step={mae_4:.3f}  predict-mean={mae_mean:.3f}")
print(f"  (a 'repeat last velocity' baseline is the honest yardstick on this smooth-trajectory data, not predict-mean)")
