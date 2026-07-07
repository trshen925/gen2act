"""Verify the trajectory overlay alignment: render a few windows of one episode with the demo EE path
drawn (blue->red) + a green dot at the current target_step. If the green dot sits on the gripper in the
image and the path traces the EE motion, the overlay is aligned. Saves PNGs to outputs/overlay_viz/."""
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from r2r_gen2act.config.load import load_config
from r2r_gen2act.data.factories import build_dataset

cfg = load_config("configs/droidex2000_C3_overlay_dualtrack.yaml")
cfg["data"]["trajectory_overlay"]["mark_current"] = True   # green dot at target_step for the check
cfg["data"]["augmentation"]["enabled"] = False             # cleaner image for inspection
ds = build_dataset(cfg, "val")
ds.split = "val"  # no coord-noise

out = ROOT / "outputs" / "overlay_viz"
out.mkdir(parents=True, exist_ok=True)

# pick the first episode with a usable path; render a few windows across the episode
from collections import defaultdict
byep = defaultdict(list)
for i, (eid, si) in enumerate(ds.samples):
    byep[eid].append(si)
ep = list(byep.keys())[0]
starts = byep[ep]
picks = [starts[0], starts[len(starts) // 3], starts[2 * len(starts) // 3], starts[-1]]
print(f"[viz] episode {ep}, num windows {len(starts)}, rendering target_steps {picks}")
for si in picks:
    s = ds.sample_window(ep, si)
    frame = s["target_history"][-1].clamp(0, 1).permute(1, 2, 0).numpy()  # HWC [0,1]
    img = (frame * 255).astype(np.uint8)
    fp = out / f"overlay_{ep}_t{s['target_step']:03d}.png"
    imageio.imwrite(fp, img)
    print(f"[viz] saved {fp}  (green dot should sit on the gripper at step {s['target_step']})")
print("[viz] DONE")
