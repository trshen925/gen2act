# Local single-GPU launcher: same srcfuture config but more dataloader workers (this box has
# 128 cores; the cluster config keeps num_workers=4 for its pid limit). Config file untouched.
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from r2r_gen2act.config.load import load_config
from r2r_gen2act.training.trainer import train

cfg = load_config(str(ROOT / "configs/droid2000new_future5_chunk4_pose6d_regression_qpos_ft4dinov2_latent128_srcfuture.yaml"))
cfg["train"]["num_workers"] = 16
print("local override: num_workers=16")
train(cfg, device="cuda")
