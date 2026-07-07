"""C2 smoke: build fused_query_flow (C1 conditioning + flow head), train forward (velocity), eval sample."""
import torch
from r2r_gen2act.config.load import load_config
from r2r_gen2act.config.schema import validate_config
from r2r_gen2act.modeling.factory import build_policy

cfg = load_config("configs/droidex2000_C2_dualtrack_causal_flow.yaml")
validate_config(cfg)
print("[C2] config OK")
m = build_policy(cfg)
B = 2
src = torch.randn(B, 8, 3, 224, 224); tgt = torch.randn(B, 1, 3, 224, 224); prop = torch.randn(B, 4)
pt = torch.randn(B, 10, 32, 2); ptc = torch.randn(B, 10, 16, 2)
at = torch.randn(B, 4, 9)
m.train()
out = m(src, tgt, prop, at, pt, point_track_causal=ptc)
print("[C2] train out:", {k: tuple(v.shape) for k, v in out.items()})
assert out["pred_velocity"].shape == (B, 4, 9) and out["target_velocity"].shape == (B, 4, 9)
m.eval()
with torch.no_grad():
    s = m(src, tgt, prop, None, pt, point_track_causal=ptc)
print("[C2] eval(sample) out:", {k: tuple(v.shape) for k, v in s.items()})
assert s["action_pred"].shape == (B, 4, 9)
print("[C2] SMOKE OK")
