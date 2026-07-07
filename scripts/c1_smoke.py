"""C1 dual-track smoke test: config valid, dataset emits BOTH point_track (global) and
point_track_causal (per-window causal), model builds with the causal encoder and forwards.
Run from project root."""
import torch
from collections import defaultdict
from r2r_gen2act.config.load import load_config
from r2r_gen2act.config.schema import validate_config
from r2r_gen2act.data.factories import build_dataset
from r2r_gen2act.modeling.factory import build_policy

cfg = load_config("configs/droidex2000_C1_dualtrack_causal.yaml")
validate_config(cfg)
print("[C1] config OK")

ds = build_dataset(cfg, "train")
byep = defaultdict(list)
ep = None
for i, (eid, si) in enumerate(ds.samples):
    byep[eid].append(i)
    if len(byep[eid]) >= 2:
        ep = eid
        break
i0, i1 = byep[ep][0], byep[ep][-1]
s0, s1 = ds[i0], ds[i1]
print("[C1] sample keys:", sorted(s0.keys()))
print(f"[C1] global {tuple(s0['point_track'].shape)}  causal {tuple(s0['point_track_causal'].shape)}")
g_same = torch.allclose(s0['point_track'], s1['point_track'])
c_same = torch.allclose(s0['point_track_causal'], s1['point_track_causal'])
print(f"[C1] across 2 windows of same ep: global identical={g_same} (expect True=whole-ep const), "
      f"causal identical={c_same} (expect False=per-window)")
assert g_same and not c_same, "expected global=const, causal=per-window varying"

m = build_policy(cfg).train()
out = m(torch.randn(2, 8, 3, 224, 224), torch.randn(2, 1, 3, 224, 224), torch.randn(2, 4), None,
        torch.randn(2, 10, 32, 2), point_track_causal=torch.randn(2, 10, 16, 2))
print("[C1] forward:", {k: (tuple(v.shape) if torch.is_tensor(v) else f'list[{len(v)}]') for k, v in out.items()})
assert out["action_pred"].shape == (2, 4, 9)
print("[C1] SMOKE OK")
