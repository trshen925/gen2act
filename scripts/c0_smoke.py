"""C0 smoke test: validate config, build the fused_query_reg model in sequence+progress mode,
run one forward with dummy tensors, and check output shapes. Run from project root."""
import torch, yaml
from r2r_gen2act.config.schema import validate_config
from r2r_gen2act.modeling.factory import build_policy as build_model

cfg = yaml.safe_load(open("configs/droidex2000_C0_honest_seq_anchor.yaml"))
validate_config(cfg)
print("[smoke] config validated OK; proprio_dim=", cfg["model"]["proprioception_dim"],
      "point_seq=", cfg["model"]["point_tracking"].get("sequence"))
m = build_model(cfg).train()  # train() so deep-supervision aux preds appear
B = 2
src = torch.randn(B, 8, 3, 224, 224)
tgt = torch.randn(B, 1, 3, 224, 224)
prop = torch.randn(B, 4)            # (u,v,depth)+progress
pt = torch.randn(B, 10, 40, 2)      # S=40 -> PointTrajSeqEncoder interpolates to num_time=32
out = m(src, tgt, prop, None, pt)
print("[smoke] out keys:", {k: (tuple(v.shape) if torch.is_tensor(v) else f"list[{len(v)}]") for k, v in out.items()})
ap = out["action_pred"]
assert ap.shape == (B, 4, 9), f"unexpected action_pred shape {tuple(ap.shape)}"
assert "aux_action_preds" in out and len(out["aux_action_preds"]) == 6, "deep supervision aux missing"
print("[smoke] FORWARD OK — action_pred", tuple(ap.shape), "aux", len(out["aux_action_preds"]))
