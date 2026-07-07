"""Cross-attention regression decoder with deep supervision (videomt-style loss, but cross-attention
information transmission instead of direct query-in-backbone readout).

Ablation head: keeps the SAME cross-attention scheme as the flow DiT (a separate set of action
queries cross-attends to the conditioning tokens), but replaces flow-matching with DIRECT regression
+ deep supervision (an action is predicted and supervised after every decoder layer, like videomt).
Isolates the variable: if this learns the hard dims (dx) where the flow head failed, the cross-
attention transmission is fine and flow was the bottleneck; if it still fails dx, the cross-attention
indirection itself is weaker than videomt's in-backbone readout.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from r2r_gen2act.modeling.flow_dit import FeedForward, _MHA


class CrossAttnRegressionDecoder(nn.Module):
    def __init__(self, cond_dim: int, pose_dims: int, horizon: int, hidden_dim: int = 768,
                 num_layers: int = 6, heads: int = 8, dropout: float = 0.1,
                 deep_supervision: bool = True) -> None:
        super().__init__()
        self.horizon = int(horizon)
        self.pose_dims = int(pose_dims)
        self.deep_supervision = bool(deep_supervision)
        self.cond_in = nn.Linear(cond_dim, hidden_dim) if cond_dim != hidden_dim else nn.Identity()
        # one learnable query per action-chunk step (queries are inherently positional).
        self.query = nn.Parameter(torch.randn(1, horizon, hidden_dim) / hidden_dim**0.5)
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "sa_norm": nn.LayerNorm(hidden_dim), "self_attn": _MHA(hidden_dim, heads, dropout),
                "ca_norm": nn.LayerNorm(hidden_dim), "cross_attn": _MHA(hidden_dim, heads, dropout),
                "ff_norm": nn.LayerNorm(hidden_dim), "ff": FeedForward(hidden_dim, dropout=dropout),
            }) for _ in range(num_layers)
        ])
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.pose_head = nn.Linear(hidden_dim, pose_dims)
        self.gripper_proj = nn.Linear(hidden_dim, 2)
        self.terminate_proj = nn.Linear(hidden_dim, 2)

    def _pose(self, q: torch.Tensor) -> torch.Tensor:
        # tanh -> normalized [-1,1] action space (matches codec.normalize targets), like videomt.
        return torch.tanh(self.pose_head(self.out_norm(q)))

    def forward(self, cond: torch.Tensor) -> dict:
        kv = self.cond_in(cond)
        q = self.query.expand(cond.shape[0], -1, -1)
        aux = []
        for L in self.layers:
            h = L["sa_norm"](q)
            q = q + L["self_attn"](h, h)
            q = q + L["cross_attn"](L["ca_norm"](q), kv)
            q = q + L["ff"](L["ff_norm"](q))
            if self.deep_supervision and self.training:
                aux.append(self._pose(q))
        out = {
            "action_pred": self._pose(q),
            "gripper_logits": self.gripper_proj(self.out_norm(q)),
            "terminate_logits": self.terminate_proj(self.out_norm(q)),
        }
        if aux:
            out["aux_action_preds"] = aux
        return out
