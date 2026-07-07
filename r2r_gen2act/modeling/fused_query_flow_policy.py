"""C2 — C1's fused dual-track conditioning + flow-matching DiT head (instead of regression).

Same conditioning as FusedQueryRegPolicy (Exp C1): query-in-backbone DINOv2 readout of the demo video
+ current frame, a per-timestep PointTrajSeqEncoder for the GLOBAL demo trajectory and a second one for
the CAUSAL recent-motion track, plus EE(u,v,depth)+progress tokens — each tagged with a type embedding.
The ONLY change vs C1 is the head: the regression decoder (which collapses to the conditional mean ->
pred_std<<tgt_std -> systematic under-prediction / trajectory drift) is replaced by the flow-matching
DiT head, which models the action DISTRIBUTION and samples actions with full magnitude.

No SelfAttentionMixer here: the flow head has its own internal VL mixer (matches QueryFlowPolicy), so the
raw conditioning tokens are fed straight to the head.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from r2r_gen2act.modeling.flow_dit import FlowMatchingDiTHead
from r2r_gen2act.modeling.vit import ViTBackbone


class FusedQueryFlowPolicy(nn.Module):
    def __init__(self, vit: ViTBackbone, head: FlowMatchingDiTHead, point_encoder,
                 source_len: int, num_queries: int = 32, segmenter_start: int = 9,
                 ee_dim: int = 4, ee_tokens: int = 8, image_size: int = 224,
                 point_encoder_causal=None, aux_traj_cfg: dict | None = None,
                 aux_progress_cfg: dict | None = None) -> None:
        super().__init__()
        self.vit = vit
        self.head = head
        self.point_encoder = point_encoder
        self.point_encoder_causal = point_encoder_causal
        self.action_head_type = "flow_dit"
        self.image_size = image_size
        self.num_queries = int(num_queries)
        self.source_len = int(source_len)
        dim = vit.hidden_dim
        n_blocks = len(self.vit.backend.blocks)
        self.segmenter_start = int(segmenter_start) % n_blocks if segmenter_start >= 0 else n_blocks + int(segmenter_start)
        self.query = nn.Parameter(torch.randn(self.num_queries, dim) / dim**0.5)
        self.source_time_embed = nn.Parameter(torch.randn(source_len, 1, dim) / dim**0.5)
        # type embeddings so the flow head's VL mixer can tell the conditioning sources apart
        self.type_source = nn.Parameter(torch.randn(1, 1, dim) / dim**0.5)
        self.type_current = nn.Parameter(torch.randn(1, 1, dim) / dim**0.5)
        self.type_point = nn.Parameter(torch.randn(1, 1, dim) / dim**0.5)
        self.type_ee = nn.Parameter(torch.randn(1, 1, dim) / dim**0.5)
        if point_encoder_causal is not None:
            self.type_point_causal = nn.Parameter(torch.randn(1, 1, dim) / dim**0.5)
        self.ee_tokens = int(ee_tokens)
        self.ee_mlp = nn.Sequential(nn.LayerNorm(ee_dim), nn.Linear(ee_dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.ee_query = nn.Parameter(torch.randn(1, ee_tokens, dim) / dim**0.5)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("image_mean", mean, persistent=False)
        self.register_buffer("image_std", std, persistent=False)
        # C17: auxiliary source-video abs-EE-pose head (same regularizer as C15v2, applied to flow policy).
        # source tokens are always first in the cond sequence: [B, source_len*num_queries, dim].
        aux_traj_cfg = aux_traj_cfg or {}
        self.aux_traj_enabled = bool(aux_traj_cfg.get("enabled", False))
        if self.aux_traj_enabled:
            self.aux_traj_norm = nn.LayerNorm(dim)
            self.aux_traj_head = nn.Linear(dim, 9)
        # C18: auxiliary temporal-progress head — predict current step's normalized demo progress ∈ [0,1]
        # from a mean-pool over ALL cond tokens (source + current), a weak demo-current alignment signal.
        aux_progress_cfg = aux_progress_cfg or {}
        self.aux_progress_enabled = bool(aux_progress_cfg.get("enabled", False))
        if self.aux_progress_enabled:
            self.aux_progress_norm = nn.LayerNorm(dim)
            self.aux_progress_head = nn.Linear(dim, 1)

    def _readout(self, video: torch.Tensor, time_embed, stream: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = video.shape
        x = ((video - self.image_mean) / self.image_std).reshape(b * t, c, h, w)
        be = self.vit.backend
        x = be.patch_embed(x)
        x = be._pos_embed(x)
        if hasattr(be, "patch_drop"):
            x = be.patch_drop(x)
        if hasattr(be, "norm_pre"):
            x = be.norm_pre(x)
        for blk in be.blocks[: self.segmenter_start]:
            x = blk(x)
        q = self.query[None].expand(b * t, -1, -1)
        x = torch.cat([q, x], dim=1)
        for blk in be.blocks[self.segmenter_start :]:
            x = blk(x)
        q_out = be.norm(x[:, : self.num_queries, :]).reshape(b, t, self.num_queries, -1)
        if time_embed is not None:
            q_out = q_out + time_embed[:t].unsqueeze(0)
        q_out = q_out + stream.unsqueeze(0)
        return q_out.reshape(b, t * self.num_queries, -1)

    def _build_cond(self, source_video, current_frame, point_track, ee, point_track_causal=None) -> torch.Tensor:
        cur = self._readout(current_frame.unsqueeze(1), None, self.type_current)
        groups = [self._readout(source_video, self.source_time_embed, self.type_source), cur]
        if self.point_encoder is not None and point_track is not None:   # C7-flow has no point tracks
            groups.append(self.point_encoder(point_track) + self.type_point)
        if self.point_encoder_causal is not None and point_track_causal is not None:
            groups.append(self.point_encoder_causal(point_track_causal) + self.type_point_causal)
        if ee.dim() == 1:
            ee = ee.unsqueeze(0)
        groups.append(self.ee_query.expand(ee.shape[0], -1, -1) + self.ee_mlp(ee).unsqueeze(1) + self.type_ee)
        return torch.cat(groups, dim=1)  # NO mixer; flow head's VL mixer handles it

    def _aux_traj_pred(self, cond: torch.Tensor) -> torch.Tensor:
        """Predict per-frame abs EE pose from the source tokens (first source_len*num_queries in cond).
        mean-pool over the num_queries dim → [B, source_len, dim] → head → [B, source_len, 9]."""
        src = cond[:, : self.source_len * self.num_queries]
        src = src.reshape(cond.shape[0], self.source_len, self.num_queries, -1).mean(dim=2)
        return torch.tanh(self.aux_traj_head(self.aux_traj_norm(src)))

    def forward(self, source_video, target_history=None, proprioception=None, action_target=None,
                point_track=None, point_track_causal=None):
        if target_history is None or proprioception is None:
            raise ValueError("FusedQueryFlowPolicy needs source_video, target_history, proprioception(EE)")
        cond = self._build_cond(source_video, target_history[:, -1], point_track, proprioception, point_track_causal)
        if self.training:
            if action_target is None:
                raise ValueError("flow_dit head requires action_target during training")
            out = self.head(cond, action_target)
        else:
            out = self.head.sample(cond)
        if self.aux_traj_enabled:
            out["traj_pred"] = self._aux_traj_pred(cond)
        if self.aux_progress_enabled:
            pooled = self.aux_progress_norm(cond.mean(dim=1))          # [B, dim]
            out["progress_pred"] = torch.sigmoid(self.aux_progress_head(pooled))  # [B, 1] ∈ (0,1)
        return out
