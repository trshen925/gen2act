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
                 aux_progress_cfg: dict | None = None, dt_time_cfg: dict | None = None,
                 current_full_patch: bool = False, max_source_len: int | None = None,
                 pad_source: bool = False) -> None:
        super().__init__()
        # C21: current obs uses all 256 DINOv2 patch tokens (no readout compression) — the current
        # frame is the most action-relevant input, full patches preserve spatial detail.
        self.current_full_patch = bool(current_full_patch)
        self.vit = vit
        self.head = head
        self.point_encoder = point_encoder
        self.point_encoder_causal = point_encoder_causal
        self.action_head_type = "flow_dit"
        self.image_size = image_size
        self.num_queries = int(num_queries)
        self.source_len = int(source_len)
        # C24: with dynamic frame count, source_time_embed must cover the largest k (positions are
        # sliced [:t] at runtime). max_source_len defaults to source_len (static case).
        self.max_source_len = int(max_source_len) if max_source_len else int(source_len)
        dim = vit.hidden_dim
        n_blocks = len(self.vit.backend.blocks)
        self.segmenter_start = int(segmenter_start) % n_blocks if segmenter_start >= 0 else n_blocks + int(segmenter_start)
        self.query = nn.Parameter(torch.randn(self.num_queries, dim) / dim**0.5)
        self.source_time_embed = nn.Parameter(torch.randn(self.max_source_len, 1, dim) / dim**0.5)
        # type embeddings so the flow head's VL mixer can tell the conditioning sources apart
        self.type_source = nn.Parameter(torch.randn(1, 1, dim) / dim**0.5)
        self.type_current = nn.Parameter(torch.randn(1, 1, dim) / dim**0.5)
        # Keep the state-dict key for checkpoint compatibility, but do not ask
        # DDP to reduce a parameter for a condition stream that is disabled.
        self.type_point = nn.Parameter(
            torch.randn(1, 1, dim) / dim**0.5,
            requires_grad=point_encoder is not None,
        )
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
        # C20: Δt time-conditioning — encode each source frame's real seconds-since-previous-sampled-
        # frame (motion pace) and ADD to the source readout tokens, on top of source_time_embed.
        # Lets the model tell a fast 3s demo from a slow 30s demo (same 8 frames, different Δt).
        dt_time_cfg = dt_time_cfg or {}
        self.dt_time_enabled = bool(dt_time_cfg.get("enabled", False))
        if self.dt_time_enabled:
            self.dt_num_freq = int(dt_time_cfg.get("num_freqs", 16))
            self.dt_max_sec = float(dt_time_cfg.get("max_sec", 5.0))
            self.dt_mlp = nn.Sequential(
                nn.Linear(2 * self.dt_num_freq, dim), nn.GELU(), nn.Linear(dim, dim))
        # C26: pad dynamic source frames to max_source_len with a LEARNABLE pad-frame embedding (not
        # zeros/const). Padded frames still get source_time_embed[pos] + type_source and PARTICIPATE
        # in attention (no mask) — so all max_source_len positions of source_time_embed get trained
        # uniformly (fixes the C24/C25 uneven-position-embedding issue), and the model can learn to
        # treat the pad marker as "no content here".
        self.pad_source = bool(pad_source)
        if self.pad_source:
            self.pad_frame_embed = nn.Parameter(torch.randn(1, 1, dim) / dim**0.5)

    def _dt_embed(self, dt_sec: torch.Tensor) -> torch.Tensor:
        """Sinusoidal encoding of per-frame Δt (seconds) → [B, T, dim]. dt_sec: [B, T]."""
        # log-spaced frequencies over [0, dt_max_sec]; normalize seconds to ~[0,1] first.
        d = (dt_sec / max(1e-6, self.dt_max_sec)).clamp(0.0, 1.0).unsqueeze(-1)  # [B,T,1]
        freqs = torch.arange(self.dt_num_freq, device=dt_sec.device, dtype=dt_sec.dtype)
        freqs = (2.0 ** freqs) * torch.pi                                        # [F]
        ang = d * freqs                                                          # [B,T,F]
        feat = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)              # [B,T,2F]
        return self.dt_mlp(feat)                                                 # [B,T,dim]

    def _readout(self, video: torch.Tensor, time_embed, stream: torch.Tensor,
                 dt_sec: torch.Tensor | None = None, pad_to: int | None = None) -> torch.Tensor:
        b, t, c, h, w = video.shape
        x = ((video - self.image_mean) / self.image_std).reshape(b * t, c, h, w)
        x, context = self.vit.prepare_tokens(x)
        x = self.vit.run_blocks(x, end=self.segmenter_start, context=context)
        q = self.query[None].expand(b * t, -1, -1)
        x = torch.cat([q, x], dim=1)
        x = self.vit.run_blocks(
            x,
            start=self.segmenter_start,
            context=context,
            extra_prefix_tokens=self.num_queries,
        )
        q_out = self.vit.normalize_tokens(x[:, : self.num_queries, :]).reshape(b, t, self.num_queries, -1)
        # C20: add per-frame Δt embedding to the REAL frames (broadcast over the num_queries axis)
        if self.dt_time_enabled and dt_sec is not None:
            q_out = q_out + self._dt_embed(dt_sec).unsqueeze(2)   # [B,t,1,dim]
        # C26: pad the frame dim to `pad_to` with a learnable pad-frame embedding, so all positions of
        # source_time_embed get trained (pad frames participate in attention, no mask).
        n = t
        if pad_to is not None and t < pad_to:
            n = int(pad_to)
            pad = self.pad_frame_embed.expand(b, n - t, self.num_queries, -1)  # [b, n-t, Q, dim]
            q_out = torch.cat([q_out, pad], dim=1)                             # [b, n, Q, dim]
        if time_embed is not None:
            q_out = q_out + time_embed[:n].unsqueeze(0)     # per-frame position embed (real + pad)
        q_out = q_out + stream.unsqueeze(0)
        return q_out.reshape(b, n * self.num_queries, -1)

    def _encode_current_full(self, frame: torch.Tensor) -> torch.Tensor:
        """Encode current obs frame as all 256 DINOv2 patch tokens (no readout-query injection).
        Returns [B, 256, dim] with type_current added."""
        x = (frame - self.image_mean) / self.image_std
        x, context = self.vit.prepare_tokens(x)
        x = self.vit.run_blocks(x, context=context)
        x = self.vit.normalize_tokens(x)
        return self.vit.patch_tokens(x) + self.type_current

    def _build_cond(self, source_video, current_frame, point_track, ee, point_track_causal=None,
                    source_dt=None) -> torch.Tensor:
        if self.current_full_patch:
            cur = self._encode_current_full(current_frame)                       # [B, 256, dim]
        else:
            cur = self._readout(current_frame.unsqueeze(1), None, self.type_current)  # [B, 32, dim]
        pad_to = self.max_source_len if self.pad_source else None
        groups = [self._readout(source_video, self.source_time_embed, self.type_source, source_dt, pad_to), cur]
        if self.point_encoder is not None and point_track is not None:   # C7-flow has no point tracks
            groups.append(self.point_encoder(point_track) + self.type_point)
        if self.point_encoder_causal is not None and point_track_causal is not None:
            groups.append(self.point_encoder_causal(point_track_causal) + self.type_point_causal)
        if ee.dim() == 1:
            ee = ee.unsqueeze(0)
        groups.append(self.ee_query.expand(ee.shape[0], -1, -1) + self.ee_mlp(ee).unsqueeze(1) + self.type_ee)
        return torch.cat(groups, dim=1)  # NO mixer; flow head's VL mixer handles it

    def _aux_traj_pred(self, cond: torch.Tensor, k: int) -> torch.Tensor:
        """Predict per-frame abs EE pose from the source tokens (first k*num_queries in cond).
        mean-pool over the num_queries dim → [B, k, dim] → head → [B, k, 9]. k = source frame count
        (runtime, dynamic under C24)."""
        src = cond[:, : k * self.num_queries]
        src = src.reshape(cond.shape[0], k, self.num_queries, -1).mean(dim=2)
        return torch.tanh(self.aux_traj_head(self.aux_traj_norm(src)))

    def forward(self, source_video, target_history=None, proprioception=None, action_target=None,
                point_track=None, point_track_causal=None, source_dt=None):
        if target_history is None or proprioception is None:
            raise ValueError("FusedQueryFlowPolicy needs source_video, target_history, proprioception(EE)")
        k_src = int(source_video.shape[1])   # runtime source frame count (dynamic under C24)
        cond = self._build_cond(source_video, target_history[:, -1], point_track, proprioception,
                                point_track_causal, source_dt)
        if self.training:
            if action_target is None:
                raise ValueError("flow_dit head requires action_target during training")
            out = self.head(cond, action_target)
        else:
            out = self.head.sample(cond)
        if self.aux_traj_enabled:
            out["traj_pred"] = self._aux_traj_pred(cond, k_src)
        if self.aux_progress_enabled:
            pooled = self.aux_progress_norm(cond.mean(dim=1))          # [B, dim]
            out["progress_pred"] = torch.sigmoid(self.aux_progress_head(pooled))  # [B, 1] ∈ (0,1)
        return out
