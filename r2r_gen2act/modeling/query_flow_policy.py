"""Query-in-backbone readout + flow-matching DiT head.

Replaces the random-init PerceiverResampler/fusion bottleneck (which compressed 8x256 patch tokens
into 128 latents BEFORE the head and lost fine spatial cues, see EXPERIMENTS.md Exp 8/11) with a
VidEoMT-style readout: learnable query tokens are concatenated with each frame's DINOv2 patch tokens
and run through the LAST few (pretrained) ViT blocks, so the queries read the image via the
pretrained attention at full patch resolution (no compression bottleneck). The per-frame query
embeddings become the conditioning token sequence for the flow DiT.

Two-stage design (matches IDM/GR00T): the expensive backbone+readout runs ONCE per sample; the small
flow DiT cross-attends to the conditioning and is the only thing re-run per denoising step.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from r2r_gen2act.modeling.flow_dit import FlowMatchingDiTHead
from r2r_gen2act.modeling.vit import ViTBackbone


class QueryFlowPolicy(nn.Module):
    def __init__(self, vit: ViTBackbone, head: FlowMatchingDiTHead, source_len: int,
                 target_history_len: int, num_queries: int = 16, segmenter_start: int = 9,
                 image_size: int = 224, proprioception_dim: int = 0, single_stream: bool = False) -> None:
        super().__init__()
        self.vit = vit
        self.head = head
        self.action_head_type = "flow_dit"
        self.image_size = image_size
        self.num_queries = int(num_queries)
        # single_stream: pure-video videomt style — only the source clip ([0,+5,+10,+15,+20] =
        # current + future frames) is read out; no target_history stream, no proprioception.
        self.single_stream = bool(single_stream)
        dim = vit.hidden_dim
        n_blocks = len(self.vit.backend.blocks)
        # segmenter_start: queries are injected before these last blocks (negative = from the end).
        self.segmenter_start = int(segmenter_start) % n_blocks if segmenter_start >= 0 else n_blocks + int(segmenter_start)
        # learnable readout queries (shared across frames/streams) + per-frame & per-stream embeddings.
        self.query = nn.Parameter(torch.randn(self.num_queries, dim) / dim**0.5)
        self.source_time_embed = nn.Parameter(torch.randn(source_len, 1, dim) / dim**0.5)
        self.target_time_embed = nn.Parameter(torch.randn(target_history_len, 1, dim) / dim**0.5)
        self.source_stream = nn.Parameter(torch.randn(1, 1, dim) / dim**0.5)
        self.target_stream = nn.Parameter(torch.randn(1, 1, dim) / dim**0.5)
        self.proprioception_dim = int(proprioception_dim)
        self.proprioception_proj = (
            nn.Sequential(nn.LayerNorm(self.proprioception_dim), nn.Linear(self.proprioception_dim, dim), nn.GELU(), nn.Linear(dim, dim))
            if self.proprioception_dim > 0 else None
        )
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("image_mean", mean, persistent=False)
        self.register_buffer("image_std", std, persistent=False)

    def _readout(self, video: torch.Tensor, time_embed: torch.Tensor, stream: torch.Tensor) -> torch.Tensor:
        """video [B,T,3,H,W] -> query conditioning tokens [B, T*num_queries, dim] via query-in-backbone."""
        if video.dim() != 5:
            raise ValueError(f"Expected [B,T,3,H,W], got {tuple(video.shape)}")
        b, t, c, h, w = video.shape
        x = ((video - self.image_mean) / self.image_std).reshape(b * t, c, h, w)
        be = self.vit.backend
        x = be.patch_embed(x)
        x = be._pos_embed(x)
        if hasattr(be, "patch_drop"):
            x = be.patch_drop(x)
        if hasattr(be, "norm_pre"):
            x = be.norm_pre(x)
        # early blocks on patches only
        for blk in be.blocks[: self.segmenter_start]:
            x = blk(x)
        # inject queries and run the (pretrained) segmenter blocks: queries read patches via
        # the pretrained attention at full resolution.
        q = self.query[None].expand(b * t, -1, -1)
        x = torch.cat([q, x], dim=1)
        for blk in be.blocks[self.segmenter_start :]:
            x = blk(x)
        q_out = be.norm(x[:, : self.num_queries, :])  # [B*T, M, dim]
        q_out = q_out.reshape(b, t, self.num_queries, -1)
        # broadcast per-frame time embed [t,1,dim]->[1,t,1,dim] and stream [1,1,dim]->[1,1,1,dim]
        q_out = q_out + time_embed[:t].unsqueeze(0) + stream.unsqueeze(0)
        return q_out.reshape(b, t * self.num_queries, -1)

    def _build_cond(self, source_video, target_history, proprioception) -> torch.Tensor:
        src = self._readout(source_video, self.source_time_embed, self.source_stream)
        if self.single_stream:
            return src  # pure-video: only the [0,+5,+10,+15,+20] clip, no target/proprio
        tgt = self._readout(target_history, self.target_time_embed, self.target_stream)
        cond = torch.cat([src, tgt], dim=1)
        if self.proprioception_proj is not None:
            if proprioception is None:
                raise ValueError("proprioception required when proprioception_dim > 0")
            if proprioception.dim() == 1:
                proprioception = proprioception.unsqueeze(0)
            cond = torch.cat([cond, self.proprioception_proj(proprioception).unsqueeze(1)], dim=1)
        return cond

    def forward(self, source_video, target_history, proprioception=None, action_target=None, point_track=None):
        cond = self._build_cond(source_video, target_history, proprioception)
        if self.training:
            if action_target is None:
                raise ValueError("flow_dit head requires action_target during training")
            return self.head(cond, action_target)
        return self.head.sample(cond)
