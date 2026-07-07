"""Query-in-backbone readout + cross-attention regression decoder (deep supervision).

Same readout as QueryFlowPolicy (learnable queries read DINOv2 patches via the pretrained last
blocks), but the head is a cross-attention REGRESSION decoder (videomt-style loss) instead of the
flow DiT. Ablation to test whether the cross-attention information-transmission scheme is sound.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from r2r_gen2act.modeling.cross_attn_reg import CrossAttnRegressionDecoder
from r2r_gen2act.modeling.vit import ViTBackbone


class QueryRegPolicy(nn.Module):
    def __init__(self, vit: ViTBackbone, head: CrossAttnRegressionDecoder, source_len: int,
                 target_history_len: int, num_queries: int = 16, segmenter_start: int = 9,
                 image_size: int = 224, single_stream: bool = True) -> None:
        super().__init__()
        self.vit = vit
        self.head = head
        self.action_head_type = "regression"
        self.image_size = image_size
        self.num_queries = int(num_queries)
        self.single_stream = bool(single_stream)
        dim = vit.hidden_dim
        n_blocks = len(self.vit.backend.blocks)
        self.segmenter_start = int(segmenter_start) % n_blocks if segmenter_start >= 0 else n_blocks + int(segmenter_start)
        self.query = nn.Parameter(torch.randn(self.num_queries, dim) / dim**0.5)
        self.source_time_embed = nn.Parameter(torch.randn(source_len, 1, dim) / dim**0.5)
        self.target_time_embed = nn.Parameter(torch.randn(target_history_len, 1, dim) / dim**0.5)
        self.source_stream = nn.Parameter(torch.randn(1, 1, dim) / dim**0.5)
        self.target_stream = nn.Parameter(torch.randn(1, 1, dim) / dim**0.5)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("image_mean", mean, persistent=False)
        self.register_buffer("image_std", std, persistent=False)

    def _readout(self, video: torch.Tensor, time_embed: torch.Tensor, stream: torch.Tensor) -> torch.Tensor:
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
        q_out = q_out + time_embed[:t].unsqueeze(0) + stream.unsqueeze(0)
        return q_out.reshape(b, t * self.num_queries, -1)

    def _build_cond(self, source_video, target_history) -> torch.Tensor:
        src = self._readout(source_video, self.source_time_embed, self.source_stream)
        if self.single_stream:
            return src
        tgt = self._readout(target_history, self.target_time_embed, self.target_stream)
        return torch.cat([src, tgt], dim=1)

    def forward(self, source_video, target_history=None, proprioception=None, action_target=None, point_track=None):
        cond = self._build_cond(source_video, target_history)
        return self.head(cond)
