"""Step 7: fused-conditioning flow-matching policy.

Assembles the four conditioning signals from the plan into one token sequence, then runs the
flow-matching DiT to predict the action chunk. CRITICAL: there is NO t+H future-frame input —
future information enters ONLY through the VideoMAEv2-encoded reference video.

  reference video (8 frames of the demo) -> VideoMAEv2          -> video tokens
  current real image (target_history[-1]) -> DINOv2 + resampler -> image tokens
  tracked EE points [N, S, 2]              -> PointLatentEncoder -> point tokens
  current EE 2D image coord (u,v)          -> linear            -> ee token
  cond = concat(video, image, point, ee) -> FlowMatchingDiTHead(noisy action, t) -> velocity
"""
from __future__ import annotations

import torch
import torch.nn as nn

from r2r_gen2act.modeling.flow_dit import FlowMatchingDiTHead
from r2r_gen2act.modeling.point_latent import PointLatentEncoder
from r2r_gen2act.modeling.resampler import PerceiverResampler
from r2r_gen2act.modeling.video_encoder import VideoMAEv2Encoder
from r2r_gen2act.modeling.vit import ViTBackbone


class FusedFlowPolicy(nn.Module):
    def __init__(self, vit: ViTBackbone, video_encoder: VideoMAEv2Encoder,
                 point_encoder: PointLatentEncoder, head: FlowMatchingDiTHead,
                 image_resampler: PerceiverResampler, dim: int = 768,
                 num_video_tokens: int = 4, ee_dim: int = 2, image_size: int = 224) -> None:
        super().__init__()
        self.vit = vit
        self.video_encoder = video_encoder
        self.point_encoder = point_encoder
        self.image_resampler = image_resampler
        self.head = head
        self.action_head_type = "flow_dit"
        self.image_size = image_size
        self.num_video_tokens = int(num_video_tokens)
        self.video_proj = nn.Sequential(
            nn.LayerNorm(video_encoder.output_dim),
            nn.Linear(video_encoder.output_dim, num_video_tokens * dim),
        )
        self.video_token_norm = nn.LayerNorm(dim)
        self.ee_proj = nn.Sequential(nn.LayerNorm(ee_dim), nn.Linear(ee_dim, dim), nn.GELU(), nn.Linear(dim, dim))
        # learned stream embeddings so the DiT can tell the conditioning sources apart
        self.video_stream = nn.Parameter(torch.randn(1, 1, dim) / dim**0.5)
        self.image_stream = nn.Parameter(torch.randn(1, 1, dim) / dim**0.5)
        self.point_stream = nn.Parameter(torch.randn(1, 1, dim) / dim**0.5)
        self.ee_stream = nn.Parameter(torch.randn(1, 1, dim) / dim**0.5)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("image_mean", mean, persistent=False)
        self.register_buffer("image_std", std, persistent=False)

    def _encode_current_image(self, image: torch.Tensor) -> torch.Tensor:
        x = (image - self.image_mean) / self.image_std
        patches = self.vit(x)  # [B, P, dim]
        return self.image_resampler(patches)  # [B, L, dim]

    def _build_cond(self, source_video, target_history, proprioception, point_track) -> torch.Tensor:
        b = source_video.shape[0]
        # video tokens (reference video -> VideoMAEv2)
        vlat = self.video_encoder(source_video)  # [B, 768]
        vtok = self.video_proj(vlat).view(b, self.num_video_tokens, -1)
        vtok = self.video_token_norm(vtok) + self.video_stream
        # current image tokens (last frame of target history)
        itok = self._encode_current_image(target_history[:, -1]) + self.image_stream
        # point tokens
        if point_track is None:
            raise ValueError("FusedFlowPolicy requires point_track input")
        ptok = self.point_encoder(point_track) + self.point_stream
        # ee coord token
        if proprioception is None:
            raise ValueError("FusedFlowPolicy requires proprioception (EE 2D coord)")
        if proprioception.dim() == 1:
            proprioception = proprioception.unsqueeze(0)
        etok = self.ee_proj(proprioception).unsqueeze(1) + self.ee_stream
        return torch.cat([vtok, itok, ptok, etok], dim=1)

    def forward(self, source_video, target_history, proprioception=None, action_target=None, point_track=None):
        cond = self._build_cond(source_video, target_history, proprioception, point_track)
        if self.training:
            if action_target is None:
                raise ValueError("flow_dit head requires action_target during training")
            return self.head(cond, action_target)
        return self.head.sample(cond)
