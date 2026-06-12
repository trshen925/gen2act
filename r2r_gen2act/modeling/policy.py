from __future__ import annotations

import torch
import torch.nn as nn

from r2r_gen2act.modeling.fusion import CrossAttentionFusion, PolicyQueryDecoder
from r2r_gen2act.modeling.heads import ActionHead
from r2r_gen2act.modeling.resampler import PerceiverResampler
from r2r_gen2act.modeling.vit import ViTBackbone


class Robot2RobotPolicy(nn.Module):
    def __init__(self, vit: ViTBackbone, source_resampler: PerceiverResampler, target_resampler: PerceiverResampler, fusion: CrossAttentionFusion, decoder: PolicyQueryDecoder, head: ActionHead, source_len: int, target_history_len: int, image_size: int = 224, proprioception_dim: int = 0) -> None:
        super().__init__()
        self.vit = vit
        self.source_resampler = source_resampler
        self.target_resampler = target_resampler
        self.fusion = fusion
        self.decoder = decoder
        self.head = head
        self.image_size = image_size
        dim = vit.hidden_dim
        self.source_time_embed = nn.Parameter(torch.randn(source_len, 1, dim) / dim**0.5)
        self.target_time_embed = nn.Parameter(torch.randn(target_history_len, 1, dim) / dim**0.5)
        self.source_stream_embed = nn.Parameter(torch.randn(1, 1, dim) / dim**0.5)
        self.target_stream_embed = nn.Parameter(torch.randn(1, 1, dim) / dim**0.5)
        self.proprioception_dim = int(proprioception_dim)
        self.proprioception_proj = nn.Sequential(nn.LayerNorm(self.proprioception_dim), nn.Linear(self.proprioception_dim, dim), nn.GELU(), nn.Linear(dim, dim)) if self.proprioception_dim > 0 else None
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("image_mean", mean, persistent=False)
        self.register_buffer("image_std", std, persistent=False)

    def encode_video(self, video: torch.Tensor, resampler: PerceiverResampler, time_embed: torch.Tensor, stream_embed: torch.Tensor) -> torch.Tensor:
        if video.dim() != 5:
            raise ValueError(f"Expected [B,T,3,H,W], got {tuple(video.shape)}")
        b, t, c, h, w = video.shape
        x = video.reshape(b * t, c, h, w)
        x = (x - self.image_mean) / self.image_std
        patch = self.vit(x).view(b, t, -1, self.vit.hidden_dim)
        if t > time_embed.shape[0]:
            raise ValueError(f"time steps {t} exceed configured embedding length {time_embed.shape[0]}")
        patch = patch + time_embed[:t].unsqueeze(0) + stream_embed.view(1, 1, 1, -1)
        return resampler(patch)

    def forward(self, source_video: torch.Tensor, target_history: torch.Tensor, proprioception: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        source_tokens = self.encode_video(source_video, self.source_resampler, self.source_time_embed, self.source_stream_embed)
        target_tokens = self.encode_video(target_history, self.target_resampler, self.target_time_embed, self.target_stream_embed)
        if self.proprioception_proj is not None:
            if proprioception is None:
                raise ValueError("proprioception input is required when model.proprioception_dim > 0")
            if proprioception.dim() == 1:
                proprioception = proprioception.unsqueeze(0)
            target_tokens = target_tokens + self.proprioception_proj(proprioception).unsqueeze(1)
        target_tokens = self.fusion(target_tokens, source_tokens)
        ctx = self.decoder(target_tokens, source_tokens)
        return self.head(ctx)
