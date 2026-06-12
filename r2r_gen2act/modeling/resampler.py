from __future__ import annotations

import torch
import torch.nn as nn


class PerceiverResampler(nn.Module):
    def __init__(self, dim: int, num_latents: int = 64, num_layers: int = 2, num_heads: int = 8, ff_mult: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_latents, dim) / dim**0.5)
        self.in_norm = nn.LayerNorm(dim)
        self.blocks = nn.ModuleList([])
        for _ in range(num_layers):
            self.blocks.append(nn.ModuleDict({
                "cross_norm": nn.LayerNorm(dim),
                "cross_attn": nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True),
                "self_norm": nn.LayerNorm(dim),
                "self_attn": nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True),
                "ff_norm": nn.LayerNorm(dim),
                "ff": nn.Sequential(nn.Linear(dim, ff_mult * dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(ff_mult * dim, dim)),
            }))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            x = x.flatten(1, 2)
        if x.dim() != 3:
            raise ValueError(f"Expected [B,N,D] or [B,T,P,D], got {tuple(x.shape)}")
        x = self.in_norm(x)
        latents = self.latents.unsqueeze(0).expand(x.shape[0], -1, -1)
        for block in self.blocks:
            cross, _ = block["cross_attn"](block["cross_norm"](latents), x, x, need_weights=False)
            latents = latents + cross
            normed = block["self_norm"](latents)
            self_out, _ = block["self_attn"](normed, normed, normed, need_weights=False)
            latents = latents + self_out
            latents = latents + block["ff"](block["ff_norm"](latents))
        return latents
