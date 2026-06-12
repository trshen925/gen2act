from __future__ import annotations

import torch
import torch.nn as nn


class CrossAttentionFusion(nn.Module):
    def __init__(self, dim: int, heads: int = 8, dropout: float = 0.0) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(dim)
        self.context_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 4 * dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(4 * dim, dim))

    def forward(self, target_tokens: torch.Tensor, source_tokens: torch.Tensor) -> torch.Tensor:
        attended, _ = self.attn(self.query_norm(target_tokens), self.context_norm(source_tokens), source_tokens, need_weights=False)
        x = target_tokens + attended
        return x + self.ff(x)


class PolicyQueryDecoder(nn.Module):
    def __init__(self, dim: int, heads: int = 8, dropout: float = 0.0) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, dim) / dim**0.5)
        self.q_norm = nn.LayerNorm(dim)
        self.t_norm = nn.LayerNorm(dim)
        self.s_norm = nn.LayerNorm(dim)
        self.target_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.source_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 4 * dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(4 * dim, dim))

    def forward(self, target_tokens: torch.Tensor, source_tokens: torch.Tensor) -> torch.Tensor:
        q = self.query.expand(target_tokens.shape[0], -1, -1)
        target_ctx, _ = self.target_attn(self.q_norm(q), self.t_norm(target_tokens), target_tokens, need_weights=False)
        q = q + target_ctx
        source_ctx, _ = self.source_attn(self.q_norm(q), self.s_norm(source_tokens), source_tokens, need_weights=False)
        q = q + source_ctx
        q = q + self.ff(q)
        return q.squeeze(1)
