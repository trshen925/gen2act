from __future__ import annotations

import torch
import torch.nn as nn


class CrossAttentionFusion(nn.Module):
    """Stack of cross-attention blocks: target tokens repeatedly attend to source tokens."""

    def __init__(self, dim: int, heads: int = 8, num_layers: int = 1, dropout: float = 0.0) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                "query_norm": nn.LayerNorm(dim),
                "context_norm": nn.LayerNorm(dim),
                "attn": nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True),
                "ff": nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 4 * dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(4 * dim, dim)),
            })
            for _ in range(max(1, int(num_layers)))
        ])

    def forward(self, target_tokens: torch.Tensor, source_tokens: torch.Tensor) -> torch.Tensor:
        x = target_tokens
        for block in self.blocks:
            attended, _ = block["attn"](block["query_norm"](x), block["context_norm"](source_tokens), source_tokens, need_weights=False)
            x = x + attended
            x = x + block["ff"](x)
        return x


class PolicyQueryDecoder(nn.Module):
    """`num_queries` learnable queries refined through a stack of (target-attn, source-attn, ff)
    blocks. Returns [B, dim] when num_queries==1 (legacy single-action), else [B, num_queries, dim]
    so each query decodes one action-chunk step."""

    def __init__(self, dim: int, heads: int = 8, num_layers: int = 1, num_queries: int = 1, dropout: float = 0.0) -> None:
        super().__init__()
        self.num_queries = max(1, int(num_queries))
        self.query = nn.Parameter(torch.randn(1, self.num_queries, dim) / dim**0.5)
        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                "q_norm": nn.LayerNorm(dim),
                "t_norm": nn.LayerNorm(dim),
                "s_norm": nn.LayerNorm(dim),
                "target_attn": nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True),
                "source_attn": nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True),
                "ff": nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 4 * dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(4 * dim, dim)),
            })
            for _ in range(max(1, int(num_layers)))
        ])

    def forward(self, target_tokens: torch.Tensor, source_tokens: torch.Tensor) -> torch.Tensor:
        q = self.query.expand(target_tokens.shape[0], -1, -1)
        for block in self.blocks:
            target_ctx, _ = block["target_attn"](block["q_norm"](q), block["t_norm"](target_tokens), target_tokens, need_weights=False)
            q = q + target_ctx
            source_ctx, _ = block["source_attn"](block["q_norm"](q), block["s_norm"](source_tokens), source_tokens, need_weights=False)
            q = q + source_ctx
            q = q + block["ff"](q)
        return q.squeeze(1) if self.num_queries == 1 else q
