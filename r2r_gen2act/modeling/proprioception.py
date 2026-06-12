from __future__ import annotations

import torch
import torch.nn as nn

from r2r_gen2act.modeling.heads import ActionHead


class ProprioceptionOnlyPolicy(nn.Module):
    def __init__(self, proprioception_dim: int, hidden_dim: int, head: ActionHead, layers: int = 2) -> None:
        super().__init__()
        blocks = [nn.LayerNorm(proprioception_dim), nn.Linear(proprioception_dim, hidden_dim), nn.GELU()]
        for _ in range(max(0, layers - 1)):
            blocks.extend([nn.Linear(hidden_dim, hidden_dim), nn.GELU()])
        self.encoder = nn.Sequential(*blocks)
        self.head = head

    def forward(self, source_video: torch.Tensor | None = None, target_history: torch.Tensor | None = None, proprioception: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        if proprioception is None:
            raise ValueError("proprioception input is required for ProprioceptionOnlyPolicy")
        if proprioception.dim() == 1:
            proprioception = proprioception.unsqueeze(0)
        ctx = self.encoder(proprioception)
        return self.head(ctx)
