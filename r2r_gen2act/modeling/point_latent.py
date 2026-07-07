"""Step 6: point-track -> point latent module.

Input: tracked EE-neighborhood points, uniformly resampled to a fixed number of timesteps.
  point_tracks: [B, N, S, 2]  (N points, S time samples, normalized (x,y) in [-1,1])
Pipeline (per the plan):
  1. temporal MLP over each point's S-step trajectory -> per-point embedding [B, N, hidden]
  2. self-attention ACROSS the N points -> contextualized point tokens [B, N, hidden]
These N tokens become conditioning tokens for the flow-matching DiT.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
import torch.nn as nn


class PointLatentEncoder(nn.Module):
    def __init__(self, num_points: int = 10, num_time: int = 60, hidden_dim: int = 384,
                 out_dim: int = 768, heads: int = 6, attn_layers: int = 2, dropout: float = 0.0) -> None:
        super().__init__()
        self.num_points = int(num_points)
        self.num_time = int(num_time)
        # temporal MLP: flatten the S-step (x,y) trajectory of each point and embed it.
        self.temporal = nn.Sequential(
            nn.Linear(num_time * 2, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.point_embed = nn.Parameter(torch.randn(1, num_points, hidden_dim) / hidden_dim**0.5)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=heads, dim_feedforward=hidden_dim * 4,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.cross_point_attn = nn.TransformerEncoder(enc_layer, num_layers=attn_layers)
        self.out = nn.Linear(hidden_dim, out_dim)

    def forward(self, point_tracks: torch.Tensor) -> torch.Tensor:
        """point_tracks: [B, N, S, 2] -> point tokens [B, N, out_dim]."""
        b, n, s, c = point_tracks.shape
        if s != self.num_time:
            # uniformly resample to num_time along the time axis (linear interp).
            x = point_tracks.permute(0, 1, 3, 2).reshape(b * n, c, s)
            x = torch.nn.functional.interpolate(x, size=self.num_time, mode="linear", align_corners=True)
            point_tracks = x.reshape(b, n, c, self.num_time).permute(0, 1, 3, 2)
        flat = point_tracks.reshape(b, n, self.num_time * 2)
        tok = self.temporal(flat) + self.point_embed[:, :n]
        tok = self.cross_point_attn(tok)
        return self.out(tok)


class PointTrajSeqEncoder(nn.Module):
    """Step 7-C: demo point trajectory -> per-TIMESTEP token sequence (NOT pooled over time).

    Input [B, N, S, 2] -> [B, S, out_dim]: each token encodes the N points' positions at one
    timestep, plus a temporal positional embedding. Keeping the trajectory as a time sequence (vs
    PointLatentEncoder which pools time into per-point tokens) lets the model attend to a specific
    phase of the demonstrated motion -> local trajectory detail (incl. grasp->lift transitions) is
    accessible via attention/localization, rather than via a causal-momentum shortcut."""

    def __init__(self, num_points: int = 10, num_time: int = 32, out_dim: int = 768,
                 hidden_dim: int = 384) -> None:
        super().__init__()
        self.num_points = int(num_points)
        self.num_time = int(num_time)
        self.embed = nn.Sequential(
            nn.Linear(num_points * 2, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, out_dim), nn.GELU(), nn.LayerNorm(out_dim),
        )
        self.time_pos = nn.Parameter(torch.randn(1, num_time, out_dim) / out_dim**0.5)

    def forward(self, point_tracks: torch.Tensor) -> torch.Tensor:
        """point_tracks: [B, N, S, 2] -> [B, num_time, out_dim] (one token per timestep)."""
        b, n, s, c = point_tracks.shape
        if s != self.num_time:
            x = point_tracks.permute(0, 1, 3, 2).reshape(b * n, c, s)
            x = F.interpolate(x, size=self.num_time, mode="linear", align_corners=True)
            point_tracks = x.reshape(b, n, c, self.num_time).permute(0, 1, 3, 2)
        # per timestep: gather the N points -> [B, S, N*2]
        per_t = point_tracks.permute(0, 2, 1, 3).reshape(b, self.num_time, n * 2)
        return self.embed(per_t) + self.time_pos
