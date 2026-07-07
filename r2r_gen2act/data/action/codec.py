from __future__ import annotations

import torch
import torch.nn.functional as F


class ActionCodec:
    def __init__(self, pose_dims: int, num_bins: int, low, high) -> None:
        self.pose_dims = int(pose_dims)
        self.num_bins = int(num_bins)
        self.low = torch.as_tensor(low, dtype=torch.float32)
        self.high = torch.as_tensor(high, dtype=torch.float32)
        if self.low.numel() != self.pose_dims or self.high.numel() != self.pose_dims:
            raise ValueError("Action bounds must match pose_dims")
        if not torch.all(self.high > self.low):
            raise ValueError("Each action bound must satisfy high > low")

    @classmethod
    def from_config(cls, cfg: dict) -> "ActionCodec":
        action = cfg["action"]
        pose_dims = int(action["pose_dims"])
        if action.get("bounds_source") == "stats_json":
            from r2r_gen2act.data.action.stats import load_action_stats, pose_bounds_from_stats
            stats = load_action_stats(action["stats_path"])
            low, high = pose_bounds_from_stats(stats, pose_dims)
        else:
            low = action.get("bounds_low", [-1.0] * pose_dims)[:pose_dims]
            high = action.get("bounds_high", [1.0] * pose_dims)[:pose_dims]
        return cls(pose_dims, int(action["num_bins"]), low, high)

    def discretize(self, pose_action: torch.Tensor) -> torch.Tensor:
        low = self.low.to(device=pose_action.device, dtype=pose_action.dtype)
        high = self.high.to(device=pose_action.device, dtype=pose_action.dtype)
        clipped = torch.minimum(torch.maximum(pose_action, low), high)
        scale = (high - low).clamp(min=1e-6)
        bins = torch.round((clipped - low) / scale * (self.num_bins - 1)).long()
        return bins.clamp_(0, self.num_bins - 1)

    def decode(self, bins: torch.Tensor) -> torch.Tensor:
        low = self.low.to(device=bins.device, dtype=torch.float32)
        high = self.high.to(device=bins.device, dtype=torch.float32)
        scale = (high - low).clamp(min=1e-6)
        return low + bins.float() * scale / max(1, self.num_bins - 1)

    def normalize(self, pose_action: torch.Tensor) -> torch.Tensor:
        """Clamp raw pose to [low, high] then map per-dim to [-1, 1] (regression targets)."""
        low = self.low.to(device=pose_action.device, dtype=pose_action.dtype)
        high = self.high.to(device=pose_action.device, dtype=pose_action.dtype)
        clipped = torch.minimum(torch.maximum(pose_action, low), high)
        scale = (high - low).clamp(min=1e-6)
        return (clipped - low) / scale * 2.0 - 1.0

    def unnormalize(self, norm: torch.Tensor) -> torch.Tensor:
        """Inverse of normalize: map [-1, 1] back to raw [low, high] units."""
        low = self.low.to(device=norm.device, dtype=torch.float32)
        high = self.high.to(device=norm.device, dtype=torch.float32)
        scale = (high - low).clamp(min=1e-6)
        return (norm.float() + 1.0) * 0.5 * scale + low


def action_loss(logits: torch.Tensor, target_bins: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target_bins.reshape(-1))
