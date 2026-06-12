from __future__ import annotations

import torch
import torch.nn as nn


class ActionHead(nn.Module):
    def __init__(self, dim: int, pose_action_dims: int = 6, num_bins: int = 256, action_mode: str = "classification") -> None:
        super().__init__()
        self.pose_action_dims = pose_action_dims
        self.num_bins = num_bins
        self.action_mode = action_mode
        self.mlp = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim), nn.GELU())
        out_dim = pose_action_dims if action_mode == "regression" else pose_action_dims * num_bins
        self.action_proj = nn.Linear(dim, out_dim)
        self.gripper_proj = nn.Linear(dim, 2)
        self.terminate_proj = nn.Linear(dim, 2)

    def forward(self, ctx: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.mlp(ctx)
        outputs = {
            "gripper_logits": self.gripper_proj(h),
            "terminate_logits": self.terminate_proj(h),
        }
        if self.action_mode == "regression":
            outputs["action_pred"] = self.action_proj(h)
        else:
            outputs["action_logits"] = self.action_proj(h).view(-1, self.pose_action_dims, self.num_bins)
        return outputs
