from __future__ import annotations

import torch
import torch.nn.functional as F

from r2r_gen2act.data.action.codec import ActionCodec


def compute_losses(outputs: dict[str, torch.Tensor], batch: dict, codec: ActionCodec, cfg: dict) -> dict[str, torch.Tensor]:
    weights = cfg["train"].get("losses", {})
    action = batch["action"][:, : codec.pose_dims]
    action_mode = str(cfg.get("action", {}).get("mode", "classification"))
    dim_losses = []
    losses: dict[str, torch.Tensor]
    if action_mode == "regression":
        action_pred = outputs["action_pred"]
        loss_type = str(weights.get("action_regression_loss", "smooth_l1"))
        if loss_type == "mse":
            per_dim = (action_pred - action).pow(2).mean(dim=0)
        else:
            beta = float(weights.get("smooth_l1_beta", 0.02))
            per_dim = F.smooth_l1_loss(action_pred, action, beta=beta, reduction="none").mean(dim=0)
        dim_losses = [per_dim[dim] for dim in range(codec.pose_dims)]
        loss_action = torch.stack(dim_losses).mean()
        abs_err = (action_pred - action).abs()
        metrics = {
            "action_mae": abs_err.mean(),
            "action_rmse": (action_pred - action).pow(2).mean().sqrt(),
        }
        for dim in range(codec.pose_dims):
            metrics[f"action_dim_{dim}_mae"] = abs_err[:, dim].mean()
    else:
        action_bins = codec.discretize(action)
        action_logits = outputs["action_logits"]
        for dim in range(codec.pose_dims):
            dim_losses.append(F.cross_entropy(action_logits[:, dim, :], action_bins[:, dim]))
        loss_action = torch.stack(dim_losses).mean()
        metrics = {}
    loss_gripper = F.cross_entropy(outputs["gripper_logits"], batch["gripper"].long())
    loss_terminate = F.cross_entropy(outputs["terminate_logits"], batch["terminate"].long())
    total = (
        float(weights.get("action_weight", 1.0)) * loss_action
        + float(weights.get("gripper_weight", 0.2)) * loss_gripper
        + float(weights.get("terminate_weight", 0.1)) * loss_terminate
    )
    losses = {"loss": total, "action_loss": loss_action, "gripper_loss": loss_gripper, "terminate_loss": loss_terminate, **metrics}
    for dim, dim_loss in enumerate(dim_losses):
        losses[f"action_dim_{dim}_loss"] = dim_loss
    return losses
