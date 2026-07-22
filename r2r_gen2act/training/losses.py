from __future__ import annotations

import torch
import torch.nn.functional as F

from r2r_gen2act.data.action.codec import ActionCodec


def compute_losses(outputs: dict[str, torch.Tensor], batch: dict, codec: ActionCodec, cfg: dict) -> dict[str, torch.Tensor]:
    weights = cfg["train"].get("losses", {})
    pose_dims = codec.pose_dims
    diffuse_gripper = bool(cfg.get("model", {}).get("flow_dit", {}).get("diffuse_gripper", False))
    # Leading dims may be [B] (single action) or [B, N] (action chunk); flatten them so the same
    # code path handles both. The last dim is always the action/pose dimension.
    action = batch["action"][..., :pose_dims].reshape(-1, pose_dims)
    action_mode = str(cfg.get("action", {}).get("mode", "classification"))
    dim_losses = []
    losses: dict[str, torch.Tensor]
    if action_mode == "flow":
        # Flow-matching DiT. Training: the head returns predicted/target velocity (loss = MSE on the
        # flow field). Eval: the head samples an action chunk (Euler integration); we then report the
        # same MAE/RMSE metrics as regression so held-out quality is directly comparable.
        flow_dims = pose_dims + int(diffuse_gripper)
        if "pred_velocity" in outputs:
            pred_v = outputs["pred_velocity"].reshape(-1, flow_dims)
            tgt_v = outputs["target_velocity"].reshape(-1, flow_dims)
            per_dim = (pred_v - tgt_v).pow(2).mean(dim=0)
            dim_losses = [per_dim[dim] for dim in range(pose_dims)]
            loss_action = per_dim.mean()
            metrics = {}
        else:
            action_pred = outputs["action_pred"].reshape(-1, flow_dims)
            target = codec.normalize(action)  # flow head predicts normalized [-1, 1] actions
            if diffuse_gripper:
                grip = batch["gripper"].reshape(-1, 1).to(device=target.device, dtype=target.dtype)
                target = torch.cat((target, grip.mul(2.0).sub(1.0)), dim=-1)
            per_dim = (action_pred - target).pow(2).mean(dim=0)
            dim_losses = [per_dim[dim] for dim in range(pose_dims)]
            loss_action = per_dim.mean()
            pred_units = codec.unnormalize(action_pred[..., :pose_dims])
            tgt_units = codec.unnormalize(target[..., :pose_dims])
            abs_err = (pred_units - tgt_units).abs()
            metrics = {
                "action_mae": abs_err.mean(),
                "action_rmse": (pred_units - tgt_units).pow(2).mean().sqrt(),
            }
            for dim in range(pose_dims):
                metrics[f"action_dim_{dim}_mae"] = abs_err[:, dim].mean()
            if diffuse_gripper:
                gripper_prob = ((action_pred[:, pose_dims] + 1.0) * 0.5).clamp(0.0, 1.0)
                gripper_target = batch["gripper"].reshape(-1).to(
                    device=gripper_prob.device, dtype=gripper_prob.dtype)
                metrics["gripper_accuracy"] = ((gripper_prob >= 0.5) == (gripper_target >= 0.5)).float().mean()
                metrics["gripper_brier"] = (gripper_prob - gripper_target).pow(2).mean()
    elif action_mode == "regression":
        action_pred = outputs["action_pred"].reshape(-1, pose_dims)
        # When regression_normalize is on, train against per-dim [-1,1] targets (clamped to
        # bounds) so xyz/rotation share one scale; report metrics back in raw units.
        normalize = bool(cfg.get("action", {}).get("regression_normalize", False))
        target = codec.normalize(action) if normalize else action
        loss_type = str(weights.get("action_regression_loss", "smooth_l1"))
        if loss_type == "mse":
            per_dim = (action_pred - target).pow(2).mean(dim=0)
        else:
            beta = float(weights.get("smooth_l1_beta", 0.02))
            per_dim = F.smooth_l1_loss(action_pred, target, beta=beta, reduction="none").mean(dim=0)
        dim_losses = [per_dim[dim] for dim in range(pose_dims)]
        loss_action = torch.stack(dim_losses).mean()
        # deep supervision (videomt-style): each decoder layer also predicts the action; supervise all.
        aux_preds = outputs.get("aux_action_preds")
        if aux_preds:
            aux_w = float(weights.get("deep_supervision_weight", 1.0))
            aux_terms = []
            for aux in aux_preds:
                ap = aux.reshape(-1, pose_dims)
                if loss_type == "mse":
                    aux_terms.append((ap - target).pow(2).mean())
                else:
                    aux_terms.append(F.smooth_l1_loss(ap, target, beta=beta))
            loss_action = loss_action + aux_w * torch.stack(aux_terms).mean()
        pred_units = codec.unnormalize(action_pred) if normalize else action_pred
        tgt_units = codec.unnormalize(target) if normalize else target
        abs_err = (pred_units - tgt_units).abs()
        metrics = {
            "action_mae": abs_err.mean(),
            "action_rmse": (pred_units - tgt_units).pow(2).mean().sqrt(),
        }
        for dim in range(pose_dims):
            metrics[f"action_dim_{dim}_mae"] = abs_err[:, dim].mean()
    else:
        action_bins = codec.discretize(action)
        action_logits = outputs["action_logits"].reshape(-1, pose_dims, codec.num_bins)
        for dim in range(pose_dims):
            dim_losses.append(F.cross_entropy(action_logits[:, dim, :], action_bins[:, dim]))
        loss_action = torch.stack(dim_losses).mean()
        metrics = {}
    if diffuse_gripper:
        # This metric is already included in loss_action; do not add it again.
        if "pred_velocity" in outputs:
            loss_gripper = (pred_v[..., pose_dims] - tgt_v[..., pose_dims]).pow(2).mean()
        else:
            loss_gripper = (action_pred[..., pose_dims] - target[..., pose_dims]).pow(2).mean()
    else:
        loss_gripper = F.cross_entropy(outputs["gripper_logits"].reshape(-1, 2), batch["gripper"].reshape(-1).long())
    loss_terminate = F.cross_entropy(outputs["terminate_logits"].reshape(-1, 2), batch["terminate"].reshape(-1).long())
    total = (
        float(weights.get("action_weight", 1.0)) * loss_action
        + (0.0 if diffuse_gripper else float(weights.get("gripper_weight", 0.2))) * loss_gripper
        + float(weights.get("terminate_weight", 0.1)) * loss_terminate
    )
    losses = {"loss": total, "action_loss": loss_action, "gripper_loss": loss_gripper, "terminate_loss": loss_terminate, **metrics}
    for dim, dim_loss in enumerate(dim_losses):
        losses[f"action_dim_{dim}_loss"] = dim_loss
    # C15: auxiliary source-video abs-EE-pose loss
    traj_pred = outputs.get("traj_pred")
    traj_target = batch.get("traj_target")
    if traj_pred is not None and torch.is_tensor(traj_target):
        aux_traj_weight = float(weights.get("aux_traj_weight", 0.0))
        if aux_traj_weight > 0.0:
            tgt_9 = traj_target[..., :9].float().to(traj_pred.device)
            bl = torch.tensor(weights.get("aux_traj_bounds_low",  [-0.60, -0.60, 0.20, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]),
                              dtype=torch.float32, device=traj_pred.device)
            bh = torch.tensor(weights.get("aux_traj_bounds_high", [ 0.60,  0.60, 1.20,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0]),
                              dtype=torch.float32, device=traj_pred.device)
            tgt_norm = (2.0 * (tgt_9 - bl) / (bh - bl).clamp(min=1e-6) - 1.0).clamp(-1.0, 1.0)
            loss_aux = F.smooth_l1_loss(traj_pred, tgt_norm, beta=0.1)
            losses["loss"] = losses["loss"] + aux_traj_weight * loss_aux
            losses["aux_traj_loss"] = loss_aux
    # C18: auxiliary temporal-progress loss (demo-current alignment)
    progress_pred = outputs.get("progress_pred")
    progress_target = batch.get("progress_target")
    if progress_pred is not None and torch.is_tensor(progress_target):
        aux_progress_weight = float(weights.get("aux_progress_weight", 0.0))
        if aux_progress_weight > 0.0:
            pt = progress_target.float().to(progress_pred.device).reshape_as(progress_pred)
            loss_prog = F.smooth_l1_loss(progress_pred, pt, beta=0.1)
            losses["loss"] = losses["loss"] + aux_progress_weight * loss_prog
            losses["aux_progress_loss"] = loss_prog
    return losses
