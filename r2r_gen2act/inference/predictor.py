from __future__ import annotations

import json
from pathlib import Path

import torch

from r2r_gen2act.data.factories import build_action_codec, build_dataset
from r2r_gen2act.modeling.factory import build_policy
from r2r_gen2act.training.checkpoint import load_checkpoint


class PolicyPredictor:
    def __init__(self, cfg: dict, checkpoint_path: str | Path, device: str | None = None, strict: bool = True) -> None:
        self.cfg = cfg
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.codec = build_action_codec(cfg)
        self.model = build_policy(cfg).to(self.device)
        self.checkpoint = load_checkpoint(checkpoint_path, self.model, self.device, strict=strict)
        self.model.eval()

    @torch.no_grad()
    def predict_batch(self, batch: dict) -> dict:
        source = batch["source_video"].to(self.device)
        target = batch["target_history"].to(self.device)
        if source.dim() == 4:
            source = source.unsqueeze(0)
            target = target.unsqueeze(0)
        proprioception = batch.get("proprioception")
        if torch.is_tensor(proprioception):
            proprioception = proprioception.to(self.device)
            if proprioception.dim() == 1:
                proprioception = proprioception.unsqueeze(0)
        point_track = batch.get("point_track")
        if torch.is_tensor(point_track):
            point_track = point_track.to(self.device)
            if point_track.dim() == 3:
                point_track = point_track.unsqueeze(0)
        extra = {}
        ptc = batch.get("point_track_causal")
        if torch.is_tensor(ptc):
            ptc = ptc.to(self.device)
            if ptc.dim() == 3:
                ptc = ptc.unsqueeze(0)
            extra["point_track_causal"] = ptc
        source_dt = batch.get("source_dt")
        if torch.is_tensor(source_dt):
            source_dt = source_dt.to(self.device)
            if source_dt.dim() == 1:
                source_dt = source_dt.unsqueeze(0)
            extra["source_dt"] = source_dt
        wrist = batch.get("wrist_current")
        if torch.is_tensor(wrist):
            wrist = wrist.to(self.device)
            if wrist.dim() == 3:
                wrist = wrist.unsqueeze(0)
            extra["wrist_current"] = wrist
        front_geometry = batch.get("front_geometry")
        if torch.is_tensor(front_geometry):
            front_geometry = front_geometry.to(self.device)
            if front_geometry.dim() == 3:
                front_geometry = front_geometry.unsqueeze(0)
            extra["front_geometry"] = front_geometry
        outputs = self.model(source, target, proprioception, None, point_track, **extra)
        if "action_pred" in outputs:
            pred = outputs["action_pred"]
            action_mode = str(self.cfg.get("action", {}).get("mode", ""))
            # flow head emits normalized [-1,1] actions; regression may too (regression_normalize).
            if action_mode == "flow" or bool(self.cfg.get("action", {}).get("regression_normalize", False)):
                pred = self.codec.unnormalize(pred)
            pose = pred.cpu()
            bins = None
        else:
            bins = outputs["action_logits"].argmax(dim=-1)
            pose = self.codec.decode(bins).cpu()
        gripper_prob = outputs["gripper_logits"].softmax(dim=-1).cpu()
        terminate_prob = outputs["terminate_logits"].softmax(dim=-1).cpu()
        result = {"pose_action": pose, "gripper_prob": gripper_prob, "terminate_prob": terminate_prob}
        if bins is not None:
            result["action_bins"] = bins.cpu()
        # 6D rotation rep: orthonormalize the predicted [3:9] into a valid rotation matrix.
        if "pose6d" in str(self.cfg.get("action", {}).get("mapping", {}).get("type", "")) and pose.shape[-1] >= 9:
            from r2r_gen2act.data.action.rotation import sixd_to_matrix
            result["rotation_matrix"] = sixd_to_matrix(pose[..., 3:9]).cpu()
            result["pose_xyz"] = pose[..., :3]
        return result


def predict_dataset_window(cfg: dict, checkpoint_path: str | Path, split: str, episode_id: str | None, start_index: int, save_path: str | Path | None = None, device: str | None = None, strict: bool = True) -> dict:
    dataset = build_dataset(cfg, split)
    if episode_id:
        sample = dataset.sample_window(episode_id, start_index)
    else:
        sample = dataset[0]
    predictor = PolicyPredictor(cfg, checkpoint_path, device=device, strict=strict)
    pred = predictor.predict_batch(sample)
    result = {
        "episode_id": sample["episode_id"],
        "start_index": int(sample["start_index"]),
        "target_step": int(sample["target_step"]),
        "pose_action": pred["pose_action"][0].tolist(),
        "gripper_prob": pred["gripper_prob"][0].tolist(),
        "terminate_prob": pred["terminate_prob"][0].tolist(),
        "checkpoint": str(checkpoint_path),
    }
    if "action_bins" in pred:
        result["action_bins"] = pred["action_bins"][0].tolist()
    if "rotation_matrix" in pred:
        result["rotation_matrix"] = pred["rotation_matrix"][0].tolist()
    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
