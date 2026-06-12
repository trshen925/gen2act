from __future__ import annotations

from r2r_gen2act.modeling.fusion import CrossAttentionFusion, PolicyQueryDecoder
from r2r_gen2act.modeling.heads import ActionHead
from r2r_gen2act.modeling.policy import Robot2RobotPolicy
from r2r_gen2act.modeling.proprioception import ProprioceptionOnlyPolicy
from r2r_gen2act.modeling.resampler import PerceiverResampler
from r2r_gen2act.modeling.vit import ViTBackbone


def build_policy(cfg: dict) -> Robot2RobotPolicy | ProprioceptionOnlyPolicy:
    model_cfg = cfg["model"]
    action_mode = str(cfg.get("action", {}).get("mode", model_cfg.get("action_mode", "classification")))
    model_type = str(model_cfg.get("type", "video_policy"))
    if model_type == "proprioception_only":
        proprioception_dim = int(model_cfg.get("proprioception_dim", 0))
        if proprioception_dim <= 0:
            raise ValueError("model.proprioception_dim must be positive for proprioception_only model")
        hidden_dim = int(model_cfg.get("hidden_dim", 256))
        head = ActionHead(hidden_dim, int(model_cfg.get("pose_action_dims", 6)), int(model_cfg.get("num_bins", 256)), action_mode)
        return ProprioceptionOnlyPolicy(proprioception_dim, hidden_dim, head, int(model_cfg.get("mlp_layers", 2)))
    backbone_cfg = model_cfg.get("backbone", {})
    vit = ViTBackbone(
        name=str(backbone_cfg.get("name", "dinov2_vitb14")),
        pretrained=bool(backbone_cfg.get("pretrained", True)),
        image_size=int(model_cfg["image_size"]),
        hidden_dim=int(model_cfg.get("hidden_dim", 768)),
        local_checkpoint=str(backbone_cfg.get("local_checkpoint", "") or ""),
        allow_random_init=bool(backbone_cfg.get("allow_random_init", False)),
    )
    if bool(backbone_cfg.get("freeze", False)):
        for p in vit.parameters():
            p.requires_grad_(False)
    dim = vit.hidden_dim
    latent_tokens = int(model_cfg.get("latent_tokens", 64))
    source_resampler = PerceiverResampler(dim, latent_tokens, int(model_cfg.get("resampler_layers", 2)), int(model_cfg.get("resampler_heads", 8)))
    target_resampler = PerceiverResampler(dim, latent_tokens, int(model_cfg.get("resampler_layers", 2)), int(model_cfg.get("resampler_heads", 8)))
    fusion = CrossAttentionFusion(dim, int(model_cfg.get("fusion_heads", 8)))
    decoder = PolicyQueryDecoder(dim, int(model_cfg.get("fusion_heads", 8)))
    action_mode = str(cfg.get("action", {}).get("mode", model_cfg.get("action_mode", "classification")))
    head = ActionHead(dim, int(model_cfg.get("pose_action_dims", 6)), int(model_cfg.get("num_bins", 256)), action_mode)
    proprioception_dim = int(model_cfg.get("proprioception_dim", 0))
    return Robot2RobotPolicy(vit, source_resampler, target_resampler, fusion, decoder, head, int(model_cfg["source_len"]), int(model_cfg["target_history_len"]), int(model_cfg["image_size"]), proprioception_dim)
