from __future__ import annotations


def validate_config(cfg: dict) -> None:
    data = cfg["data"]
    model = cfg["model"]
    action = cfg["action"]
    if int(data["source_len"]) <= 0 or int(data["target_history_len"]) <= 0:
        raise ValueError("source_len and target_history_len must be positive")
    if int(data["image_size"]) != int(model["image_size"]):
        raise ValueError("data.image_size must match model.image_size")
    if int(action["pose_dims"]) != int(model["pose_action_dims"]):
        raise ValueError("action.pose_dims must match model.pose_action_dims")
    action_mode = str(action.get("mode", "classification"))
    if action_mode not in ("classification", "regression"):
        raise ValueError("action.mode must be classification or regression")
    if action_mode == "classification" and int(action["num_bins"]) != int(model["num_bins"]):
        raise ValueError("action.num_bins must match model.num_bins")
    if int(model["latent_tokens"]) <= 0:
        raise ValueError("model.latent_tokens must be positive")
    prop_cfg = data.get("proprioception", {})
    prop_enabled = bool(prop_cfg.get("enabled", False))
    prop_dim = int(model.get("proprioception_dim", 0))
    if prop_enabled and prop_dim <= 0:
        raise ValueError("model.proprioception_dim must be positive when data.proprioception.enabled is true")
    if not prop_enabled and prop_dim != 0:
        raise ValueError("model.proprioception_dim must be 0 when data.proprioception.enabled is false")
    if prop_enabled and int(prop_cfg.get("dims", prop_dim)) != prop_dim:
        raise ValueError("data.proprioception.dims must match model.proprioception_dim")
