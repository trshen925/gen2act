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
    if action_mode not in ("classification", "regression", "flow"):
        raise ValueError("action.mode must be classification, regression, or flow")
    if action_mode == "classification" and int(action["num_bins"]) != int(model["num_bins"]):
        raise ValueError("action.num_bins must match model.num_bins")
    if int(model["latent_tokens"]) <= 0:
        raise ValueError("model.latent_tokens must be positive")
    chunk_size = int(action.get("chunk_size", 1))
    num_queries = int(model.get("num_queries", 1))
    if chunk_size <= 0:
        raise ValueError("action.chunk_size must be positive")
    # flow_dit predicts the whole chunk as a token sequence (horizon=chunk_size), so it has no
    # per-step query decoder; the chunk==num_queries constraint only applies to the query heads.
    if action_mode != "flow" and chunk_size != num_queries:
        raise ValueError(f"action.chunk_size ({chunk_size}) must match model.num_queries ({num_queries})")
    prop_cfg = data.get("proprioception", {})
    prop_enabled = bool(prop_cfg.get("enabled", False))
    prop_dim = int(model.get("proprioception_dim", 0))
    if prop_enabled and prop_dim <= 0:
        raise ValueError("model.proprioception_dim must be positive when data.proprioception.enabled is true")
    if not prop_enabled and prop_dim != 0:
        raise ValueError("model.proprioception_dim must be 0 when data.proprioception.enabled is false")
    if prop_enabled:
        # Model receives proprioception.dims values plus optional scalar state cues.
        expected = int(prop_cfg.get("dims", prop_dim))
        if bool(prop_cfg.get("append_progress", False)):
            expected += 1
        if bool(prop_cfg.get("append_current_gripper", False)):
            expected += 1
        if expected != prop_dim:
            raise ValueError(
                "data.proprioception.dims (+ optional progress/current-gripper dims) "
                "must match model.proprioception_dim")

    backbone = model.get("backbone", {}) or {}
    backbone_name = str(backbone.get("name", "dinov2_vitb14")).lower()
    wan_names = {"wan_vae", "wan2.1_vae", "wan2_1_vae", "wan_video_vae"}
    source_micro_clip = data.get("source_micro_clip", {}) or {}
    source_micro_clip_enabled = bool(source_micro_clip.get("enabled", False))
    if source_micro_clip_enabled:
        frames = int(source_micro_clip.get("frames", 1))
        stride = int(source_micro_clip.get("stride", 1))
        alignment = str(source_micro_clip.get("alignment", "causal"))
        if backbone_name not in wan_names:
            raise ValueError("data.source_micro_clip currently requires a Wan-VAE backbone")
        if frames <= 1 or (frames - 1) % 4:
            raise ValueError(
                "Wan source_micro_clip.frames must be a legal temporal length 1+4n and >1")
        if stride <= 0:
            raise ValueError("data.source_micro_clip.stride must be positive")
        if alignment != "causal":
            raise ValueError("data.source_micro_clip.alignment currently supports only causal")
    if backbone_name in wan_names:
        if str(model.get("type", "video_policy")) != "fused_query_flow":
            raise ValueError("Wan-VAE requires model.type=fused_query_flow")
        if int(model["image_size"]) % 8:
            raise ValueError("Wan-VAE requires model.image_size divisible by 8")
        if not bool(backbone.get("pretrained", True)):
            raise ValueError("Wan-VAE must use pretrained weights")
        if not bool(backbone.get("freeze", True)):
            raise ValueError("Wan-VAE is a frozen encoder; set model.backbone.freeze=true")
        if int(backbone.get("latent_dim", 16)) != 16:
            raise ValueError("Wan 2.1 VAE checkpoints require model.backbone.latent_dim=16")
        backend = str(backbone.get("backend", "official")).lower()
        fallback_backends = backbone.get("fallback_backends", [])
        if isinstance(fallback_backends, str):
            fallback_backends = [fallback_backends]
        known_backends = {"official", "diffusers", "diffsynth"}
        if backend not in known_backends or any(str(name).lower() not in known_backends for name in fallback_backends):
            raise ValueError("Wan-VAE backend/fallback_backends must use official, diffusers, or diffsynth")
        if backend in {str(name).lower() for name in fallback_backends}:
            raise ValueError("model.backbone.fallback_backends must not repeat model.backbone.backend")
        if "diffusers" in (backend, *(str(name).lower() for name in fallback_backends)) and not str(
            backbone.get("diffusers_model_id", "") or ""
        ):
            raise ValueError(
                "Diffusers Wan-VAE backend/fallback requires model.backbone.diffusers_model_id"
            )
        if str(backbone.get("dtype", "bfloat16")).lower() not in {
            "float32", "fp32", "bfloat16", "bf16", "float16", "fp16"
        }:
            raise ValueError("Unsupported model.backbone.dtype for Wan-VAE")
        if bool(model.get("current_full_patch", False)):
            raise ValueError("model.current_full_patch is DINO-only and incompatible with Wan-VAE")
        if bool((model.get("front_depth", {}) or {}).get("enabled", False)):
            raise ValueError("DINO patch geometry is incompatible with Wan-VAE")
        readout = model.get("query_readout", {}) or {}
        heads = int(readout.get("heads", 8))
        hidden_dim = int(model.get("hidden_dim", 768))
        if heads <= 0 or hidden_dim % heads:
            raise ValueError("model.query_readout.heads must divide model.hidden_dim for Wan-VAE")
