from __future__ import annotations

from r2r_gen2act.modeling.flow_dit import FlowMatchingDiTHead
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
    if model_type == "fused_flow":
        return _build_fused_flow(cfg)
    if model_type == "query_flow":
        return _build_query_flow(cfg)
    if model_type == "query_reg":
        return _build_query_reg(cfg)
    if model_type == "fused_query_reg":
        return _build_fused_query_reg(cfg)
    if model_type == "fused_query_flow":
        return _build_fused_query_flow(cfg)
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
        unfrozen = vit.unfreeze_last_blocks(int(backbone_cfg.get("unfreeze_last_n_blocks", 0)))
        if unfrozen:
            print(f"[build_policy] backbone frozen except last {unfrozen} block(s) + final norm")
    dim = vit.hidden_dim
    latent_tokens = int(model_cfg.get("latent_tokens", 64))
    source_resampler = PerceiverResampler(dim, latent_tokens, int(model_cfg.get("resampler_layers", 2)), int(model_cfg.get("resampler_heads", 8)))
    target_resampler = PerceiverResampler(dim, latent_tokens, int(model_cfg.get("resampler_layers", 2)), int(model_cfg.get("resampler_heads", 8)))
    fusion = CrossAttentionFusion(dim, int(model_cfg.get("fusion_heads", 8)), int(model_cfg.get("fusion_layers", 1)))
    action_mode = str(cfg.get("action", {}).get("mode", model_cfg.get("action_mode", "classification")))
    pose_dims = int(model_cfg.get("pose_action_dims", 6))
    proprioception_dim = int(model_cfg.get("proprioception_dim", 0))
    if action_mode == "flow":
        # Flow-matching DiT head: cross-attends over the fused token sequence (no query decoder).
        fd = model_cfg.get("flow_dit", {}) or {}
        head = FlowMatchingDiTHead(
            cond_dim=dim,
            action_dim=pose_dims,
            horizon=int(cfg.get("action", {}).get("chunk_size", 1)),
            hidden_dim=int(fd.get("hidden_dim", 1024)),
            num_layers=int(fd.get("num_layers", 6)),
            heads=int(fd.get("heads", 16)),
            num_inference_steps=int(fd.get("num_inference_steps", 16)),
            dropout=float(fd.get("dropout", 0.0)),
            num_eval_samples=int(fd.get("num_eval_samples", 1)),
            time_sampling=str(fd.get("time_sampling", "beta")),
            noise_beta_alpha=float(fd.get("noise_beta_alpha", 1.5)),
            noise_beta_beta=float(fd.get("noise_beta_beta", 1.0)),
            noise_s=float(fd.get("noise_s", 0.999)),
            vl_mixer_layers=int(fd.get("vl_mixer_layers", 4)),
            interleave_self_attention=bool(fd.get("interleave_self_attention", True)),
        )
        n_params = sum(p.numel() for p in head.parameters())
        print(f"[build_policy] FlowMatchingDiTHead: {n_params/1e6:.1f}M params "
              f"(hidden={int(fd.get('hidden_dim', 1024))}, layers={int(fd.get('num_layers', 6))}, heads={int(fd.get('heads', 16))})")
        return Robot2RobotPolicy(vit, source_resampler, target_resampler, fusion, None, head, int(model_cfg["source_len"]), int(model_cfg["target_history_len"]), int(model_cfg["image_size"]), proprioception_dim, action_head_type="flow_dit")
    decoder = PolicyQueryDecoder(dim, int(model_cfg.get("fusion_heads", 8)), int(model_cfg.get("decoder_layers", 1)), int(model_cfg.get("num_queries", 1)))
    head = ActionHead(dim, pose_dims, int(model_cfg.get("num_bins", 256)), action_mode)
    return Robot2RobotPolicy(vit, source_resampler, target_resampler, fusion, decoder, head, int(model_cfg["source_len"]), int(model_cfg["target_history_len"]), int(model_cfg["image_size"]), proprioception_dim, action_head_type=action_mode)


def _build_fused_query_reg(cfg: dict):
    """Step 7 fused model: query-in-backbone readout (source video + current frame) + point latent +
    EE pos/depth -> self-attention mixer -> cross-attention regression decoder (deep supervision)."""
    from r2r_gen2act.modeling.cross_attn_reg import CrossAttnRegressionDecoder
    from r2r_gen2act.modeling.flow_dit import SelfAttentionMixer
    from r2r_gen2act.modeling.fused_query_reg_policy import FusedQueryRegPolicy
    from r2r_gen2act.modeling.point_latent import PointLatentEncoder, PointTrajSeqEncoder

    model_cfg = cfg["model"]
    backbone_cfg = model_cfg.get("backbone", {})
    vit = ViTBackbone(
        name=str(backbone_cfg.get("name", "dinov2_vitb14")),
        pretrained=bool(backbone_cfg.get("pretrained", True)),
        image_size=int(model_cfg["image_size"]),
        hidden_dim=int(model_cfg.get("hidden_dim", 768)),
        local_checkpoint=str(backbone_cfg.get("local_checkpoint", "") or ""),
        allow_random_init=bool(backbone_cfg.get("allow_random_init", False)),
    )
    dim = vit.hidden_dim
    pose_dims = int(model_cfg.get("pose_action_dims", 9))
    qr = model_cfg.get("query_readout", {}) or {}
    pt = model_cfg.get("point_tracking", {}) or {}
    fz = model_cfg.get("fused", {}) or {}
    # C0 (Step 7-C): sequence mode keeps the demo trajectory as per-timestep tokens (PointTrajSeqEncoder)
    # instead of pooling time into per-point tokens (PointLatentEncoder), so the model can attend to a
    # specific phase of the demonstrated motion.
    point_seq = bool(pt.get("sequence", False))
    # C4 (overlay-isolation): toggle off the demo-video readout and the global track tokens to test
    # whether the trajectory OVERLAY on the current frame alone carries the demo info.
    use_source_video = bool(model_cfg.get("use_source_video", True))
    use_global_track = bool(pt.get("use_global", True))
    # C1: optional SECOND causal point-track encoder (recent-motion momentum), built when the data
    # config sets point_tracking.causal_window. It runs alongside the global demo-trajectory encoder.
    data_pt = (cfg.get("data", {}).get("point_tracking", {}) or {})
    causal_on = point_seq and bool(data_pt.get("causal_window"))
    point_encoder = None
    point_encoder_causal = None
    if point_seq:
        if use_global_track:
            point_encoder = PointTrajSeqEncoder(
                num_points=int(pt.get("num_points", 10)), num_time=int(pt.get("num_time", 32)),
                out_dim=dim, hidden_dim=int(pt.get("hidden_dim", 384)),
            )
        if causal_on:
            point_encoder_causal = PointTrajSeqEncoder(
                num_points=int(pt.get("num_points", 10)),
                num_time=int(data_pt.get("causal_num_time", pt.get("num_time", 32))),
                out_dim=dim, hidden_dim=int(pt.get("hidden_dim", 384)),
            )
    elif use_global_track:
        point_encoder = PointLatentEncoder(
            num_points=int(pt.get("num_points", 10)), num_time=int(pt.get("num_time", 60)),
            hidden_dim=int(pt.get("hidden_dim", 384)), out_dim=dim,
            heads=int(pt.get("heads", 6)), attn_layers=int(pt.get("attn_layers", 2)),
        )
    mixer = SelfAttentionMixer(dim, int(fz.get("mixer_heads", 8)), int(fz.get("mixer_layers", 4)), float(fz.get("dropout", 0.1)))
    horizon = int(cfg.get("action", {}).get("chunk_size", 1))
    current_full_patch = bool(model_cfg.get("current_full_patch", False))
    joint_action = bool(model_cfg.get("joint_action", False))
    if joint_action:
        head = None   # action queries live inside the mixer; no separate cross-attn decoder
    else:
        cr = model_cfg.get("cross_attn_reg", {}) or {}
        head = CrossAttnRegressionDecoder(
            cond_dim=dim, pose_dims=pose_dims, horizon=horizon,
            hidden_dim=int(cr.get("hidden_dim", dim)), num_layers=int(cr.get("num_layers", 6)),
            heads=int(cr.get("heads", 8)), dropout=float(cr.get("dropout", 0.1)),
            deep_supervision=bool(cr.get("deep_supervision", True)),
        )
    depth_cfg = model_cfg.get("depth", {}) or {}
    aux_traj_cfg = model_cfg.get("aux_traj", {}) or {}
    model = FusedQueryRegPolicy(
        vit, head, point_encoder, mixer, int(model_cfg["source_len"]),
        num_queries=int(qr.get("num_queries", 32)), segmenter_start=int(qr.get("segmenter_start", 9)),
        ee_dim=int(model_cfg.get("proprioception_dim", 3)), ee_tokens=int(fz.get("ee_tokens", 8)),
        point_tokens=int(fz.get("point_tokens", 32)), heads=int(fz.get("mixer_heads", 8)),
        image_size=int(model_cfg["image_size"]), point_seq=point_seq,
        point_encoder_causal=point_encoder_causal,
        use_source_video=use_source_video, use_global_track=use_global_track,
        depth_cfg=depth_cfg,
        current_full_patch=current_full_patch,
        joint_action=joint_action, horizon=horizon, pose_dims=pose_dims,
        aux_traj_cfg=aux_traj_cfg,
    )
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[fused_query_reg] total={sum(p.numel() for p in model.parameters())/1e6:.1f}M trainable={n_tr/1e6:.1f}M "
          f"src_video={use_source_video} global_track={use_global_track} causal={causal_on} overlay-isolation={'YES' if not (use_source_video or use_global_track or causal_on) else 'no'}")
    return model


def _build_fused_query_flow(cfg: dict):
    """C2: C1's fused dual-track conditioning (query readout + global & causal PointTrajSeqEncoder +
    EE/progress) but with the flow-matching DiT head instead of regression. No SelfAttentionMixer — the
    flow head's internal VL mixer handles it."""
    from r2r_gen2act.modeling.fused_query_flow_policy import FusedQueryFlowPolicy
    from r2r_gen2act.modeling.point_latent import PointLatentEncoder, PointTrajSeqEncoder

    model_cfg = cfg["model"]
    backbone_cfg = model_cfg.get("backbone", {})
    vit = ViTBackbone(
        name=str(backbone_cfg.get("name", "dinov2_vitb14")),
        pretrained=bool(backbone_cfg.get("pretrained", True)),
        image_size=int(model_cfg["image_size"]),
        hidden_dim=int(model_cfg.get("hidden_dim", 768)),
        local_checkpoint=str(backbone_cfg.get("local_checkpoint", "") or ""),
        allow_random_init=bool(backbone_cfg.get("allow_random_init", False)),
    )
    dim = vit.hidden_dim
    pose_dims = int(model_cfg.get("pose_action_dims", 9))
    qr = model_cfg.get("query_readout", {}) or {}
    pt = model_cfg.get("point_tracking", {}) or {}
    fz = model_cfg.get("fused", {}) or {}
    point_seq = bool(pt.get("sequence", False))
    data_pt = (cfg.get("data", {}).get("point_tracking", {}) or {})
    causal_on = point_seq and bool(data_pt.get("causal_window"))
    # C7-flow: no point tracks at all (point_tracking disabled / use_global false) -> point_encoder=None
    use_global = bool(pt.get("use_global", True)) and bool(data_pt.get("enabled", True))
    point_encoder = None
    point_encoder_causal = None
    if point_seq and use_global:
        point_encoder = PointTrajSeqEncoder(
            num_points=int(pt.get("num_points", 10)), num_time=int(pt.get("num_time", 32)),
            out_dim=dim, hidden_dim=int(pt.get("hidden_dim", 384)),
        )
    if point_seq and causal_on:
        point_encoder_causal = PointTrajSeqEncoder(
            num_points=int(pt.get("num_points", 10)),
            num_time=int(data_pt.get("causal_num_time", pt.get("num_time", 32))),
            out_dim=dim, hidden_dim=int(pt.get("hidden_dim", 384)),
        )
    elif not point_seq and use_global:
        point_encoder = PointLatentEncoder(
            num_points=int(pt.get("num_points", 10)), num_time=int(pt.get("num_time", 60)),
            hidden_dim=int(pt.get("hidden_dim", 384)), out_dim=dim,
            heads=int(pt.get("heads", 6)), attn_layers=int(pt.get("attn_layers", 2)),
        )
    fd = model_cfg.get("flow_dit", {}) or {}
    head = FlowMatchingDiTHead(
        cond_dim=dim, action_dim=pose_dims, horizon=int(cfg.get("action", {}).get("chunk_size", 1)),
        hidden_dim=int(fd.get("hidden_dim", 1024)), num_layers=int(fd.get("num_layers", 8)),
        heads=int(fd.get("heads", 16)), num_inference_steps=int(fd.get("num_inference_steps", 16)),
        dropout=float(fd.get("dropout", 0.1)), num_eval_samples=int(fd.get("num_eval_samples", 1)),
        time_sampling=str(fd.get("time_sampling", "beta")),
        noise_beta_alpha=float(fd.get("noise_beta_alpha", 1.5)),
        noise_beta_beta=float(fd.get("noise_beta_beta", 1.0)), noise_s=float(fd.get("noise_s", 0.999)),
        vl_mixer_layers=int(fd.get("vl_mixer_layers", 4)),
        interleave_self_attention=bool(fd.get("interleave_self_attention", True)),
    )
    model = FusedQueryFlowPolicy(
        vit, head, point_encoder, int(model_cfg["source_len"]),
        num_queries=int(qr.get("num_queries", 32)), segmenter_start=int(qr.get("segmenter_start", 9)),
        ee_dim=int(model_cfg.get("proprioception_dim", 4)), ee_tokens=int(fz.get("ee_tokens", 8)),
        image_size=int(model_cfg["image_size"]), point_encoder_causal=point_encoder_causal,
        aux_traj_cfg=model_cfg.get("aux_traj", {}) or {},
        aux_progress_cfg=model_cfg.get("aux_progress", {}) or {},
        dt_time_cfg=model_cfg.get("dt_time_embed", {}) or {},
        current_full_patch=bool(model_cfg.get("current_full_patch", False)),
        max_source_len=int(model_cfg.get("max_source_len", model_cfg["source_len"])),
        pad_source=bool(model_cfg.get("pad_source", False)),
        wrist_current_enabled=bool((model_cfg.get("wrist_current", {}) or {}).get("enabled", False)),
        separate_stream_queries=bool(qr.get("separate_streams", False)),
        front_depth_cfg=model_cfg.get("front_depth", {}) or {},
        current_history_len=len(cfg.get("data", {}).get("current_history_offsets", [0])),
    )
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[fused_query_flow] total={sum(p.numel() for p in model.parameters())/1e6:.1f}M trainable={n_tr/1e6:.1f}M "
          f"head={sum(p.numel() for p in head.parameters())/1e6:.1f}M point_seq={point_seq} causal={causal_on} "
          f"num_eval_samples={int(fd.get('num_eval_samples', 1))}")
    return model


def _build_query_reg(cfg: dict):
    """Query-in-backbone readout + cross-attention regression decoder (deep supervision). Ablation:
    same cross-attention transmission as query_flow, but regression loss instead of flow."""
    from r2r_gen2act.modeling.cross_attn_reg import CrossAttnRegressionDecoder
    from r2r_gen2act.modeling.query_reg_policy import QueryRegPolicy

    model_cfg = cfg["model"]
    backbone_cfg = model_cfg.get("backbone", {})
    vit = ViTBackbone(
        name=str(backbone_cfg.get("name", "dinov2_vitb14")),
        pretrained=bool(backbone_cfg.get("pretrained", True)),
        image_size=int(model_cfg["image_size"]),
        hidden_dim=int(model_cfg.get("hidden_dim", 768)),
        local_checkpoint=str(backbone_cfg.get("local_checkpoint", "") or ""),
        allow_random_init=bool(backbone_cfg.get("allow_random_init", False)),
    )
    dim = vit.hidden_dim
    pose_dims = int(model_cfg.get("pose_action_dims", 9))
    cr = model_cfg.get("cross_attn_reg", {}) or {}
    head = CrossAttnRegressionDecoder(
        cond_dim=dim, pose_dims=pose_dims, horizon=int(cfg.get("action", {}).get("chunk_size", 1)),
        hidden_dim=int(cr.get("hidden_dim", dim)), num_layers=int(cr.get("num_layers", 6)),
        heads=int(cr.get("heads", 8)), dropout=float(cr.get("dropout", 0.1)),
        deep_supervision=bool(cr.get("deep_supervision", True)),
    )
    qr = model_cfg.get("query_readout", {}) or {}
    model = QueryRegPolicy(
        vit, head, int(model_cfg["source_len"]), int(model_cfg["target_history_len"]),
        num_queries=int(qr.get("num_queries", 16)), segmenter_start=int(qr.get("segmenter_start", 9)),
        image_size=int(model_cfg["image_size"]), single_stream=bool(qr.get("single_stream", True)),
    )
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[query_reg] total={sum(p.numel() for p in model.parameters())/1e6:.1f}M trainable={n_tr/1e6:.1f}M "
          f"decoder_layers={int(cr.get('num_layers', 6))} deep_supervision={bool(cr.get('deep_supervision', True))}")
    return model


def _build_query_flow(cfg: dict):
    """Query-in-backbone readout (VidEoMT-style) + IDM flow head. Replaces the resampler/fusion
    bottleneck: learnable queries read DINOv2 patches via the pretrained last blocks."""
    from r2r_gen2act.modeling.query_flow_policy import QueryFlowPolicy

    model_cfg = cfg["model"]
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
        unfrozen = vit.unfreeze_last_blocks(int(backbone_cfg.get("unfreeze_last_n_blocks", 0)))
        if unfrozen:
            print(f"[query_flow] DINOv2 frozen except last {unfrozen} block(s)")
    dim = vit.hidden_dim
    pose_dims = int(model_cfg.get("pose_action_dims", 9))
    fd = model_cfg.get("flow_dit", {}) or {}
    head = FlowMatchingDiTHead(
        cond_dim=dim, action_dim=pose_dims, horizon=int(cfg.get("action", {}).get("chunk_size", 1)),
        hidden_dim=int(fd.get("hidden_dim", 1024)), num_layers=int(fd.get("num_layers", 8)),
        heads=int(fd.get("heads", 16)), num_inference_steps=int(fd.get("num_inference_steps", 16)),
        dropout=float(fd.get("dropout", 0.1)), num_eval_samples=int(fd.get("num_eval_samples", 1)),
        time_sampling=str(fd.get("time_sampling", "beta")),
        noise_beta_alpha=float(fd.get("noise_beta_alpha", 1.5)),
        noise_beta_beta=float(fd.get("noise_beta_beta", 1.0)), noise_s=float(fd.get("noise_s", 0.999)),
        vl_mixer_layers=int(fd.get("vl_mixer_layers", 4)),
        interleave_self_attention=bool(fd.get("interleave_self_attention", True)),
    )
    qr = model_cfg.get("query_readout", {}) or {}
    model = QueryFlowPolicy(
        vit, head, int(model_cfg["source_len"]), int(model_cfg["target_history_len"]),
        num_queries=int(qr.get("num_queries", 16)), segmenter_start=int(qr.get("segmenter_start", 9)),
        image_size=int(model_cfg["image_size"]), proprioception_dim=int(model_cfg.get("proprioception_dim", 0)),
        single_stream=bool(qr.get("single_stream", False)),
    )
    n_tot = sum(p.numel() for p in model.parameters())
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[query_flow] total={n_tot/1e6:.1f}M trainable={n_tr/1e6:.1f}M head={sum(p.numel() for p in head.parameters())/1e6:.1f}M "
          f"num_queries={int(qr.get('num_queries', 16))} segmenter_start={int(qr.get('segmenter_start', 9))}")
    return model


def _build_fused_flow(cfg: dict):
    """Step 7 fused-conditioning flow policy (video + image + point + ee -> flow DiT)."""
    from r2r_gen2act.modeling.fused_policy import FusedFlowPolicy
    from r2r_gen2act.modeling.point_latent import PointLatentEncoder
    from r2r_gen2act.modeling.video_encoder import VideoMAEv2Encoder

    model_cfg = cfg["model"]
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
        unfrozen = vit.unfreeze_last_blocks(int(backbone_cfg.get("unfreeze_last_n_blocks", 0)))
        if unfrozen:
            print(f"[fused_flow] DINOv2 frozen except last {unfrozen} block(s)")
    dim = vit.hidden_dim
    image_resampler = PerceiverResampler(dim, int(model_cfg.get("latent_tokens", 64)), int(model_cfg.get("resampler_layers", 3)), int(model_cfg.get("resampler_heads", 8)))

    ve_cfg = model_cfg.get("video_encoder", {}) or {}
    video_encoder = VideoMAEv2Encoder(
        checkpoint=str(ve_cfg.get("checkpoint", "")),
        all_frames=int(ve_cfg.get("all_frames", 16)),
        tubelet_size=int(ve_cfg.get("tubelet_size", 2)),
        freeze=bool(ve_cfg.get("freeze", True)),
    )
    pt_cfg = model_cfg.get("point_tracking", {}) or {}
    point_encoder = PointLatentEncoder(
        num_points=int(pt_cfg.get("num_points", 10)),
        num_time=int(pt_cfg.get("num_time", 60)),
        hidden_dim=int(pt_cfg.get("hidden_dim", 384)),
        out_dim=dim,
        heads=int(pt_cfg.get("heads", 6)),
        attn_layers=int(pt_cfg.get("attn_layers", 2)),
    )
    pose_dims = int(model_cfg.get("pose_action_dims", 9))
    fd = model_cfg.get("flow_dit", {}) or {}
    head = FlowMatchingDiTHead(
        cond_dim=dim, action_dim=pose_dims, horizon=int(cfg.get("action", {}).get("chunk_size", 1)),
        hidden_dim=int(fd.get("hidden_dim", 1024)), num_layers=int(fd.get("num_layers", 6)),
        heads=int(fd.get("heads", 16)), num_inference_steps=int(fd.get("num_inference_steps", 16)),
        dropout=float(fd.get("dropout", 0.0)), num_eval_samples=int(fd.get("num_eval_samples", 1)),
        time_sampling=str(fd.get("time_sampling", "beta")),
        noise_beta_alpha=float(fd.get("noise_beta_alpha", 1.5)),
        noise_beta_beta=float(fd.get("noise_beta_beta", 1.0)), noise_s=float(fd.get("noise_s", 0.999)),
        vl_mixer_layers=int(fd.get("vl_mixer_layers", 4)),
        interleave_self_attention=bool(fd.get("interleave_self_attention", True)),
    )
    model = FusedFlowPolicy(
        vit, video_encoder, point_encoder, head, image_resampler, dim=dim,
        num_video_tokens=int(ve_cfg.get("num_video_tokens", 4)),
        ee_dim=int(model_cfg.get("proprioception_dim", 2)), image_size=int(model_cfg["image_size"]),
    )
    n_tot = sum(p.numel() for p in model.parameters())
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[fused_flow] total={n_tot/1e6:.1f}M trainable={n_tr/1e6:.1f}M head={sum(p.numel() for p in head.parameters())/1e6:.1f}M")
    return model
