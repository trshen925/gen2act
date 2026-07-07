"""Step 7 — fused multi-condition model with the reliable cross-attention regression architecture.

Conditions (all encoded to the SAME hidden width and comparable token counts for alignment):
  - source/reference video (linspace frames of the demo)  -> query-in-backbone DINOv2 readout
  - current frame (target_step)                           -> query-in-backbone DINOv2 readout
                                                             OR all 256 patch tokens (current_full_patch=True)
  - tracked EE-neighborhood points                        -> point encoder + summary queries
  - current EE image position + depth (u,v,z)             -> MLP -> a few tokens
All condition tokens (each tagged with a learnable type embedding) are processed by a SELF-ATTENTION
mixer, then:
  - (default) action queries CROSS-ATTEND to the mixed conditions (CrossAttnRegressionDecoder) with
    DEEP SUPERVISION; or
  - (joint_action=True) learnable action queries are ADDED TO THE MIXER INPUT and read out directly
    from mixer output via a simple linear head (no separate decoder, no deep supervision).
No flow head, no VideoMAEv2. Future info enters ONLY via the source video.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from r2r_gen2act.modeling.cross_attn_reg import CrossAttnRegressionDecoder
from r2r_gen2act.modeling.depth_lifting import DepthTo3DPatchPositions
from r2r_gen2act.modeling.flow_dit import FeedForward, SelfAttentionMixer, _MHA
from r2r_gen2act.modeling.point_latent import PointLatentEncoder
from r2r_gen2act.modeling.vit import ViTBackbone


class FusedQueryRegPolicy(nn.Module):
    def __init__(self, vit: ViTBackbone, head, point_encoder: PointLatentEncoder,
                 mixer: SelfAttentionMixer, source_len: int, num_queries: int = 32, segmenter_start: int = 9,
                 ee_dim: int = 3, ee_tokens: int = 8, point_tokens: int = 32, heads: int = 8,
                 image_size: int = 224, point_seq: bool = False, point_encoder_causal=None,
                 use_source_video: bool = True, use_global_track: bool = True,
                 depth_cfg: dict | None = None,
                 current_full_patch: bool = False,
                 joint_action: bool = False, horizon: int = 8, pose_dims: int = 9,
                 aux_traj_cfg: dict | None = None) -> None:
        super().__init__()
        self.vit = vit
        self.head = head
        self.point_encoder = point_encoder
        # C1: optional second encoder for the causal recent-motion track (own token group + type embed).
        self.point_encoder_causal = point_encoder_causal
        # C4 (overlay-isolation): toggle whole condition streams off to test the trajectory OVERLAY alone.
        self.use_source_video = bool(use_source_video)       # ① demo video readout
        self.use_global_track = bool(use_global_track) and point_encoder is not None  # ③ global track tokens
        self.mixer = mixer
        self.action_head_type = "regression"
        self.image_size = image_size
        self.num_queries = int(num_queries)
        # C0 (Step 7-C): when point_seq, point_encoder already emits a per-TIMESTEP token sequence
        # [B, S, dim] (PointTrajSeqEncoder); use those tokens directly. Otherwise point_encoder emits
        # per-POINT tokens [B, N, dim] (PointLatentEncoder) summarized to point_tokens via attention.
        self.point_seq = bool(point_seq)
        dim = vit.hidden_dim
        n_blocks = len(self.vit.backend.blocks)
        self.segmenter_start = int(segmenter_start) % n_blocks if segmenter_start >= 0 else n_blocks + int(segmenter_start)
        # query-in-backbone readout queries (shared by source video + current frame)
        self.query = nn.Parameter(torch.randn(self.num_queries, dim) / dim**0.5)
        self.source_time_embed = nn.Parameter(torch.randn(source_len, 1, dim) / dim**0.5)
        # learnable type embeddings so the mixer/decoder can tell the conditions apart
        self.type_source = nn.Parameter(torch.randn(1, 1, dim) / dim**0.5)
        self.type_current = nn.Parameter(torch.randn(1, 1, dim) / dim**0.5)
        self.type_point = nn.Parameter(torch.randn(1, 1, dim) / dim**0.5)
        self.type_ee = nn.Parameter(torch.randn(1, 1, dim) / dim**0.5)
        if point_encoder_causal is not None:
            self.type_point_causal = nn.Parameter(torch.randn(1, 1, dim) / dim**0.5)
        # point summary: project the per-point tokens to `point_tokens` aligned tokens
        # (only needed in pooled mode; in seq mode the encoder's per-timestep tokens are used as-is).
        self.point_tokens = int(point_tokens)
        if not self.point_seq:
            self.point_summary_q = nn.Parameter(torch.randn(1, point_tokens, dim) / dim**0.5)
            self.point_summary_attn = _MHA(dim, heads)
            self.point_summary_norm = nn.LayerNorm(dim)
        # EE position+depth -> embedding -> ee_tokens learnable tokens carrying it
        self.ee_tokens = int(ee_tokens)
        self.ee_mlp = nn.Sequential(nn.LayerNorm(ee_dim), nn.Linear(ee_dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.ee_query = nn.Parameter(torch.randn(1, ee_tokens, dim) / dim**0.5)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("image_mean", mean, persistent=False)
        self.register_buffer("image_std", std, persistent=False)
        # ── depth 3D lifting (3D Diffuser Actor approach) ─────────────────────
        # For each frame, compute per-patch 3D camera-frame positions from depth +
        # camera intrinsics, encode to dim-dim tokens, pool to n_depth_tok tokens,
        # and add as an extra condition group alongside the visual readout tokens.
        depth_cfg = depth_cfg or {}
        self.depth_enabled = bool(depth_cfg.get("enabled", False))
        if self.depth_enabled:
            n_depth_tok = int(depth_cfg.get("num_tokens", 32))
            self.n_depth_tok = n_depth_tok
            # pure geometry: no trainable params in the lifter
            self.depth_lifter = DepthTo3DPatchPositions(
                patch_size=14, image_size=image_size,
                depth_scale=float(depth_cfg.get("depth_scale_mm", 1.0)),
                max_depth_m=float(depth_cfg.get("max_depth_m", 10.0)),
            )
            # 3D position (3) → dim embedding
            self.depth_3d_enc = nn.Sequential(
                nn.LayerNorm(3), nn.Linear(3, dim), nn.GELU(), nn.Linear(dim, dim)
            )
            # pool 256 patch-position tokens → n_depth_tok aligned tokens per frame
            self.depth_pool_q    = nn.Parameter(torch.randn(1, n_depth_tok, dim) / dim ** 0.5)
            self.depth_pool_attn = _MHA(dim, heads)
            self.depth_pool_norm = nn.LayerNorm(dim)
            self.type_depth      = nn.Parameter(torch.randn(1, 1, dim) / dim ** 0.5)
        # ── C12: current-obs full-patch + joint-mixer action queries ──────────
        # current_full_patch: bypass readout for current frame, use all 256 DINOv2 patch tokens.
        # joint_action: add learnable action queries into the mixer (no separate cross-attn decoder).
        self.source_len = int(source_len)
        self.current_full_patch = bool(current_full_patch)
        self.joint_action = bool(joint_action)
        if self.joint_action:
            self.horizon = int(horizon)
            self.act_query = nn.Parameter(torch.randn(1, horizon, dim) / dim ** 0.5)
            self.type_action = nn.Parameter(torch.randn(1, 1, dim) / dim ** 0.5)
            self.act_out_norm = nn.LayerNorm(dim)
            self.act_pose_head = nn.Linear(dim, int(pose_dims))
            self.act_gripper_head = nn.Linear(dim, 2)
            self.act_terminate_head = nn.Linear(dim, 2)
        # C15: auxiliary source-video abs-EE-pose head (regularizer for underdetermined delta BC)
        aux_traj_cfg = aux_traj_cfg or {}
        self.aux_traj_enabled = bool(aux_traj_cfg.get("enabled", False))
        if self.aux_traj_enabled:
            self.aux_traj_norm = nn.LayerNorm(dim)
            self.aux_traj_head = nn.Linear(dim, 9)

    def _readout(self, video: torch.Tensor, time_embed: torch.Tensor | None, stream: torch.Tensor) -> torch.Tensor:
        """video [B,T,3,H,W] -> [B, T*num_queries, dim] via query-in-backbone."""
        b, t, c, h, w = video.shape
        x = ((video - self.image_mean) / self.image_std).reshape(b * t, c, h, w)
        be = self.vit.backend
        x = be.patch_embed(x)
        x = be._pos_embed(x)
        if hasattr(be, "patch_drop"):
            x = be.patch_drop(x)
        if hasattr(be, "norm_pre"):
            x = be.norm_pre(x)
        for blk in be.blocks[: self.segmenter_start]:
            x = blk(x)
        q = self.query[None].expand(b * t, -1, -1)
        x = torch.cat([q, x], dim=1)
        for blk in be.blocks[self.segmenter_start :]:
            x = blk(x)
        q_out = be.norm(x[:, : self.num_queries, :]).reshape(b, t, self.num_queries, -1)
        if time_embed is not None:
            q_out = q_out + time_embed[:t].unsqueeze(0)
        q_out = q_out + stream.unsqueeze(0)
        return q_out.reshape(b, t * self.num_queries, -1)

    def _encode_current_full(self, frame: torch.Tensor) -> torch.Tensor:
        """Encode current obs frame as all 256 DINOv2 patch tokens (no readout-query injection).
        Returns [B, 256, dim] with type_current added."""
        x = (frame - self.image_mean) / self.image_std
        be = self.vit.backend
        x = be.patch_embed(x)
        x = be._pos_embed(x)          # [B, 1+256, dim]
        if hasattr(be, "patch_drop"):
            x = be.patch_drop(x)
        if hasattr(be, "norm_pre"):
            x = be.norm_pre(x)
        for blk in be.blocks:
            x = blk(x)
        x = be.norm(x)                 # [B, 257, dim]
        return x[:, 1:] + self.type_current   # [B, 256, dim], skip CLS at idx 0

    def _build_cond(self, source_video, current_frame, point_track, ee,
                    point_track_causal=None,
                    depth_video=None, camera_K=None) -> torch.Tensor:
        if self.current_full_patch:
            cur = self._encode_current_full(current_frame)                                   # [B, 256, dim]
        else:
            cur = self._readout(current_frame.unsqueeze(1), None, self.type_current)         # [B, Q, dim]
        groups = []
        if self.use_source_video:                                                            # ① demo video
            groups.append(self._readout(source_video, self.source_time_embed, self.type_source))  # [B, 8*Q, dim]
        groups.append(cur)
        if self.use_global_track and point_track is not None:                                # ③ global track tokens
            if self.point_seq:
                pts = self.point_encoder(point_track) + self.type_point                       # [B, num_time, dim]
            else:
                pts = self.point_encoder(point_track)                                         # [B, N, dim]
                sq = self.point_summary_q.expand(pts.shape[0], -1, -1)
                pts = self.point_summary_attn(sq, self.point_summary_norm(pts)) + self.type_point
            groups.append(pts)
        # C1: causal recent-motion track as its own token group (in addition to the global track above).
        if self.point_encoder_causal is not None and point_track_causal is not None:
            pts_c = self.point_encoder_causal(point_track_causal) + self.type_point_causal     # [B, causal_T, dim]
            groups.append(pts_c)
        # EE position+depth -> ee tokens
        if ee.dim() == 1:
            ee = ee.unsqueeze(0)
        ee_tok = self.ee_query.expand(ee.shape[0], -1, -1) + self.ee_mlp(ee).unsqueeze(1) + self.type_ee
        groups.append(ee_tok)
        # ── depth 3D lifting tokens (3D Diffuser Actor style) ─────────────────
        if self.depth_enabled and depth_video is not None and camera_K is not None:
            B, T = depth_video.shape[0], depth_video.shape[1]
            # [B*T, H_d, W_d] + [B*T, 4] → [B*T, N_patches, 3]
            depth_flat = depth_video.reshape(B * T, *depth_video.shape[2:])
            K_flat = camera_K.unsqueeze(1).expand(-1, T, -1).reshape(B * T, 4)
            pos3d = self.depth_lifter(depth_flat, K_flat)               # [B*T, 256, 3]
            tok = self.depth_3d_enc(pos3d)                              # [B*T, 256, dim]
            sq = self.depth_pool_q.expand(B * T, -1, -1)
            tok = self.depth_pool_attn(sq, self.depth_pool_norm(tok))   # [B*T, n_tok, dim]
            tok = tok.reshape(B, T * self.n_depth_tok, -1) + self.type_depth
            groups.append(tok)
        if self.joint_action:
            B = current_frame.shape[0]
            aq = self.act_query.expand(B, -1, -1) + self.type_action   # [B, horizon, dim]
            groups.append(aq)
        cond = torch.cat(groups, dim=1)
        return self.mixer(cond)

    def forward(self, source_video, target_history=None, proprioception=None, action_target=None,
                point_track=None, point_track_causal=None,
                depth_video=None, camera_K=None):
        if target_history is None or proprioception is None:
            raise ValueError("FusedQueryRegPolicy needs target_history and proprioception(EE)")
        current_frame = target_history[:, -1]
        mixed = self._build_cond(source_video, current_frame, point_track, proprioception,
                                 point_track_causal, depth_video, camera_K)
        if self.joint_action:
            act_tok = mixed[:, -self.horizon:]           # [B, horizon, dim]
            normed = self.act_out_norm(act_tok)
            out = {
                "action_pred": torch.tanh(self.act_pose_head(normed)),
                "gripper_logits": self.act_gripper_head(normed),
                "terminate_logits": self.act_terminate_head(normed),
            }
            if self.aux_traj_enabled and self.use_source_video:
                # source tokens are always first in the mixer output: [B, source_len*num_queries, dim]
                src_tok = mixed[:, : self.source_len * self.num_queries]
                # mean-pool over the num_queries dimension → [B, source_len, dim]
                src_tok = src_tok.reshape(mixed.shape[0], self.source_len, self.num_queries, -1).mean(dim=2)
                out["traj_pred"] = torch.tanh(self.aux_traj_head(self.aux_traj_norm(src_tok)))
            return out
        return self.head(mixed)
