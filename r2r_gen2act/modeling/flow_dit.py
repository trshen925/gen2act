from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta

# Self-contained flow-matching DiT action head (no diffusers dependency), a FAITHFUL port of the
# GR00T-Dreams IDM design (cross_attention_dit.py DiT + flow_matching_action_head_idm.py +
# IDM_dump/base.yaml). Structure mirrors IDM as closely as possible:
#   - conditioning tokens are projected to hidden_dim and mixed by a SelfAttentionTransformer
#     (the "VL self-attention" mixer, 4 layers in IDM) before the DiT;
#   - the noisy action chunk is embedded together with the flow timestep (MultiEmbodimentActionEncoder
#     analogue) so time enters the DiT input tokens;
#   - the DiT is `num_layers` BasicTransformerBlocks with AdaLayerNorm timestep conditioning
#     (scale+shift, NOT adaLN-Zero gating) and a single attention per block, INTERLEAVED between
#     cross-attention to the conditioning (even blocks) and self-attention over the action tokens
#     (odd blocks), matching interleave_self_attention=True;
#   - a final AdaLayerNorm output head from the timestep embedding (proj_out_1 / norm_out / proj_out_2);
#   - the network predicts the flow VELOCITY (action - noise); loss is MSE; inference Euler-integrates;
#   - flow timesteps are sampled from a Beta distribution (IDM), not uniformly.
# Differences from IDM (kept for gen2act compatibility): conditioning comes from gen2act's
# DINOv2 resampler/fusion tokens (not SigLIP); separate gripper/terminate binary heads are added on
# the pooled conditioning (gen2act predicts these as classification, not as continuous action dims);
# an optional multi-sample inference average is available (default 1 = IDM single-sample).
#
# IDM reference sizes (IDM_dump/base.yaml): hidden 1024, DiT num_layers 8, heads 16 x head_dim 64,
# VL self-attn mixer 4 layers, num_inference_timesteps 16, Beta(1.5, 1.0), noise_s 0.999.


def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """Sinusoidal embedding of continuous timesteps t in [0,1]. t: [B] -> [B, dim]."""
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = t.float()[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class _MHA(nn.Module):
    """Multi-head attention; self-attention when kv is q, cross-attention otherwise."""

    def __init__(self, dim: int, heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        out, _ = self.attn(q, kv, kv, need_weights=False)
        return out


class TimeActionEncoder(nn.Module):
    """Encode the noisy action chunk together with the flow timestep, mirroring GR00T IDM's
    MultiEmbodimentActionEncoder (minus the per-embodiment weights): the action embedding is
    concatenated with a sinusoidal time embedding and mixed (swish MLP), so the DiT input tokens
    already carry 'where on the noise trajectory we are' (time enters both here and the adaLN)."""

    def __init__(self, action_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.w1 = nn.Linear(action_dim, hidden_dim)
        self.w2 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.w3 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, actions: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        a = self.w1(actions)
        tau = timestep_embedding(t, self.hidden_dim).to(a.dtype)
        tau = tau[:, None, :].expand(-1, a.shape[1], -1)
        x = F.silu(self.w2(torch.cat([a, tau], dim=-1)))
        return self.w3(x)


class AdaLayerNorm(nn.Module):
    """IDM AdaLayerNorm: timestep embedding -> (scale, shift) modulation of a non-affine LayerNorm.
    No zero-init gate (this is plain ada_norm, not adaLN-Zero)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, 2 * dim)
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        scale, shift = self.linear(self.silu(temb)).chunk(2, dim=-1)
        return self.norm(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class FeedForward(nn.Module):
    def __init__(self, dim: int, mult: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        inner = dim * mult
        self.net = nn.Sequential(
            nn.Linear(dim, inner), nn.GELU(), nn.Dropout(dropout), nn.Linear(inner, dim), nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BasicTransformerBlock(nn.Module):
    """IDM BasicTransformerBlock: AdaLayerNorm(temb) -> single attention (self OR cross) -> residual
    -> plain LayerNorm -> FeedForward -> residual. The conditioning enters only in cross-attn blocks
    (encoder_hidden_states); self-attn blocks attend over the action tokens. No gating."""

    def __init__(self, dim: int, heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = AdaLayerNorm(dim)
        self.attn = _MHA(dim, heads, dropout)
        self.norm3 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, dropout=dropout)

    def forward(self, x: torch.Tensor, temb: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        h = self.norm1(x, temb)
        kv = h if cond is None else cond
        x = x + self.attn(h, kv)
        x = x + self.ff(self.norm3(x))
        return x


class SelfAttentionMixer(nn.Module):
    """IDM SelfAttentionTransformer: stack of self-attention blocks (plain LayerNorm, no temb) that
    mixes the conditioning tokens before the DiT cross-attends to them."""

    def __init__(self, dim: int, heads: int, num_layers: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [nn.ModuleDict({
                "norm1": nn.LayerNorm(dim), "attn": _MHA(dim, heads, dropout),
                "norm2": nn.LayerNorm(dim), "ff": FeedForward(dim, dropout=dropout),
            }) for _ in range(num_layers)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for b in self.blocks:
            h = b["norm1"](x)
            x = x + b["attn"](h, h)
            x = x + b["ff"](b["norm2"](x))
        return x


class FlowMatchingDiTHead(nn.Module):
    """Flow-matching DiT action head, faithful to GR00T IDM. Conditioning is a token sequence
    [B, S, cond_dim] (gen2act resampler/fusion tokens). Predicts an action chunk
    [B, horizon, action_dim] (pose, normalized). Gripper/terminate are separate per-step binary heads
    on the pooled (mixed) conditioning, kept for compatibility with the gen2act losses."""

    def __init__(self, cond_dim: int, action_dim: int, horizon: int, hidden_dim: int = 1024,
                 num_layers: int = 8, heads: int = 16, num_inference_steps: int = 16,
                 dropout: float = 0.1, num_eval_samples: int = 1, time_sampling: str = "beta",
                 noise_beta_alpha: float = 1.5, noise_beta_beta: float = 1.0, noise_s: float = 0.999,
                 vl_mixer_layers: int = 4, interleave_self_attention: bool = True) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.horizon = int(horizon)
        self.hidden_dim = int(hidden_dim)
        self.num_inference_steps = int(num_inference_steps)
        self.num_eval_samples = int(num_eval_samples)
        self.interleave = bool(interleave_self_attention)
        # flow timestep sampling: "beta" matches IDM (Beta(alpha,beta) via t=(s-sample)/s); "uniform"
        # is the plain torch.rand baseline.
        self.time_sampling = str(time_sampling)
        self.noise_s = float(noise_s)
        self.beta_dist = Beta(float(noise_beta_alpha), float(noise_beta_beta))
        # conditioning: project to hidden then mix with a self-attention transformer (IDM VL mixer).
        self.cond_in = nn.Linear(cond_dim, hidden_dim) if cond_dim != hidden_dim else nn.Identity()
        self.vl_mixer = SelfAttentionMixer(hidden_dim, heads, vl_mixer_layers, dropout)
        # time-conditioned action encoder (IDM-style) + learned action-chunk positional embedding.
        self.action_encoder = TimeActionEncoder(action_dim, hidden_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, horizon, hidden_dim) / hidden_dim**0.5)
        # timestep encoder -> temb (for AdaLayerNorm).
        self.t_embed = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        # DiT blocks (single-attention, interleaved cross/self).
        self.blocks = nn.ModuleList([BasicTransformerBlock(hidden_dim, heads, dropout) for _ in range(num_layers)])
        # IDM output head: AdaLayerNorm(temb) over the final hidden, then project to the action dim.
        self.norm_out = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.proj_out_1 = nn.Linear(hidden_dim, 2 * hidden_dim)
        self.action_out = nn.Linear(hidden_dim, action_dim)
        # auxiliary per-horizon-step binary heads on pooled (mixed) conditioning.
        self.gripper_proj = nn.Linear(hidden_dim, horizon * 2)
        self.terminate_proj = nn.Linear(hidden_dim, horizon * 2)

    def _aux(self, vl: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pooled = vl.mean(dim=1)
        b = pooled.shape[0]
        gripper = self.gripper_proj(pooled).view(b, self.horizon, 2)
        terminate = self.terminate_proj(pooled).view(b, self.horizon, 2)
        return gripper, terminate

    def _mix_cond(self, cond: torch.Tensor) -> torch.Tensor:
        return self.vl_mixer(self.cond_in(cond))

    def _velocity(self, noisy: torch.Tensor, t: torch.Tensor, vl: torch.Tensor) -> torch.Tensor:
        # noisy: [B, H, action_dim], t: [B] in [0,1], vl: [B, S, hidden] (already mixed conditioning)
        temb = self.t_embed(timestep_embedding(t, self.hidden_dim))
        x = self.action_encoder(noisy, t) + self.pos_embed[:, : noisy.shape[1]]
        for idx, blk in enumerate(self.blocks):
            cond = None if (self.interleave and idx % 2 == 1) else vl
            x = blk(x, temb, cond)
        scale, shift = self.proj_out_1(F.silu(temb)).chunk(2, dim=-1)
        x = self.norm_out(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        return self.action_out(x)

    def _sample_t(self, b: int, device) -> torch.Tensor:
        """Flow timestep in [0,1]. Beta-biased (IDM) or uniform."""
        if self.time_sampling == "beta":
            sample = self.beta_dist.sample((b,)).to(device)
            return ((self.noise_s - sample) / self.noise_s).clamp(0.0, 1.0)
        return torch.rand(b, device=device)

    def forward(self, cond: torch.Tensor, target_actions: torch.Tensor) -> dict:
        """Training: flow-matching loss on the pose chunk. target_actions: [B, H, action_dim] (normalized)."""
        vl = self._mix_cond(cond)
        b = target_actions.shape[0]
        noise = torch.randn_like(target_actions)
        t = self._sample_t(b, target_actions.device)
        tt = t[:, None, None]
        noisy = (1 - tt) * noise + tt * target_actions
        velocity = target_actions - noise
        pred_v = self._velocity(noisy, t, vl)
        gripper, terminate = self._aux(vl)
        return {
            "pred_velocity": pred_v,
            "target_velocity": velocity,
            "gripper_logits": gripper,
            "terminate_logits": terminate,
        }

    def _sample_once(self, vl: torch.Tensor) -> torch.Tensor:
        b = vl.shape[0]
        x = torch.randn(b, self.horizon, self.action_dim, device=vl.device)
        dt = 1.0 / self.num_inference_steps
        for i in range(self.num_inference_steps):
            t = torch.full((b,), i * dt, device=vl.device)
            x = x + dt * self._velocity(x, t, vl)
        return x

    @torch.no_grad()
    def sample(self, cond: torch.Tensor, num_samples: int | None = None) -> dict:
        """Inference: Euler-integrate the velocity field from noise -> action chunk. With
        num_samples>1 (default self.num_eval_samples), average several samples (conditional mean)."""
        vl = self._mix_cond(cond)
        k = max(1, int(num_samples if num_samples is not None else self.num_eval_samples))
        acc = self._sample_once(vl)
        for _ in range(k - 1):
            acc = acc + self._sample_once(vl)
        x = acc / k
        gripper, terminate = self._aux(vl)
        return {
            "action_pred": x,
            "gripper_logits": gripper,
            "terminate_logits": terminate,
        }
