"""Step 5: VideoMAEv2 reference-video encoder (8 frames -> video latent).

Wraps the VideoMAEv2 ViT-B (distilled-from-giant, K710) backbone as a frozen video encoder.
The reference video is sampled to 8 frames upstream; the pretrained temporal positional
embedding expects 16 frames (tubelet_size=2 -> 8 temporal tokens), so we temporally upsample
the 8 input frames to 16 before encoding. Returns a pooled [B, 768] latent (used as one or a few
conditioning tokens by the flow-matching DiT). This is the ONLY path future information enters the
model — there is no t+H future-frame input.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# VideoMAEv2 repo lives at <repo_root>/VideoMAEv2 (sibling of the gen2act training package).
_VMAE_ROOT = Path(__file__).resolve().parents[3] / "VideoMAEv2"


def _import_vmae():
    if str(_VMAE_ROOT) not in sys.path:
        sys.path.insert(0, str(_VMAE_ROOT))
    from models.modeling_finetune import vit_base_patch16_224  # noqa: E402
    return vit_base_patch16_224


# Kinetics/ImageNet normalization used by VideoMAEv2.
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


class VideoMAEv2Encoder(nn.Module):
    def __init__(self, checkpoint: str, all_frames: int = 16, tubelet_size: int = 2,
                 freeze: bool = True) -> None:
        super().__init__()
        builder = _import_vmae()
        self.all_frames = int(all_frames)
        self.model = builder(num_classes=710, all_frames=all_frames, tubelet_size=tubelet_size)
        if checkpoint:
            sd = torch.load(checkpoint, map_location="cpu")
            sd = sd.get("module", sd.get("model", sd))
            missing, unexpected = self.model.load_state_dict(sd, strict=False)
            print(f"[VideoMAEv2Encoder] loaded {checkpoint}: missing={len(missing)} unexpected={len(unexpected)}")
        self.model.head = nn.Identity()
        self.output_dim = 768
        self.frozen = bool(freeze)
        if self.frozen:
            for p in self.model.parameters():
                p.requires_grad_(False)
            self.model.eval()
        self.register_buffer("mean", torch.tensor(_MEAN).view(1, 3, 1, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(_STD).view(1, 3, 1, 1, 1), persistent=False)

    def _prep(self, frames: torch.Tensor) -> torch.Tensor:
        """frames: [B, T, 3, H, W] in [0,1] -> [B, 3, all_frames, H, W] normalized."""
        b, t, c, h, w = frames.shape
        x = frames.permute(0, 2, 1, 3, 4)  # [B,3,T,H,W]
        if t != self.all_frames:
            x = F.interpolate(x, size=(self.all_frames, h, w), mode="nearest")
        return (x - self.mean) / self.std

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        x = self._prep(frames)
        if self.frozen:
            with torch.no_grad():
                feat = self.model.forward_features(x)
        else:
            feat = self.model.forward_features(x)
        return feat  # [B, 768]
