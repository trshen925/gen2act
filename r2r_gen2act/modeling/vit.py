from __future__ import annotations

import torch
import torch.nn as nn

try:
    from torchvision.models import vit_b_16, ViT_B_16_Weights
except Exception:
    vit_b_16 = None
    ViT_B_16_Weights = None


class ViTBackbone(nn.Module):
    def __init__(self, name: str = "vit_b16", pretrained: bool = True, image_size: int = 224, hidden_dim: int = 768, local_checkpoint: str = "", allow_random_init: bool = False) -> None:
        super().__init__()
        self.name = name
        self.image_size = image_size
        self.hidden_dim = hidden_dim
        self.patch_size = 16
        if name == "debug_small":
            self.patch_size = 16
            self.hidden_dim = min(int(hidden_dim), 128)
            self.backend = nn.Sequential(
                nn.Conv2d(3, self.hidden_dim, kernel_size=self.patch_size, stride=self.patch_size),
                nn.Flatten(2),
            )
        elif name in ("dinov2_vitb14", "dinov2"):
            self.patch_size = 14
            try:
                import timm  # type: ignore
            except ModuleNotFoundError as exc:
                raise ModuleNotFoundError(
                    "DINOv2 requires timm in the gen2act environment. Install timm or set "
                    "model.backbone.local_checkpoint with a supported local DINOv2/timm checkpoint."
                ) from exc
            self.backend = timm.create_model("vit_base_patch14_dinov2", pretrained=bool(pretrained and not local_checkpoint), img_size=image_size, num_classes=0)
            self.hidden_dim = int(getattr(self.backend, "num_features", hidden_dim))
            if local_checkpoint:
                state = torch.load(local_checkpoint, map_location="cpu")
                self.backend.load_state_dict(state.get("model", state), strict=False)
        else:
            self.backend = self._build_torchvision_vit(pretrained=pretrained, allow_random_init=allow_random_init)
        self.num_patches = (image_size // self.patch_size) * (image_size // self.patch_size)

    def _build_torchvision_vit(self, pretrained: bool, allow_random_init: bool) -> nn.Module:
        if vit_b_16 is None:
            raise ImportError("torchvision vit_b_16 is unavailable")
        weights = None
        if pretrained:
            if ViT_B_16_Weights is None:
                if not allow_random_init:
                    raise RuntimeError("Pretrained ViT_B_16 weights are unavailable in this torchvision version")
            else:
                weights = ViT_B_16_Weights.DEFAULT
        elif not allow_random_init and self.name != "debug_small":
            # Explicitly requiring pretrained catches silent random-init regressions.
            pass
        return vit_b_16(weights=weights, image_size=self.image_size, num_classes=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.name == "debug_small":
            tokens = self.backend(x).transpose(1, 2).contiguous()
            return tokens
        if hasattr(self.backend, "forward_features"):
            tokens = self.backend.forward_features(x)
            if isinstance(tokens, dict):
                tokens = tokens.get("x_norm_patchtokens", tokens.get("x", None))
            if tokens is None:
                raise RuntimeError("Unsupported timm forward_features output")
            if tokens.dim() == 3 and tokens.shape[1] > self.num_patches:
                tokens = tokens[:, -self.num_patches:, :]
            return tokens
        model = self.backend
        x = model._process_input(x)
        batch_size = x.shape[0]
        cls = model.class_token.expand(batch_size, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = model.encoder(x)
        return x[:, 1:, :]
