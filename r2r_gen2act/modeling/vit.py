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
        elif name in ("dinov2_vitb14", "dinov2", "dinov3_vitl16", "dinov3_vitl", "dinov3_large"):
            self.patch_size = 14
            try:
                import timm  # type: ignore
            except ModuleNotFoundError as exc:
                raise ModuleNotFoundError(
                    "DINO backbones require timm in the gen2act environment. Install timm or set "
                    "model.backbone.local_checkpoint with a supported local timm checkpoint."
                ) from exc
            if name in ("dinov3_vitl16", "dinov3_vitl", "dinov3_large"):
                self.patch_size = 16
                timm_name = "vit_large_patch16_dinov3"
            else:
                timm_name = "vit_base_patch14_dinov2"
            self.backend = timm.create_model(
                timm_name,
                pretrained=bool(pretrained and not local_checkpoint),
                img_size=image_size,
                num_classes=0,
            )
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

    def unfreeze_last_blocks(self, n: int) -> int:
        """Unfreeze the last n transformer blocks (+ final norm) of a timm ViT backbone.

        Returns the number of blocks actually unfrozen. No-op for backbones without `.blocks`
        (e.g. debug_small). Pair with a small backbone_lr_multiplier so these layers actually train.
        """
        n = int(n)
        if n <= 0:
            return 0
        blocks = getattr(self.backend, "blocks", None)
        if blocks is None:
            return 0
        n = min(n, len(blocks))
        for blk in blocks[-n:]:
            for p in blk.parameters():
                p.requires_grad_(True)
        final_norm = getattr(self.backend, "norm", None)
        if final_norm is not None:
            for p in final_norm.parameters():
                p.requires_grad_(True)
        return n

    def prepare_tokens(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """Prepare timm ViT tokens while retaining optional DINOv3 rotary embeddings."""
        backend = self.backend
        x = backend.patch_embed(x)
        pos_out = backend._pos_embed(x)
        if isinstance(pos_out, tuple):
            x, rope = pos_out
        else:
            x, rope = pos_out, None
        patch_drop = getattr(backend, "patch_drop", None)
        # DINOv3's _pos_embed already applies patch_drop because it must update RoPE too.
        if patch_drop is not None and not isinstance(pos_out, tuple):
            x = patch_drop(x)
        norm_pre = getattr(backend, "norm_pre", None)
        if norm_pre is not None:
            x = norm_pre(x)
        return x, {"rope": rope}

    def run_blocks(
        self,
        x: torch.Tensor,
        start: int = 0,
        end: int | None = None,
        context: dict | None = None,
        extra_prefix_tokens: int = 0,
    ) -> torch.Tensor:
        """Run a block range for DINOv2 or DINOv3.

        DINOv3 uses RoPE on patch tokens. Query-in-backbone prepends action readout
        tokens, so those queries must temporarily count as additional prefix tokens;
        otherwise RoPE is applied to the wrong positions and the sequence lengths differ.
        """
        blocks = self.backend.blocks
        stop = len(blocks) if end is None else int(end)
        rope = (context or {}).get("rope")
        rope_mixed = bool(getattr(self.backend, "rope_mixed", False))
        for index in range(int(start), stop):
            block = blocks[index]
            block_rope = rope[index] if rope_mixed and rope is not None else rope
            if block_rope is None:
                x = block(x)
                continue
            attn = getattr(block, "attn", None)
            original_prefix = getattr(attn, "num_prefix_tokens", None)
            if original_prefix is not None and extra_prefix_tokens:
                attn.num_prefix_tokens = int(original_prefix) + int(extra_prefix_tokens)
            try:
                x = block(x, rope=block_rope)
            finally:
                if original_prefix is not None and extra_prefix_tokens:
                    attn.num_prefix_tokens = original_prefix
        return x

    def normalize_tokens(self, x: torch.Tensor) -> torch.Tensor:
        return self.backend.norm(x)

    def patch_tokens(self, x: torch.Tensor) -> torch.Tensor:
        prefix = int(getattr(self.backend, "num_prefix_tokens", 1))
        return x[:, prefix:]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.name == "debug_small":
            tokens = self.backend(x).transpose(1, 2).contiguous()
            return tokens
        if hasattr(self.backend, "forward_features") and hasattr(self.backend, "blocks"):
            tokens, context = self.prepare_tokens(x)
            tokens = self.run_blocks(tokens, context=context)
            return self.patch_tokens(self.normalize_tokens(tokens))
        model = self.backend
        x = model._process_input(x)
        batch_size = x.shape[0]
        cls = model.class_token.expand(batch_size, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = model.encoder(x)
        return x[:, 1:, :]
