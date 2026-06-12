from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def image_to_tensor(image: np.ndarray, image_size: int) -> torch.Tensor:
    tensor = torch.as_tensor(image)
    if tensor.ndim == 2:
        tensor = tensor[:, :, None].expand(-1, -1, 3)
    if tensor.ndim != 3 or tensor.shape[-1] not in (1, 3, 4):
        raise ValueError(f"Unexpected image shape: {tuple(tensor.shape)}")
    if tensor.shape[-1] == 4:
        tensor = tensor[..., :3]
    if tensor.shape[-1] == 1:
        tensor = tensor.expand(-1, -1, 3)
    tensor = tensor.permute(2, 0, 1).contiguous().float()
    if tensor.numel() and tensor.max().item() > 1.5:
        tensor = tensor / 255.0
    return resize_center_crop(tensor, image_size)


def resize_center_crop(tensor: torch.Tensor, image_size: int) -> torch.Tensor:
    if tensor.shape[-2:] == (image_size, image_size):
        return tensor
    h, w = tensor.shape[-2], tensor.shape[-1]
    if h < w:
        new_h = image_size
        new_w = int(round(w * image_size / h))
    else:
        new_w = image_size
        new_h = int(round(h * image_size / w))
    x = F.interpolate(tensor.unsqueeze(0), size=(new_h, new_w), mode="bilinear", align_corners=False).squeeze(0)
    top = max(0, (new_h - image_size) // 2)
    left = max(0, (new_w - image_size) // 2)
    return x[:, top:top + image_size, left:left + image_size]


def apply_image_augmentation(video: torch.Tensor, cfg: dict) -> torch.Tensor:
    """Apply action-safe photometric augmentation without changing geometry.

    The same random brightness/contrast/saturation factors are applied to all frames
    in one clip to avoid temporal flicker. No crop/flip/rotation is used because
    this project predicts robot actions in the original camera/action frame.
    """
    if not bool(cfg.get("enabled", False)):
        return video
    p = float(cfg.get("p", 1.0))
    if p < 1.0 and torch.rand(()) > p:
        return video
    x = video
    brightness = float(cfg.get("brightness", 0.0))
    contrast = float(cfg.get("contrast", 0.0))
    saturation = float(cfg.get("saturation", 0.0))
    noise_std = float(cfg.get("noise_std", 0.0))
    if brightness > 0:
        delta = (torch.rand((), dtype=x.dtype, device=x.device) * 2.0 - 1.0) * brightness
        x = x + delta
    if contrast > 0:
        factor = 1.0 + (torch.rand((), dtype=x.dtype, device=x.device) * 2.0 - 1.0) * contrast
        mean = x.mean(dim=(-2, -1), keepdim=True)
        x = (x - mean) * factor + mean
    if saturation > 0:
        factor = 1.0 + (torch.rand((), dtype=x.dtype, device=x.device) * 2.0 - 1.0) * saturation
        gray = x.mean(dim=-3, keepdim=True)
        x = (x - gray) * factor + gray
    if noise_std > 0:
        x = x + torch.randn_like(x) * noise_std
    return x.clamp_(0.0, 1.0)
