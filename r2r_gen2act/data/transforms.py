from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional


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


def _gaussian_kernel(sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """1D Gaussian kernel of size 2*ceil(3σ)+1."""
    k = int(2 * int(np.ceil(3.0 * sigma)) + 1)
    k = k if k % 2 == 1 else k + 1
    k = max(k, 3)
    coords = torch.arange(k, dtype=torch.float32, device=device) - k // 2
    g = torch.exp(-coords ** 2 / (2.0 * sigma ** 2))
    g = g / g.sum()
    kernel = g.outer(g).to(dtype)  # [k, k]
    return kernel


def _blur_patch(patch: torch.Tensor, sigma: float) -> torch.Tensor:
    """Gaussian blur a [C, H, W] patch via separable conv2d."""
    C, H, W = patch.shape
    kernel = _gaussian_kernel(sigma, patch.device, patch.dtype)  # [k, k]
    k = kernel.shape[0]
    kernel4d = kernel.unsqueeze(0).unsqueeze(0).expand(C, 1, k, k)  # [C,1,k,k]
    pad = k // 2
    x4d = patch.unsqueeze(0)  # [1, C, H, W]
    return F.conv2d(x4d, kernel4d, padding=pad, groups=C).squeeze(0)


def _ee_patch_box(uc: int, vc: int, size: int, H: int, W: int) -> tuple[int, int]:
    """(top, left) so a size×size box centred at (uc, vc) fits inside the image."""
    top  = max(0, min(vc - size // 2, H - size))
    left = max(0, min(uc - size // 2, W - size))
    return top, left


def apply_structural_augmentation(
    video: torch.Tensor,
    cfg: dict,
    ee_fracs: Optional[np.ndarray] = None,
) -> torch.Tensor:
    """Per-frame structural augmentation to simulate generated-video gripper gap.

    A. Per-frame independent photometric jitter (brightness/contrast/saturation/noise).
    B. EE-centred local Gaussian blur  (patch around gripper; background unchanged).
    C. EE-centred local scale jitter   (±scale_range, resize patch in-place).
    D. EE-centred rectangular mask     (fill with mean colour; most aggressive).

    B/C/D require ee_fracs [T, 2] ∈ [0, 1] (normalised 224×224 coords); skipped if None.
    video: [T, C, H, W] float32 in [0, 1].
    """
    if not bool(cfg.get("enabled", False)):
        return video
    p = float(cfg.get("p", 1.0))
    if p < 1.0 and torch.rand(()) > p:
        return video

    T, C, H, W = video.shape
    x = video.clone()

    # A: per-frame independent photometric jitter
    pf = cfg.get("per_frame_photometric", {})
    if bool(pf.get("enabled", False)):
        brightness = float(pf.get("brightness", 0.0))
        contrast   = float(pf.get("contrast",   0.0))
        saturation = float(pf.get("saturation", 0.0))
        noise_std  = float(pf.get("noise_std",  0.0))
        for t in range(T):
            f = x[t]
            if brightness > 0:
                delta = (torch.rand((), dtype=f.dtype, device=f.device) * 2.0 - 1.0) * brightness
                f = f + delta
            if contrast > 0:
                factor = 1.0 + (torch.rand((), dtype=f.dtype, device=f.device) * 2.0 - 1.0) * contrast
                m = f.mean(dim=(-2, -1), keepdim=True)
                f = (f - m) * factor + m
            if saturation > 0:
                factor = 1.0 + (torch.rand((), dtype=f.dtype, device=f.device) * 2.0 - 1.0) * saturation
                gray = f.mean(dim=0, keepdim=True)
                f = (f - gray) * factor + gray
            if noise_std > 0:
                f = f + torch.randn_like(f) * noise_std
            x[t] = f

    # EE pixel coordinates in 224×224 frame
    if ee_fracs is not None:
        uc_arr = np.clip((ee_fracs[:, 0] * (W - 1)).astype(int), 0, W - 1)
        vc_arr = np.clip((ee_fracs[:, 1] * (H - 1)).astype(int), 0, H - 1)

        # B: EE local Gaussian blur
        blur_cfg = cfg.get("ee_blur", {})
        if bool(blur_cfg.get("enabled", False)):
            size_frac = float(blur_cfg.get("size_frac", 0.30))
            sigma     = float(blur_cfg.get("sigma",     5.0))
            prob      = float(blur_cfg.get("prob",      0.6))
            sz = max(3, int(min(H, W) * size_frac))
            for t in range(T):
                if torch.rand(()) < prob:
                    top, left = _ee_patch_box(int(uc_arr[t]), int(vc_arr[t]), sz, H, W)
                    patch = x[t, :, top:top + sz, left:left + sz]
                    x[t, :, top:top + sz, left:left + sz] = _blur_patch(patch, sigma)

        # C: EE local scale jitter
        scale_cfg = cfg.get("ee_scale", {})
        if bool(scale_cfg.get("enabled", False)):
            size_frac   = float(scale_cfg.get("size_frac",   0.30))
            scale_range = float(scale_cfg.get("scale_range", 0.20))
            prob        = float(scale_cfg.get("prob",        0.5))
            sz = max(4, int(min(H, W) * size_frac))
            for t in range(T):
                if torch.rand(()) < prob:
                    top, left = _ee_patch_box(int(uc_arr[t]), int(vc_arr[t]), sz, H, W)
                    patch = x[t:t + 1, :, top:top + sz, left:left + sz].clone()
                    scale = 1.0 + (torch.rand(()).item() * 2.0 - 1.0) * scale_range
                    nh = max(1, int(sz * scale))
                    nw = max(1, int(sz * scale))
                    scaled = F.interpolate(patch, size=(nh, nw), mode="bilinear", align_corners=False)
                    fill = x[t].mean().item()
                    if scale >= 1.0:
                        ct = (nh - sz) // 2
                        cl = (nw - sz) // 2
                        x[t, :, top:top + sz, left:left + sz] = scaled[0, :, ct:ct + sz, cl:cl + sz]
                    else:
                        pt = (sz - nh) // 2
                        pl = (sz - nw) // 2
                        x[t, :, top:top + sz, left:left + sz] = fill
                        x[t, :, top + pt:top + pt + nh, left + pl:left + pl + nw] = scaled[0]

        # D: EE rectangular mask (most aggressive)
        cutout_cfg = cfg.get("ee_cutout", {})
        if bool(cutout_cfg.get("enabled", False)):
            size_frac = float(cutout_cfg.get("size_frac", 0.25))
            prob      = float(cutout_cfg.get("prob",      0.3))
            sz = max(2, int(min(H, W) * size_frac))
            for t in range(T):
                if torch.rand(()) < prob:
                    top, left = _ee_patch_box(int(uc_arr[t]), int(vc_arr[t]), sz, H, W)
                    x[t, :, top:top + sz, left:left + sz] = x[t].mean().item()

    return x.clamp_(0.0, 1.0)
