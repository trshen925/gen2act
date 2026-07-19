"""Depth-based 3D patch position computation (3D Diffuser Actor approach).

For each DINOv2 patch in a 224×224 frame, computes its 3D camera-frame position
by averaging the depth values within the patch footprint and backprojecting via the
pinhole camera model (intrinsics scaled to 224×224 after resize_center_crop).

This is purely geometric — no learnable parameters.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthTo3DPatchGeometry(nn.Module):
    """Align metric depth with the RGB crop and lift each ViT patch to 3D.

    ``K`` is expressed in the final square model-input coordinates. Invalid
    depth is resized with normalized mask weighting, so zero-valued holes do
    not pull neighbouring surfaces towards the camera.

    Returns ``[X, Y, Z, valid_ratio]`` in camera coordinates for every patch.
    """

    def __init__(self, patch_size: int = 14, image_size: int = 224,
                 max_depth_m: float = 20.0) -> None:
        super().__init__()
        if image_size % patch_size:
            raise ValueError(f"image_size={image_size} must be divisible by patch_size={patch_size}")
        self.patch_size = int(patch_size)
        self.image_size = int(image_size)
        self.max_depth_m = float(max_depth_m)
        side = self.image_size // self.patch_size
        centers = torch.arange(side, dtype=torch.float32) * self.patch_size + self.patch_size / 2.0
        py, px = torch.meshgrid(centers, centers, indexing="ij")
        self.register_buffer("patch_px", px.reshape(1, -1), persistent=False)
        self.register_buffer("patch_py", py.reshape(1, -1), persistent=False)

    def _resize_center_crop_valid(self, depth_m: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, h, w = depth_m.shape
        if h < w:
            new_h = self.image_size
            new_w = int(round(w * self.image_size / h))
        else:
            new_w = self.image_size
            new_h = int(round(h * self.image_size / w))
        valid = ((depth_m > 0) & (depth_m < self.max_depth_m)).to(torch.float32)
        weighted = F.interpolate(
            (depth_m * valid).unsqueeze(1), (new_h, new_w), mode="bilinear", align_corners=False)
        weights = F.interpolate(
            valid.unsqueeze(1), (new_h, new_w), mode="bilinear", align_corners=False)
        resized = weighted / weights.clamp_min(1e-6)
        top = max(0, (new_h - self.image_size) // 2)
        left = max(0, (new_w - self.image_size) // 2)
        return (
            resized[:, :, top:top + self.image_size, left:left + self.image_size],
            weights[:, :, top:top + self.image_size, left:left + self.image_size],
        )

    def forward(self, depth_mm: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
        if depth_mm.ndim != 3 or K.ndim != 2 or K.shape[-1] != 4:
            raise ValueError(
                f"Expected depth [B,H,W] and K [B,4], got {tuple(depth_mm.shape)} and {tuple(K.shape)}")
        depth_m = depth_mm.to(torch.float32) / 1000.0
        depth_crop, valid_crop = self._resize_center_crop_valid(depth_m)
        ps = self.patch_size
        valid_sum = F.avg_pool2d(valid_crop, ps, ps).flatten(1)
        depth_sum = F.avg_pool2d(depth_crop * valid_crop, ps, ps).flatten(1)
        patch_depth = depth_sum / valid_sum.clamp_min(1e-6)
        valid_ratio = valid_sum.clamp(0.0, 1.0)
        patch_depth = patch_depth * (valid_ratio > 0).to(patch_depth.dtype)

        fx, fy, cx, cy = (K[:, i:i + 1].to(torch.float32) for i in range(4))
        x = patch_depth * (self.patch_px - cx) / fx.clamp_min(1e-6)
        y = patch_depth * (self.patch_py - cy) / fy.clamp_min(1e-6)
        return torch.stack((x, y, patch_depth, valid_ratio), dim=-1)


class DepthTo3DPatchPositions(nn.Module):
    """Backproject depth patch averages to 3D camera-frame positions.

    Input:
        depth   [B, H_d, W_d]  uint16 or float depth in mm (raw sensor output)
        K       [B, 4]          camera intrinsics (fx, fy, cx, cy) scaled to 224×224
    Output:
        [B, N_patches, 3]  3D positions in metres (camera frame, x-right y-down z-forward)

    N_patches = (image_size // patch_size)^2 = (224 // 14)^2 = 256.
    """

    def __init__(self, patch_size: int = 14, image_size: int = 224,
                 depth_scale: float = 1.0, max_depth_m: float = 10.0) -> None:
        super().__init__()
        self.patch_size = int(patch_size)
        self.image_size = int(image_size)
        self.depth_scale = float(depth_scale)   # multiply raw values → mm (usually 1.0)
        self.max_depth_m = float(max_depth_m)
        self.n_patches_per_side = image_size // patch_size   # 16
        self.n_patches = self.n_patches_per_side ** 2        # 256

        # Patch centre pixel coordinates in the 224×224 frame (fixed, precomputed)
        # patch (i, j): row i, col j → centre at (j*ps+ps/2, i*ps+ps/2) = (x, y) pixels
        ps = self.patch_size
        ns = self.n_patches_per_side
        cols = torch.arange(ns, dtype=torch.float32) * ps + ps / 2.0   # x (width)
        rows = torch.arange(ns, dtype=torch.float32) * ps + ps / 2.0   # y (height)
        grid_y, grid_x = torch.meshgrid(rows, cols, indexing='ij')    # [ns, ns]
        # flatten in row-major order: patch k → (px_k, py_k)
        px = grid_x.reshape(-1)   # [N_patches] pixel x (column)
        py = grid_y.reshape(-1)   # [N_patches] pixel y (row)
        self.register_buffer('patch_px', px, persistent=False)  # [N]
        self.register_buffer('patch_py', py, persistent=False)  # [N]

    def forward(self, depth: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
        """
        depth : [B, H_d, W_d]  — uint16 or float, values in mm
        K     : [B, 4]          — (fx, fy, cx, cy) in 224×224 pixel units
        returns [B, N_patches, 3]
        """
        B = depth.shape[0]
        device = depth.device
        dtype = torch.float32

        # ── convert to float metres ──────────────────────────────────────────
        d = depth.to(dtype)
        if self.depth_scale != 1.0:
            d = d * self.depth_scale
        d = d / 1000.0        # mm → metres

        # ── resize depth to image_size × image_size ──────────────────────────
        # Use bilinear for smooth depth values; keepdim adds channel dim
        d4 = d.unsqueeze(1)   # [B, 1, H_d, W_d]
        d_resized = F.interpolate(d4, size=(self.image_size, self.image_size),
                                  mode='bilinear', align_corners=False).squeeze(1)  # [B, H, W]

        # ── average depth within each patch ──────────────────────────────────
        # Fold depth into (patch_size × patch_size) blocks, then mean
        ps = self.patch_size
        ns = self.n_patches_per_side
        # reshape [B, H, W] → [B, ns, ps, ns, ps] → mean over patch dims
        d_fold = d_resized.reshape(B, ns, ps, ns, ps)
        patch_depth = d_fold.mean(dim=(2, 4))    # [B, ns, ns]
        patch_depth = patch_depth.reshape(B, self.n_patches)   # [B, N]

        # ── clamp invalid depth ───────────────────────────────────────────────
        # zero-depth pixels (missing/out-of-range) become invalid; set to 0 in output
        valid = (patch_depth > 0) & (patch_depth < self.max_depth_m)

        # ── backproject via pinhole camera model ──────────────────────────────
        # K: [B, 4] = (fx, fy, cx, cy)
        fx = K[:, 0:1]   # [B, 1]
        fy = K[:, 1:2]
        cx = K[:, 2:3]
        cy = K[:, 3:4]

        px = self.patch_px.to(device).unsqueeze(0).expand(B, -1)  # [B, N]
        py = self.patch_py.to(device).unsqueeze(0).expand(B, -1)  # [B, N]

        d_val = patch_depth                          # [B, N]
        X = d_val * (px - cx) / fx                  # [B, N]
        Y = d_val * (py - cy) / fy                  # [B, N]
        Z = d_val                                    # [B, N]

        # Zero out invalid patches
        mask = valid.float()                         # [B, N]
        X = X * mask
        Y = Y * mask
        Z = Z * mask

        return torch.stack([X, Y, Z], dim=-1)        # [B, N_patches, 3]
