from __future__ import annotations

import importlib
from pathlib import Path
import sys

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from r2r_gen2act.config.load import default_config
from r2r_gen2act.config.schema import validate_config
from r2r_gen2act.data.adapters.base import WindowedRobotDataset
from r2r_gen2act.data.types import EpisodeRecord
from r2r_gen2act.modeling.fused_query_flow_policy import FusedQueryFlowPolicy
from r2r_gen2act.modeling.wan_vae import WanVAEBackbone


class _FakeWanBackbone(nn.Module):
    encoder_kind = "wan_vae"
    hidden_dim = 32
    latent_dim = 3

    def __init__(self) -> None:
        super().__init__()
        self.last_input_shape: tuple[int, ...] | None = None

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        self.last_input_shape = tuple(video.shape)
        pixels = video.transpose(1, 2)
        return F.avg_pool3d(pixels, kernel_size=(1, 8, 8), stride=(4, 8, 8))


class _UnusedHead(nn.Module):
    pass


def _fake_official_wan(root: Path) -> None:
    (root / "wan" / "modules").mkdir(parents=True)
    package_guard = "raise RuntimeError('Wan package initialization must not run')\n"
    (root / "wan" / "__init__.py").write_text(package_guard, encoding="utf-8")
    (root / "wan" / "modules" / "__init__.py").write_text(
        package_guard, encoding="utf-8")
    (root / "wan" / "modules" / "vae.py").write_text(
        """import torch
import torch.nn as nn
import torch.nn.functional as F

class _Model(nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        self.conv = nn.Conv3d(3, z_dim, 1)

    def encode(self, x, scale):
        latent = F.avg_pool3d(self.conv(x), (1, 8, 8), (4, 8, 8))
        mean, inv_std = scale
        return (latent - mean.view(1, -1, 1, 1, 1)) * inv_std.view(1, -1, 1, 1, 1)

class WanVAE:
    def __init__(self, z_dim=16, vae_pth='', dtype=torch.float32, device='cpu'):
        self.model = _Model(z_dim)
        self.model.load_state_dict(torch.load(vae_pth, map_location='cpu', weights_only=True))
""",
        encoding="utf-8",
    )


def _fake_diffsynth(root: Path) -> None:
    (root / "diffsynth" / "models").mkdir(parents=True)
    (root / "diffsynth" / "__init__.py").write_text("", encoding="utf-8")
    (root / "diffsynth" / "models" / "__init__.py").write_text("", encoding="utf-8")
    (root / "diffsynth" / "models" / "wan_video_vae.py").write_text(
        """import torch
import torch.nn as nn
import torch.nn.functional as F

class _Model(nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        self.conv = nn.Conv3d(3, z_dim, 1)

    def encode(self, x, scale):
        return F.avg_pool3d(self.conv(x), (1, 8, 8), (4, 8, 8))

class WanVideoVAE(nn.Module):
    def __init__(self, z_dim=16):
        super().__init__()
        self.model = _Model(z_dim)
        self.scale = [torch.zeros(z_dim), torch.ones(z_dim)]
""",
        encoding="utf-8",
    )


def _remove_test_modules() -> None:
    for name in tuple(sys.modules):
        if (name == "wan" or name.startswith("wan.")
                or name.startswith("_r2r_gen2act_wan_vae_")
                or name == "diffsynth" or name.startswith("diffsynth.")):
            sys.modules.pop(name, None)
    importlib.invalidate_caches()


def test_official_wan_backend_is_default_and_encodes(tmp_path: Path) -> None:
    root = tmp_path / "Wan2.1"
    checkpoint = tmp_path / "Wan2.1_VAE.pth"
    _fake_official_wan(root)
    torch.save({
        "conv.weight": torch.randn(16, 3, 1, 1, 1),
        "conv.bias": torch.randn(16),
    }, checkpoint)

    backbone = WanVAEBackbone(
        str(checkpoint), official_root=str(root), dtype="float32")
    latent = backbone(torch.rand(2, 5, 3, 16, 24))

    assert backbone.backend == "official"
    assert latent.shape == (2, 16, 2, 2, 3)
    assert not backbone.vae.training
    assert all(not parameter.requires_grad for parameter in backbone.parameters())
    _remove_test_modules()


def test_diffsynth_is_used_only_when_listed_as_fallback(tmp_path: Path) -> None:
    checkpoint = tmp_path / "Wan2.1_VAE.pth"
    diffsynth_root = tmp_path / "DiffSynth-Studio"
    _fake_diffsynth(diffsynth_root)
    torch.save({
        "model.conv.weight": torch.randn(16, 3, 1, 1, 1),
        "model.conv.bias": torch.randn(16),
    }, checkpoint)

    with pytest.raises(RuntimeError, match="official"):
        WanVAEBackbone(
            str(checkpoint), official_root=str(tmp_path / "missing"),
            diffsynth_root=str(diffsynth_root), dtype="float32")

    with pytest.warns(RuntimeWarning, match="fallback='diffsynth'"):
        backbone = WanVAEBackbone(
            str(checkpoint), official_root=str(tmp_path / "missing"),
            diffsynth_root=str(diffsynth_root), fallback_backends=["diffsynth"],
            dtype="float32")

    assert backbone.backend == "diffsynth"
    assert backbone(torch.rand(1, 5, 3, 16, 16)).shape == (1, 16, 2, 2, 2)
    _remove_test_modules()


def test_fused_policy_resamples_wan_latents_to_per_frame_queries() -> None:
    backbone = _FakeWanBackbone()
    policy = FusedQueryFlowPolicy(
        backbone, _UnusedHead(), None, source_len=5,
        num_queries=3, ee_dim=2, ee_tokens=1, vae_readout_heads=4,
        dt_time_cfg={"enabled": True, "num_freqs": 2, "max_sec": 2.0},
    )
    video = torch.rand(2, 5, 3, 16, 24)
    source_dt = torch.rand(2, 5)

    tokens = policy._readout(
        video, policy.source_time_embed, policy.type_source, source_dt,
        stream_name="source")
    tokens.sum().backward()

    assert tokens.shape == (2, 15, 32)
    assert backbone.last_input_shape == (10, 1, 3, 16, 24)
    assert policy.vae_latent_proj.weight.grad is not None
    assert policy.query.grad is not None


def test_fused_policy_wan_readout_keeps_frames_independent() -> None:
    policy = FusedQueryFlowPolicy(
        _FakeWanBackbone(), _UnusedHead(), None, source_len=3,
        num_queries=2, ee_dim=2, ee_tokens=1, vae_readout_heads=4,
    ).eval()
    video = torch.rand(1, 3, 3, 16, 16)

    before = policy._readout(
        video, policy.source_time_embed, policy.type_source,
        stream_name="source").reshape(1, 3, 2, 32)
    changed = video.clone()
    changed[:, 1] = 1.0 - changed[:, 1]
    after = policy._readout(
        changed, policy.source_time_embed, policy.type_source,
        stream_name="source").reshape(1, 3, 2, 32)

    torch.testing.assert_close(after[:, 0], before[:, 0])
    torch.testing.assert_close(after[:, 2], before[:, 2])
    assert not torch.allclose(after[:, 1], before[:, 1])


def test_source_micro_clips_are_sorted_and_read_in_one_40_frame_request() -> None:
    dataset = object.__new__(WindowedRobotDataset)
    dataset.source_micro_clip_enabled = True
    dataset.source_micro_clip_frames = 5
    dataset.source_micro_clip_stride = 1
    dataset.source_micro_clip_alignment = "causal"
    episode = EpisodeRecord(
        "ep", 100, Path("front.mp4"), Path("front.mp4"), Path("metadata.json"))
    anchors = [2, 12, 22, 32, 42, 52, 62, 72]
    calls: list[list[int]] = []

    def read_once(path: Path, indices: list[int]) -> torch.Tensor:
        assert path == episode.source_video_path
        calls.append(list(indices))
        return torch.as_tensor(indices, dtype=torch.float32).view(-1, 1, 1, 1)

    dataset._read_video_indices = read_once
    frames = dataset._read_source_indices(episode, anchors)

    assert len(calls) == 1
    assert len(calls[0]) == 40
    assert calls[0] == sorted(calls[0])
    assert frames.shape == (8, 5, 1, 1, 1)
    assert frames[:, -1, 0, 0, 0].tolist() == anchors
    assert frames[0, :, 0, 0, 0].tolist() == [0.0, 0.0, 0.0, 1.0, 2.0]


def test_fused_policy_wan_readout_encodes_eight_five_frame_microclips() -> None:
    backbone = _FakeWanBackbone()
    policy = FusedQueryFlowPolicy(
        backbone, _UnusedHead(), None, source_len=8,
        num_queries=3, ee_dim=2, ee_tokens=1, vae_readout_heads=4,
        dt_time_cfg={"enabled": True, "num_freqs": 2, "max_sec": 2.0},
    ).eval()
    video = torch.rand(2, 8, 5, 3, 16, 24)
    source_dt = torch.rand(2, 8)

    before = policy._readout(
        video, policy.source_time_embed, policy.type_source, source_dt,
        stream_name="source").reshape(2, 8, 3, 32)
    assert backbone.last_input_shape == (16, 5, 3, 16, 24)
    assert before.shape == (2, 8, 3, 32)

    changed = video.clone()
    changed[:, 4, -1] = 1.0 - changed[:, 4, -1]
    after = policy._readout(
        changed, policy.source_time_embed, policy.type_source, source_dt,
        stream_name="source").reshape(2, 8, 3, 32)

    torch.testing.assert_close(after[:, :4], before[:, :4])
    torch.testing.assert_close(after[:, 5:], before[:, 5:])
    assert not torch.allclose(after[:, 4], before[:, 4])


def test_config_validation_rejects_invalid_wan_backends() -> None:
    cfg = default_config()
    cfg["model"]["type"] = "fused_query_flow"
    cfg["model"]["backbone"].update({
        "name": "wan_vae", "pretrained": True, "freeze": True,
        "dtype": "bfloat16", "latent_dim": 16, "backend": "not-a-backend",
    })
    cfg["model"]["query_readout"] = {"heads": 8}
    with pytest.raises(ValueError, match="backend"):
        validate_config(cfg)

    cfg["model"]["backbone"]["backend"] = "official"
    cfg["model"]["front_depth"] = {"enabled": True}
    with pytest.raises(ValueError, match="patch geometry"):
        validate_config(cfg)


def test_config_validation_accepts_wan_five_frame_microclips() -> None:
    cfg = default_config()
    cfg["model"]["type"] = "fused_query_flow"
    cfg["model"]["backbone"].update({
        "name": "wan_vae", "pretrained": True, "freeze": True,
        "dtype": "bfloat16", "latent_dim": 16, "backend": "official",
    })
    cfg["model"]["query_readout"] = {"heads": 8}
    cfg["data"]["source_micro_clip"] = {
        "enabled": True, "frames": 5, "stride": 1, "alignment": "causal",
    }

    validate_config(cfg)
