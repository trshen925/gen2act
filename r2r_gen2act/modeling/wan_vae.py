from __future__ import annotations

import hashlib
import importlib
import importlib.util
import os
import sys
import warnings
from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn


_WAN_VAE_NAMES = {"wan_vae", "wan2.1_vae", "wan2_1_vae", "wan_video_vae"}
_WanBackend = Literal["official", "diffusers", "diffsynth"]
_OFFICIAL_MODULE_PREFIX = "_r2r_gen2act_wan_vae"
_WAN21_MEAN = (
    -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
    0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921,
)
_WAN21_STD = (
    2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
    3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160,
)


def is_wan_vae_name(name: str) -> bool:
    return str(name).lower() in _WAN_VAE_NAMES


def _resolve_optional_path(value: str, env_name: str) -> Path | None:
    raw = str(value or "").strip() or str(os.environ.get(env_name, "")).strip()
    return Path(raw).expanduser().resolve() if raw else None


def resolve_wan_checkpoint(value: str = "", env_name: str = "WAN_VAE_CHECKPOINT") -> Path | None:
    return _resolve_optional_path(value, env_name)


def _resolve_official_vae_source(path: Path | None, env_name: str) -> Path:
    if path is None:
        raise FileNotFoundError(
            f"Wan2.1 repository root is required. Set model.backbone.official_root "
            f"or {env_name}."
        )
    if (path / "wan").is_dir():
        root = path
    elif path.name == "wan" and path.is_dir():
        root = path.parent
    else:
        raise FileNotFoundError(f"Expected a Wan2.1 repository root, got: {path}")
    source = root / "wan" / "modules" / "vae.py"
    if not source.is_file():
        raise FileNotFoundError(f"Official Wan VAE source not found: {source}")
    return source


def _import_official_wan(root: str, env_name: str):
    source = _resolve_official_vae_source(
        _resolve_optional_path(root, env_name), env_name)
    digest = hashlib.sha256(str(source).encode("utf-8")).hexdigest()[:16]
    module_name = f"{_OFFICIAL_MODULE_PREFIX}_{digest}"
    module = sys.modules.get(module_name)
    if module is None:
        spec = importlib.util.spec_from_file_location(module_name, source)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load official Wan VAE source: {source}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(module_name, None)
            raise
    vae_class = getattr(module, "WanVAE", None)
    if vae_class is None:
        raise ImportError(f"Official Wan VAE source does not export WanVAE: {source}")
    return vae_class


def _resolve_diffsynth_root(value: str, env_name: str) -> Path | None:
    root = _resolve_optional_path(value, env_name)
    candidates: list[Path] = []
    if root is not None:
        candidates.append(root)
    else:
        # Developer checkout layout: DreamFlyWheel/{gen2act,DiffSynth-Studio}.
        candidates.append(Path(__file__).resolve().parents[3] / "DiffSynth-Studio")
    return next((path for path in candidates if (path / "diffsynth").is_dir()), None)


def _import_diffsynth_wan(root: str, env_name: str):
    try:
        module = importlib.import_module("diffsynth.models.wan_video_vae")
    except ModuleNotFoundError:
        path = _resolve_diffsynth_root(root, env_name)
        if path is None:
            raise ModuleNotFoundError(
                "DiffSynth fallback was requested but DiffSynth-Studio was not found. Set "
                "model.backbone.diffsynth_root or DIFFSYNTH_ROOT."
            )
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
        module = importlib.import_module("diffsynth.models.wan_video_vae")
    vae_class = getattr(module, "WanVideoVAE", None)
    if vae_class is None:
        raise ImportError("DiffSynth Wan-VAE module does not export WanVideoVAE")
    return vae_class


def _torch_dtype(name: str) -> torch.dtype:
    choices = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
    }
    try:
        return choices[str(name).lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported Wan-VAE dtype: {name}") from exc


def _load_diffsynth_state_dict(checkpoint: Path) -> dict[str, torch.Tensor]:
    """Read a checkpoint in the key layout expected by DiffSynth's wrapper.

    Wan's official ``Wan2.1_VAE.pth`` is a bare state dict, while some
    compatibility checkpoints are saved beneath ``state_dict`` or
    ``model_state``.  DiffSynth's ``WanVideoVAE`` owns the actual VAE below a
    ``model`` attribute, so add that prefix only when the checkpoint does not
    already contain it.
    """
    if checkpoint.suffix.lower() == ".safetensors":
        try:
            from safetensors.torch import load_file
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Loading a .safetensors DiffSynth fallback checkpoint requires safetensors"
            ) from exc
        state = load_file(str(checkpoint), device="cpu")
    else:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if isinstance(state, dict):
        for container_key in ("model_state", "state_dict"):
            nested = state.get(container_key)
            if isinstance(nested, dict):
                state = nested
                break
    if not isinstance(state, dict):
        raise TypeError("DiffSynth fallback checkpoint must be a state_dict mapping")

    converted: dict[str, torch.Tensor] = {}
    for raw_key, value in state.items():
        if not isinstance(raw_key, str) or not torch.is_tensor(value):
            continue
        key = raw_key.removeprefix("module.")
        converted[key if key.startswith("model.") else f"model.{key}"] = value
    if not converted:
        raise TypeError("DiffSynth fallback checkpoint contains no tensor parameters")
    return converted


class WanVAEBackbone(nn.Module):
    """Frozen Wan video VAE with the official Wan2.1 implementation as default.

    The adapter accepts Gen2Act video layout ``[B,T,3,H,W]`` in ``[0,1]`` and
    produces a deterministic latent ``[B,16,T_lat,H/8,W/8]``. ``backend`` is
    deliberately explicit: use the official `Wan-Video/Wan2.1` code by default,
    choose Diffusers only for a verified converted checkpoint, and leave
    DiffSynth-Studio as a compatibility fallback.
    """

    encoder_kind = "wan_vae"

    def __init__(
        self,
        checkpoint_path: str = "",
        *,
        hidden_dim: int = 768,
        checkpoint_env: str = "WAN_VAE_CHECKPOINT",
        backend: _WanBackend = "official",
        fallback_backends: tuple[str, ...] | list[str] | str = (),
        official_root: str = "",
        official_root_env: str = "WAN21_ROOT",
        diffusers_model_id: str = "",
        diffsynth_root: str = "",
        diffsynth_root_env: str = "DIFFSYNTH_ROOT",
        dtype: str = "bfloat16",
        latent_dim: int = 16,
    ) -> None:
        super().__init__()
        if int(latent_dim) != 16:
            raise ValueError("Wan 2.1 VAE has a fixed 16-channel latent")
        checkpoint = resolve_wan_checkpoint(checkpoint_path, checkpoint_env)
        selected = str(backend).lower()
        if isinstance(fallback_backends, str):
            fallback_backends = (fallback_backends,)
        fallbacks = tuple(str(name).lower() for name in fallback_backends)
        valid = {"official", "diffusers", "diffsynth"}
        if selected not in valid or any(name not in valid for name in fallbacks):
            raise ValueError(f"Wan-VAE backend must be one of {sorted(valid)}")
        if selected in fallbacks:
            raise ValueError("Wan-VAE fallback_backends must not repeat backend")
        if "diffusers" in (selected, *fallbacks) and not str(diffusers_model_id).strip():
            raise ValueError(
                "Diffusers Wan-VAE backend/fallback requires model.backbone.diffusers_model_id"
            )
        if selected != "diffusers" and checkpoint is None:
            raise ValueError(
                "Wan-VAE checkpoint is required. Set model.backbone.local_checkpoint "
                f"or {checkpoint_env}."
            )
        if checkpoint is not None and not checkpoint.is_file():
            raise FileNotFoundError(f"Wan-VAE checkpoint not found: {checkpoint}")
        self.name = "wan_vae"
        self.hidden_dim = int(hidden_dim)
        self.latent_dim = int(latent_dim)
        self.checkpoint_path = str(checkpoint) if checkpoint is not None else ""
        self.compute_dtype = _torch_dtype(dtype)
        self.vae: nn.Module
        self.backend: _WanBackend
        self._official_scale: tuple[torch.Tensor, torch.Tensor] | None = None
        self._init_backend(
            selected, fallbacks, checkpoint, official_root, official_root_env,
            str(diffusers_model_id), diffsynth_root, diffsynth_root_env)
        self.vae.eval()
        self.vae.requires_grad_(False)

    def _init_backend(
        self,
        primary: str,
        fallbacks: tuple[str, ...],
        checkpoint: Path | None,
        official_root: str,
        official_root_env: str,
        diffusers_model_id: str,
        diffsynth_root: str,
        diffsynth_root_env: str,
    ) -> None:
        failures: list[str] = []
        for candidate in (primary, *fallbacks):
            try:
                if candidate == "official":
                    assert checkpoint is not None
                    vae_class = _import_official_wan(official_root, official_root_env)
                    self.vae = vae_class(
                        z_dim=self.latent_dim, vae_pth=str(checkpoint),
                        dtype=self.compute_dtype, device="cpu").model
                    self.vae.to(dtype=self.compute_dtype)
                    mean = torch.tensor(_WAN21_MEAN, dtype=self.compute_dtype)
                    inv_std = torch.tensor(_WAN21_STD, dtype=self.compute_dtype).reciprocal()
                    self._official_scale = (mean, inv_std)
                elif candidate == "diffusers":
                    from diffusers import AutoencoderKLWan

                    self.vae = AutoencoderKLWan.from_pretrained(
                        diffusers_model_id, subfolder="vae", torch_dtype=self.compute_dtype,
                        local_files_only=True)
                    self._official_scale = None
                else:
                    assert checkpoint is not None
                    vae_class = _import_diffsynth_wan(diffsynth_root, diffsynth_root_env)
                    self.vae = vae_class(z_dim=self.latent_dim)
                    self.vae.load_state_dict(_load_diffsynth_state_dict(checkpoint), strict=True)
                    self.vae.to(dtype=self.compute_dtype)
                    self._official_scale = None
                self.backend = candidate  # type: ignore[assignment]
                if candidate != primary:
                    warnings.warn(
                        f"Wan-VAE backend={primary!r} failed; using fallback={candidate!r}. "
                        + " | ".join(failures),
                        RuntimeWarning,
                        stacklevel=2,
                    )
                return
            except Exception as exc:
                failures.append(f"{candidate}: {type(exc).__name__}: {exc}")
        raise RuntimeError("Could not initialize Wan-VAE. Tried " + " | ".join(failures))

    def train(self, mode: bool = True) -> WanVAEBackbone:
        super().train(mode)
        self.vae.eval()
        return self

    def _encode_official(self, pixels: torch.Tensor) -> torch.Tensor:
        assert self._official_scale is not None
        mean, inv_std = (value.to(pixels.device) for value in self._official_scale)
        return self.vae.encode(pixels, [mean, inv_std])

    def _encode_diffusers(self, pixels: torch.Tensor) -> torch.Tensor:
        encoded = self.vae.encode(pixels)
        latent_dist = getattr(encoded, "latent_dist", encoded)
        latent = getattr(latent_dist, "mean", None)
        if latent is None:
            raise RuntimeError("Diffusers AutoencoderKLWan.encode() did not return latent_dist.mean")

        def latent_parameter(value):
            tensor = torch.as_tensor(value, device=latent.device, dtype=latent.dtype)
            if tensor.ndim == 1:
                if tensor.numel() != latent.shape[1]:
                    raise RuntimeError(
                        "Diffusers Wan-VAE latent normalization has an unexpected channel count"
                    )
                tensor = tensor.view(1, -1, 1, 1, 1)
            return tensor

        config = getattr(self.vae, "config", None)
        mean = getattr(config, "latents_mean", None)
        std = getattr(config, "latents_std", None)
        if mean is not None and std is not None:
            return (latent - latent_parameter(mean)) / latent_parameter(std)

        # Retain compatibility with older/generic AutoencoderKL configs.
        shift = getattr(config, "shift_factor", None)
        scaling = getattr(config, "scaling_factor", None)
        if shift is not None and scaling is not None:
            latent = (latent - latent_parameter(shift)) * latent_parameter(scaling)
        return latent

    def _encode_diffsynth(self, pixels: torch.Tensor) -> torch.Tensor:
        # DiffSynth stores scale as ordinary CPU tensors instead of buffers, so
        # ``module.to(device)`` does not move them with the VAE.
        scale = [
            value.to(device=pixels.device, dtype=pixels.dtype)
            for value in self.vae.scale
        ]
        return self.vae.model.encode(pixels, scale)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        if video.ndim != 5 or video.shape[2] != 3:
            raise ValueError(f"Wan-VAE expects [B,T,3,H,W], got {tuple(video.shape)}")
        if video.shape[-2] % 8 or video.shape[-1] % 8:
            raise ValueError(
                f"Wan-VAE spatial dimensions must be divisible by 8, got {tuple(video.shape[-2:])}")
        pixels = video.transpose(1, 2).to(dtype=self.compute_dtype).mul(2.0).sub(1.0)
        with torch.no_grad():
            if self.backend == "official":
                latent = self._encode_official(pixels)
            elif self.backend == "diffusers":
                latent = self._encode_diffusers(pixels)
            else:
                latent = self._encode_diffsynth(pixels)
        if latent.ndim != 5 or latent.shape[1] != self.latent_dim:
            raise RuntimeError(
                f"Wan-VAE returned {tuple(latent.shape)}; expected [B,{self.latent_dim},T,H,W]")
        return latent
