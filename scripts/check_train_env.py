from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import os
import subprocess
import sys
from pathlib import Path


REQUIREMENTS = {
    "torch": "torch",
    "torchvision": "torchvision",
    "timm": "timm>=1.0.27",
    "huggingface_hub": "huggingface-hub",
    "einops": "einops",
    "safetensors": "safetensors",
    "tqdm": "tqdm",
    "numpy": "numpy",
    "scipy": "scipy",
    "imageio": "imageio",
    "imageio_ffmpeg": "imageio-ffmpeg",
    "PIL": "pillow",
    "pyarrow": "pyarrow",
    "h5py": "h5py",
    "yaml": "pyyaml",
    "matplotlib": "matplotlib",
    "wandb": "wandb==0.20.1",
}


def _version(module_name: str) -> str:
    package_name = REQUIREMENTS[module_name].split("=", 1)[0].split(">", 1)[0]
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _missing_requirements() -> tuple[list[str], list[str]]:
    missing: list[str] = []
    errors: list[str] = []
    for module_name, requirement in REQUIREMENTS.items():
        try:
            importlib.import_module(module_name)
        except Exception as exc:
            missing.append(requirement)
            errors.append(f"{module_name}: {exc}")

    if "timm>=1.0.27" not in missing:
        import timm

        if "vit_large_patch16_dinov3" not in timm.list_models():
            missing.append("timm>=1.0.27")
            errors.append("timm: vit_large_patch16_dinov3 is not registered")
    if "wandb==0.20.1" not in missing and _version("wandb") != "0.20.1":
        missing.append("wandb==0.20.1")
        errors.append(f"wandb: expected 0.20.1, found {_version('wandb')}")
    return sorted(set(missing)), errors


def _install(requirements: list[str]) -> None:
    if not requirements:
        return
    print("[preflight] installing missing packages into", sys.executable, flush=True)
    print("[preflight] packages:", " ".join(requirements), flush=True)
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--upgrade", *requirements]
    )


def _wan_backend_error(backend: str, backbone_cfg: dict) -> str | None:
    """Return a diagnostic if a configured Wan backend cannot be imported.

    This intentionally checks one backend at a time: a missing optional
    fallback must not prevent training when the configured primary backend is
    usable.
    """
    if backend in {"official", "diffsynth"}:
        from r2r_gen2act.modeling.wan_vae import resolve_wan_checkpoint

        checkpoint_env = str(backbone_cfg.get("checkpoint_env", "WAN_VAE_CHECKPOINT"))
        checkpoint = resolve_wan_checkpoint(
            str(backbone_cfg.get("local_checkpoint", "") or ""), checkpoint_env
        )
        if checkpoint is None or not checkpoint.is_file():
            return (
                "Wan-VAE weights are unavailable; set model.backbone.local_checkpoint "
                f"or {checkpoint_env}"
            )
    if backend == "official":
        from r2r_gen2act.modeling.wan_vae import _import_official_wan

        try:
            _import_official_wan(
                str(backbone_cfg.get("official_root", "") or ""),
                str(backbone_cfg.get("official_root_env", "WAN21_ROOT")),
            )
        except Exception as exc:
            return (
                "official Wan2.1 code is unavailable; set WAN21_ROOT to the official "
                f"Wan2.1 repository root or install its `wan` package ({type(exc).__name__}: {exc})"
            )
        return None
    if backend == "diffusers":
        try:
            importlib.import_module("diffusers")
        except Exception as exc:
            return f"Diffusers is unavailable ({type(exc).__name__}: {exc})"
        return None
    if backend == "diffsynth":
        from r2r_gen2act.modeling.wan_vae import _import_diffsynth_wan

        try:
            _import_diffsynth_wan(
                str(backbone_cfg.get("diffsynth_root", "") or ""),
                str(backbone_cfg.get("diffsynth_root_env", "DIFFSYNTH_ROOT")),
            )
        except Exception as exc:
            return f"DiffSynth-Studio is unavailable ({type(exc).__name__}: {exc})"
        return None
    return f"unsupported backend {backend!r}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the Gen2Act training environment")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-gpus", type=int, default=1)
    parser.add_argument("--install", action="store_true")
    args = parser.parse_args()

    if sys.version_info < (3, 11):
        raise SystemExit(f"Python >=3.11 is required; found {sys.version.split()[0]}")

    missing, errors = _missing_requirements()
    if missing and args.install:
        for error in errors:
            print(f"[preflight] missing/broken: {error}")
        _install(missing)
        # Re-exec with a clean interpreter so newly installed binary modules are
        # not affected by partial imports from the first check.
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--config",
            str(args.config),
            "--expected-gpus",
            str(args.expected_gpus),
        ]
        raise SystemExit(subprocess.call(cmd))
    if missing:
        for error in errors:
            print(f"[preflight] missing/broken: {error}")
        print("[preflight] rerun with --install to install:", " ".join(missing))
        raise SystemExit(2)

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available in this Python environment/job")
    visible_gpus = torch.cuda.device_count()
    if visible_gpus != args.expected_gpus:
        raise SystemExit(
            f"GPU visibility mismatch: launcher selected {args.expected_gpus}, "
            f"but torch sees {visible_gpus}"
        )
    if args.expected_gpus > 1 and not torch.distributed.is_nccl_available():
        raise SystemExit("NCCL is unavailable; multi-GPU CUDA DDP cannot start")

    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from r2r_gen2act.config.load import load_config
    from r2r_gen2act.config.schema import validate_config

    cfg = load_config(args.config)
    validate_config(cfg)
    amp_dtype = str(cfg["train"].get("amp_dtype", "bfloat16")).lower()
    if bool(cfg["train"].get("amp", True)) and amp_dtype in ("bfloat16", "bf16"):
        if not torch.cuda.is_bf16_supported():
            raise SystemExit("this GPU does not support the configured bfloat16 AMP training")
    data_root = Path(cfg["data"].get("root") or cfg["data"].get("hdf5_path") or "")
    if not data_root.exists():
        raise SystemExit(f"dataset path does not exist on this node: {data_root}")

    backbone = str(cfg["model"].get("backbone", {}).get("name", ""))
    weight_cache = ""
    backbone_cfg = cfg["model"].get("backbone", {}) or {}
    if backbone.startswith("dinov3") and bool(backbone_cfg.get("pretrained", True)):
        if not str(backbone_cfg.get("local_checkpoint", "") or ""):
            from huggingface_hub import snapshot_download

            try:
                weight_cache = snapshot_download(
                    "timm/vit_large_patch16_dinov3.lvd1689m",
                    local_files_only=True,
                )
            except Exception:
                if os.environ.get("HF_HUB_OFFLINE", "0").lower() in ("1", "true", "yes"):
                    raise SystemExit(
                        "DINOv3-L weights are absent from the HF cache while HF_HUB_OFFLINE=1"
                    )
                weight_cache = "not cached; rank 0 will download before training"
    if backbone.lower() in {"wan_vae", "wan2.1_vae", "wan2_1_vae", "wan_video_vae"}:
        from r2r_gen2act.modeling.wan_vae import resolve_wan_checkpoint

        checkpoint = resolve_wan_checkpoint(
            str(backbone_cfg.get("local_checkpoint", "") or ""),
            str(backbone_cfg.get("checkpoint_env", "WAN_VAE_CHECKPOINT")),
        )
        backend = str(backbone_cfg.get("backend", "official")).lower()
        fallbacks = backbone_cfg.get("fallback_backends", [])
        if isinstance(fallbacks, str):
            fallbacks = [fallbacks]
        primary_error = _wan_backend_error(backend, backbone_cfg)
        if primary_error:
            raise SystemExit(f"Wan-VAE primary backend={backend!r}: {primary_error}")
        for fallback in (str(name).lower() for name in fallbacks):
            fallback_error = _wan_backend_error(fallback, backbone_cfg)
            if fallback_error:
                print(
                    f"[preflight] warning: Wan-VAE fallback backend={fallback!r} is unavailable: "
                    f"{fallback_error}",
                    file=sys.stderr,
                )
        weight_cache = (
            str(checkpoint)
            if checkpoint is not None
            else f"diffusers:{backbone_cfg.get('diffusers_model_id')} (local cache only)"
        )
    per_gpu_batch = int(cfg["train"]["batch_size"])
    gpu_desc = []
    for index in range(visible_gpus):
        props = torch.cuda.get_device_properties(index)
        gpu_desc.append(f"{index}:{props.name}({props.total_memory / 2**30:.0f}GiB)")
    print(
        f"[preflight] OK python={sys.version.split()[0]} torch={torch.__version__} "
        f"cuda={torch.version.cuda} gpus={visible_gpus} nccl={torch.distributed.is_nccl_available()}"
    )
    print(
        f"[preflight] config={args.config} backbone={backbone} data_root={data_root} "
        f"batch={per_gpu_batch}x{visible_gpus}={per_gpu_batch * visible_gpus} "
        f"timm={_version('timm')} pyarrow={_version('pyarrow')}"
    )
    if cfg["data"].get("max_episodes") not in (None, ""):
        print(f"[preflight] dataset is frozen to sorted first {int(cfg['data']['max_episodes'])} candidate episodes")
    print("[preflight] devices=" + ", ".join(gpu_desc))
    if weight_cache:
        weight_label = "dinov3_weights" if backbone.startswith("dinov3") else f"{backbone}_weights"
        print(f"[preflight] {weight_label}=" + weight_cache)


if __name__ == "__main__":
    main()
