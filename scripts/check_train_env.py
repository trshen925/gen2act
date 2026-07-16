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
    "numpy": "numpy",
    "scipy": "scipy",
    "imageio": "imageio",
    "imageio_ffmpeg": "imageio-ffmpeg",
    "PIL": "pillow",
    "pyarrow": "pyarrow",
    "yaml": "pyyaml",
    "matplotlib": "matplotlib",
}


def _version(module_name: str) -> str:
    package_name = REQUIREMENTS[module_name].split(">", 1)[0]
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
    return sorted(set(missing)), errors


def _install(requirements: list[str]) -> None:
    if not requirements:
        return
    print("[preflight] installing missing packages into", sys.executable, flush=True)
    print("[preflight] packages:", " ".join(requirements), flush=True)
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--upgrade", *requirements]
    )


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
        print("[preflight] dinov3_weights=" + weight_cache)


if __name__ == "__main__":
    main()
