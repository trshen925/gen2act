from __future__ import annotations

from pathlib import Path

import torch


def save_checkpoint(
    path: str | Path,
    model,
    optimizer,
    cfg: dict,
    epoch: int,
    metrics: dict,
    *,
    ema_state_dict: dict[str, torch.Tensor] | None = None,
    ema_decay: float | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    training_state = model.state_dict()
    checkpoint = {
        "epoch": epoch,
        # Inference/export always sees the smoothed weights when EMA is enabled.
        "model_state_dict": ema_state_dict if ema_state_dict is not None else training_state,
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "config": cfg,
        "metrics": metrics,
    }
    if ema_state_dict is not None:
        checkpoint["training_model_state_dict"] = training_state
        checkpoint["ema_state_dict"] = ema_state_dict
        checkpoint["ema_decay"] = float(ema_decay) if ema_decay is not None else None
    torch.save(checkpoint, path)


def save_slim_checkpoint(path: str | Path, checkpoint: dict) -> None:
    """Save an eval/deploy checkpoint without optimizer state."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    slim = {
        "epoch": checkpoint.get("epoch"),
        "model_state_dict": checkpoint["model_state_dict"],
        "config": checkpoint.get("config"),
        "metrics": checkpoint.get("metrics"),
        "slim_checkpoint": True,
    }
    torch.save(slim, path)


def load_checkpoint(
    path: str | Path,
    model,
    device,
    strict: bool = True,
    exclude_prefixes: tuple[str, ...] = (),
) -> dict:
    ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    else:
        # Also support plain state_dict files for deployment/export tooling.
        state = ckpt
        ckpt = {"model_state_dict": state}
    compat = getattr(model, "checkpoint_state_dict_compat", None)
    if callable(compat):
        state = compat(state)
    if exclude_prefixes:
        state = {k: v for k, v in state.items() if not k.startswith(exclude_prefixes)}
    if strict and not exclude_prefixes:
        model.load_state_dict(state, strict=True)
        return ckpt
    # Non-strict: load matching-shape params directly; for shape-mismatched params (e.g.
    # source_time_embed [8,1,d] → [16,1,d] when growing the frame count), copy the overlapping
    # prefix along each dim so learned positions are preserved and new ones stay random-init.
    own = model.state_dict()
    to_load, grown = {}, []
    for k, v in state.items():
        if k not in own:
            continue
        if own[k].shape == v.shape:
            to_load[k] = v
        else:
            dst = own[k].clone()
            slices = tuple(slice(0, min(a, b)) for a, b in zip(dst.shape, v.shape))
            dst[slices] = v[slices]
            to_load[k] = dst
            grown.append(f"{k}: {tuple(v.shape)}→{tuple(own[k].shape)}")
    missing = [k for k in own if k not in to_load]
    model.load_state_dict(to_load, strict=False)
    if grown:
        print(f"[load_checkpoint] prefix-copied shape-mismatched params: {grown}")
    if missing:
        print(f"[load_checkpoint] left random-init (not in ckpt): {missing}")
    return ckpt
