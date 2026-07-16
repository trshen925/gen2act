from __future__ import annotations

from pathlib import Path

import torch


def save_checkpoint(path: str | Path, model, optimizer, cfg: dict, epoch: int, metrics: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "config": cfg,
        "metrics": metrics,
    }, path)


def load_checkpoint(path: str | Path, model, device, strict: bool = True) -> dict:
    ckpt = torch.load(path, map_location=device)
    state = ckpt["model_state_dict"]
    if strict:
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

