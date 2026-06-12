from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from r2r_gen2act.config.schema import validate_config
from r2r_gen2act.data.factories import build_action_codec, build_dataset
from r2r_gen2act.modeling.factory import build_policy
from r2r_gen2act.training.checkpoint import save_checkpoint
from r2r_gen2act.training.losses import compute_losses
from r2r_gen2act.training.seed import seed_everything


def _move_batch(batch: dict, device: torch.device) -> dict:
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device) if torch.is_tensor(v) else v
    return out


def _param_groups(model, cfg: dict):
    opt_cfg = cfg["train"].get("optimizer", {})
    lr = float(opt_cfg.get("lr", 1e-4))
    wd = float(opt_cfg.get("weight_decay", 0.05))
    mult = float(opt_cfg.get("backbone_lr_multiplier", 0.1))
    vit = getattr(model, "vit", None)
    vit_params = list(vit.parameters()) if vit is not None else []
    vit_ids = {id(p) for p in vit_params}
    other_params = [p for p in model.parameters() if id(p) not in vit_ids and p.requires_grad]
    groups = []
    vit_trainable = [p for p in vit_params if p.requires_grad]
    if vit_trainable:
        groups.append({"params": vit_trainable, "lr": lr * mult, "weight_decay": wd})
    if other_params:
        groups.append({"params": other_params, "lr": lr, "weight_decay": wd})
    return groups


def _write_loss_history(out_dir: Path, history: list[dict]) -> None:
    (out_dir / "loss_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    if history:
        preferred = ["epoch", "train_loss", "val_loss", "train_action_loss", "val_action_loss", "train_gripper_loss", "val_gripper_loss", "train_terminate_loss", "val_terminate_loss"]
        dynamic = sorted({k for row in history for k in row if k not in preferred})
        keys = preferred + dynamic
        lines = [",".join(keys)]
        for row in history:
            lines.append(",".join(str(row.get(k, "")) for k in keys))
        (out_dir / "loss_history.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[warn] Could not plot loss curve: {exc}")
        return
    epochs = [row["epoch"] for row in history]
    if not epochs:
        return
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, [row["train_loss"] for row in history], label="train/loss")
    plt.plot(epochs, [row["val_loss"] for row in history], label="val/loss")
    plt.plot(epochs, [row["train_action_loss"] for row in history], label="train/action", alpha=0.7)
    plt.plot(epochs, [row["val_action_loss"] for row in history], label="val/action", alpha=0.7)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Robot2Robot Gen2Act loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "loss_curve.png", dpi=160)
    plt.close()

    dim_keys = sorted(k for k in history[0] if k.startswith("train_action_dim_") and k.endswith("_loss"))
    if dim_keys:
        plt.figure(figsize=(12, 7))
        for train_key in dim_keys:
            suffix = train_key[len("train_"):]
            val_key = "val_" + suffix
            dim_name = suffix.replace("action_dim_", "dim ").replace("_loss", "")
            plt.plot(epochs, [row[train_key] for row in history], label=f"train/{dim_name}")
            if val_key in history[0]:
                plt.plot(epochs, [row[val_key] for row in history], linestyle="--", label=f"val/{dim_name}")
        plt.xlabel("epoch")
        plt.ylabel("cross entropy")
        plt.title("Per-dimension action loss")
        plt.grid(True, alpha=0.3)
        plt.legend(ncol=2, fontsize=8)
        plt.tight_layout()
        plt.savefig(out_dir / "loss_curve_per_dim.png", dpi=160)
        plt.close()


def run_epoch(model, loader, codec, cfg, device, optimizer=None, train: bool = True) -> dict[str, float]:
    model.train(train)
    totals: dict[str, float] = {}
    steps = 0
    amp_enabled = bool(cfg["train"].get("amp", True)) and device.type == "cuda"
    for batch in loader:
        batch = _move_batch(batch, device)
        with torch.set_grad_enabled(train):
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                proprioception = batch.get("proprioception")
                outputs = model(batch.get("source_video"), batch.get("target_history"), proprioception)
                losses = compute_losses(outputs, batch, codec, cfg)
                if not torch.isfinite(losses["loss"]):
                    raise FloatingPointError(f"Non-finite loss in {'train' if train else 'val'} epoch step {steps}: {float(losses['loss'].detach().cpu())}")
            if train:
                optimizer.zero_grad(set_to_none=True)
                losses["loss"].backward()
                grad_clip = float(cfg["train"].get("grad_clip_norm", 1.0))
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
        for k, v in losses.items():
            totals[k] = totals.get(k, 0.0) + float(v.detach().cpu())
        steps += 1
    denom = max(1, steps)
    return {k: v / denom for k, v in totals.items()}


def train(cfg: dict, device: str | None = None) -> Path:
    validate_config(cfg)
    if bool(cfg["train"].get("debug", {}).get("anomaly_detection", False)):
        torch.autograd.set_detect_anomaly(True)
    seed_everything(int(cfg["train"].get("seed", 42)))
    device_obj = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    train_ds = build_dataset(cfg, "train")
    val_ds = build_dataset(cfg, "val")
    if not set(e.episode_id for e in train_ds.episodes).isdisjoint(set(e.episode_id for e in val_ds.episodes)):
        raise RuntimeError("Train/val episode split overlaps")
    codec = build_action_codec(cfg)
    train_loader = DataLoader(train_ds, batch_size=int(cfg["train"]["batch_size"]), shuffle=bool(cfg["train"].get("shuffle", True)), num_workers=int(cfg["train"].get("num_workers", 0)), pin_memory=device_obj.type == "cuda")
    val_loader = DataLoader(val_ds, batch_size=int(cfg["train"]["batch_size"]), shuffle=False, num_workers=int(cfg["train"].get("num_workers", 0)), pin_memory=device_obj.type == "cuda")
    model = build_policy(cfg).to(device_obj)
    opt_cfg = cfg["train"].get("optimizer", {})
    optimizer = torch.optim.AdamW(_param_groups(model, cfg), betas=tuple(opt_cfg.get("betas", [0.9, 0.95])))
    out_dir = Path(cfg["experiment"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config_snapshot.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print(f"train_episodes={len(train_ds.episodes)} val_episodes={len(val_ds.episodes)} train_windows={len(train_ds)} val_windows={len(val_ds)}")
    best = float("inf")
    latest_path = out_dir / "latest.pt"
    history: list[dict] = []
    for epoch in range(1, int(cfg["train"].get("epochs", 1)) + 1):
        t0 = time.time()
        train_metrics = run_epoch(model, train_loader, codec, cfg, device_obj, optimizer, train=True)
        with torch.no_grad():
            val_metrics = run_epoch(model, val_loader, codec, cfg, device_obj, None, train=False)
        metrics = {"train": train_metrics, "val": val_metrics}
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
            "train_action_loss": train_metrics["action_loss"],
            "val_action_loss": val_metrics["action_loss"],
            "train_gripper_loss": train_metrics["gripper_loss"],
            "val_gripper_loss": val_metrics["gripper_loss"],
            "train_terminate_loss": train_metrics["terminate_loss"],
            "val_terminate_loss": val_metrics["terminate_loss"],
        }
        for key, value in train_metrics.items():
            if key.startswith("action_dim_") or key in ("action_mae", "action_rmse"):
                row[f"train_{key}"] = value
        for key, value in val_metrics.items():
            if key.startswith("action_dim_") or key in ("action_mae", "action_rmse"):
                row[f"val_{key}"] = value
        history.append(row)
        _write_loss_history(out_dir, history)
        print(f"epoch={epoch} train_loss={train_metrics['loss']:.4f} val_loss={val_metrics['loss']:.4f} time={time.time()-t0:.1f}s loss_plot={out_dir / 'loss_curve.png'}")
        save_checkpoint(latest_path, model, optimizer, cfg, epoch, metrics)
        if val_metrics["loss"] < best:
            best = val_metrics["loss"]
            save_checkpoint(out_dir / "best.pt", model, optimizer, cfg, epoch, metrics)
    return latest_path
