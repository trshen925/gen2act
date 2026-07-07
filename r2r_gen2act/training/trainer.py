from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from r2r_gen2act.config.schema import validate_config
from r2r_gen2act.data.factories import build_action_codec, build_dataset
from r2r_gen2act.modeling.factory import build_policy
from r2r_gen2act.training.checkpoint import save_checkpoint
from r2r_gen2act.training.losses import compute_losses
from r2r_gen2act.training.seed import seed_everything


def _init_distributed() -> tuple[int, int, int, bool]:
    """Read torchrun env. Returns (rank, local_rank, world_size, is_dist). Single-process when
    not launched by torchrun (WORLD_SIZE<=1)."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    is_dist = world_size > 1
    if is_dist and not dist.is_initialized():
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    return rank, local_rank, world_size, is_dist


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
    llrd = float(opt_cfg.get("llrd", 1.0))
    vit = getattr(model, "vit", None)
    if vit is not None and llrd < 1.0 - 1e-9:
        return _llrd_param_groups(model, vit, lr, wd, llrd, mult)
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


def _llrd_param_groups(model, vit, base_lr: float, wd: float, llrd: float, backbone_mult: float = 1.0):
    """Layer-wise LR decay over the (fully unfrozen) DINOv2 backbone: the deepest block gets
    base_lr*backbone_mult, each shallower block decays by *llrd; non-block backbone params
    (patch/pos-embed, final norm) get the lowest lr; everything above the backbone
    (resampler/fusion/decoder/head/proprio/embeds) gets full base_lr. backbone_mult keeps the
    whole backbone gentle (e.g. 0.3) while the randomly-init upper modules learn at base_lr.
    Norms/biases/embeddings get no weight decay."""
    backend = getattr(vit, "backend", None)
    blocks = list(getattr(backend, "blocks", [])) if backend is not None else []
    n_blocks = len(blocks)
    block_of = {}
    for i, blk in enumerate(blocks):
        for p in blk.parameters():
            block_of[id(p)] = i
    vit_ids = {id(p) for p in vit.parameters()}
    groups: dict[tuple, dict] = {}
    no_wd_keys = ("pos_embed", "cls_token", "reg_token", ".token")
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        pid = id(p)
        if pid in block_of:
            scale = backbone_mult * (llrd ** (n_blocks - 1 - block_of[pid]))
        elif pid in vit_ids:
            scale = backbone_mult * (llrd ** n_blocks)  # patch_embed / pos_embed / tokens / norms -> lowest
        else:
            scale = 1.0  # upper modules (non-backbone)
        no_wd = p.ndim <= 1 or any(k in name for k in no_wd_keys)
        key = (round(scale, 10), no_wd)
        g = groups.setdefault(key, {"params": [], "lr": base_lr * scale, "weight_decay": 0.0 if no_wd else wd})
        g["params"].append(p)
    return list(groups.values())


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
    if not history:
        return

    def series(key: str):
        # (epochs, values) keeping only rows where the key is present (val/inference is periodic).
        xs, ys = [], []
        for row in history:
            v = row.get(key)
            if v not in (None, ""):
                xs.append(row["epoch"])
                ys.append(v)
        return xs, ys

    # Total loss: train (every epoch) vs inference (held-out 100 cases, periodic).
    plt.figure(figsize=(10, 6))
    xt, yt = series("train_loss")
    xi, yi = series("val_loss")
    plt.plot(xt, yt, "-", color="C0", label="train")
    if xi:
        plt.plot(xi, yi, "o-", color="C1", label="inference (held-out)")
    xa, ya = series("train_action_loss")
    xia, yia = series("val_action_loss")
    plt.plot(xa, ya, "--", color="C0", alpha=0.6, label="train/action")
    if xia:
        plt.plot(xia, yia, "s--", color="C1", alpha=0.6, label="inference/action")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Robot2Robot Gen2Act — train vs inference loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "loss_curve.png", dpi=160)
    plt.close()

    dim_keys = sorted(k for k in history[0] if k.startswith("train_action_dim_") and k.endswith("_mae")) or \
        sorted(k for k in history[0] if k.startswith("train_action_dim_") and k.endswith("_loss"))
    if dim_keys:
        plt.figure(figsize=(12, 7))
        for i, train_key in enumerate(dim_keys):
            suffix = train_key[len("train_"):]
            dim_name = suffix.replace("action_dim_", "dim ").replace("_loss", "").replace("_mae", "")
            xtr, ytr = series(train_key)
            plt.plot(xtr, ytr, "-", color=f"C{i % 10}", label=f"train/{dim_name}")
            xv, yv = series("val_" + suffix)
            if xv:
                plt.plot(xv, yv, "o--", color=f"C{i % 10}", alpha=0.6)
        plt.xlabel("epoch")
        plt.ylabel("per-dim action error")
        plt.title("Per-dimension action error (solid=train, dashed=inference)")
        plt.grid(True, alpha=0.3)
        plt.legend(ncol=2, fontsize=8)
        plt.tight_layout()
        plt.savefig(out_dir / "loss_curve_per_dim.png", dpi=160)
        plt.close()


def _amp_dtype(cfg: dict) -> torch.dtype:
    # Default to bf16: the trainer has no GradScaler, so fp16 autocast would be unsafe.
    name = str(cfg["train"].get("amp_dtype", "bfloat16")).lower()
    return {"bfloat16": torch.bfloat16, "bf16": torch.bfloat16, "float16": torch.float16, "fp16": torch.float16}.get(name, torch.bfloat16)


def _build_scheduler(optimizer, cfg: dict, steps_per_epoch: int):
    sch_cfg = cfg["train"].get("scheduler", {})
    name = str(sch_cfg.get("name", "none")).lower()
    if name in ("none", ""):
        return None
    total = max(1, int(cfg["train"].get("epochs", 1)) * max(1, steps_per_epoch))
    warmup = int(sch_cfg.get("warmup_steps", 0))
    if sch_cfg.get("warmup_ratio") is not None:
        warmup = int(float(sch_cfg["warmup_ratio"]) * total)
    warmup = max(0, min(warmup, total - 1))
    min_ratio = float(sch_cfg.get("min_lr_ratio", 0.0))

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(1, warmup)
        progress = (step - warmup) / max(1, total - warmup)
        if name == "cosine":
            return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        if name == "linear":
            return min_ratio + (1.0 - min_ratio) * max(0.0, 1.0 - progress)
        return 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def run_epoch(model, loader, codec, cfg, device, optimizer=None, train: bool = True, scheduler=None, world_size: int = 1) -> dict[str, float]:
    model.train(train)
    totals: dict[str, float] = {}
    steps = 0
    amp_enabled = bool(cfg["train"].get("amp", True)) and device.type == "cuda"
    amp_dtype = _amp_dtype(cfg)
    for batch in loader:
        batch = _move_batch(batch, device)
        with torch.set_grad_enabled(train):
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                proprioception = batch.get("proprioception")
                action_target = None
                if train and str(cfg.get("action", {}).get("mode", "")) == "flow":
                    # flow-matching head needs the normalized pose chunk to build the flow target;
                    # eval samples from noise so no target is passed.
                    pose_dims = codec.pose_dims
                    action_target = codec.normalize(batch["action"][..., :pose_dims].to(device))
                point_track = batch.get("point_track")
                extra = {}
                ptc = batch.get("point_track_causal")
                if ptc is not None:
                    extra["point_track_causal"] = ptc
                # C11 depth 3D lifting: pass depth frames + camera intrinsics when available
                dv = batch.get("depth_video")
                if dv is not None:
                    extra["depth_video"] = dv.float()
                ck = batch.get("camera_K")
                if ck is not None:
                    extra["camera_K"] = ck
                outputs = model(batch.get("source_video"), batch.get("target_history"), proprioception, action_target, point_track, **extra)
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
                if scheduler is not None:
                    scheduler.step()
        for k, v in losses.items():
            totals[k] = totals.get(k, 0.0) + float(v.detach().cpu())
        steps += 1
    if world_size > 1 and dist.is_initialized():
        # Average metrics across ranks so logged loss reflects the global batch.
        keys = sorted(totals.keys())
        packed = torch.tensor([totals[k] for k in keys] + [float(steps)], device=device, dtype=torch.float64)
        dist.all_reduce(packed, op=dist.ReduceOp.SUM)
        totals = {k: packed[i].item() for i, k in enumerate(keys)}
        steps = packed[-1].item()
    denom = max(1.0, float(steps))
    return {k: v / denom for k, v in totals.items()}


def train(cfg: dict, device: str | None = None) -> Path:
    validate_config(cfg)
    rank, local_rank, world_size, is_dist = _init_distributed()
    is_main = rank == 0
    if bool(cfg["train"].get("debug", {}).get("anomaly_detection", False)):
        torch.autograd.set_detect_anomaly(True)
    seed_everything(int(cfg["train"].get("seed", 42)) + rank)
    if torch.cuda.is_available():
        if device and device not in ("cuda", "gpu"):
            device_obj = torch.device(device)
        else:
            device_obj = torch.device(f"cuda:{local_rank % torch.cuda.device_count()}")
        torch.cuda.set_device(device_obj)
    else:
        device_obj = torch.device(device) if device else torch.device("cpu")

    train_ds = build_dataset(cfg, "train")
    val_ds = build_dataset(cfg, "val")
    if len(train_ds) == 0:
        root = cfg["data"].get("root") or cfg["data"].get("hdf5_path")
        eff = int(cfg["data"].get("future_horizon", 0)) * max(1, int(cfg.get("action", {}).get("chunk_size", 1)))
        raise RuntimeError(
            f"No training windows: train_episodes={len(train_ds.episodes)}, windows=0. "
            f"Check that data.root='{root}' is mounted/accessible on this node and episode_glob matches, "
            f"and that episodes are longer than target_history_len+future_horizon*chunk_size (={eff}). "
            f"(0 episodes almost always means the dataset path is missing on this machine.)"
        )
    if not set(e.episode_id for e in train_ds.episodes).isdisjoint(set(e.episode_id for e in val_ds.episodes)):
        raise RuntimeError("Train/val episode split overlaps")
    codec = build_action_codec(cfg)
    batch_size = int(cfg["train"]["batch_size"])  # per-GPU
    num_workers = int(cfg["train"].get("num_workers", 0))
    pin = device_obj.type == "cuda"
    train_sampler = DistributedSampler(train_ds, shuffle=bool(cfg["train"].get("shuffle", True))) if is_dist else None
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=train_sampler, shuffle=(bool(cfg["train"].get("shuffle", True)) and not is_dist), num_workers=num_workers, pin_memory=pin)
    # Inference set (held-out cases) is evaluated on the main rank only.
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin)

    model = build_policy(cfg).to(device_obj)
    raw_model = model
    # Optional warm-restart: load weights from a prior checkpoint and continue training (fresh
    # optimizer + a new, typically smaller-LR schedule). Used to keep training a converged run at
    # low LR for more epochs.
    resume_ckpt = str(cfg["train"].get("resume_checkpoint", "") or "")
    if resume_ckpt:
        from r2r_gen2act.training.checkpoint import load_checkpoint
        load_checkpoint(resume_ckpt, raw_model, device_obj, strict=False)
        if is_main:
            print(f"[resume] loaded weights from {resume_ckpt} (fresh optimizer/scheduler)")
    if is_dist:
        model = DDP(model, device_ids=[device_obj.index] if device_obj.type == "cuda" else None, find_unused_parameters=False)
    opt_cfg = cfg["train"].get("optimizer", {})
    optimizer = torch.optim.AdamW(_param_groups(raw_model, cfg), betas=tuple(opt_cfg.get("betas", [0.9, 0.95])))
    scheduler = _build_scheduler(optimizer, cfg, len(train_loader))

    out_dir = Path(cfg["experiment"]["output_dir"])
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "config_snapshot.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        print(f"world_size={world_size} per_gpu_batch={batch_size} train_episodes={len(train_ds.episodes)} infer_episodes={len(val_ds.episodes)} train_windows={len(train_ds)} infer_windows={len(val_ds)}")

    eval_every = max(1, int(cfg["train"].get("eval_every_epochs", 1)))
    epochs = int(cfg["train"].get("epochs", 1))
    best = float("inf")
    latest_path = out_dir / "latest.pt"
    history: list[dict] = []
    start_epoch = 1

    # Full resume: restore model + optimizer state and fast-forward scheduler.
    full_resume = str(cfg["train"].get("resume_full_checkpoint", "") or "")
    if full_resume:
        ckpt = torch.load(full_resume, map_location=device_obj, weights_only=False)
        raw_model.load_state_dict(ckpt["model_state_dict"], strict=True)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = int(ckpt["epoch"]) + 1
        # Restore best val loss: prefer checkpoint metrics, fall back to CSV min
        val_m = (ckpt.get("metrics") or {}).get("val") or {}
        if val_m and val_m.get("loss") is not None:
            best = float(val_m["loss"])
        else:
            csv_path_tmp = out_dir / "loss_history.csv"
            if csv_path_tmp.exists():
                import csv as _csv2
                with csv_path_tmp.open(newline="") as fh2:
                    for row2 in _csv2.DictReader(fh2):
                        v = row2.get("val_loss", "")
                        if v not in ("", None):
                            best = min(best, float(v))
        # Fast-forward scheduler to match completed steps (no gradient, just counters)
        done_steps = (start_epoch - 1) * len(train_loader)
        for _ in range(done_steps):
            scheduler.step()
        # Reload loss history from CSV so plots stay continuous
        csv_path = out_dir / "loss_history.csv"
        if csv_path.exists():
            import csv as _csv
            with csv_path.open(newline="") as fh:
                reader = _csv.DictReader(fh)
                for row in reader:
                    history.append({k: (float(v) if v != "" else None) for k, v in row.items()})
        if is_main:
            print(f"[resume_full] loaded {full_resume} @ epoch {start_epoch - 1}, continuing from epoch {start_epoch}, best={best:.4f}")

    for epoch in range(start_epoch, epochs + 1):
        if is_dist:
            train_sampler.set_epoch(epoch)
        t0 = time.time()
        train_metrics = run_epoch(model, train_loader, codec, cfg, device_obj, optimizer, train=True, scheduler=scheduler, world_size=world_size)

        do_eval = (epoch % eval_every == 0) or (epoch == epochs)
        val_metrics = None
        if do_eval and is_main:
            raw_model.eval()
            with torch.no_grad():
                val_metrics = run_epoch(raw_model, val_loader, codec, cfg, device_obj, None, train=False, world_size=1)

        if is_main:
            row = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_action_loss": train_metrics["action_loss"],
                "train_gripper_loss": train_metrics["gripper_loss"],
                "train_terminate_loss": train_metrics["terminate_loss"],
            }
            for key, value in train_metrics.items():
                if key.startswith("action_dim_") or key in ("action_mae", "action_rmse"):
                    row[f"train_{key}"] = value
            if val_metrics is not None:
                row["val_loss"] = val_metrics["loss"]
                row["val_action_loss"] = val_metrics["action_loss"]
                row["val_gripper_loss"] = val_metrics["gripper_loss"]
                row["val_terminate_loss"] = val_metrics["terminate_loss"]
                for key, value in val_metrics.items():
                    if key.startswith("action_dim_") or key in ("action_mae", "action_rmse"):
                        row[f"val_{key}"] = value
            history.append(row)
            _write_loss_history(out_dir, history)
            msg = f"epoch={epoch} train_loss={train_metrics['loss']:.4f}"
            if val_metrics is not None:
                msg += f" infer_loss={val_metrics['loss']:.4f}"
            print(f"{msg} time={time.time()-t0:.1f}s loss_plot={out_dir / 'loss_curve.png'}")
            save_checkpoint(latest_path, raw_model, optimizer, cfg, epoch, {"train": train_metrics, "val": val_metrics})
            if val_metrics is not None and val_metrics["loss"] < best:
                best = val_metrics["loss"]
                save_checkpoint(out_dir / "best.pt", raw_model, optimizer, cfg, epoch, {"train": train_metrics, "val": val_metrics})
        if is_dist:
            dist.barrier()

    if is_dist:
        dist.destroy_process_group()
    return latest_path
