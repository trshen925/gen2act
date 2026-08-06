from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from r2r_gen2act.config.load import load_config
from r2r_gen2act.data.episode_sampler import EpisodeLocalSampler
from r2r_gen2act.data.factories import build_action_codec, build_dataset
from r2r_gen2act.modeling.factory import build_policy
from r2r_gen2act.training.checkpoint import load_checkpoint
from r2r_gen2act.training.losses import compute_losses
from r2r_gen2act.training.trainer import _amp_dtype, _move_batch, _param_groups


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the real C40 training input path")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("/tmp/c40_loader_benchmark.json"))
    args = parser.parse_args()

    cfg = load_config(args.config)
    torch.manual_seed(int(cfg["train"].get("seed", 42)))
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    dataset = build_dataset(cfg, "train")
    sampler = EpisodeLocalSampler(
        dataset,
        shuffle=bool(cfg["train"].get("shuffle", True)),
        seed=int(cfg["train"].get("seed", 42)),
    )
    sampler.set_epoch(1)
    num_workers = int(cfg["train"].get("num_workers", 0))
    loader = DataLoader(
        dataset,
        batch_size=int(cfg["train"]["batch_size"]),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=bool(cfg["train"].get("persistent_workers", False)),
        prefetch_factor=int(cfg["train"].get("prefetch_factor", 2)),
    )

    codec = build_action_codec(cfg)
    model = build_policy(cfg).to(device)
    checkpoint = str(cfg["train"].get("resume_checkpoint", "") or "")
    if checkpoint:
        load_checkpoint(checkpoint, model, device, strict=False)
    opt_cfg = cfg["train"].get("optimizer", {})
    optimizer = torch.optim.AdamW(
        _param_groups(model, cfg), betas=tuple(opt_cfg.get("betas", [0.9, 0.95]))
    )
    model.train()

    wait_times: list[float] = []
    compute_times: list[float] = []
    unique_episodes: set[str] = set()
    episode_switches = 0
    previous_episode: str | None = None
    samples = 0
    amp_enabled = bool(cfg["train"].get("amp", True))
    amp_dtype = _amp_dtype(cfg)
    iterator = iter(loader)
    started = time.perf_counter()

    for step in range(args.steps):
        wait_started = time.perf_counter()
        batch = next(iterator)
        wait_seconds = time.perf_counter() - wait_started

        episode_ids = [str(value) for value in batch["episode_id"]]
        unique_episodes.update(episode_ids)
        for episode_id in episode_ids:
            if previous_episode is not None and episode_id != previous_episode:
                episode_switches += 1
            previous_episode = episode_id
        samples += len(episode_ids)

        compute_started = time.perf_counter()
        batch = _move_batch(batch, device)
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
            pose_dims = codec.pose_dims
            action_target = codec.normalize(batch["action"][..., :pose_dims])
            if bool(cfg["model"].get("flow_dit", {}).get("diffuse_gripper", False)):
                gripper_cfg = cfg.get("action", {}).get("gripper", {}) or {}
                gripper_key = "gripper_value" if bool(gripper_cfg.get("continuous", False)) else "gripper"
                gripper = batch[gripper_key].to(dtype=action_target.dtype).unsqueeze(-1)
                gripper = codec.normalize_scalar(
                    gripper,
                    float(gripper_cfg.get("bounds_low", 0.0)),
                    float(gripper_cfg.get("bounds_high", 1.0)),
                )
                action_target = torch.cat((action_target, gripper), dim=-1)
            extra = {}
            if batch.get("source_dt") is not None:
                extra["source_dt"] = batch["source_dt"]
            if batch.get("wrist_current") is not None:
                extra["wrist_current"] = batch["wrist_current"]
            outputs = model(
                batch.get("source_video"),
                batch.get("target_history"),
                batch.get("proprioception"),
                action_target,
                batch.get("point_track"),
                **extra,
            )
            losses = compute_losses(outputs, batch, codec, cfg)
        optimizer.zero_grad(set_to_none=True)
        losses["loss"].backward()
        grad_clip = float(cfg["train"].get("grad_clip_norm", 1.0))
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        torch.cuda.synchronize(device)
        compute_seconds = time.perf_counter() - compute_started

        if step >= args.warmup_steps:
            wait_times.append(wait_seconds)
            compute_times.append(compute_seconds)
        if step == 0 or (step + 1) % 25 == 0:
            print(
                f"step={step + 1}/{args.steps} wait={wait_seconds:.3f}s "
                f"compute={compute_seconds:.3f}s episodes={len(unique_episodes)}",
                flush=True,
            )

    elapsed = time.perf_counter() - started
    result = {
        "steps": args.steps,
        "warmup_steps_excluded": args.warmup_steps,
        "samples": samples,
        "unique_episodes": len(unique_episodes),
        "episode_switches": episode_switches,
        "elapsed_seconds": elapsed,
        "throughput_samples_per_second": samples / elapsed,
        "wait_seconds": {
            "mean": float(np.mean(wait_times)),
            "p50": percentile(wait_times, 50),
            "p95": percentile(wait_times, 95),
            "p99": percentile(wait_times, 99),
            "max": max(wait_times),
        },
        "compute_seconds": {
            "mean": float(np.mean(compute_times)),
            "p50": percentile(compute_times, 50),
            "p95": percentile(compute_times, 95),
            "p99": percentile(compute_times, 99),
            "max": max(compute_times),
        },
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
