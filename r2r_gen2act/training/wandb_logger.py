from __future__ import annotations

import os
from pathlib import Path
import warnings

import torch
import torch.distributed as dist


class WandbLogger:
    """Rank-safe scalar logging with a persistent run ID for restarts."""

    def __init__(self, cfg: dict, output_dir: Path, *, is_main: bool, world_size: int) -> None:
        logging_cfg = cfg["train"].get("logging", {}) or {}
        self.cfg = logging_cfg.get("wandb", {}) or {}
        self.enabled = bool(self.cfg.get("enabled", False))
        self.log_every = max(1, int(logging_cfg.get("log_every", 100)))
        self.is_main = bool(is_main)
        self.world_size = int(world_size)
        self.run = None
        if not self.enabled or not self.is_main:
            return

        try:
            import wandb

            credential = str(self.cfg.get("credential", "") or "")
            if credential and "WANDB_API_KEY" not in os.environ:
                credential_path = Path(credential).expanduser()
                if credential_path.is_file():
                    os.environ["WANDB_API_KEY"] = credential_path.read_text(
                        encoding="utf-8").strip()

            output_dir.mkdir(parents=True, exist_ok=True)
            run_id_path = output_dir / "wandb_id.txt"
            if run_id_path.is_file():
                run_id = run_id_path.read_text(encoding="utf-8").strip()
            else:
                run_id = wandb.util.generate_id()
                temporary = run_id_path.with_suffix(".tmp")
                temporary.write_text(run_id + "\n", encoding="utf-8")
                os.replace(temporary, run_id_path)

            mode = str(os.environ.get("WANDB_MODE", self.cfg.get("mode", "online")))
            self.run = wandb.init(
                id=run_id,
                resume="allow",
                entity=self.cfg.get("entity") or None,
                project=str(self.cfg.get("project", "gen2act")),
                group=self.cfg.get("group") or None,
                name=str(self.cfg.get("name") or cfg["experiment"]["name"]),
                config=cfg,
                dir=str(output_dir),
                mode=mode,
            )
            print(
                f"[wandb] initialized project={self.cfg.get('project', 'gen2act')} "
                f"name={self.cfg.get('name') or cfg['experiment']['name']} mode={mode} id={run_id}")
        except Exception as exc:
            warnings.warn(f"WandB initialization failed; continuing without it: {exc}")
            self.enabled = False
            self.run = None

    def log_train_step(
        self,
        losses: dict[str, torch.Tensor],
        *,
        global_step: int,
        epoch: int,
        step_in_epoch: int,
        steps_per_epoch: int,
        optimizer,
        device: torch.device,
    ) -> None:
        if not self.enabled or global_step % self.log_every != 0:
            return
        keys = sorted(losses)
        values = torch.stack([losses[key].detach().float() for key in keys]).to(device)
        if self.world_size > 1 and dist.is_initialized():
            dist.all_reduce(values, op=dist.ReduceOp.SUM)
            values /= self.world_size
        if not self.is_main or self.run is None:
            return
        payload = {f"train/{key}": float(value.cpu()) for key, value in zip(keys, values)}
        payload.update({
            "optimizer/lr": float(optimizer.param_groups[0]["lr"]),
            "optimizer/global_step": int(global_step),
            "progress/epoch": int(epoch),
            "progress/epoch_fraction": float(step_in_epoch) / max(1, steps_per_epoch),
        })
        try:
            self.run.log(payload, step=global_step)
        except Exception as exc:
            warnings.warn(f"WandB train logging failed at step {global_step}: {exc}")

    def log_epoch(self, row: dict, *, global_step: int) -> None:
        if not self.enabled or not self.is_main or self.run is None:
            return
        payload = {}
        for key, value in row.items():
            if key == "epoch" or value is None:
                continue
            prefix, name = ("val", key[4:]) if key.startswith("val_") else (
                "train", key[6:]) if key.startswith("train_") else ("epoch", key)
            payload[f"{prefix}/{name}"] = float(value)
        payload["progress/epoch"] = int(row["epoch"])
        try:
            self.run.log(payload, step=global_step)
        except Exception as exc:
            warnings.warn(f"WandB epoch logging failed at step {global_step}: {exc}")

    def log_checkpoint(self, path: Path, *, global_step: int) -> None:
        if not self.enabled or not self.is_main or self.run is None:
            return
        try:
            self.run.log({"checkpoint/saved": 1, "checkpoint/path": str(path)}, step=global_step)
        except Exception as exc:
            warnings.warn(f"WandB checkpoint logging failed at step {global_step}: {exc}")

    def finish(self) -> None:
        if self.is_main and self.run is not None:
            try:
                self.run.finish()
            except Exception as exc:
                warnings.warn(f"WandB finish failed: {exc}")
