from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from r2r_gen2act.config.load import load_config
from r2r_gen2act.training.trainer import train


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    try:
        latest = train(cfg, device=args.device)
    finally:
        # train() cleans up on success; this covers exceptions such as a DDP
        # reducer failure so NCCL communicators are not leaked on the worker.
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
    if int(os.environ.get("RANK", "0")) == 0:
        print("latest_checkpoint", latest)


if __name__ == "__main__":
    main()
