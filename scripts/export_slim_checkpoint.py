from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from r2r_gen2act.training.checkpoint import save_slim_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Export an eval/deploy checkpoint without optimizer state.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Full training checkpoint.")
    parser.add_argument("--output", type=Path, required=True, help="Slim checkpoint output path.")
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
        raise ValueError(f"Expected a training checkpoint with model_state_dict: {args.checkpoint}")

    save_slim_checkpoint(args.output, ckpt)
    size_gib = args.output.stat().st_size / 1024**3
    epoch = ckpt.get("epoch", "unknown")
    print(f"wrote {args.output} ({size_gib:.2f} GiB, epoch={epoch}, optimizer_state_dict=removed)")


if __name__ == "__main__":
    main()
