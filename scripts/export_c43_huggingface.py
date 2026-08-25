"""Export the C43 EMA weights and release metadata for Hugging Face."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("ema_state_dict") or checkpoint["model_state_dict"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(state_dict, args.output_dir / "pytorch_model.pt")

    config = json.loads(args.config.read_text(encoding="utf-8"))
    (args.output_dir / "config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    metadata = {
        "source_checkpoint": str(args.checkpoint),
        "epoch": int(checkpoint["epoch"]),
        "weights": "ema_state_dict",
        "metrics": checkpoint.get("metrics", {}),
    }
    (args.output_dir / "release_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
