from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from r2r_gen2act.config.load import load_config
from r2r_gen2act.inference.predictor import predict_dataset_window


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--episode-id", type=str, default="")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--save-path", type=Path, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--allow-partial-load", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    result = predict_dataset_window(cfg, args.checkpoint, args.split, args.episode_id or None, args.start_index, args.save_path, args.device, strict=not args.allow_partial_load)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
