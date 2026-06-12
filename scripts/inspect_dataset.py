from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from r2r_gen2act.config.load import load_config
from r2r_gen2act.config.schema import validate_config
from r2r_gen2act.tools.inspect_dataset import inspect_dataset, print_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", type=str, default="train")
    args = parser.parse_args()
    cfg = load_config(args.config)
    validate_config(cfg)
    print_report(inspect_dataset(cfg, args.split))


if __name__ == "__main__":
    main()
