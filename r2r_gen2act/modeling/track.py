from __future__ import annotations

import sys
from pathlib import Path


def ensure_cotracker_path() -> Path:
    cotracker_root = Path("/mnt/afs/shentingrui/code/dreamwheel/gen2act/co-tracker")
    if cotracker_root.is_dir() and str(cotracker_root) not in sys.path:
        sys.path.insert(0, str(cotracker_root))
    return cotracker_root
