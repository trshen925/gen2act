from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from r2r_gen2act.modeling.track import ensure_cotracker_path


def main() -> None:
    path = ensure_cotracker_path()
    print(f"CoTracker path: {path}")
    print("Track caching is intentionally disabled by default; implement dataset-specific caching here when enabled.")


if __name__ == "__main__":
    main()
