from __future__ import annotations

import json
from pathlib import Path


def load_action_stats(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def pose_bounds_from_stats(stats: dict, pose_dims: int = 6) -> tuple[list[float], list[float]]:
    per_dim = stats.get("per_dim", {})
    low = list(per_dim.get("min", []))[:pose_dims]
    high = list(per_dim.get("max", []))[:pose_dims]
    if len(low) != pose_dims or len(high) != pose_dims:
        raise ValueError(f"Stats do not contain {pose_dims}D min/max bounds")
    return [float(x) for x in low], [float(x) for x in high]
