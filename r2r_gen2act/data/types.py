from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EpisodeRecord:
    episode_id: str
    num_steps: int
    source_video_path: Path | None
    target_video_path: Path | None
    metadata_path: Path | None
    split: str | None = None
    extra: dict[str, Any] | None = None
