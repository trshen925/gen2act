from __future__ import annotations

from typing import Iterable

from r2r_gen2act.data.types import EpisodeRecord


def build_windows(
    episodes: Iterable[EpisodeRecord],
    source_len: int,
    target_history_len: int,
    target_offset: int = 0,
    stride: int = 1,
    max_windows: int | None = None,
    future_horizon: int = 0,
) -> list[tuple[str, int]]:
    samples: list[tuple[str, int]] = []
    for episode in episodes:
        last_start = episode.num_steps - target_history_len - target_offset - max(0, int(future_horizon))
        if last_start < 0:
            continue
        for start in range(0, last_start + 1, max(1, stride)):
            target_step = start + target_history_len - 1 + target_offset
            if 0 <= target_step and target_step + max(0, int(future_horizon)) < episode.num_steps:
                samples.append((episode.episode_id, start))
                if max_windows is not None and len(samples) >= max_windows:
                    return samples
    return samples
