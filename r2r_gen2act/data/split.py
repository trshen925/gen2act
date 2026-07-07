from __future__ import annotations

import random
from collections.abc import Sequence


def split_episode_ids(ids: Sequence[str], val_ratio: float, seed: int, val_count: int | None = None) -> tuple[set[str], set[str]]:
    """Deterministically split episode ids into (train, val/inference).

    If val_count is given (>0) hold out exactly that many episodes (clamped to len-1);
    otherwise fall back to a fraction val_ratio.
    """
    ids = list(sorted(ids))
    rng = random.Random(seed)
    rng.shuffle(ids)
    if val_count is not None and int(val_count) > 0:
        n_val = min(int(val_count), max(0, len(ids) - 1))
    else:
        n_val = max(1, int(round(len(ids) * val_ratio))) if len(ids) > 1 else 0
    val = set(ids[:n_val])
    train = set(ids[n_val:])
    return train, val
