from __future__ import annotations

import random
from collections.abc import Sequence


def split_episode_ids(ids: Sequence[str], val_ratio: float, seed: int) -> tuple[set[str], set[str]]:
    ids = list(sorted(ids))
    rng = random.Random(seed)
    rng.shuffle(ids)
    val_count = max(1, int(round(len(ids) * val_ratio))) if len(ids) > 1 else 0
    val = set(ids[:val_count])
    train = set(ids[val_count:])
    return train, val
