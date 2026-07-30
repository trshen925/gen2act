from __future__ import annotations

import math
import random

from torch.utils.data import Sampler


class EpisodeLocalSampler(Sampler[int]):
    """Shuffle episodes while reading each episode's windows in temporal order.

    DDP ranks receive contiguous equal-length slices of the resulting order. This
    preserves MP4 reader locality and guarantees every rank executes the same
    number of optimizer steps.
    """

    def __init__(
        self,
        dataset,
        *,
        rank: int = 0,
        world_size: int = 1,
        shuffle: bool = True,
        seed: int = 0,
    ) -> None:
        if rank < 0 or rank >= world_size:
            raise ValueError(f"Invalid rank={rank} for world_size={world_size}")
        samples = dataset._samples
        grouped: dict[str, list[int]] = {}
        for index, (episode_id, _) in enumerate(samples):
            grouped.setdefault(str(episode_id), []).append(index)
        self._groups = []
        for episode_id in sorted(grouped):
            indices = grouped[episode_id]
            indices.sort(key=lambda index: int(samples[index][1]))
            self._groups.append(indices)
        self._size = len(samples)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0
        self.num_samples = int(math.ceil(self._size / self.world_size)) if self._size else 0
        self.start_index = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def set_start_index(self, start_index: int) -> None:
        start_index = int(start_index)
        if start_index < 0 or start_index > self.num_samples:
            raise ValueError(
                f"start_index must be in [0, {self.num_samples}], got {start_index}")
        self.start_index = start_index

    def _group_order(self) -> list[int]:
        order = list(range(len(self._groups)))
        if self.shuffle:
            random.Random(self.seed + self.epoch).shuffle(order)
        return order

    def __iter__(self):
        if self._size == 0:
            return iter(())
        order = self._group_order()
        shard_start = self.rank * self.num_samples
        shard_stop = min(shard_start + self.num_samples, self._size)

        def indices_for_slice(start: int, stop: int):
            cursor = 0
            for group_index in order:
                group = self._groups[group_index]
                next_cursor = cursor + len(group)
                if next_cursor > start and cursor < stop:
                    lo = max(0, start - cursor)
                    hi = min(len(group), stop - cursor)
                    yield from group[lo:hi]
                cursor = next_cursor
                if cursor >= stop:
                    break

        shard = list(indices_for_slice(shard_start, shard_stop))
        if len(shard) < self.num_samples:
            shard.extend(indices_for_slice(0, self.num_samples - len(shard)))
        return iter(shard[self.start_index:])

    def __len__(self) -> int:
        return self.num_samples - self.start_index
