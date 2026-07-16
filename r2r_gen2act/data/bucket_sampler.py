"""C24: batch sampler that groups windows by their source-frame count k, so every batch has a
uniform k and `source_video`/`traj_target` collate cleanly (no padding, no attention masks).

k depends only on episode length, so it's precomputed per window (WindowedRobotDataset.window_k()).
"""
from __future__ import annotations

import torch
from torch.utils.data import Sampler


class KBucketBatchSampler(Sampler):
    def __init__(self, window_k: list[int], batch_size: int, shuffle: bool = True,
                 drop_last: bool = False, seed: int = 0) -> None:
        self.window_k = list(window_k)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self._epoch = 0
        # group window indices by k
        buckets: dict[int, list[int]] = {}
        for idx, k in enumerate(self.window_k):
            buckets.setdefault(int(k), []).append(idx)
        self.buckets = buckets

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def _make_batches(self) -> list[list[int]]:
        g = torch.Generator()
        g.manual_seed(self.seed + self._epoch)
        batches: list[list[int]] = []
        for k in sorted(self.buckets):
            idxs = list(self.buckets[k])
            if self.shuffle:
                perm = torch.randperm(len(idxs), generator=g).tolist()
                idxs = [idxs[i] for i in perm]
            for start in range(0, len(idxs), self.batch_size):
                batch = idxs[start:start + self.batch_size]
                if self.drop_last and len(batch) < self.batch_size:
                    continue
                batches.append(batch)
        if self.shuffle:
            order = torch.randperm(len(batches), generator=g).tolist()
            batches = [batches[i] for i in order]
        return batches

    def __iter__(self):
        self._epoch += 1
        for batch in self._make_batches():
            yield batch

    def __len__(self) -> int:
        n = 0
        for k in self.buckets:
            c = len(self.buckets[k])
            n += c // self.batch_size if self.drop_last else (c + self.batch_size - 1) // self.batch_size
        return n
