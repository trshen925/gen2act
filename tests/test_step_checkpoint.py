from __future__ import annotations

from pathlib import Path
import tempfile

import torch

from r2r_gen2act.data.episode_sampler import EpisodeLocalSampler
from r2r_gen2act.training.checkpoint import save_checkpoint


class _Dataset:
    def __init__(self) -> None:
        self._samples = [
            ("ep_b", 3), ("ep_a", 2), ("ep_a", 1),
            ("ep_b", 1), ("ep_c", 0), ("ep_b", 2),
        ]


def test_episode_sampler_can_resume_from_local_sample_offset() -> None:
    sampler = EpisodeLocalSampler(_Dataset(), shuffle=False)
    complete = list(sampler)

    sampler.set_start_index(3)

    assert len(sampler) == len(complete) - 3
    assert list(sampler) == complete[3:]


def test_checkpoint_persists_scheduler_and_mid_epoch_progress_atomically() -> None:
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    progress = {
        "step_in_epoch": 6000,
        "global_step": 6000,
        "steps_per_epoch": 41667,
        "world_size": 4,
        "batch_size": 12,
    }

    with tempfile.TemporaryDirectory() as temporary_directory:
        path = Path(temporary_directory) / "step_000006000.pt"
        save_checkpoint(
            path, model, optimizer, {}, 1, {},
            scheduler_state_dict=scheduler.state_dict(),
            progress=progress,
        )
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)

        assert checkpoint["progress"] == progress
        assert checkpoint["scheduler_state_dict"] == scheduler.state_dict()
        assert checkpoint["optimizer_state_dict"] is not None
        assert not list(path.parent.glob(".*.tmp-*"))
