from __future__ import annotations

import pytest
import torch

from r2r_gen2act.training.trainer import _build_scheduler


def test_piecewise_linear_scheduler_hits_epoch_lr_targets() -> None:
    parameter = torch.nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.AdamW([parameter], lr=1.0e-5)
    cfg = {
        "train": {
            "epochs": 4,
            "scheduler": {
                "name": "piecewise_linear",
                "epoch_points": [
                    [0.0, 0.288],
                    [0.2, 1.0],
                    [1.0, 1.0],
                    [2.0, 0.8],
                    [3.0, 0.4],
                    [4.0, 0.1],
                ],
            },
        }
    }

    scheduler = _build_scheduler(optimizer, cfg, steps_per_epoch=10)
    assert scheduler is not None
    assert optimizer.param_groups[0]["lr"] == pytest.approx(2.88e-6)

    targets = {2: 1.0e-5, 10: 1.0e-5, 20: 8.0e-6, 30: 4.0e-6, 40: 1.0e-6}
    for step in range(1, 41):
        optimizer.step()
        scheduler.step()
        if step in targets:
            assert optimizer.param_groups[0]["lr"] == pytest.approx(targets[step])


@pytest.mark.parametrize(
    "points",
    [[], [[0.0, 1.0]], [[0.0, 1.0], [0.0, 0.5]], [[0.0, -1.0], [1.0, 0.5]]],
)
def test_piecewise_linear_scheduler_rejects_invalid_points(points: list[list[float]]) -> None:
    parameter = torch.nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.AdamW([parameter], lr=1.0e-5)
    cfg = {
        "train": {
            "epochs": 4,
            "scheduler": {"name": "piecewise_linear", "epoch_points": points},
        }
    }

    with pytest.raises(ValueError):
        _build_scheduler(optimizer, cfg, steps_per_epoch=10)
