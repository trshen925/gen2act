from __future__ import annotations

from r2r_gen2act.config.schema import validate_config


def validate(cfg: dict) -> None:
    validate_config(cfg)
