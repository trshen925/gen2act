from __future__ import annotations

from torch.utils.data import Dataset

from r2r_gen2act.data.action.codec import ActionCodec


def build_dataset(cfg: dict, split: str) -> Dataset:
    dtype = str(cfg["data"].get("dataset_type", "openx_droid"))
    if dtype == "hdf5_franka":
        from r2r_gen2act.data.adapters.hdf5_franka import HDF5FrankaDataset

        return HDF5FrankaDataset(cfg, split=split)
    if dtype == "openx_droid":
        from r2r_gen2act.data.adapters.openx_droid import OpenXDroidDataset

        return OpenXDroidDataset(cfg, split=split)
    if dtype == "openx_toto":
        from r2r_gen2act.data.adapters.openx_toto import OpenXTotoDataset

        return OpenXTotoDataset(cfg, split=split)
    if dtype == "pointworld_droid":
        from r2r_gen2act.data.adapters.pointworld_droid import PointWorldDroidDataset

        return PointWorldDroidDataset(cfg, split=split)
    if dtype == "droid_ex_out":
        from r2r_gen2act.data.adapters.droid_ex_out import DroidExOutDataset

        return DroidExOutDataset(cfg, split=split)
    raise ValueError(f"Unknown dataset_type={dtype}")


def build_action_codec(cfg: dict) -> ActionCodec:
    return ActionCodec.from_config(cfg)
