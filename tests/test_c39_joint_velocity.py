from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from r2r_gen2act.data.action.codec import ActionCodec
from r2r_gen2act.data.action.mappings import droid_action
from r2r_gen2act.data.adapters.base import WindowedRobotDataset
from r2r_gen2act.data.types import EpisodeRecord
from r2r_gen2act.inference.predictor import PolicyPredictor
from r2r_gen2act.modeling.flow_dit import FlowMatchingDiTHead
from r2r_gen2act.training.checkpoint import load_checkpoint, save_checkpoint
from r2r_gen2act.training.losses import compute_losses
from r2r_gen2act.training.trainer import ModelEMA


class C39JointVelocityTest(unittest.TestCase):
    @staticmethod
    def _source_crop_dataset(split: str) -> WindowedRobotDataset:
        dataset = object.__new__(WindowedRobotDataset)
        dataset.split = split
        dataset.fps = 15.0
        dataset.source_len = 8
        dataset.target_history_len = 1
        dataset.target_offset = 0
        dataset.dynamic_source_enabled = False
        dataset.source_float_enabled = False
        dataset.source_time_crop_enabled = True
        dataset.source_time_crop_min_seconds = 10.0
        dataset.source_time_crop_max_seconds = 30.0
        dataset.source_jitter_cfg = {"enabled": False}
        dataset.data_cfg = {"source_sampling": "linspace"}
        dataset._clip_length = lambda _: 1000
        return dataset

    def test_source_time_crop_is_bounded_and_contains_current_frame(self) -> None:
        train_dataset = self._source_crop_dataset("train")
        episode = EpisodeRecord("ep", 1000, Path("rgb.mp4"), Path("rgb.mp4"), None)
        for seed in range(20):
            torch.manual_seed(seed)
            indices = train_dataset._compute_source_indices(episode, start_index=500)
            self.assertEqual(len(indices), 8)
            self.assertLessEqual(indices[0], 500)
            self.assertGreaterEqual(indices[-1], 500)
            self.assertGreaterEqual((indices[-1] - indices[0]) / 15.0, 10.0)
            self.assertLessEqual((indices[-1] - indices[0]) / 15.0, 30.0)

        val_dataset = self._source_crop_dataset("val")
        self.assertEqual(val_dataset._source_time_crop_bounds(1000, 500), (275, 725))
        self.assertEqual(val_dataset._source_time_crop_bounds(1000, 20), (0, 450))
        self.assertEqual(val_dataset._source_time_crop_bounds(100, 50), (0, 99))

    def test_native_joint_velocity_mapping_starts_at_current_action(self) -> None:
        velocity = np.arange(30 * 7, dtype=np.float32).reshape(30, 7)
        gripper = np.linspace(0.0, 1.0, 30, dtype=np.float32)
        payload = {"action_dict": {"joint_velocity": velocity, "gripper_position": gripper}}

        action = droid_action(
            payload,
            step=4,
            mapping_type="droid_action_dict_joint_velocity",
            chunk_size=15,
            mapping_cfg={"chunk_start_offset": 0, "chunk_stride": 1},
        )

        expected = np.concatenate((velocity[4:19], gripper[4:19, None]), axis=1)
        np.testing.assert_array_equal(action, expected)

    def test_continuous_gripper_normalization_round_trip(self) -> None:
        raw = torch.tensor([0.0, 0.2, 0.5, 0.9998])
        normalized = ActionCodec.normalize_scalar(raw, 0.0, 0.9998)
        restored = ActionCodec.unnormalize_scalar(normalized, 0.0, 0.9998)

        torch.testing.assert_close(restored, raw)
        self.assertGreaterEqual(float(normalized.min()), -1.0)
        self.assertLessEqual(float(normalized.max()), 1.0)

    def test_event_windows_survive_idle_filtering(self) -> None:
        dataset = object.__new__(WindowedRobotDataset)
        dataset.data_cfg = {
            "gripper_threshold": 0.5,
            "native_action_sampling": {
                "velocity_idle_threshold": 1e-3,
                "max_idle_run": 0,
                "event_before": 2,
                "event_after": 2,
                "normal_ratio": 0.0,
                "close_ratio": 0.5,
                "release_ratio": 0.5,
                "num_samples": 20,
                "seed": 7,
            },
        }
        dataset.cfg = {"train": {"seed": 7}}
        dataset.target_history_len = 1
        dataset.target_offset = 0
        dataset.chunk_start_offset = 0
        dataset.chunk_stride = 1
        dataset.chunk_size = 5
        episode = EpisodeRecord("ep", 30, None, None, Path("meta.json"))
        dataset._episode_by_id = {"ep": episode}
        gripper = np.ones(30, dtype=np.float32)
        gripper[10:20] = 0.0
        payload = {
            "action_dict": {
                "joint_velocity": np.zeros((30, 7), dtype=np.float32),
                "gripper_position": gripper,
            }
        }
        dataset._read_action_payload = lambda _: payload

        selected = dataset._build_native_action_sample_index([("ep", i) for i in range(26)])

        starts = [start for _, start in selected]
        self.assertEqual(len(starts), 20)
        self.assertTrue(all(8 <= start <= 12 or 18 <= start <= 22 for start in starts))
        self.assertTrue(any(8 <= start <= 12 for start in starts))
        self.assertTrue(any(18 <= start <= 22 for start in starts))

    def test_joint_flow_training_and_inference_shapes(self) -> None:
        codec = ActionCodec(7, 64, [-1.0] * 7, [1.0] * 7)
        head = FlowMatchingDiTHead(
            cond_dim=16,
            action_dim=8,
            horizon=15,
            hidden_dim=32,
            num_layers=2,
            heads=4,
            num_inference_steps=2,
            dropout=0.0,
            vl_mixer_layers=1,
            diffuse_gripper=True,
        )
        cond = torch.randn(2, 6, 16)
        raw_action = torch.rand(2, 15, 8) * 2.0 - 1.0
        gripper_value = torch.rand(2, 15)
        gripper = (gripper_value >= 0.5).long()
        target = torch.cat(
            (codec.normalize(raw_action[..., :7]), ActionCodec.normalize_scalar(gripper_value, 0.0, 1.0)[..., None]),
            dim=-1,
        )
        batch = {
            "action": raw_action,
            "gripper_value": gripper_value,
            "gripper": gripper,
            "terminate": torch.zeros(2, 15, dtype=torch.long),
        }
        cfg = {
            "action": {"mode": "flow", "gripper": {"continuous": True, "bounds_low": 0.0, "bounds_high": 1.0}},
            "model": {"flow_dit": {"diffuse_gripper": True}},
            "train": {"losses": {}},
        }

        train_outputs = head(cond, target)
        self.assertEqual(tuple(train_outputs["pred_velocity"].shape), (2, 15, 8))
        train_losses = compute_losses(train_outputs, batch, codec, cfg)
        self.assertTrue(torch.isfinite(train_losses["loss"]))
        train_losses["loss"].backward()

        eval_outputs = head.sample(cond)
        self.assertEqual(tuple(eval_outputs["action_pred"].shape), (2, 15, 8))
        eval_losses = compute_losses(eval_outputs, batch, codec, cfg)
        self.assertTrue(torch.isfinite(eval_losses["loss"]))

    def test_predictor_clips_and_returns_eight_execution_steps(self) -> None:
        class FakePolicy(torch.nn.Module):
            def forward(self, *args, **kwargs):
                action = torch.full((1, 15, 8), 3.0)
                action[..., 7] = 1.0
                return {
                    "action_pred": action,
                    "terminate_logits": torch.zeros(1, 15, 2),
                }

        predictor = object.__new__(PolicyPredictor)
        predictor.cfg = {
            "action": {
                "mode": "flow",
                "regression_normalize": True,
                "gripper": {"bounds_low": 0.0, "bounds_high": 1.0, "execution_threshold": 0.5},
                "execution": {"clip_low": -1.0, "clip_high": 1.0, "execute_steps": 8},
            },
            "model": {"flow_dit": {"diffuse_gripper": True}},
        }
        predictor.device = torch.device("cpu")
        predictor.codec = ActionCodec(7, 64, [-2.0] * 7, [2.0] * 7)
        predictor.model = FakePolicy()
        batch = {
            "source_video": torch.zeros(1, 1, 3, 4, 4),
            "target_history": torch.zeros(1, 1, 3, 4, 4),
        }

        result = predictor.predict_batch(batch)

        self.assertEqual(tuple(result["pose_action"].shape), (1, 15, 7))
        self.assertEqual(tuple(result["execution_pose_action"].shape), (1, 8, 7))
        self.assertTrue(torch.all(result["pose_action"] == 1.0))
        self.assertTrue(torch.all(result["gripper_action"] == 1))

    def test_ema_swap_checkpoint_and_warm_start_exclusion(self) -> None:
        model = torch.nn.Sequential()
        model.add_module("body", torch.nn.Linear(2, 2, bias=False))
        model.add_module("head", torch.nn.Linear(2, 1, bias=False))
        with torch.no_grad():
            model.body.weight.fill_(1.0)
            model.head.weight.fill_(1.0)
        ema = ModelEMA(model, decay=0.5)
        with torch.no_grad():
            model.body.weight.fill_(3.0)
            model.head.weight.fill_(3.0)
        ema.update(model)

        with ema.apply_to(model):
            torch.testing.assert_close(model.body.weight, torch.full_like(model.body.weight, 2.0))
        torch.testing.assert_close(model.body.weight, torch.full_like(model.body.weight, 3.0))

        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint = Path(tmp_dir) / "ema.pt"
            save_checkpoint(checkpoint, model, None, {}, 1, {}, ema_state_dict=ema.state, ema_decay=0.5)
            saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
            torch.testing.assert_close(
                saved["model_state_dict"]["body.weight"], torch.full_like(model.body.weight, 2.0))
            torch.testing.assert_close(saved["training_model_state_dict"]["body.weight"], model.body.weight)

            target = torch.nn.Sequential()
            target.add_module("body", torch.nn.Linear(2, 2, bias=False))
            target.add_module("head", torch.nn.Linear(2, 1, bias=False))
            original_head = target.head.weight.detach().clone()
            load_checkpoint(checkpoint, target, "cpu", strict=False, exclude_prefixes=("head",))
            torch.testing.assert_close(target.body.weight, torch.full_like(target.body.weight, 2.0))
            torch.testing.assert_close(target.head.weight, original_head)


if __name__ == "__main__":
    unittest.main()
