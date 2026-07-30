# Robot-to-Robot Gen2Act

Refactored robot-to-robot Gen2Act-style training package for datasets under `franka_dataset`.

Use neutral stream names:

- `source_video`: generated/reference/source robot video
- `target_history`: target robot observation history used to predict the next action

Default target is 7D `[dx, dy, dz, rx, ry, rz, gripper]`; the first 6 dims are trained with discretized action bins, while gripper and terminate are separate binary heads.

## Quick commands

```bash
/mnt/afs/shentingrui/anaconda3/envs/gen2act/bin/python scripts/inspect_dataset.py --config configs/debug_smoke.yaml
/mnt/afs/shentingrui/anaconda3/envs/gen2act/bin/python scripts/train.py --config configs/debug_smoke.yaml
```

## Environment setup

The portable environment definition deliberately does not pin a CUDA toolkit or
PyTorch CUDA build. Create an environment with the default PyTorch package for
the target machine:

```bash
bash scripts/create_gen2act_env.sh gen2act
```

When the machine or cluster requires a specific PyTorch wheel index, supply it
at install time instead of editing the environment definition:

```bash
TORCH_INDEX_URL=<site-or-platform-specific-index> \
  bash scripts/create_gen2act_env.sh gen2act
```

The installer removes image-level `PIP_CONSTRAINT` settings that can otherwise
force an unrelated PyTorch version. It reports the installed PyTorch build,
bundled CUDA runtime, visible GPUs, and bf16 support when finished. Validate a
training allocation with its actual visible GPU count:

```bash
conda run -n gen2act python scripts/check_train_env.py \
  --config configs/droidFULL_C40_jointvelocity_pi05_letterbox_fulltrain.yaml \
  --expected-gpus 3
```

The short C40 continuation loads only EMA model weights from step 12,000 and
starts a fresh optimizer and four-epoch piecewise LR schedule. It writes model
outputs, logs, and W&B metadata under a new experiment name:

```bash
PYTHON_BIN=/path/to/gen2act/bin/python \
  bash scripts/run_train_c40_step12k_short4ep.sh
```

Its head LR follows `2.88e-6 -> 1e-5 (epoch 0.2) -> 1e-5 (epoch 1)
-> 8e-6 (epoch 2) -> 4e-6 (epoch 3) -> 1e-6 (epoch 4)`. Backbone LR groups
remain lower according to the inherited `0.3` multiplier and `0.7` LLRD.

For remote pretrained downloads, set proxies first:

```bash
export http_proxy=http://'galbot:sK0aZ5bZ9v'@10.119.176.202:3128
export https_proxy=http://'galbot:sK0aZ5bZ9v'@10.119.176.202:3128
```
