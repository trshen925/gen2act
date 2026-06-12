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

For remote pretrained downloads, set proxies first:

```bash
export http_proxy=http://'galbot:sK0aZ5bZ9v'@10.119.176.202:3128
export https_proxy=http://'galbot:sK0aZ5bZ9v'@10.119.176.202:3128
```
