# C30 Camera-Corrected Evaluation

## Model

- Architecture: DINOv2-B/14 + flow DiT16 + VL mixer8
- Checkpoint stage epoch: 11 (legacy C30 total epoch 24 followed by corrected-camera adaptation)
- Checkpoint: `outputs/droidexFULL_C30_camera_corrected_cont12_lr3e5/c30_camera_corrected_ep11.pt`
- SHA256: `85fc5ae7e9410062c57de09a9a41fb81e7e6d1f55af449cb6f8b9a7768817981`
- Hugging Face: `Reparameterization/gen2act-c30-camera-corrected` (private)

Download the checkpoint with an authenticated Hugging Face account:

```bash
HF_HUB_OFFLINE=0 hf download \
  Reparameterization/gen2act-c30-camera-corrected \
  c30_camera_corrected_ep11.pt \
  --local-dir outputs/droidexFULL_C30_camera_corrected_cont12_lr3e5
```

## Formal Evaluation

```bash
cd /mnt/pfs/users/shentingrui/code/robo/video_gen/gen2act/gen2act
bash scripts/run_eval_c30_camera_corrected.sh
```

Equivalent direct command:

```bash
/root/miniconda3/envs/gen2act/bin/python scripts/diagnose_actions.py \
  --config configs/droidexFULL_C30_camera_corrected_cont12_lr3e5_eval32.yaml \
  --checkpoint outputs/droidexFULL_C30_camera_corrected_cont12_lr3e5/c30_camera_corrected_ep11.pt \
  --split val --max-windows 800 --batch-size 32 --device cuda
```

The eval config fixes the candidate pool to `00000..35695`, selects the video camera from `meta.camera`, applies the same resize-center-crop geometry to proprioception UV, and uses 32 ODE inference steps with 16 samples.

## Result

- Corrected val: 487 episodes / 5,210 windows; first 800 windows evaluated
- XYZ MAE: **1.582 cm**
- dx/dy/dz MAE: `1.502 / 1.560 / 1.684 cm`
- dx/dy/dz correlation: `0.916 / 0.936 / 0.938`
- Predict-mean baseline: `4.564 cm`; improvement: `65.3%`

The XYZ output is a future delta in the selected exterior camera coordinate system. The eight chunk steps correspond to approximately +333 ms through +2667 ms.
