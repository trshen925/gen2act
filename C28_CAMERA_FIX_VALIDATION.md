# C28 Camera Fix Validation

Date: 2026-07-16

## Fix

- `data.camera_selection: from_meta` maps:
  - `exterior_image_1_left -> exterior_1`
  - `exterior_image_2_left -> exterior_2`
- Unknown camera names or missing selected extrinsics fail instead of silently falling back.
- Corrected training configs retain both `GT` and `Pred` calibrations. A stratified 32-sample Pred audit found that most align; calibration provenance remains in each payload for later audits.
- `proprioception.projection_image_space: model_input` applies the same resize-center-crop geometry as the 224x224 model image.
- Historical C28/C29/C30 configs explicitly retain `legacy_exterior_1` and `original` projection space.

## Frozen Candidate Audit

Candidate directories `00000..35695`:

| Camera | Clips | Fraction |
|---|---:|---:|
| exterior_image_1_left | 17,698 | 49.58% |
| exterior_image_2_left | 17,998 | 50.42% |

Selected-camera calibration sources in the frozen candidate pool:

| Camera | GT | Pred |
|---|---:|---:|
| exterior_image_1_left | 14,086 | 3,610 |
| exterior_image_2_left | 14,401 | 3,597 |

Predicted calibrations include both `Reprojection_error` and `num_matches` metrics. A stratified visual audit found a small number of suspect projections but a majority of usable Pred calibrations, so the current corrected dataset retains them.

Corrected dataset after calibration/projection filtering (GT + Pred):

| Camera | Episodes | Windows | Window fraction |
|---|---:|---:|---:|
| exterior_image_1_left | 17,121 | 178,941 | 49.59% |
| exterior_image_2_left | 17,375 | 181,874 | 50.41% |
| Total | 34,496 | 360,815 | 100% |

## Verification

- Clip `07970` now selects `exterior_2`, serial `26638268`.
- Corrected proprioception at step 0 is `[-0.11884552, -0.13036942, 0.73732126]`.
- Decoded model-input pixel is `(98.25, 96.96)`, aligned with the gripper.
- Camera translation action round-trip back to base coordinates has max absolute error `3.8e-11`.
- C30 corrected end-to-end smoke forward passed:
  - `action_pred`: `(1, 8, 9)`
  - `traj_pred`: `(1, 8, 9)`
- Calibration source, metric type, and quality value are retained in each payload.

## Saved Visualizations

- `outputs/c28_camera_audit/07970_dataset_camera_compare.png`
- `outputs/c28_camera_audit/multi_episode_dataset_camera_compare.png`
- `outputs/c28_camera_audit/corrected_dataset_alignment_20_samples.png`
- `outputs/c28_camera_audit/suspect_raw_sequences.png`
- `outputs/c28_camera_audit/corrected_gt_alignment_20_samples.png` (final GT-only audit)
- `outputs/c28_camera_audit/pred_calibration_stratified_32_samples.png`

In the comparison images, green is the corrected meta-selected camera projection and red is the legacy exterior-1 projection.
