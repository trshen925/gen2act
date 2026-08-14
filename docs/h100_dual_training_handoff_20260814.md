# 双基线 H100 训练记录与交接（2026-08-14）

本文记录在共享存储上并行运行的两次 full-DROID 训练：主线 C41（DINOv2）和
`feature/wan-vae` 基线（Wan-VAE）。内容覆盖代码血缘、本地化配置、数据映射、
训练进度、checkpoint 与重启方法。

> 快照时间：2026-08-14 10:37 UTC。H100 节点会重启并更换 IP，因此 IP 只表示
> 本次快照；数据、代码、环境和输出均在共享 `/mnt` 上。

## 1. 当前结论

两项训练均在 `10.119.110.203` 正常运行，使用同一个 8×H100 节点并各占四张卡：

| 实验 | GPU | W&B step | 总预算 | 估算进度 | 最近完整验证损失 | 最近 step ckpt |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| C41 gripper-event2s | 0–3 | 348,600 | 416,680 | 83.7% | 0.07688（epoch 16） | `step_000348344.pt` |
| Wan-VAE full DROID | 4–7 | 243,500 | 312,510 | 77.9% | 0.18682（epoch 11） | `step_000239174.pt` |

W&B 在快照时将两项 run 都标为 `running`。GPU 0–3 占用约 43.8–44.1 GiB，
利用率 71–90%；GPU 4–7 占用约 36.9–37.2 GiB，利用率 84–100%。

W&B：

- C41：[`local_C41_gripper_event2s_full_fromscratch`](https://wandb.ai/trshen925-peking-university/gen2act/runs/uzhnqy7r)，run id `uzhnqy7r`
- Wan-VAE：[`local_wanvae_full_droid_4gpu`](https://wandb.ai/trshen925-peking-university/gen2act/runs/ioipn2th)，run id `ioipn2th`

## 2. 代码血缘

两项实验来自不同 Git 历史，不能把它们当作同一工作树中的两个配置：

| 实验 | 工作树 | 分支/来源 | 训练时的本地 HEAD | 上游基点 |
| --- | --- | --- | --- | --- |
| C41 | `/mnt/home/sujiayi/Projects/gen2act` | 主线，本地 `master` | `d258e5dacb0e8f91f32e82d1d4bb89879bbd66dd` | `origin/master` 的 `79084e383f32a20bc4f0a4993b2cc6261fd382c0` |
| Wan-VAE | `/mnt/home/sujiayi/Projects/gen2act-wan-vae` | `local/wan-vae`，跟踪 `origin/feature/wan-vae` | `598ca5f8e9e860e3b60241218a20d36a562a2937` | `8ee93b7f0e386921da307abc1537e811d15507be` |

C41 主线的 `d258e5d` 是“远程模型/训练更新”和本地集群适配的 merge；其本地适配父提交为
`a7b366191c8fa52081f2fe164fd47845c77217f1`。Wan-VAE 的 `598ca5f` 是在同事的
`feature/wan-vae` 上增加本地集群适配的提交。

训练启动时还使用了工作树中的本地配置/脚本修改。本次文档提交没有把这些未整理改动
混入 commit；复现实验时应在上述共享工作树使用这些文件，或先另行审核并提交它们。

### C41 相关文件

- `configs/local_C40_jointvelocity_pi05_letterbox_fromscratch.yaml`：共享数据路径、JPEG 帧缓存、from-scratch 优化器/调度器和 checkpoint 策略。
- `configs/local_C41_gripper_event2s_full_fromscratch.yaml`：同事 C41 gripper-event 采样策略的本地 full-DROID 版本。
- `configs/local_C41_gripper_event2s_full_4gpu_resume.yaml`：将 8×12 改成 4×24，保持 global batch 96 和每 epoch 20,834 step。
- `scripts/run_train_c41_local_fromscratch.sh`：最初从零启动。
- `scripts/run_train_c41_local_4gpu_resume.sh`：GPU 0–3 完整状态续训。
- `r2r_gen2act/training/trainer.py`：允许从“完整 epoch”checkpoint 切换 DDP 几何；epoch 中间的 checkpoint 仍强制保持 world size、batch 和 steps/epoch。
- `scripts/extract_frames_raw_droid.py`：把 front/wrist 视频预抽成 JPEG；缺缓存的 episode 会回退到 MP4。

### Wan-VAE 相关文件

- `configs/local_wanvae_full_droid_4gpu.yaml`：官方 Wan VAE backend、全量 DROID、本地路径和 4-GPU 参数。
- `configs/local_wanvae_smoke_4gpu.yaml`：16 episode/384 window 的 smoke 配置。
- `scripts/run_train_local_wanvae_4gpu.sh`：GPU 4–7 启动/续训入口。
- 官方 Wan 源码：`/mnt/project/simvla/world_policy_storage/gen2act/third_party/Wan2.1`，commit `9737cba`。
- 官方 VAE 权重：`/mnt/project/simvla/world_policy_storage/gen2act/pretrained/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth`。
- 权重 SHA-256：`38071ab59bd94681c686fa51d75a1968f64e470262043be31f7a094e442fd981`。

## 3. 数据与 mapping 本地化

两项训练使用同一套本地 full-DROID 数据和 pi0.5 非静止过滤结果：

```text
数据根目录  /mnt/project/simvla/world_policy_storage/gen2act/droid-decompressed-1.0.1
过滤 manifest /mnt/project/simvla/world_policy_storage/gen2act/artifacts/raw_droid_1_0_1_pi05_manifest.json
训练 episode 55,318
验证 episode 500
每 epoch 训练 window 2,000,000
验证 window 11,852
```

同事配置中的 `/mnt/pfs/...` 绝对路径没有直接沿用。mapping 的“转换”实际是保留相同
采样语义，以本地 manifest/DROID 路径生成或加载等价索引：

- C41 mapping：`/mnt/project/simvla/world_policy_storage/gen2act/artifacts/c41_raw_droid_gripper_target_pre3s_post1s_window_index.json`
  - `event_before: 15`，`event_after: 45`；实现语义为围绕事件覆盖 target frame
    `[-45, +15]`，对应 DROID 15 FPS 下事件前三秒至后一秒。
  - normal/close/release 比例为 `0.20/0.40/0.40`，seed 42。
- Wan-VAE mapping：`/mnt/project/simvla/world_policy_storage/gen2act/artifacts/c40_raw_droid_jointvelocity_window_index.json`
  - 沿用 C40 的 joint-velocity/pi0.5 采样语义；normal/close/release 为
    `0.60/0.22/0.18`，seed 42。

因此模型行为和同事配置保持一致，本地化只替换机器相关的代码/权重/数据/输出路径，
并按当前 GPU 划分调整单卡 batch。

## 4. 训练配置对照

| 项目 | C41 | Wan-VAE |
| --- | --- | --- |
| 模型视觉表示 | DINOv2 ViT-B/14 主线 | 官方 Wan 2.1 VAE backend |
| 初始化 | 从零训练任务头/DiT；仅保留 DINOv2 预训练 | 从零训练任务模型；加载官方 Wan VAE 权重 |
| GPU / 单卡 batch | 0–3 / 24 | 4–7 / 24 |
| global batch | 96 | 96 |
| epoch / steps per epoch | 20 / 20,834 | 15 / 20,834 |
| 优化器 | AdamW | AdamW |
| LR / scheduler | `1e-4` / cosine，5% warmup | `3e-5` / cosine（继承 baseline） |
| loader | 8 workers/rank，prefetch 2，非 persistent | 8 workers/rank（本地配置） |
| checkpoint | 每 epoch + 每 5,000 step，保留最近 2 个 step ckpt | 每 epoch + 每 5,000 step，保留最近 2 个 step ckpt |
| 输出目录 | `.../outputs/local_C41_gripper_event2s_full_fromscratch` | `.../outputs/local_wanvae_full_droid_4gpu` |

共同环境：`/mnt/home/sujiayi/miniconda3/envs/gen2act`，Python 3.11.15，
PyTorch 2.6.0+cu124。

## 5. 训练进度

### C41

最近几个完整 epoch 的 validation loss 持续下降：

| 完成的 epoch 字段 | val loss |
| ---: | ---: |
| 12 | 0.0834 |
| 13 | 0.0812 |
| 14 | 0.0794 |
| 15 | 0.0785 |
| 16 | 0.0769 |

快照时 W&B 的 `train/loss` 为 0.07259，step 348,600；最新已落盘 step
checkpoint 是 348,344。`latest.pt` 是完成 epoch 16 时的完整状态，下一次中断时优先使用
最新 step checkpoint（保持当前 4-GPU 几何），需要改变 DDP 几何时使用 completed-epoch
`latest.pt`。

### Wan-VAE

最近几个完整 epoch 的 validation loss 同样持续下降：

| 完成的 epoch 字段 | val loss |
| ---: | ---: |
| 8 | 0.1958 |
| 9 | 0.1912 |
| 10 | 0.1896 |
| 11 | 0.1868 |

快照时 W&B 的 `train/loss` 为 0.14208，step 243,500；最新已落盘 step
checkpoint 是 239,174，训练仍在其后继续。

## 6. 节点迁移记录

训练经历了多次集群重启；所有 checkpoint 和代码均在共享 `/mnt`，因此迁移节点无需复制：

| 节点 | 启动/恢复点 | 该节点结束前记录 |
| --- | --- | --- |
| `10.119.101.221` | C41 从 8 GPU 改为 4 GPU；Wan-VAE 正式启动 | C41 93,336；Wan 5,000 |
| `10.119.99.131` | C41 93,336；Wan 5,000 | C41 234,174；Wan 140,004 |
| `10.119.97.60` | C41 234,174；Wan 140,004 | C41 265,008；Wan 166,672 |
| `10.119.108.191` | C41 265,008；Wan 166,672 | C41 completed epoch 16（333,344）；Wan completed epoch 11（229,174） |
| `10.119.110.203` | C41 333,344；Wan 229,174 | 本文快照时仍在运行 |

恢复到相同 W&B run 后，日志可能出现 “step less than current step” 警告。这是因为本地
checkpoint 略早于 W&B 已接收的 step；追平旧的 W&B step 之前重复记录会被忽略，但训练、
优化器和 checkpoint 仍正常前进。

## 7. 重启操作

当前节点通过跳板机连接：

```bash
ssh -J galbot@192.168.94.233 sujiayi@10.119.110.203
```

C41（GPU 0–3）：

```bash
cd /mnt/home/sujiayi/Projects/gen2act
CUDA_VISIBLE_DEVICES=0,1,2,3 \
RESUME_FULL_CHECKPOINT=/mnt/project/simvla/world_policy_storage/gen2act/outputs/local_C41_gripper_event2s_full_fromscratch/latest.pt \
bash scripts/run_train_c41_local_4gpu_resume.sh
```

Wan-VAE（GPU 4–7）：

```bash
cd /mnt/home/sujiayi/Projects/gen2act-wan-vae
CUDA_VISIBLE_DEVICES=4,5,6,7 \
RESUME_FULL_CHECKPOINT=/mnt/project/simvla/world_policy_storage/gen2act/outputs/local_wanvae_full_droid_4gpu/latest.pt \
bash scripts/run_train_local_wanvae_4gpu.sh
```

注意：

1. `latest.pt` 是 completed-epoch checkpoint，适合节点/GPU 几何变化后的安全恢复。
2. `step_*.pt` 可减少丢失的训练量，但 epoch 中间恢复必须保持 4 GPU × 24、global batch 96
   和 20,834 steps/epoch。
3. 若要重新从零开始，必须先换新的 `experiment.name`、`output_dir` 和 W&B run id，避免覆盖
   现有 checkpoint 或把新实验写进旧 run。
4. 不要在两个工作树之间交叉使用 checkpoint；两条模型分支的状态字典并不等价。

## 8. OneDrive checkpoint 备份

以下上传均已通过脚本的目标端大小与哈希校验：

```text
onedrive:/wp_dev/gen2act/checkpoints/2026-08-13/C41_step_000234174.pt
onedrive:/wp_dev/gen2act/checkpoints/2026-08-13/WanVAE_step_000140004.pt
onedrive:/wp_dev/gen2act/checkpoints/2026-08-14/C41_step_000343344.pt
onedrive:/wp_dev/gen2act/checkpoints/2026-08-14/WanVAE_step_000239174.pt
```

2026-08-14 的备份是上传任务启动时选中的最新 checkpoint；训练随后继续产生了更新的
本地 checkpoint，因此 OneDrive 记录是灾备锚点，不代表本文快照时最新落盘的 step。
