# Gen2Act 使用 Wan 2.1 VAE 作为 Encoder 的 Smoke Test

## 目标

在当前目录的 `uv` 环境中，验证 Gen2Act 能否使用官方 Wan 2.1 VAE 代替 DINO 视觉编码器，并依次跑通：

1. Wan VAE 适配层单元测试。
2. 官方 `Wan2.1_VAE.pth` 的真实 GPU 编码。
3. Gen2Act 完整 policy 的单卡前向、反向与 optimizer step。
4. 8 卡 RTX 4090 的单机 DDP smoke test。

测试对象是 Wan 2.1 的 VAE，不需要下载 Wan 1.3B/14B diffusion model 或 T5 权重。

## 当前结论

截至 2026-08-10，仓库已经包含 Wan VAE 适配代码、专用配置和单元测试：

- 适配器：`r2r_gen2act/modeling/wan_vae.py`
- policy 接入：`r2r_gen2act/modeling/factory.py` 和 `r2r_gen2act/modeling/fused_query_flow_policy.py`
- 配置：`configs/droidFULL_wanvae_jointvelocity_pi05_letterbox_fulltrain.yaml`
- 测试：`tests/test_wan_vae.py`

已完成的验证：

- 当前 `.venv` 为 Python 3.11.15。
- PyTorch 为 `2.11.0+cu128`，torchvision 为 `0.26.0+cu128`。
- `uv lock --check` 通过。
- `tests/test_wan_vae.py` 结果为 `4 passed`。
- Wan VAE 专用配置 schema 校验通过。
- 适配器接受 `[B,T,3,H,W]`、值域 `[0,1]` 的 Gen2Act 视频，转换为 Wan 使用的 `[-1,1]`，输出 16 通道时空 latent。
- Wan VAE 参数被冻结；训练发生在 latent-to-query adapter 和 action policy。

当前尚未完成真实 GPU 测试，原因是执行检查时所在会话中：

- `nvidia-smi` 无法连接 NVIDIA driver。
- `torch.cuda.is_available()` 为 `False`，PyTorch 可见 GPU 数为 0。
- 官方 Wan2.1 仓库和 `Wan2.1_VAE.pth` 尚不存在。
- 正式配置引用的 `/mnt/pfs/...` DROID 数据、manifest、采样缓存和 intrinsics 在当前节点不可见。

因此，当前结论是：**代码级适配已通过测试，但是否能在这台 8 卡 4090 上使用真实官方权重完成训练，仍需按下面流程验证。**

## 1. 检查 8 卡与 uv 环境

```bash
cd /mnt/afs/dongyifei/DreamFlyWheel/gen2act
export UV_CACHE_DIR=/tmp/gen2act-uv-cache

nvidia-smi -L
uv sync --frozen

uv run --no-sync python - <<'PY'
import torch

print("torch:", torch.__version__)
print("cuda runtime:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("visible GPUs:", torch.cuda.device_count())
print("bf16 supported:", torch.cuda.is_bf16_supported())

assert torch.cuda.is_available()
assert torch.cuda.device_count() == 8
assert torch.cuda.is_bf16_supported()
PY
```

预期结果：PyTorch 可见 8 张 GPU，且支持 bf16。若这里失败，应先修复宿主机驱动、容器 GPU 映射或作业调度资源，不能继续归因于 Wan VAE。

执行一次 `uv sync --frozen` 后，后续命令使用 `uv run --no-sync`，避免 smoke test 期间自动同步环境或移除临时安装的 Wan 依赖。

## 2. 获取官方 Wan2.1 源码

建议把官方仓库放在 Gen2Act 目录之外，避免把第三方仓库加入当前 git worktree。Gen2Act
直接加载 `wan/modules/vae.py`，不执行 Wan 的包初始化，也不需要安装 Wan 的整套依赖：

```bash
cd /mnt/afs/dongyifei/DreamFlyWheel
git clone --depth 1 https://github.com/Wan-Video/Wan2.1.git Wan2.1
git -C Wan2.1 rev-parse HEAD

cd gen2act
uv run --no-sync python - <<'PY'
from pathlib import Path

source = Path("../Wan2.1/wan/modules/vae.py").resolve()
assert source.is_file(), source
print(source)
PY
```

记录 `git rev-parse HEAD` 的输出，便于固定测试所用 Wan2.1 版本。不要执行
`uv pip install -r ../Wan2.1/requirements.txt`；Wan VAE 源文件所需的 PyTorch 和 `einops`
已由 Gen2Act 环境提供，安装整套 Wan 依赖会引入 NumPy 版本冲突。
`9737cba9c1c3c4d04b33fcad41c111989865d315`

## 3. 只下载 Wan VAE 权重

```bash
cd /mnt/afs/dongyifei/DreamFlyWheel/gen2act
mkdir -p ../wan-checkpoints/Wan2.1-T2V-1.3B

uv run --no-sync hf download \
  Wan-AI/Wan2.1-T2V-1.3B \
  Wan2.1_VAE.pth \
  --local-dir ../wan-checkpoints/Wan2.1-T2V-1.3B

export WAN21_ROOT="$(realpath ../Wan2.1)"
export WAN_VAE_CHECKPOINT="$(realpath ../wan-checkpoints/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth)"

test -f "$WAN21_ROOT/wan/modules/vae.py"
ls -lh "$WAN_VAE_CHECKPOINT"
sha256sum "$WAN_VAE_CHECKPOINT"
```

后续启动训练的每个 shell 都必须设置 `WAN21_ROOT` 和 `WAN_VAE_CHECKPOINT`。

## 4. 运行代码级测试

```bash
uv run --no-sync python -m pytest -q tests/test_wan_vae.py

uv run --no-sync python scripts/validate_config.py \
  --config configs/droidFULL_wanvae_jointvelocity_pi05_letterbox_fulltrain.yaml
```

预期结果：

- pytest 显示 `4 passed`。
- 配置检查显示 `Config OK`。

这里使用的是 fake Wan 实现，只能验证接口、shape、fallback 规则和 adapter 梯度，不能代替真实权重测试。

## 5. 使用真实官方权重测试 Encoder

先只使用 GPU 0 测试 1 帧和正式配置使用的 8 帧输入：

```bash
CUDA_VISIBLE_DEVICES=0 uv run --no-sync python - <<'PY'
import torch
from r2r_gen2act.modeling.wan_vae import WanVAEBackbone

vae = WanVAEBackbone(dtype="bfloat16").cuda().eval()
assert vae.backend == "official", vae.backend

for frames, expected_latent_frames in ((1, 1), (8, 2)):
    torch.cuda.reset_peak_memory_stats()
    video = torch.rand(1, frames, 3, 224, 224, device="cuda")
    latent = vae(video)
    torch.cuda.synchronize()
    peak_gib = torch.cuda.max_memory_allocated() / 2**30
    print(frames, tuple(latent.shape), latent.dtype, f"peak={peak_gib:.2f} GiB")
    assert latent.shape == (1, 16, expected_latent_frames, 28, 28)
    assert torch.isfinite(latent).all()
PY
```

必须显式检查 `vae.backend == "official"`。正式配置允许回退到已有的 DiffSynth-Studio；如果只检查程序是否退出成功，可能会掩盖官方 Wan2.1 接口不兼容的问题。

## 6. 准备有限数据 smoke 配置

`raw_droid` 不直接读取 TFDS/RLDS 的 `.tfrecord` 分片。它要求每个 episode 都是
一个目录，目录中有 `episode.parquet`、`metadata.json` 和三路 MP4。因此不要把
`.../droid/droid_100/1.0.0` 直接填到 `data.root`。

如果本机已有官方 DROID100 TFRecord，以及由
`IDM_dump/convert_official_droid100_to_franka_lerobot.py` 生成的 LeRobot 三路视频，
可运行下面的无损 smoke 转换。它从**原始 TFRecord**写出 commanded
`joint_velocity`/`gripper_position`，只将 LeRobot 的三路 MP4 建立符号链接，不使用
LeRobot 派生的 28 维 Franka action：

```bash
cd /mnt/afs/dongyifei/DreamFlyWheel/gen2act

uv run --no-sync python scripts/convert_droid100_to_raw_smoke.py \
  --tfrecord-root /mnt/afs/dongyifei/DreamFlyWheel/GR00T-Dreams/dataset/droid/droid_100/1.0.0 \
  --lerobot-root /mnt/afs/dongyifei/DreamFlyWheel/GR00T-Dreams/IDM_dump/data/droid100_official_franka \
  --output-root /mnt/afs/dongyifei/DreamFlyWheel/gen2act/artifacts/droid100_raw_smoke \
  --manifest /mnt/afs/dongyifei/DreamFlyWheel/gen2act/artifacts/droid100_raw_smoke_pi05_manifest.json \
  --max-episodes 16
```

首次运行会创建输出目录；为防止意外混用旧数据，如果输出或 manifest 已存在，脚本会
停止而不会覆盖。若要重建，先人工确认后删除这两个明确的 `artifacts/droid100_raw_smoke*`
目标。

创建 `configs/droidFULL_wanvae_smoke.yaml`：

```yaml
_base_: droidFULL_wanvae_jointvelocity_pi05_letterbox_fulltrain.yaml

experiment:
  name: wanvae_smoke
  output_dir: outputs/wanvae_smoke

data:
  root: /mnt/afs/dongyifei/DreamFlyWheel/gen2act/artifacts/droid100_raw_smoke
  filter_manifest: /mnt/afs/dongyifei/DreamFlyWheel/gen2act/artifacts/droid100_raw_smoke_pi05_manifest.json
  max_episodes: 16
  max_windows: 64
  val_count: 2
  validate_manifest_paths: true
  native_action_sampling:
    enabled: false
  wrist_current:
    raw_root: /mnt/afs/dongyifei/DreamFlyWheel/gen2act/artifacts/droid100_raw_smoke

model:
  backbone:
    # Smoke test 必须强制使用 official backend，不能静默回退。
    fallback_backends: []

train:
  batch_size: 1
  epochs: 1
  num_workers: 0
  persistent_workers: false
  logging:
    log_every: 1
    wandb:
      enabled: false
  checkpoint:
    every_steps: 0
```

Raw-DROID adapter 强制要求与 `data.root` 匹配的 filtering manifest。如果数据在本机路径与原配置不同，必须同时修改 root、manifest 和 wrist root。这里的 `wrist_current.raw_root` 与 `data.root` 保持相同，是为了清楚表达三路视频都来自同一转换根目录；当前 `RawDroidDataset` 已直接从该根目录的 episode 中解析 wrist MP4。

## 7. 检查数据与单卡完整训练

```bash
CUDA_VISIBLE_DEVICES=5 python scripts/check_train_env.py \
  --config configs/droidFULL_wanvae_smoke.yaml \
  --expected-gpus 1

CUDA_VISIBLE_DEVICES=5 python scripts/inspect_dataset.py \
  --config configs/droidFULL_wanvae_smoke.yaml

CUDA_VISIBLE_DEVICES=5 python scripts/train.py \
  --config configs/droidFULL_wanvae_smoke.yaml
```

单卡成功判据：

- 日志包含 `Wan-VAE backend=official`。
- 日志包含 `vision=wan_vae`。
- 至少完成一次 forward、backward 和 optimizer step。
- loss 为有限值，没有 NaN/Inf。
- 生成 `outputs/wanvae_smoke/latest.pt`。
- GPU 0 没有 OOM，显存使用稳定。

这一步同时覆盖真实数据读取、官方 VAE、latent readout、Flow DiT、损失、反向传播和优化器状态分配。

## 8. 运行 8 卡 DDP smoke test

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

uv run --no-sync python scripts/check_train_env.py \
  --config configs/droidFULL_wanvae_smoke.yaml \
  --expected-gpus 8

TORCH_NCCL_ASYNC_ERROR_HANDLING=1 NCCL_DEBUG=INFO \
uv run --no-sync torchrun \
  --standalone \
  --nnodes=1 \
  --nproc-per-node=8 \
  scripts/train.py \
  --config configs/droidFULL_wanvae_smoke.yaml
```

8 卡成功判据：

- 日志显示 `world_size=8 per_gpu_batch=1 global_batch=8`。
- 8 个 rank 均成功初始化 NCCL。
- 8 张 GPU 都有合理且相近的显存占用。
- 所有 rank 正常完成，未出现 collective timeout 或 unused parameter 错误。
- rank 0 写出 `outputs/wanvae_smoke/latest.pt`。

不要无参数直接调用 `scripts/run_distributed_train.sh`，它默认使用 `/root/miniconda3/envs/gen2act`。本测试应直接使用上述 `uv run --no-sync torchrun`，或者显式把该脚本的 `PYTHON`、`TORCHRUN` 指向当前 `.venv/bin`。

## 9. 探测 4090 可用 batch size

正式配置继承的 `train.batch_size` 是每卡 12；8 卡时全局 batch 为 96。RTX 4090 只有 24 GiB 显存，不能直接假定每卡 12 可运行。

建议固定其他参数，依次测试每卡 batch：

```text
1 -> 2 -> 4 -> 8 -> 12
```

每档至少连续执行几十个训练 step，并用以下命令监控：

```bash
watch -n 1 nvidia-smi
```

选择满足以下条件的最大 batch：

- 连续运行无 OOM。
- 峰值显存建议不超过约 22 GiB，给 CUDA/NCCL 波动留余量。
- loss 保持有限。
- 8 张卡负载和显存基本均衡。

当前 trainer 没有实际使用 gradient accumulation。若正式训练需要降低每卡 batch，必须接受全局 batch 改变并重新评估学习率，或者另行实现梯度累积，不能仅在配置中添加一个未被 trainer 消费的字段。

## 最终判定

只有下面三层全部通过，才能确认“Gen2Act 能在这台 8 卡 4090 上使用 Wan VAE 作为 encoder 运行”：

1. 真实官方 `Wan2.1_VAE.pth` 在单卡输出正确、有限的 latent，且 `backend == official`。
2. 单卡有限数据训练完成 forward、backward、optimizer step 并写出 checkpoint。
3. 8 卡 DDP 有限数据训练正常结束，所有 rank 和 GPU 均参与计算。

上述 smoke test 只能证明工程链路可运行，不代表 Wan VAE 特征对 Gen2Act 动作预测有效。模型效果仍需通过正式训练、验证集 loss、动作误差和与 DINO baseline 的对照实验评估。
