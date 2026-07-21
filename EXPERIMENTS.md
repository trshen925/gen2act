# 实验记录 (Experiments Log)

任务：Robot-to-Robot Gen2Act，从 `source_video`(参考机器人视频) + `target_history`(目标机器人观测) + `qpos` 预测目标机器人未来动作。
数据：`/mnt/pfs/share/shentingrui/dataset/droid-2000-new`（2965 episodes，留出 100 作 inference 集，`val_count=100`）。

> ⚠️ 故意保留的"泄漏"：`source_video = gt.mp4`（与 target 同一视频，且 linspace 采样覆盖整段，含未来帧）。本意是先看泄漏能否被利用。

判断"学会"的尺子（不看 loss 绝对值）：
1. 明显赢过傻瓜基线（回归比"输出均值"，分类比"猜多数类"）；
2. 预测随输入变化且与真值相关（corr>0.5、pred_std≈tgt_std）；
3. train/infer 同步下降（gap 小）。
经验门槛：xyz held-out **MAE ≤1.5cm 且 corr>0.6** 才算平移"学会"。

---

## Exp 1 — 6D 回归 (regression)
- **config**: `droid2000new_future5_chunk4_pose6d_regression_qpos_ft4dinov2_latent128_aug50`
- **设置**: 6D 旋转回归(pose=xyz3+6D6=9)，action chunk=4(预测+5/+10/+15/+20)，4 query token，latent_tokens=128；DINOv2 冻结+解冻最后4层(lr×0.1)，~127M 可训练；batch=32/卡，bf16，cosine+warmup，lr=3e-4；50 epochs。
- **loss**: train 0.236→0.053；**inference 在 ep4 触底(0.211)，之后单调升到 0.365** → 过拟合。
  - infer/action 全程平 ~0.098–0.103（ep4 起几乎不动）；infer/gripper 爆炸 0.52→1.31（train→0.022）。
- **诊断 (best.pt @ep4，800 held-out 窗口)**：
  - XYZ MAE = **2.90cm**，"输出均值"基线 = 3.05cm（仅好 ~5%）。
  - 相关性：dx 0.30 / dy 0.02 / dz **0.48**；旋转 corr≈0，pred_std 仅为 tgt_std 的 ~10%（塌成常数）。
  - 夹爪 acc=0.797 vs 多数类 0.576（学到了，但 CE 高=过度自信/标定差）。
- **结论**: **大部分在"输出均值"**。只有 dz（+一点 dx）有弱信号，旋转完全没学，dy 没学。best.pt=ep4。

## Exp 2 — 256-bin 分类 (cls256，P0 mode-collapse 修复尝试)
- **config**: `droid2000new_future5_chunk4_pose6d_cls256_qpos_ft4dinov2_latent128_aug50`
- **设置**: 同 Exp1，但动作头换成离散化分类(256 bins/维)；只跑了 **6 epochs**（快速 sanity）。
- **loss**: train 3.91→3.66；inference ep4 触底(3.785)，ep6 回升(3.794)。action CE ~3.7（随机基线 ln256=5.545）。
- **诊断 (best.pt @ep4)**：
  - top1 bin 准确率 vs 多数类：dx 0.056/0.031，dy 0.051/0.037，dz 0.095/0.055（xyz 略赢基线）；旋转各维 ≈ 多数类（没赢）。
  - 相关性：dx 0.29 / dz 0.34 / dy -0.04；旋转 ≈0。
  - 解码回连续：**XYZ MAE = 5.15cm，比"输出均值"基线 2.76cm 还差**（argmax 偶尔自信选到远 bin）。
- **结论**: **分类头没解决 mode collapse，xyz 反而更差**；旋转仍塌成常数。仅 6 epoch 偏短，但趋势同 Exp1。

## 跨实验小结
两种输出头（回归/分类）都塌到傻瓜基线 → 瓶颈大概率**不在输出头，而在输入信号**：模型可能无法从 `source(linspace 采 gt 8帧)+target history+qpos` 读出"未来 +5/+10/+15/+20 步的具体位姿"。
**下一步**：把 source 改成显式覆盖未来帧的采样，验证"泄漏到底能不能被利用"——区分是"输入没信息" vs "模型/损失问题"。见 Exp 3。

## Exp 3 — source 覆盖未来帧 (泄漏可利用性测试)  ✅ 结论：泄漏可被利用
- **config**: `droid2000new_future5_chunk4_pose6d_regression_qpos_ft4dinov2_latent128_srcfuture`（与 Exp1 唯一差别：`source_sampling: future_window`，source 采 `[target_step, target_step+20]` 即被预测的未来帧；关 source_jitter）。
- **加速**: 预解码所有帧成 jpg（`scripts/extract_frames.py`，398567 帧）+ dataset 改读 jpg（`data.frames_subdir: frames`）→ GPU 利用率 0→**99%**，~6min/epoch（之前数据受限 GPU 饿死）。本机单卡 num_workers=16。
- **loss 趋势**: infer/action 从 Exp1 的 ~0.098 平台**持续下降到 0.079（ep12，仍在降）**，且 train≈infer（泛化好，不像 Exp1 过拟合）。
- **诊断（latest.pt @ep13，800 held-out 窗口）vs Exp1(linspace)**：
  | 维度 | Exp1 corr | srcfuture corr |
  |---|---|---|
  | dz | 0.48 | **0.92** |
  | dx | 0.30 | 0.37 |
  | r00/r20/r11 | ≈0 | **0.40–0.45** |
  | dy, r10/r01/r21 | ≈0 | 仍 ≈0（这几维数据近常数/低信号）|
  - **XYZ MAE = 2.18cm**，"输出均值"基线 3.05cm（**↓29%，明显赢**）；Exp1 是 2.90cm≈持平。
- **最终（20 epoch 跑完，latest ep20）**: infer/action 在 ep14 见底 ~0.0775 后平台震荡(ep20=0.078)，train 继续降(0.069)→ep16 起轻微过拟合。逐维 corr：dx 0.43 / **dz 0.92** / **dy 0.00** / 旋转 r00 0.45,r20 0.41,r11 0.41（学到）vs r10/r01/r21 ≈0（没学）。XYZ MAE **2.19cm**（基线 3.05cm，↓28%）。
- **结论**: 喂未来帧后模型确实学会用 source（dz 0.48→0.92，xyz 从持平基线→明显赢）→ 之前学不动**部分是输入采样问题**，非纯模型/损失。但**有明确上限**：dy + 半数旋转维始终学不到（见下）。
- **🔑 根因（坐标系）**: 动作是**机器人基座系**下的位姿（确认：x 恒正~前方、y 对称于0~左右、z~高度），而 source 是**外参未知且逐 episode 变化的外部相机**(exterior1)。→ z 因重力对齐、视角不变而可学；x/y 与方位旋转映射到的图像方向随机位变化、模型不知相机外参 → **数学上欠定，学不了**。这解释了"学得会的维 vs 学不会的维"的分裂。
- **启示**: ① source 采样方式决定能否利用参考视频；② **要救 x/y/旋转：把动作预测改到相机系，或喂相机外参，或用腕部相机**（自我中心、几何固定）；③ 真实(无泄漏)场景要让生成的 source 含未来动作相关信息；④ 数据侧：预解码帧是单机训练必要优化（GPU 0→99%）。

## Exp 4 — 相机系动作（droid-ex extrinsics）  ⏳ 待跑
- **dataset**: `/mnt/pfs/share/shentingrui/dataset/droid-ex/droid_2000_with_extrinsics_filtered`（3313 clips；每个 episode 的 `data.json.calibration.extrinsics` 带单个 source camera 6D 外参，另有 `calibration.json` 备份）。
- **config**: `droidex2000_future5_chunk4_pose6d_camera_regression_qpos_ft4dinov2_latent128_aug50`
- **launcher**: `bash scripts/run_train_camera.sh`
- **代码改动**: 新增 action mapping `droid_observation_cartesian_future_delta_pose6d_camera`。translation target 从 base-frame `p_future - p_now` 变为 camera-frame `R_cam_base @ (p_future - p_now)`；rotation target 从 base-frame `R_future @ R_now^T` 变为 camera-frame `R_cam_base @ (R_future @ R_now^T) @ R_cam_base^T`，再取 Zhou 6D 前两列。
- **外参约定**: 当前按 `calibration.extrinsics[serial] = camera pose in robot base = [x,y,z,roll,pitch,yaw]` 处理，因此 `R_cam_base = R_base_camera^T`。配置里保留 `action.mapping.extrinsics_convention: camera_pose_in_base`，若后续发现数据实际是 base->camera，可改成 `base_to_camera` 直接重跑。
- **目的**: 验证 Exp3 的坐标系欠定诊断。把标签变到 source camera 坐标系后，x/y 和旋转不再需要模型隐式猜每个 episode 的外参，理论上应比 base-frame target 更容易从 exterior video 学到。
- **判断指标**: 对比 Exp1/Exp3 的 held-out corr 和 MAE，重点看 dx/dy、r10/r01/r21 是否从近 0 提升；如果 dz 保持高相关且 x/y/旋转提升，说明坐标系改动有效。
- **外参可视化检查**: `scripts/visualize_ee_projection.py` 可把 `observations.cartesian_position[:3]` 投影回 `gt.mp4`。示例：`/root/miniconda3/envs/gen2act/bin/python scripts/visualize_ee_projection.py --episode 0001 --frame 30 --output outputs/projection_checks/0001_frame030_ee_projection.png`；整段视频加 `--all`。当前 `0001` 在 `camera_pose_in_base` 约定下 projected_inside=90/90，可作为 sanity check。

## Exp 5 — 相机系动作 + 显式未来帧 source offsets  ⏸️ 配置已准备
- **背景**: Exp4 的 10-epoch 试跑在 epoch 6 暂停。该 run 使用 `source_sampling: linspace`，即 source 从整段 `gt.mp4` 均匀采 8 帧；这不是想要的 `+3,+6,...` 未来帧设置。
- **代码改动**: `WindowedRobotDataset._read_source_video` 新增 `source_sampling: future_offsets`。source frame 由 `target_step + data.source_future_offsets[i]` 得到，并继续从预解码 `frames/` 读取。
- **config**: `droidex2000_future_offsets_pose6d_camera_regression_qpos_ft4dinov2_latent128_aug50`
- **source 设置**: `source_future_offsets: [3, 6, 9, 12, 15, 18, 21, 24]`，`source_jitter.enabled: false`，避免破坏固定 offset 语义。例：`target_step=3` 时 source indices = `[6,9,12,15,18,21,24,27]`。
- **归一化尺度分析（camera-frame xyz, action chunk 后全部 future targets）**:
  - train actions=138776: x/y/z std≈0.050/0.057/0.054m；99.5% abs≈0.217/0.233/0.219m；99.9% abs≈0.285/0.314/0.292m。
  - val actions=4244: x/y/z std≈0.052/0.059/0.053m；99.5% abs≈0.227/0.258/0.215m；99.9% abs≈0.290/0.335/0.285m。
  - 旧 `[-0.10,0.10]` 会裁掉约 x 6-7%、y 9%、z 7% 的标签，偏紧。
- **新 bounds**: xyz 改为 `[-0.30, 0.30]`，rotation 6D 保持 `[-1,1]`。理由：覆盖约 99.5%-99.9% 的 camera-frame xyz target，避免 10cm bounds 系统性裁剪较大动作，同时不被极端点过度拉宽。
- **proprioception 修改**: 不再输入 `observations.joint_position` 7D qpos；改为 target_step 末端在 source camera 图像上的 2D 投影坐标 `[u_norm, v_norm]`。图像边界映射到 `[-1,1]`，不做 clamp，出画点自然超过 1 或小于 -1。模型 `proprioception_dim: 2`。
- **projection 数据过滤**: camera-projection proprioception 需要 intrinsics，因此缺/坏 intrinsics 的 episode 会跳过；同时过滤投影质量差的 episode（任意 z<=0，或超过 `[-3,3]` 的 projection 比例 >5%）。过滤后约 train=3096 episodes/33272 windows，val=93 episodes/993 windows。抽样 2000 个 train windows 的 projection 范围：u≈[-1.60,1.62]，v≈[-3.12,1.52]，`abs>1` 比例约 u 0.6%、v 5.0%。

## Exp 6 — videomt(ViT-S) 纯视频，相机系，修复优化器  ✅ 三轴平移全部学到
- **代码库**: `videomt`（VidEoMT, ViT-S/DINOv2 + query propagation），`tools/train_gen2act_standalone.py`。数据/动作同 droid-ex 相机系 future5（source frames offsets `[0,5,10,15,20]` 含未来帧；6D 旋转；bounds xyz±0.30）。**纯视频，无 proprioception**。
- **背景（坏版本）**: 初版用 `AdamW(model.parameters(), lr=1e-4)` 平 lr、无 warmup/LLRD/grad-clip → epoch 3 仍 **mean-collapse**：pred_std≈0、gripper loss 卡 ln2(0.693)、val xyz L2≈70mm(=均值基线)、gripper acc 0.50。诊断为 timm vit_small_patch14_reg4_dinov2 在缓存可加载，问题在训练配方。
- **修复（照搬原版 videomt + gen2act Exp4/5 配方）**: `build_param_groups` 加 **LLRD=0.7**（ViT 深层满 lr、浅层指数衰减、head/query 满 base_lr，norm/bias/embed 不加 WD）+ **warmup(3%)+cosine** schedule + **grad_clip=1.0** + AdamW betas(0.9,0.95)。新增 CLI `--llrd --warmup-steps --grad-clip --min-lr-ratio --weight-decay`。
- **过拟合 sanity**: 16 样本 40 步 pose 0.279→0.027、gripper 0.70→0.31（坏版本 6000 步不动）→ 确认是优化器问题。
- **正式 run** (`outputs/gen2act_droid_camera_vits_future5_llrd`, batch16, 30ep, ~12 it/s ~3min/ep): val xyz L2 单调降 61.7→45.5mm，gripper acc ~0.88。
- **逐维诊断 (epoch 20, 800 held-out)**: x **0.62** / y **0.89** / z **0.67**（pred_std/tgt 0.63/0.87/0.73）；旋转 r00/r20/r11/r21 corr 0.44–0.55，r10/r01 弱 0.15；XYZ L2 **46.9mm**(基线71.0mm,↓34%)，xyz_corr均值 0.73；旋转角误差 mean 9.3°/p50 6.6°。
- **平移三轴全部学到**（之前每个 gen2act 实验总有一轴卡 0：Exp1 dy0, Exp3 dy0, Exp5 x0.06）。videomt 反超带 proprio 的 Exp5（46.9 vs 59.5mm），且纯视频。
- **结论**: videomt 之前差**不是架构问题，是训练配方被简化**（平 lr/无 warmup/LLRD/clip）。原版 videomt 靠 LLRD+warmup+grad-clip+长 schedule + 密集监督才 work；补回优化配方后纯视频版即学到三轴平移+旋转。
- **gripper（更正）**: 早先report"acc 0.859 < 多数类 0.910"是**诊断脚本 bug**——对 `{-1,+1}` 标签误用 `max(mean,1-mean)`(mean(±1)=0.09→误得0.91)。**真实多数类基线 0.545**（train/val gripper 都平衡 +1 55%/-1 45%）。故 **gripper acc 0.859 ≫ 0.545 = 学得不错**，无类别不平衡问题。

## Exp 7 — videomt 续训 10 epoch（gripper 实为误报，主要为再榨性能）  ⏳ 进行中
- **背景更正**: gripper 本就学得不错（见 Exp6 更正），无需类别重加权。仍按用户要求从 Exp6 续训 10 epoch 再榨性能，并小幅提升 gripper 关注。
- **代码改动（保留备用）**: `action_losses` 加 `gripper_pos_weight`(BCE pos_weight)；trainer 加 `--resume`(载权重续训)、`--gripper-weight`(覆写)、`--gripper-pos-weight`。
- **方案**: 从 Exp6 epoch_0030 resume，+10 epoch，gripper_weight 0.2→0.3(小幅)，fine-tune lr~3e-5 + 短 warmup + cosine。
- **结果**: _(待填)_ 看 xyz L2 / 三轴 corr 是否再降、gripper acc(对 0.545 基线) 是否再升、且不退化。


## Exp 8 — flow-matching DiT 头（步骤1 可行性）  ⚠️ 中途负信号（ep9）
- **config**: `droidex2000_future_offsets_pose6d_camera_flowdit_qpos_llrd`（= Exp5 完全同设置，仅把回归 ActionHead 换成 164.3M 的 `FlowMatchingDiTHead`：hidden1024/6层/16头；cond=concat(fused_target,source) 256 tokens；rectified-flow velocity=action−noise，16 步 Euler 采样）。全模型 328.8M，LLRD0.7/warmup/cosine/grad-clip，batch24，24ep。
- **训练曲线（异常）**: train velocity-MSE 不降反升 ep4 0.171→ep6-9 0.26；infer(采样后 MSE)≈0.117–0.173 持平。
- **诊断（latest.pt @ep9，800 held-out 窗口，采样）**:
  | 维 | corr | MAE | pred_std/tgt_std |
  |---|---|---|---|
  | dx/dy/dz | 0.02 / 0.02 / -0.00 | 4.35/5.43/4.69cm | 0.75/0.88/0.82 |
  | 旋转 r0..r5 | 0.16/-0.05/0.03/-0.04/0.07/0.01 | — | 0.66–0.86 |
  - **XYZ MAE 4.82cm，比"输出均值"基线 3.47cm 还差 39%**。
- **判读**: pred_std≈tgt_std（非常数塌缩）但 **corr≈0** → 模型学到了动作的**边缘分布**(正确的每维方差)，却**没用上 conditioning**(条件映射没学到)。这是 flow-matching 版的"塌缩"：采样像在采先验，而非条件分布。当前**远不如回归头**(Exp5/Exp3 同数据 dz corr 能到 0.9)。
- **可能原因/假设**: ① DiT(adaLN-Zero 零初始化 + action_out 零初始化)起步是恒等映射，9ep 没学会用 cond；② DiT 头满 base_lr=3e-4 可能偏高(train loss 上升像 LR 过大)，或需更长 warmup/更小 head lr；③ flow-matching 对弱条件天然容易忽略 → 可能需要更强条件注入(如把 cond 也喂 adaLN，或 classifier-free guidance，或 x0-prediction 而非 velocity)。
- **处置**: 不杀，先跑完 24ep（后半 cosine 降 LR 也许收敛）；ep24 再诊断。若仍 corr≈0 → 结论"flow 头在此设置下不及回归头"，按上述假设调整(降 head lr / x0-pred / CFG / 条件 token 加位置&类型 embed)重训。
- **对比基准**: 回归头(Exp3 srcfuture) dz corr 0.92、XYZ MAE 2.19cm(↓28% vs 基线)。flow 头要"可行"至少需 corr>0.5 且 XYZ MAE<基线。

### Exp 8 最终（ep24，24 epoch 跑完）— flow 头确实能学，但未全面胜过回归头
- **训练曲线**: train velocity-MSE 前半平台(ep4-11 ~0.25，假阴性) → LR cosine 衰减后半段骤降，ep24 **0.096**(infer 采样MSE ~0.10–0.12)。
- **终诊断（latest.pt @ep24，800 held-out 采样）**:
  | 维 | corr | MAE | pred_std/tgt_std |
  |---|---|---|---|
  | dx | **0.018** | 4.38cm | 0.82 |
  | dy | **0.764** | 2.66cm | 0.95 |
  | dz | 0.182 | 4.14cm | 0.95 |
  | r0/r4 | 0.26/0.22 | 0.02 | ~0.85/1.08 |
  | r1/r2/r3/r5 | ≈0 | — | ~1.0 |
  - **XYZ MAE 3.73cm，比"输出均值"基线 3.47cm 差 7.6%**。
- **结论**: ① **flow-matching DiT 头能学到条件分布**——dy corr 0.76、pred_std≈tgt_std 是硬证据(非塌缩、非采先验)；ep9 的"corr≈0"是 LR 还高、后半才学到的假阴性。② 但**总体未胜回归头**：dx 仍学不到(相机系 x 轴老大难，回归头同样)，且 flow 的**采样发散**在学不到的维(dx)上把 MAE 顶得比"均值预测"还高(回归头在学不到的维会退化成输出均值，MAE 反而低)。③ 对比回归头 Exp3(srcfuture) dz0.92/XYZ 2.19cm(↓28%)：flow 头此快测设置下**逊于**回归头。
- **启示 → 步骤7**: dx/旋转学不到是**信号不足**(单外参视频+2D投影)，正是步骤7 引入 **VideoMAEv2 参考视频 latent + 10 点轨迹 + 当前图像** 想补的。flow 头本身 OK；要让它全面发力需更强条件(步骤7) 或针对采样发散的改进(x0-prediction / CFG / 推理步数↑ / 对 unlearned 维退化)。
- **判定**: 步骤1 可行性 = **部分通过**(flow 能学，dy 强；但需更强条件才能全面超越回归)。继续推进步骤7。

## Exp 9 — 融合 flow（步骤7：VideoMAE 参考视频 + 10 点轨迹 + 当前图 + EE）  ⚠️ 反而不如步骤1（corr 诊断待补）
- **config**: `droidex2000_fused_flowdit_videomae_points`。371.6M(trainable 285.4M, VideoMAE-B 冻结)。条件 = VideoMAEv2(参考视频 8 帧 linspace→4 pooled tokens) + DINOv2(当前帧→64 image tokens) + PointLatent(10 点×60 步→10 tokens) + EE 2D(1 token) → flow DiT。**source_sampling=linspace**(整段 demo)，与步骤1 的 future_offsets 不同。
- **训练曲线（loss_history，采样 val）**: train_loss 0.345→0.102 平滑降；**val_loss 0.13→0.24 上升，但主要是 gripper 过拟合**(val_gripper 0.51→1.07)；**val_action_loss(采样动作 MSE) 全程平 ~0.027**，未改善。
- **采样 held-out xyz MAE(逐 epoch, cm)**: ep2[4.39,5.14,4.53]→ep8[4.15,4.67,4.40]→ep16[3.91,4.28,4.20]→**ep24[4.22,4.31,4.27]**。三轴**全程≈基线(3.47cm)、几乎不降**。
- **关键对比**: 步骤1(Exp8 ep24) dy MAE **2.61cm**(corr0.76)；步骤7 dy MAE **4.31cm** ——**dy 反而退化**，xyz 三轴都卡基线。
- **判读(待 corr 诊断确认)**: 融合模型**没学到条件映射**(动作 MSE 不降、MAE 卡基线)，且比步骤1 更差。最可能原因：**条件反而变弱了**——① 步骤7 用 `linspace` 整段采样 + VideoMAE **池化成 4 个 token**(空间细节大量丢失)，远不如步骤1 的 `future_offsets` 近未来帧 + DINOv2 **256 个 patch token** 的强信号(那正是 dy 0.76 的来源)；② 点轨迹/当前图/EE 没补上，可能还稀释了注意力；③ 371M 更大更易过拟合(gripper 已现)。
- **TODO（下次 Bash 可用时）**: 跑 `diagnose_actions.py` 对 latest.pt(ep24) 与 best.pt 出逐维 corr/pred_std，确认是否全维 corr≈0；据此定 v2。
- **v2 假设**: ① 参考视频改回 future_offsets 或让 VideoMAE 输出 **token 序列(非池化)** / 用 DINOv2 多帧 patch 作参考；② 点轨迹 token 加强(更多 token/更深)；③ gripper_weight 调小或早停；④ 视频/点条件做 dropout 防过拟合。
- **注**: 本轮诊断时 Bash 安全分类器临时不可用，corr 数值待补；以上结论基于 loss_history 的采样 MAE/loss（足以判定"未改善、且 dy 退化"）。

## Exp 11 — step1 IDM-faithful flow 头（DINOv2 + 完整复刻 IDM 头，12ep 压缩 cosine）  ⚠️ 与 v1 持平，未突破
- **config**: `droidex2000_future_offsets_pose6d_camera_flowdit_qpos_llrd_idm`。flow 头重写为 IDM 端口：8 层单注意力 **interleave** cross/self DiT（AdaLayerNorm 非零门控）+ **4 层 VL self-attn mixer** + 时间注入 action encoder（TimeActionEncoder）+ 输出端 adaLN + **Beta(1.5,1.0) 时间步采样** + 16 步 Euler + dropout0.2 + betas(0.95,0.999)。宽 1024/16 头/8 层，head **176.1M**/total 340.6M。**图像 encoder 仍 DINOv2**（经讨论 encoder 非瓶颈）。**epochs 12**（cosine 压缩到 12 内衰减完）。
- **训练**: train_loss ep1 0.29→**ep12 0.083**（压缩调度生效，收敛比 v1 24ep 快一倍）；infer(采样) ~0.083–0.093 平。
- **终诊断（latest.pt @ep12，800 held-out 采样，num_eval_samples=1）**:
  | 维 | corr | MAE | pred_std/tgt_std |
  |---|---|---|---|
  | dx | **0.046** | 4.36cm | 0.78 |
  | dy | **0.689** | 3.04cm | 0.85 |
  | dz | **0.239** | 3.99cm | 0.82 |
  | r0/r4/r5 | 0.19/0.19/0.13 | — | ~0.8 |
  | r1/r2/r3 | ≈0/-0.06 | — | ~0.9 |
  - **XYZ MAE 3.80cm，比"输出均值"基线 3.465cm 差 9.6%**。
- **对比 v1（Exp8，24ep）**: dx 0.018→0.046、dy 0.764→**0.689**(略降,但只用一半epoch)、dz 0.182→0.239、XYZ 3.73→3.80cm。**基本持平**。
- **结论**: **IDM-faithful 的头部改动（Beta 采样 + 时间注入 + 交替单注意力块 + VL mixer + adaLN 输出）没有突破**——和 v1 一样：**dy 学到(~0.7)、dx 几乎没学(~0.05)、dz 弱(0.24)、旋转弱、XYZ 整体≈基线**。距 videomt(Exp6) 的 x0.62/y0.89/z0.67、XYZ 46.9mm 仍有明显差距。压缩 12ep 的价值是"用一半时间拿到同等结果"，但**没抬高天花板**。
- **判读（重要）**: 两个 flow 变体(v1 + IDM-faithful)都卡在同一形态，而**回归头(videomt Exp6)能学全三轴** → 瓶颈**不在 flow 头的结构细节**，而在更根本处：① flow-matching 建模分布、采样发散，对弱条件维(dx)不如回归头直接收敛到条件均值；② 或 gen2act 的 resampler/fusion 条件通路 不如 videomt 的 query-propagation 会抽 dx 信号。
- **下一步候选**（若要继续逼近回归头）: ① 推理多采样取均值(num_eval_samples↑)看 XYZ MAE 能否压到基线下；② flow 头加 x0-prediction 或 classifier-free guidance；③ 换 videomt 式 query-propagation 条件通路；④ 或直接结论"此任务回归头优于 flow 头"。

## 三个 flow 变体最终对比（同数据/同相机系/同条件信号，仅头不同）
| 变体 | epochs | dx corr | dy corr | dz corr | XYZ MAE | vs 基线3.47 |
|---|---|---|---|---|---|---|
| v1 (Exp8, adaLN-Zero 合并块+均匀采样) | 24 | 0.018 | **0.764** | 0.182 | 3.73cm | 差7.6% |
| IDM-faithful (Exp11, 交替块+Beta+VLmixer+时间注入) | 12 | 0.046 | 0.689 | 0.239 | 3.80cm | 差9.6% |
| 融合 (Exp9, +VideoMAE参考视频+点轨迹+图, linspace) | 24 | — | (dy MAE 4.31 退化) | — | ~baseline | 差 |
| **参照: 回归头 videomt (Exp6)** | 30 | **0.62** | **0.89** | **0.67** | **46.9mm** | 好34% |
- **总结**: 在 droid-ex 相机系 + future_offsets 条件下，**flow-matching DiT 头(无论 v1 还是 IDM-faithful)都只学到 dy 一轴、整体≈基线，明显逊于回归头(videomt 三轴全学)**。flow 头的瓶颈是范式/条件通路，非头部结构细节或参数量。融合更多条件(步骤7)在当前实现下反而更差(条件被 VideoMAE 池化削弱)。

## Exp 13 — 纯视频单流(videomt同款输入) + query-in-backbone读出 + IDM flow头  ✅ 关键: 多采样翻盘
- **config**: `droidex2000_purevideo_queryflow_idm`。输入 = **videomt(Exp6) 完全相同**: 单个 5 帧 clip `[0,5,10,15,20]`(当前+4未来), **无双流/无proprio**。读出 = 16 query 拼进每帧 patch 过 DINOv2 后3层(预训练, 无resampler瓶颈) → 5×16=80 cond tokens → IDM flow头(8层interleave DiT+4层VL mixer+Beta采样)。**与 Exp6 唯一区别 = flow头 vs 回归头**。batch32/12ep, total 261.9M。关proprio后过滤放宽 train3212/val100。
- **训练**: train 0.295→ep12 0.082, 收敛快(~5min/ep)。
- **诊断对比(latest.pt @ep12, 800 held-out)**:
  | 维 | 单次采样 K=1 | **K=32 均值** | videomt回归(Exp6) |
  |---|---|---|---|
  | dx | corr 0.019 | **0.005** | **0.62** |
  | dy | 0.666 | **0.808** | 0.89 |
  | dz | 0.196 | **0.438** | 0.67 |
  | 旋转r0/r4 | 0.20/0.15 | (类似) | r00-r21 0.44-0.55 |
  | XYZ MAE | 4.16cm(差基线17%) | **2.95cm(胜基线3.55的17%)** | 46.9mm(胜34%) |
- **★关键发现: 多采样取均值是 flow 头的必需操作**。单次采样(IDM默认 num_eval_samples=1)严重低估 flow: dy 0.67→0.81、dz 0.20→0.44、XYZ 从"差基线17%"翻转到"胜基线17%"。原因: flow 采样有方差, 单次评估的噪声盖住了已学到的条件信号; 取32次均值≈条件均值, 信号显现。**之前 Exp8/11 判 flow"不如回归"部分是被单采样评估坑了**。
- **结论(修正)**:
  1. **flow 头是有效的**——配多采样, dy(0.81)/dz(0.44) 接近 videomt(0.89/0.67), 整体 XYZ 胜基线。flow 范式可用。
  2. **但 dx 是 flow 的真实死角**——即使多采样 corr 仍 0.005(pred_std 塌成~常数), 而 videomt 回归学到 dx 0.62。同输入同读出, 差别只剩 head: 回归直接 commit 条件均值, 弱信号维(dx)也能挤出弱相关; flow 建模分布, 最弱维学成了~marginal。
  3. **dx gap 的可能补法**: videomt 还有**逐帧递归 query 传播 + 每层深监督**(aux loss), 密集梯度更能抽最弱信号; flow 只在输出端监督一次。→ 候选: 给 flow 加深监督/逐帧readout, 或对最难维退化成回归。
- **行动项**: 把 flow 评估/推理默认 `num_eval_samples` 从 1 提到 ~16-32(否则系统性低估)。

## Exp 14 — 纯视频 query-flow, 小LR续训到ep20 (多采样eval)  ✅ 小LR有用, dy近videomt, dx仍死
- **背景**: Exp13 的 12ep run 在 ep11/12 train_loss 仍快降→早期LR偏高、低LR才仔细学。给 trainer 加 `resume_checkpoint` 暖启动: 从 ep12 latest.pt 续, **lr 3e-4→5e-5**, 8 epoch(=总ep13-20), eval `num_eval_samples=16`。
- **续训曲线**: train_loss contEp1 0.085(确认续训非重启)→contEp8 0.072 持续降; val xyz MAE(多采样) 总ep14[3.39,2.44,3.0]→**ep20[3.35,1.98,3.01]cm** dy/dz降。
- **终诊断(ep20, K=16, 800 held-out)**:
  | 维 | corr | MAE | pred_std/tgt |
  |---|---|---|---|
  | dx | **0.056** | 3.36cm | 0.20(塌成~常数) |
  | dy | **0.851** | 2.07cm | 0.78 |
  | dz | **0.439** | 3.10cm | 0.52 |
  | r0/r4/r5 | 0.37/0.37/0.21 | — | ~0.3 |
  | XYZ MAE | **2.85cm 胜基线3.55的 19.8%** | | |
- **对比**: Exp13 ep12(K=32) dx0.005/dy0.808/dz0.438 XYZ2.95 → ep20 dx0.056/dy**0.851**/dz0.439 XYZ**2.85**, 旋转r0/r4 0.2→0.37。**小LR续训提升 dy+旋转+整体, dz持平, dx仍~0**。
- **vs videomt回归(Exp6)**: dy 0.85≈0.89(近), dz 0.44<0.67(差), **dx 0.06<<0.62(死角)**。
- **结论**: ① 用户"低LR续训"判断正确, flow头配多采样+足够低LR训练后 **整体胜基线~20%、dy 已接近回归头**。② **dx 仍是 flow 唯一硬伤**(pred_std 塌成常数=对最弱信号维放弃、退化成输出均值), 而 videomt 回归能学 dx 0.62。③ dx gap 的结构性原因仍指向: 回归直接 commit 条件均值 + videomt 的逐帧递归/深监督密集梯度; flow 建模分布对最弱维天然吃亏。

## Exp 15 — cross-attn 回归+深监督(同flow读出/传输, 仅换损失)  ✅✅ 决定性: cross-attn通路OK, flow是病根
- **config**: `droidex2000_purevideo_queryreg_deepsup`。**与 Exp13/14 flow 版唯一区别 = 头/损失**: 同 query-in-backbone 读出(16query过DINOv2后3层) + 同 cross-attn 信息传输(4个action query cross-attend 80 cond tokens, 6层), 但 **flow velocity → 直接回归(tanh)+6层深监督**(每层出动作算aux loss, videomt式)。纯视频5帧[0,5,10,15,20]单流。total 142.5M, batch32/20ep/lr3e-4。
- **训练**: train 0.237→ep20 0.025; val xyz MAE ep2[3.43,2.24,3.07]→ep16[2.09,1.25,2.02], ep18起 infer_loss略升(gripper/aux轻微过拟合, pose MAE稳定)。
- **终诊断(latest.pt @ep20, 800 held-out, 回归确定性单次)**:
  | 维 | corr | MAE | pred_std/tgt |
  |---|---|---|---|
  | dx | **0.762** | 2.14cm | 0.81 |
  | dy | **0.925** | 1.24cm | 0.87 |
  | dz | **0.768** | 2.05cm | 0.91 |
  | r0/r1/r2/r3/r4/r5 | 0.61/0.48/0.67/0.48/0.61/0.61 | — | 0.47-0.71 |
  | XYZ MAE | **1.81cm 胜基线3.55的 49%** | | |
- **★决定性结论**:
  1. **cross-attention 信息传输是对的**——同样的 cross-attn 通路+读出, 换成回归+深监督, **dx corr 0.056→0.762**(比 videomt 0.62 还高), 三轴+全旋转全学到。**之前 dx 学不到不是 cross-attn 的锅**。
  2. **flow 头才是 dx(及dz/旋转)的病根**: flow 建模分布、采样发散, 对弱信号维学成 marginal(corr~0); 回归直接 commit 条件均值, 弱信号也挤得出强相关。
  3. **深监督(每层aux loss)也是关键助力**: 密集梯度灌进 backbone+decoder 每层, 帮抽最弱的 dx。
  4. 此版**全面超过 videomt(Exp6)**: dx 0.76>0.62, dy 0.92>0.89, dz 0.77>0.67, 旋转 0.48-0.67 vs 0.44-0.55, XYZ MAE 1.81cm。说明 cross-attn 读出+深监督回归 是这个任务的强方案。

## Flow 变体 vs 回归 最终对比表 (纯视频 [0,5,10,15,20] 同输入)
| 头/方案 | dx | dy | dz | 旋转 | XYZ MAE | vs基线 |
|---|---|---|---|---|---|---|
| flow DiT (Exp14, ep20+多采样K16) | 0.06 | 0.85 | 0.44 | 0.37 | 2.85cm | 胜20% |
| **cross-attn 回归+深监督 (Exp15)** | **0.76** | **0.92** | **0.77** | **0.48-0.67** | **1.81cm** | **胜49%** |
| videomt 回归(直接读出,Exp6, 不同实现) | 0.62 | 0.89 | 0.67 | 0.44-0.55 | 46.9mm(L2) | 胜34% |
- **总判决**: **cross-attention 通路完全没问题, flow 头是 dx/弱信号维的瓶颈**(分布采样 vs 直接回归)。回归+深监督+cross-attn读出 在此任务上最强, 超 videomt。若目标是性能, 应弃 flow 用回归+深监督; 若必须用 flow(为多峰/生成), dx 这类弱维需额外补(深监督已加, 仍不够, 是 flow 范式天花板)。

## Exp 16 — step7 融合(全局条件)  ❌ 没学到(per-episode常量条件无法定位per-window动作)
- **config**: `droidex2000_step7_fused_queryreg`。融合模型(cross-attn回归+深监督, 177.9M): **source video=linspace整段demo(8帧) + 点轨迹=整段轨迹 + 当前帧 + EE(u,v,depth)**。20ep。注: 与 Exp15(直接未来帧[0,5,10,15,20])不同, step7 当前帧只1帧、未来信息走"参考demo"通道。
- **训练**: train_loss **全程死平 ~0.242**(ep4→ep20 基本不动), infer 0.189。val xyz MAE ep2[3.79,3.82,3.54]→**ep20[3.25,3.93,3.58]cm 全部≈基线3.5, 几乎不降**。**连训练集都拟合不了**。
- **诊断**: corr 诊断待 Bash 恢复补, 但 train+val MAE 已铁证: **模型没学到**(若学到 train 必降)。
- **结论**: **全局 per-episode 常量条件失败**。source video(linspace整段)和点轨迹(整段)对同一 episode 的所有 window 完全相同, 只有当前帧+EE 在变 → 这两个"载未来"的条件无法区分 episode 内不同时刻该做什么动作; 而光靠当前帧+EE 预测未来 delta 是欠定的。→ 必须把条件改成 **per-window 对齐**(方向 B, 见 Step7-B/Exp17)。
- **对比**: Exp15(直接未来帧, dx0.76/dy0.92/dz0.77 XYZ1.81cm) ≫ Exp16(全局条件, ≈基线)。差别=Exp15 的未来帧是 per-window 对齐的强信号; Exp16 的未来信号是 per-episode 全局常量。

## Exp 17 — step7-B per-window 对齐(lowLR 1.2e-4 稳定)  ✅ 学到三轴(但用了对齐未来帧=泄漏, 不可部署)
- **config**: `droidex2000_step7B_perwindow`。融合模型(177.9M)同 Exp16, 但条件改 **per-window 对齐**: source video=`future_offsets [0,4,..,28]`相对target_step(局部demo片段) + 点轨迹=`window[0,24]`相对target_step。lr 3e-4→**1.2e-4**+warmup0.10(原3e-4崩溃: ep2-3降0.17后反弹回0.24)。
- **训练(低LR版)**: train_loss 平滑降 ep1 0.247→ep20 0.050, **全程无崩溃**(对比旧3e-4版ep4崩)。
- **终val xyz MAE(ep20)**: **dx2.27 / dy1.47 / dz2.38 cm**(旋转维 mae 0.013-0.086), 三轴全破基线3.5, **基本追平 Exp15(2.14/1.24/2.05)**。corr诊断待Bash(MAE已铁证学到; 与Exp15 MAE相当→corr应近Exp15的0.76/0.92/0.77)。
- **结论**: ① **per-window 对齐 + 低LR 让融合模型学到三轴**(vs Exp16全局失败) → 验证方向B的两点: 条件必须per-window对齐 + 多条件融合需更低LR稳。② 旧3e-4崩溃证实多条件融合对LR敏感。
- **⚠️ 重大 caveat(部署不可迁移)**: Exp17(及Exp15)的 source `future_offsets` = **真实执行的未来帧、精确对齐预测窗口** = 泄漏。真实部署 source 是**生成的语义视频**(不时间对齐、夹爪几何不准), **没有 future_offsets**。所以 Exp17 的成功**不能迁移到部署**。诚实的部署设定 = Exp16(linspace语义demo), 它失败了。→ 真实方向见下 Step7-C 计划。

## Step7-C 计划 — 诚实部署版(无对齐未来帧, 语义 demo)
- **真实场景**: source=生成语义视频(几何不准/不对齐), 无future_offsets; 当前观测=真实帧+EE。
- **三支柱改动**: ① 可部署per-window锚=**因果点轨迹window[-K,0]**(最近运动动量)+短历史帧+进度信号; ② demo编码走**语义**(DINOv2+强增强 或 VideoMAE/SigLIP), 当前帧走DINOv2精确读出; ③ 显式**定位机制**(当前↔demo cross-attn)+训练期辅助定位loss(用gt.mp4对齐监督)。
- **训练→部署桥**: 训练用linspace整段demo(不用future_offsets)+模拟生成视频gap的增强; 有生成视频后再微调。
- **预期**: 语义demo给意图/方向, 精确delta欠定 → step7定位为"高层意图策略", 精确执行交下层控制器。

## Exp C0 — 诚实部署设定 + 序列轨迹 + 定位锚  ⏳ 训练进行中(已启动, ep1 loss 0.2642, ~474s/ep, 20ep≈2.7h)
- **目标**: 在**诚实/可部署**设定下(无对齐未来帧, 去掉 future_offsets), 验证"整段轨迹做成**序列** + **定位锚**"能否让融合模型超过 **Exp16**(诚实设定的失败基线)。上界参照 Exp17(泄漏版 dx2.27/dy1.47/dz2.38), 大概率达不到, 但只要明显好于 Exp16(≈基线 3.5cm)即证明"序列轨迹+定位锚"让诚实设定可学。
- **config**: `droidex2000_C0_honest_seq_anchor`。融合模型(fused_query_reg, cross-attn回归+深监督), 同 Exp16/17 架构(4层self-attn mixer + 6层cross-attn回归 + 深监督)。
- **5 类条件(全 768 维, 各带 type embedding), 共 328 cond tokens**:
  | 条件 | 来源 | 编码 | tokens |
  |---|---|---|---|
  | ① demo 视频 | gt.mp4 **linspace 整段** 8 帧 (诚实, 无未来帧) | query-in-backbone DINOv2 (32 q/帧)+时间PE | 8×32=256 |
  | ② demo 点轨迹(序列) | track_points.npy **整段** [T,10,2], 采 32 时刻 | **PointTrajSeqEncoder**: 逐时刻 token(10点20值→MLP)+时间PE | 32 |
  | ③ 当前帧 | target_step 真实帧(1帧) | query-in-backbone DINOv2 | 32 |
  | ④⑤ EE+进度 | (u,v,depth)+progress(target_step/num_steps) | append_progress→4维→ee_mlp→ee tokens | 8 |
- **与 Exp16/17 的关键区别**:
  | | demo视频 | demo轨迹 | 定位 | 泄漏 |
  |---|---|---|---|---|
  | Exp16(失败) | linspace全局 | **池化latent** | 无 | 否(诚实) |
  | Exp17(成功但泄漏) | future_offsets对齐 | window切片 | — | **是** |
  | **C0** | linspace全局 | **整段序列** | **EE+进度+当前帧** | 否(诚实) |
  C0 = 把 Exp16 的"池化轨迹/无定位"换成"序列轨迹/有定位锚", 其余保持诚实。
- **代码改动(相对 Step7-B / Exp16)**:
  1. `point_latent.py::PointTrajSeqEncoder` — 整段轨迹做成**逐时刻 token 序列**(非池化), [B,N,S,2]→[B,32,768]+时间PE。
  2. `fused_query_reg_policy.py` — 加 `point_seq` 分支: seq 模式下直接用逐时刻 token(+type embed), 跳过池化的 point-summary attention。
  3. `factory.py::_build_fused_query_reg` — `model.point_tracking.sequence: true` 时构建 PointTrajSeqEncoder。
  4. `base.py::sample_window` — `proprioception.append_progress: true` 时把归一化进度 target_step/(num_steps-1) 拼到 proprioception(3→4维)。
  5. `schema.py` — append_progress 时 proprioception_dim 期望值 +1。
- **训练设定(沿用 Exp17 稳定配方防崩)**: lr 1.2e-4 + warmup 0.10 + cosine; LLRD 0.7, backbone_mult 0.3; batch 24; 20 epoch; grad_clip 1.0; bf16。source_sampling=linspace, source_len=8(无 future_offsets); point_tracking num_time=32, sequence=true, **无 window**(整段); proprioception dims=3 + append_progress。
- **成功判据**: 超过 Exp16 — train_loss 能降(不卡 0.24), val xyz MAE 破基线 3.5cm, 逐维 corr>0。额外: 单看相变帧子集误差, 验证没有惯性捷径。
- **训练曲线(20ep, 并行消融致~1000s/ep)**: train_loss **ep1 0.2642 → ep20 0.1036** 平滑单调降, **决定性跌破 Exp16 死守的 ~0.24 平台**(Exp16 连训练集都拟合不了)。val_loss ep2 0.182→ep6 0.145 见底后 **ep10 起走平/微升到 0.16**(gripper 轻微过拟合; pose MAE 稳定)。
- **终诊断(latest.pt @ep20, 800 held-out, 回归确定性单次)**:
  | 维 | corr | MAE | pred_std/tgt |
  |---|---|---|---|
  | dx | **0.259** | 3.12cm | 0.19(塌成~常数) |
  | dy | **0.606** | 3.21cm | 0.54 |
  | dz | **0.351** | 3.17cm | 0.42 |
  | r0/r4 | 0.46/0.36 | 0.02/0.01 | ~0.28 |
  | r1/r2/r3/r5 | 0.09-0.19 | — | ~0.2 |
  | **XYZ MAE** | **3.167cm** | 胜"输出均值"基线 3.467cm **8.6%** | |
  - best.pt(低 val) 反而更差: XYZ 3.355cm 胜基线 3.2%(val_loss 含 gripper 过拟合, 不代表 pose; **latest 比 best 在 pose 上更好**)。
- **★结论(诚实评估, 修正训练期的过度乐观)**:
  1. **C0 明确好于 Exp16(诚实失败基线)**: Exp16 train_loss 死平 0.24、连训练集都拟合不了、held-out≈基线(0% 胜); C0 train 拟合到 0.104、held-out **胜基线 8.6%**。**"序列轨迹+定位锚"让诚实设定从"学不动"变成"能学"** → **方向成立**。
  2. **但 held-out 增益弱**: 仅胜基线 ~9%, 远不及 Exp17 泄漏版(胜~30%, dx2.27/dy1.47/dz2.38)。只有 **dy 中等学到(corr 0.606)**, dx(0.259)/dz(0.351)/旋转(0.09-0.46) 都弱, 且 dx pred_std 塌成常数(0.19)——和历次实验一样, **相机系 dx 是老大难**。
  3. **train 强拟合 vs held-out 弱泛化 = 过拟合**: train_loss 一路降只证明"可训练", 不代表 held-out 质量; ep10 起 val 走平即信号。**诚实设定(linspace 整段 demo, 无对齐未来帧)的天花板就在这**: 精确 future delta 本质欠定, demo 只能给"高层意图/方向", 不能给"精确 +5/+10 步该到哪"。
  4. 符合计划预期: "C0 大概率达不到 Exp17(没对齐未来帧), 但明显好于 Exp16 即证明方向成立"——**两条都验证了**。
- **消融对照**: 见下 "Exp C0-消融(序列无进度)" —— 用于分离"轨迹序列化"vs"进度定位"哪个是 C0 相对 Exp16 进步的主力。

## Exp C0-消融 — 序列轨迹，去掉进度信号 (隔离"进度定位"的贡献)  ✅ 完成
- **config**: `droidex2000_C0_ablate_noprogress`。与 C0 **唯一区别 = 去掉进度信号**(`proprioception.append_progress: false`, proprioception_dim 4→3)。轨迹仍是逐时刻**序列**(PointTrajSeqEncoder), demo 仍 linspace 整段(诚实), 其余完全同 C0。与 C0 并行训练(共享单卡, 故前期~1000s/ep, C0 跑完后提速~480s/ep)。
- **训练曲线**: train_loss **ep1 0.2746 → ep20 0.1047**, 与 C0(0.2642→0.1036)**几乎完全重合**(ep3: B0.190/C0.192; ep7: B0.142/C0.148)。→ **进度信号对"可训练性"完全冗余**(train_loss 看不出差别)。val_loss 同样 ep~8 见底后微升(0.138→0.172, gripper 过拟合)。
- **终诊断(latest.pt @ep20, 800 held-out)**:
  | 维 | corr | MAE | pred_std/tgt |
  |---|---|---|---|
  | dx | **0.199** | 3.19cm | 0.14(塌成~常数) |
  | dy | **0.537** | 3.29cm | 0.54 |
  | dz | **0.323** | 3.26cm | 0.39 |
  | r0/r4 | 0.38/0.35 | — | ~0.27 |
  | r1/r2/r3/r5 | 0.06-0.14 | — | ~0.2 |
  | **XYZ MAE** | **3.244cm** | 胜基线 3.467cm **6.4%** | |
  - best.pt 同样更差(XYZ 3.458cm, 仅胜 0.3%; val_loss 含 gripper, latest 的 pose 更好)。

## ★★ Exp16 / C0-消融 / C0 — 2×2 因子分解最终结论 (序列化 vs 进度定位)
| | 池化轨迹 | 序列轨迹 |
|---|---|---|
| **无进度** | **Exp16**: ≈基线(胜 **0%**, train 死平 0.24=学不动) | **C0-消融(B)**: 胜基线 **6.4%** (dx0.20/dy0.54/dz0.32) |
| **有进度** | (未跑, 第四角) | **C0**: 胜基线 **8.6%** (dx0.26/dy0.61/dz0.35) |

- **Exp16 → B (加"轨迹序列化", 都无进度)**: 胜基线 **0% → 6.4%**, train 从"死平 0.24 学不动"→"拟合到 0.10"。**→ 轨迹序列化是主力**: 它独自解决了 Exp16"连训练集都拟合不了"的根因。机制: 池化把整段轨迹压成固定 blob, 对同一 episode 所有 window 是同一个常量, 无法定位/索引; 序列化成逐时刻 token 后, 模型可用当前帧+EE 与序列做内容匹配自行定位、读取相应阶段 → 常量 demo 第一次能产出 per-window 变化的预测。
- **B → C0 (加"进度信号", 都序列)**: 胜基线 **6.4% → 8.6%**(+2.2pp), 逐维 corr 全部小幅上抬(dx0.20→0.26, dy0.54→0.61, dz0.32→0.35)。**→ 进度是真实但小的加成**: 它在 **train_loss 上完全看不出**(对训练冗余, B/C0 曲线重合), 但对 **held-out 泛化**有稳定小幅帮助——把"模型靠内容匹配隐式定位"换成"显式告知在 demo 的第几 % ", 省了一点定位负担、提升泛化。
- **总判决**: **诚实部署版(C0)成立的主力是"把轨迹做成可索引的序列"(Exp16→B 占了 6.4/8.6≈74% 的增益), 进度信号是锦上添花(占~26%)。** 修正了训练期"只看 train_loss → 误判进度冗余"的过度乐观: train_loss 只测可训练性, held-out 诊断才是真尺子。
- **共同天花板**: 三者(及所有诚实设定)都受限——XYZ 最多胜基线 ~9%, **dx 始终学不动(corr<0.26, pred_std 塌成常数)**, 远不及 Exp17 泄漏版(胜~30%)。诚实设定下 demo 只给"高层意图/方向", 精确 future delta 本质欠定。要再上一层需: 真实生成语义视频微调 / 腕部相机 / 或把 step7 定位为高层意图策略+下层控制器执行。 → C1 用"因果最近运动"补这个欠定, 见下。

## Exp C1 — 双路点追踪(全局 demo 轨迹 + 因果最近运动 window[-24,0])  ✅ 明显胜 C0, 救活 dx
- **动机(承接 C0 诊断)**: C0 是 train-limited(pose 不过拟合, train≈val), 瓶颈是诚实全局 demo 欠定。补一个 **per-window 因果信号 = 当前 EE 邻域点在 [target_step-24, target_step] 的最近运动(动量)**, 它在部署时可得(只用过去帧, 不像 Exp17 用未来=泄漏), 强约束"下一步往哪动"。
- **设计(对齐部署计划)**: source video 最终是**生成语义视频**(夹爪形变/大小/角度不准、无 future_offsets), 全帧点追踪承载语义意图、**必须保留**。所以 C1 = C0 的全局轨迹 **不删**, **叠加**第二路因果轨迹, 两路各自 PointTrajSeqEncoder + 各自 type embedding 进 mixer:
  - `point_track`(全局整段, 序列, num_time 32) = demo 语义意图 (= C0)
  - `point_track_causal`(window[-24,0], 序列, num_time 16) = 当前最近运动动量 (新增)
- **config**: `droidex2000_C1_dualtrack_causal`。代码: base.py `_read_point_track` 参数化窗口/时长 + `causal_window` 配置出第二路; factory 配 causal 才建第二 encoder; FusedQueryRegPolicy 加 `point_encoder_causal`+`type_point_causal` 独立 token 组; trainer/predictor/diagnose 仅在 batch 含 causal 时传该 kwarg(其他实验不受影响)。修 diagnose_actions.py 不传 point_track 的 bug(顺带)。
- **训练**: train_loss ep1 0.260 → **ep20 0.0953**(全程略低于 C0 0.1036, 因果信号把 train floor 压低)。train_action_mae **0.0485** ≈ val_action_mae **0.0485** → **pose 仍不过拟合**(C0 是 0.0504/0.0495; C1 train+val 都更低 = 补的是真确定性信号, 非记忆)。
- **终诊断(latest.pt @ep20, 800 held-out)**:
  | 维 | C1 corr | C1 MAE | C1 pred_std/tgt | vs C0 |
  |---|---|---|---|---|
  | dx | **0.482** | 2.89cm | **0.44** | C0 0.259/3.12/**0.19(塌)** → **dx 救活** |
  | dy | **0.653** | 3.06cm | 0.58 | C0 0.606/3.21 |
  | dz | **0.388** | 3.13cm | 0.44 | C0 0.351/3.17 |
  | r0/r2/r4/r5 | 0.43/0.28/0.36/0.21 | — | — | 普遍↑(C0 r2 0.09→0.28) |
  | **XYZ MAE** | **3.027cm** | 胜基线 3.467 **12.7%** | | C0 **8.6%** |
  - best.pt 同样更差(XYZ 3.275cm 胜 5.5%; val_loss 含 gripper 过拟合, latest 才准)。
- **★结论**:
  1. **因果最近运动是有效的可部署信号**: XYZ 胜基线 8.6%→**12.7%**(margin 提升 ~48%), 且 **dx corr 0.259→0.482、pred_std 0.19→0.44** —— **历次诚实实验都学不动、塌成常数的 dx 这次被救活**。机制: 最近运动动量强烈约束 immediate-next dx(刚才在 +x 动, 下一步多半继续), 正好补 C0 欠定的那一块。
  2. **是真泛化非记忆/过拟合**: train_action_mae≈val_action_mae 且**两者都比 C0 低** → 因果信号降低了欠定地板本身(train 能更低), 同时 held-out 同步改善。
  3. **仍诚实可部署**: 因果窗口只用过去帧, 不泄漏未来(区别于 Exp17)。距 Exp17 泄漏版(~30%)还有差距, 但方向明确——**给 per-window 因果运动信号是抬诚实设定 eval 的有效杠杆**。
- **后续候选**: 调因果窗口长度 K / num_time; 因果轨迹也给 best.pt 选择换成按 val_action(非含 gripper 的 val_loss)挑; 腕部相机继续攻 dx; gripper 早停修 val_loss 数字。

### Exp C1 — 条件流置换重要性 (source video 影响: 轨迹主导 vs 视频主导)
- **方法**: `scripts/stream_importance.py`——拿训练好的 C1(latest.pt), 不重训, 在 800 held-out 上把某一路输入在 batch 内 roll-by-1 打乱(失去与动作的对应、仍在分布内), 看 XYZ MAE 涨多少。涨得多=该路对当前模型越重要。
- **结果(XYZ MAE cm, Δ vs 完整 3.027)**:
  | 打乱的流 | XYZ MAE | Δ | dx corr |
  |---|---|---|---|
  | 完整 | 3.027 | — | 0.482 |
  | 当前帧 | 3.247 | **+0.221** | 0.406 |
  | 因果轨迹 | 3.116 | **+0.089** | **0.371**(砸dx最狠) |
  | 全局轨迹 | 3.082 | +0.055 | 0.425 |
  | EE+进度 | 3.080 | +0.054 | 0.473 |
  | demo 视频 | 3.062 | **+0.035** | 0.457 |
  | 两路轨迹 | 3.059 | +0.032 | — |
- **结论**: ① **当前帧主导**(+0.221, 锚定现在场景)。② **因果轨迹第二**(+0.089), 且**专门撑 dx**(打乱后 dx 0.482→0.371)——dx 的提升确由因果动量贡献。③ **demo 视频几乎无关**(+0.035, ~1%)——**模型是轨迹/当前观测主导, 不是视频主导**。④ 部署利好: 模型主要靠当前真实帧+因果近期运动(部署时都是真实观测), 故将来用**不完美生成语义视频**做 demo 几乎无损。
- **⚠️ caveat**: 此处 source_video=gt.mp4=执行同一段(自演示), demo 视频与当前帧**高度冗余**→ 忽略它正常。真实部署 demo 是**另一段、载任务目标**, 当前帧给不了。故**不能下"视频无用"结论**——自演示代理里冗余, 真实部署里要扛"任务意图"。更深: 当前模型更像"当前状态+动量→下一步"的动力学模型, 而非"看懂参考视频做新任务"; 要逼出后者需**跨任务/跨视频训练设定(demo≠执行视频)**, 自演示代理测不出。

### Exp C1 — 旋转预测评估 (测地角误差)  ❌ 旋转实质没学到
- **方法**: `scripts/rotation_eval.py`——把预测/真值 6D 经 Gram-Schmidt 还原成旋转矩阵, 算测地角误差(度), 对比两个基线: "预测零旋转(单位阵)"(其误差=GT旋转幅度本身)、"预测均值旋转"。
- **结果(latest.pt, 800 held-out, 度)**:
  | | mean | median | p90 |
  |---|---|---|---|
  | GT 旋转幅度(=预测零旋转误差) | 9.44 | 6.37 | 22.3 |
  | 预测均值旋转基线 | 9.48 | 6.33 | 22.3 |
  | **模型预测** | **9.41** | 6.39 | 21.6 |
  - 模型逐样本只在 **49.0%**(=随机) 上赢过"预测零旋转"。
- **结论**: **旋转基本没学到**——模型误差 9.41° 与"预测零旋转"(9.44°)、"预测均值"(9.48°) 统计持平。诊断里逐 6D 分量 corr 0.16-0.43 是**假象**: ① 旋转本身很小(中位 6.4°, 近噪声地板); ② pred_std 仅真值的 ~0.3(往均值缩), 重建矩阵≈常数。这是 **dx 塌缩的旋转版**(弱信号→回归均值; 而均值旋转≈零旋转因为旋转都小)。**平移(xyz)才是真正学到的(胜基线12.7%), 旋转这条线(C0/C1 皆然)实质没学**。部署上旋转宜交下层控制器或当小量默认("保持当前朝向"已是强默认)。

### Exp C1/C0 — 整段轨迹 rollout 累积误差 (open-loop 位置积分)  ⚠️ open-loop 不可用, 系统性漂移
- **方法**: `scripts/trajectory_rollout.py`——从起点把每窗口预测的 **+5 步相机系位移 delta** 在 t=0,5,10,… 累加(open-loop 积分)重建整段 EE 轨迹, 比真值漂移多少。观测 teacher-forcing(GT帧/EE/轨迹, 无仿真器渲染预测状态)→ 误差**纯来自 delta 积分**(真实闭环里这类误差不累积; 故这是**最坏情况下界**)。基线: 积分全局**均值 delta**(按平均速度 dead-reckon)。40 held-out episodes。
- **结果**:
  | | C0(无因果) | C1(有因果) | 匀速基线 |
  |---|---|---|---|
  | per-step +5 MAE | 1.58cm | **1.48cm** | — |
  | 终点漂移 | 27.6cm(41%路长) | **24.7cm(37%)** | 24.9cm |
  | 漂移@25/50/75/100% | 14/19/25/28 | 14/18/23/**25** | — |
  | 胜匀速基线 | 42%(**输**) | 55%(勉强赢) | — |
  - GT 路径长 ~74cm。
- **结论**:
  1. **open-loop 灾难性、不可用**: per-window 1.5cm 串起来终点漂移 **~25cm = 37-41% 路长**。
  2. **系统性累积非随机游走**: 独立误差应 1.5×√15≈6cm, 实际 25cm → **pred_std<tgt_std 的"欠预测"偏置线性累积**(回归均值的恶果)。
  3. **per-window 12.7% 优势积分后几乎蒸发**: C1 终点 24.7 vs 匀速基线 24.9, **只赢 1%**——轨迹层面模型有效信息≈"按平均速度走"。
  4. **因果轨迹在轨迹层面也有用**: C1(24.7cm,55%胜) > C0(27.6cm,42%输), C1 是唯一勉强赢匀速基线的(呼应 dx 救活)。
- **部署含义**: 此策略**必须闭环**(每步重观测真实态+重预测, 误差不累积), **不能 open-loop 当轨迹生成器**。但**系统性欠预测(回归均值)闭环里也跑不掉**→机械臂会一致滞后/欠到位; 治本需解决 pred_std 塌缩(更强条件 / 换建模让它敢预测大动作)。
- **逐轴分解(C1, 相机系, 终点漂移)**: 深度**不是罪魁**, 三轴均匀:
  | 轴 | step MAE | 终点漂移 | GT位移幅度 | 漂移/位移 | 占比 |
  |---|---|---|---|---|---|
  | x(图像水平) | 1.33cm | 11.8cm | 13.6cm | **0.87** | 32% |
  | y(图像垂直) | 1.61cm | 11.9cm | 11.3cm | 1.05 | 32% |
  | z(深度) | 1.50cm | 13.5cm | 12.9cm | 1.05 | 36% |
  - **z(深度)只比 x/y 略差(占 36% vs 32%)**, step MAE 也夹在中间, 不突出。**关键"漂移/位移比"三轴都 ~0.87-1.05**=每轴终点漂移≈该轴真实移动幅度, **没有任何一个轴整段可靠**。唯一稍好是 x(0.87<1, 因果救活的 dx 轴)。
  - **判读**: 瓶颈不在深度通道(proprioception 的 depth channel 没成短板), 而在**所有轴共有的系统性欠预测(pred_std 塌缩)**, 均匀砸在 xyz。→ 治本(解决 pred_std 塌缩)可一次改善三轴, 无需为深度特殊设计。

## Exp C2 — 双路因果输入 + flow 头 (换头治 pred_std 塌缩)  ❌ 治塌缩部分有效, 但不改善精度/漂移
- **动机/假设**: C1 回归头塌成条件均值(pred_std<<tgt)→系统性欠预测→dx弱/旋转没学/轨迹线性漂移。flow 建模**分布**、采样**保幅度**, 理论上 ① pred_std/tgt→~1, ② 单采样轨迹漂移从线性降到 √N。重测时机: 之前判 flow 差是缺因果信号(Exp8/11/13/14), 现在 C1 双路因果补了 dx 信号, 给 flow 公平复审。
- **设计**: **唯一改头**——C1 的双路因果条件(读出+全局轨迹seq+因果window[-24,0]seq+EE/进度)完全不变, 去掉 SelfAttentionMixer(flow头自带vl_mixer), 接 FlowMatchingDiTHead(1024/8层/16头, IDM式Beta采样, 16步Euler)。新策略 `FusedQueryFlowPolicy` + factory `_build_fused_query_flow`(model.type=fused_query_flow) + action.mode=flow。config `droidex2000_C2_dualtrack_causal_flow`。263M(head 176M)。
- **训练**: velocity-MSE ep1 0.358→ep20 0.0626 平滑降(flow loss 与回归 smooth-L1 不可比绝对值)。~510s/ep。
- **诊断(latest.pt @ep20, 800 held-out, K=8 多采样)** vs C1(回归):
  | 维 | C2 corr | C2 MAE | **C2 pred_std/tgt** | C1 pred_std/tgt |
  |---|---|---|---|---|
  | dx | 0.455 | 2.90cm | **0.60** | 0.44 |
  | dy | 0.630 | 3.16cm | **0.66** | 0.58 |
  | dz | 0.408 | 3.24cm | **0.65** | 0.44 |
  | 旋转 r0-r5 | 0.05-0.35 | — | **0.44-0.57** | ~0.2 |
  | **XYZ MAE** | 3.099cm | 胜基线 **10.6%** | | C1 胜 12.7% |
- **轨迹 rollout(40 episodes)**: K=8 终点漂移 **26.57cm**, K=1 单采样 **29.48cm**(C1 回归 24.71cm)。per-step +5 MAE: K=8 1.46cm / K=1 1.66cm。
- **★结论(干净的负结果)**:
  1. **flow 确实部分修复了塌缩**: pred_std/tgt 从 C1 的 0.44 抬到 0.60(旋转 0.2→0.5), 即使 K=8 平均后仍比回归更有幅度 → 分布建模保住了更多 spread。**假设①方向对**。
  2. **但精度/漂移没改善, 反而略差**: 逐维 corr/MAE ≈ 或略低 C1(XYZ 10.6% vs 12.7%); 轨迹漂移 K=8 26.6/K=1 29.5cm **都 ≥ C1 的 24.7cm**。
  3. **假设②(√N漂移)被证伪**: 单采样 K=1 漂移最大(29.5cm), 因为单采样每步噪声(1.66cm>均值1.46)**累积而非抵消**; K=8 平均又退回均值≈回归。两头都不赢回归。
  4. **根本判读**: 回归"塌成均值"是对**真欠定**的最优响应——那份 spread 是真实 aleatoric 噪声; flow 把它建成分布后, 采样(单)注入噪声进轨迹、平均(多)又回均值, 所以 **flow ≤ 回归**。**换头不是杠杆**。呼应 C1 诊断"train-limited 非过拟合": 瓶颈是**信息量**不是输出头。
- **行动启示**: 要提升必须 **降低欠定**(更强确定性输入: 因果轨迹已验证有效 / 腕部相机 / 更密历史) 或 **改目标为可确定量**(方向/子目标, 交下层控制器执行), 而非在回归↔flow 之间换头。修 trajectory_rollout.py 加 `--num-eval-samples` 支持 K 覆写。

### ★ 动量分析 — 因果轨迹的 dx/dz 提升是"运动惯性外推", 非"定位+读 demo"
- **问题**: C1 的因果轨迹把 dx 0.26→0.48, 是 ① 运动惯性(过去运动→未来运动, 靠轨迹平滑, 不需懂 demo) 还是 ② 用因果定位当前位置 + 从全局轨迹读未来方向?
- **方法**: `scripts/momentum_autocorr.py`——纯数据(无模型)算 camera-frame +5 delta 的"过去 vs 未来"自相关 = 动量能达到的 corr 天花板。
- **结果(val 1595 windows)**:
  | 轴 | 1步前动量 corr | ~20步动量 corr | C1 实际 corr |
  |---|---|---|---|
  | dx | **0.763** | 0.469 | 0.482 |
  | dy | 0.755 | 0.406 | **0.653** |
  | dz | **0.774** | 0.455 | 0.388 |
  - +5 delta XYZ MAE(cm): **动量-1步前 1.04** / 动量-20步 1.53 / predict-mean 1.54。
- **★结论**:
  1. **dx/dz 是动量(H1)**: ~20步动量(0.47/0.46) ≈ C1(0.48/0.39)。dx/dz 拿到的正是动量能给的; 全局轨迹对 dx 几乎无贡献(置换: 打乱全局 dx 仅 0.48→0.43)。→ **因果轨迹的 dx/dz 提升 = 惯性外推, 不是懂 demo**。
  2. **dy 是部分例外**: C1 dy 0.653 > 20步动量 0.406 → dy 有 ~0.25 信号在动量之外(可能用到 demo), 但仍 < 1步动量 0.76。
  3. **predict-mean 基线太弱**: 平滑遥操+短时域(+5)下, 诚实基线是**动量基线**(重复上一步速度: corr 0.76 / +5 MAE 1.04cm ≪ predict-mean 1.54)。**对动量基线, 我们所有模型基本无增量** → 模型很可能没在"理解 demo", 只在外推惯性。
  4. **根因=自演示**: 因果轨迹取"真实执行的最近运动"(gt=执行), "过去→未来"靠平滑成立, 模型不需查 demo。部署时近期真实运动仍有用(继续当前动作), 但**无法启动新任务、非懂 demo**; 该扛任务意图的全局 demo 轨迹被忽略。
- **判别性后续**: 在**相变帧**(抓取↔抬起, 运动方向反转)上 eval——动量必错, 只有懂 demo 才对。这是 C0 原定成功判据"单独看相变帧验证无惯性捷径"的兑现, 几分钟可跑, 比 causal-only 重训更直接分离 动量 vs demo理解。见下。

### ★★ 相变帧 eval — 模型 vs 动量基线 (`scripts/phase_transition_eval.py`)
- **方法**: 每个 window 比"过去运动方向 vs 未来运动方向", cos<0.3 或夹爪开合 = 相变帧(动量必错)。模型预测 vs 动量预测(=上一窗口的 GT +5 delta), 分 全部/相变/平滑 三子集对比。
- **C1 结果(1446 windows, 相变 150=10%)**:
  | 子集 | 模型 MAE | 动量 MAE | 模型 dx/dy/dz | 动量 dx/dy/dz |
  |---|---|---|---|---|
  | 全部 | 1.67cm | **1.21cm** | 0.48/0.60/0.34 | 0.77/0.76/0.79 |
  | **相变** | **1.30cm** | 1.78cm | 0.24/**0.40**/0.12 | 0.03/**−0.13**/0.20 |
  | 平滑 | 1.71cm | **1.15cm** | 0.49/0.61/0.35 | 0.81/0.80/0.82 |
- **★结论(比纯动量分析更细、更乐观)**:
  1. **模型不是纯动量**: 相变帧上模型赢动量(1.30 vs 1.78cm; corr dx 0.24 vs 0.03、dy 0.40 vs **−0.13**)。方向反转处动量彻底失效(corr~0/负), 模型保住正相关 → **demo/因果轨迹确实学到了"该转向/该抓"**, 部分给因果轨迹平反。
  2. **但整体净输给动量**: 1.67 vs 1.21cm。因 90% 是平滑帧, 那里动量近乎完美(1.15cm)、模型平庸(1.71cm)。**模型价值全在 10% 相变帧, 被平滑帧平庸拖垮**。
  3. **指向"预测动量残差"**: 模型(擅相变)与动量(擅平滑)**互补**。改目标为 `未来位移 − 过去位移`(动量残差)→ 模型只学相对"继续当前方向"的偏离(=demo理解部分: 减速/反向/抓取), 部署时 `动作=动量+模型残差`。模型容量投在擅长的相变, 平滑帧白嫖动量。**有数据支撑的下一个实验**。
  4. predict-mean 基线作废: 此后所有 eval 应对**动量基线**报, 模型的及格线 = 在相变帧赢动量、整体不输动量。
- **跨模型对比(相变帧 n=150 / 全部 n=1446, XYZ MAE cm)**:
  | 模型 | 相变 MAE | 相变 dx/dy/dz | 全部 MAE |
  |---|---|---|---|
  | C0(仅全局) | 1.299 | 0.20/0.28/0.13 | 1.760 |
  | C1(双路因果) | 1.297 | 0.24/**0.40**/0.12 | 1.669 |
  | C2(flow) | 1.341 | **0.29**/0.29/0.15 | 1.690 |
  | **动量** | 1.785 | 0.03/**−0.13**/0.20 | **1.215** |
  - **三模型相变帧全赢动量(~1.30 vs 1.78)、全保正 corr** → demo 被用上、学到相变(模型无关稳健结论)。但**全部帧全输动量**(被平滑帧拖垮)。**C1≈C0 于相变帧** → 因果轨迹在相变帧无帮助、其全局优势全在平滑帧(=动量) → 再证因果轨迹=动量。
  - **★动量残差天花板**: oracle 混合(平滑用动量1.15 + 相变用模型1.30) XYZ MAE **≈1.165cm**, **同时胜纯动量(1.21)和所有现模型(1.67)**。残差目标(模型预测"相对动量偏离": 平滑≈0→输出≈动量, 相变大→纠正)=该混合的可学版本。**下一个实验: 目标改 `未来位移−过去位移`, 部署 `动作=动量+残差`**。

## Exp C3/C4 — HAMSTER 式轨迹 overlay (把整段 demo EE 路径画到当前帧)  ✅ overlay 真带方向信息, 但单独不充分
- **动机**: stream-importance 显示全局轨迹(抽象 token)几乎被忽略(+0.03), 模型靠动量而非读 demo。HAMSTER 做法: 把 demo 2D 路径**画到 RGB 上**, 同一 backbone 锚定场景→视觉定位+读未来方向。
- **代码**: `data/overlay.py`(numpy 折线光栅化 + `raw_to_display_px` 按 resize_center_crop 几何对齐到 224); openx_droid `_ee_path_pixels`(整段 EE 投影像素路径); base.py 在 augment 后的当前帧上画蓝→红折线(按 episode 缓存, 训练加坐标噪声=HAMSTER path-aug); 模型加 `use_source_video`/`point_tracking.use_global` 开关可整路关闭 demo视频/全局token。`scripts/viz_overlay.py` 验证对齐(绿点随 target_step 沿路径移动, 已目视确认对齐)。
- **C3(=C1+overlay, 全保留)**: 中途叫停——confound: 全局 demo 同时在 ①demo视频/②overlay/③token 三处 + ④动量, 分不清谁的功劳。改做 C4 干净隔离。
- **C4(干净隔离, 仅 ②当前帧+overlay + ⑤EE(u,v,depth), 删 ①③④+进度)**: 模型"做什么"的唯一来源=画在当前帧的轨迹。train 252s/ep(无 source readout, 半速), train_loss ep20 0.085(< C1 0.104)。
  - **诊断(latest.pt, 800 held-out)**: dx0.332/dy0.563/dz0.305, XYZ MAE **3.334cm**(胜 predict-mean 仅 3.8%; C1 12.7%)。train 低但 held-out 差 = **更过拟合**(没了动量这条可泛化信号)。
  - **相变帧 eval**:
    | 子集 | C4(仅overlay) | C1(token+动量) | 动量 |
    |---|---|---|---|
    | 相变 MAE | **1.40**(dx.18/dy.23) | 1.30(dx.24/dy.40) | 1.78(dx.03/dy−.13) |
    | 全部 MAE | 1.85 | 1.67 | **1.21** |
- **★结论**:
  1. **overlay 真带方向信息=YES**: C4 **零动量输入**却在相变帧赢动量(1.40 vs 1.78, 保正 corr 而动量塌成0/负) → **画在图像上的轨迹被模型读出来了**(模型从图像 get 到 demo 方向的硬证据, 核心假设成立)。
  2. **但 overlay 单独不充分(C4 < C1, 非≈)**: 整体 1.85 vs 1.67(丢动量→平滑90%吃亏), 相变 1.40 vs 1.30(overlay 单独 < C1 的 demo视频+token 合力)。删掉的流**不是纯冗余**: 动量管平滑、demo视频/token 在相变也有贡献。
  3. **最优组合 = overlay(管相变方向) + 动量(管平滑)**, 而非只用 overlay。即 C3 去掉冗余的全局token/demo视频 = overlay + 因果动量(+残差)。仍受**整体输动量基线(1.21)**的天花板制约——所有诚实模型未破。

## ★★ Exp C5 — 路径作为「值」(vs C4 同路径作为 RGB overlay)  ✅ 值 >> RGB, 决定性
- **设计(apples-to-apples)**: 与 C4 唯一区别=demo 路径的**注入方式**。同一条 **EE 投影路径**, C4 画成 RGB overlay(②), C5 喂成**单点坐标值序列**(32时刻 → PointTrajSeqEncoder/MLP, `data.point_tracking.source: ee_projection`, num_points=1)。其余相同(当前帧 plain + EE(u,v,depth); 无 demo视频/动量/cotracker/进度)。代码: base.py `_read_ee_projection_track`(复用 overlay 的 EE 路径缓存, 归一化 [-1,1])。
- **诊断(latest.pt, 800 held-out)** vs C4:
  | 维 | C5(值) corr | C5 MAE | C4(RGB) corr | C4 MAE |
  |---|---|---|---|---|
  | dx | **0.623** | 2.57cm | 0.332 | 3.21 |
  | dy | **0.655** | 2.94cm | 0.563 | 3.39 |
  | dz | **0.454** | 3.12cm | 0.305 | 3.41 |
  | XYZ | — | **2.873cm 胜基线17.1%** | — | 3.334 胜3.8% |
  - C5 dx pred_std/tgt **0.78**(几乎不塌缩) vs C4 0.42。
- **相变帧 eval(+5)**:
  | 子集 | C5(值) | C4(RGB) | C1(token+动量) | 动量 |
  |---|---|---|---|---|
  | 相变 MAE | **1.284**(dx.45/dy.39) | 1.40(dx.18/dy.23) | 1.30(dx.24/dy.40) | 1.78(dx.03/dy−.13) |
  | 全部 MAE | 1.61 | 1.85 | 1.67 | **1.21** |
- **★结论**:
  1. **值 >> RGB(用户假设证实, 决定性)**: 同一条路径, 喂值 dx0.62/XYZ2.87(胜17%) ≫ 画像素 dx0.33/XYZ3.33(胜3.8%)。机制: RGB 把精确轨迹光栅化成 ~3px 线(有损, 且 DINOv2 没学过读画线), 值直接把精确坐标给 MLP。**对轨迹这类精确几何信号, 值远胜像素。** → 推翻"必须 overlay grounding"的判断; 此前 C4 overlay 只是聊胜于无。
  2. **C5 = 目前最强诚实模型(全 chunk)**: XYZ 2.87cm(胜17.1%) > C1 3.03(12.7%) > C4 3.33, 且**输入最简**(当前帧+EE路径值+EE, 无 demo视频/动量/cotracker)。相变帧 1.28 亦最优, dx corr 0.45 历史最高、pred_std 0.78 几乎不塌缩。
  3. **解释了 C1 里 ③(全局 token)被忽略之谜**: 那是**有噪 cotracker 10点 + 动量捷径**双重拖累; 换成**干净 EE 投影路径 + 作为唯一 demo 通道**, 模型强烈使用。→ 关键是"信号干净 + 没有动量可偷懒", 不是 token 形式本身不行。
  4. **仍输动量于 +5 单步整体**(1.61 vs 1.21, 平滑帧无动量输入吃亏)。→ **下一步: C5 值路径 + 因果动量 组合**(方向用路径值、平滑用动量), 冲整体破动量基线。

## Exp C6 — cotracker 10点轨迹作为值 (vs C5 EE投影单点)  ⚠️ 部署可得的 cotracker 轨迹明显弱于干净 EE 投影
- **设计**: = C5 但路径源从 **EE投影单点** 换成 **cotracker 10点轨迹**(track_points.npy, num_points=10, 值)。意义: cotracker 轨迹**部署可得**(对生成视频跑追踪, 不需相机标定/cartesian), EE投影则需标定+cartesian(训练才有)。其余同 C5。
- **诊断(latest.pt, 800 held-out)**:
  | | C6(cotracker10pt) | C5(EE投影1pt) | C4(RGB) |
  |---|---|---|---|
  | dx | 0.287 | **0.623** | 0.332 |
  | dy/dz | 0.476/0.357 | 0.655/0.454 | 0.563/0.305 |
  | XYZ MAE | 3.378cm 胜2.6% | **2.873 胜17.1%** | 3.334 胜3.8% |
  | 相变帧 MAE | 1.414(dx.22/dy.33) | **1.284(dx.45/dy.39)** | 1.40 |
  | 全部 MAE | 1.868 | 1.61 | 1.85 |
- **★结论**:
  1. **cotracker 10点 ≪ 干净 EE 投影单点**: dx 0.62→0.29(腰斩), XYZ 胜基线 17.1%→2.6%(≈ 和 RGB overlay 一样弱)。**C5 的强结果靠的是干净的 EE 中心信号, 不是"值"这个形式本身。**
  2. **原因**: cotracker 10点是**臂 mask 内 EE 邻域的散点**(非 EE 本身), ① 追踪有噪/漂移/遮挡 ② 不是 EE 中心、是带偏移和旋转噪声的散点 ③ 10×2=20 维更难解耦。干净 EE 投影(精确中心、单点)信号干净得多。
  3. **仍赢动量于相变帧**(1.41 vs 1.78) → cotracker 轨迹**带一点 demo 方向信息, 只是弱**。
  4. **⚠️ 部署 gap**: C5 的强依赖**训练才有的 EE 投影(标定+cartesian)**; 部署可得的 cotracker 散点轨迹弱得多。→ 部署要可用, 需要**干净的单条 EE/夹爪尖端轨迹**(如夹爪关键点检测 / 取10点均值 / 只追尖端一点), 而非 10 个散乱臂点。下一步可测: cotracker-10点取**均值**当单点(隔离"10散点噪声" vs "cotracker质量")。

## ★★★ Exp C7 — 视频到轨迹 (读 8 未来帧 → 逐帧相机系绝对位姿)  ✅✅✅ 决定性突破: 范式从"预测"变"感知"
- **重构**: 之前 C0-C6 都是"从过去**预测**未来"(欠定→动量/塌缩, 整体打不过动量基线)。C7 把**未来帧变成输入**, 模型"从给定帧**读出**位姿"(well-determined 感知; 答案在画面里)。这是层次策略的**高层模块**: 生成 demo 视频 → 6DOF EE 轨迹 → 下层控制器执行。
  - 输入: 当前帧 + **8 未来帧**(均匀抽到 episode 末 + 抖动); 8 帧塞进 source_video 槽, 复用 query-in-backbone 读出/mixer/8-query 解码器(零新模型)。
  - 输出: 逐帧**相机系绝对位姿** [pos(3)+6D旋转(6)] + gripper。
  - 代码: openx_droid `_camera_abs_pose_at`(逐帧相机系绝对位姿, 复用投影/外参); base.py `future_traj`(抽帧+逐帧位姿目标); config `droidex2000_C7_videotraj`(chunk=8, num_queries=8, 相机系绝对位姿 bounds)。
- **诊断(latest.pt, held-out, 注意目标=绝对位姿, std~18cm 远大于 delta 的 3-5cm)**:
  | 维 | corr | MAE | pred_std/tgt |
  |---|---|---|---|
  | cam-x | **0.916** | 5.97cm | 0.83 |
  | cam-y | **0.926** | 4.17cm | 1.00 |
  | cam-z(深度) | **0.886** | 5.79cm | 0.86 |
  | 旋转 r0-r5 | **0.78-0.85** | — | 0.91-0.98 |
  | **XYZ MAE** | — | **5.31cm vs predict-mean 14.86cm = 胜 64.3%** | |
- **★★结论(对比之前所有预测 setup: 最好胜基线~17%、dx0.48、旋转≈0、pred_std塌缩)**:
  1. **xyz corr 全 0.88-0.93**——从帧读 EE 位置很准, **连深度 z corr 0.886**(8帧+夹爪表观大小给深度线索, 固定相机)。
  2. **旋转第一次学到(corr 0.78-0.85)**——"从帧读朝向"well-determined(夹爪朝向画面可见), 不像预测未来旋转那样欠定。
  3. **pred_std/tgt ≈0.83-1.0, 完全不塌缩**——欠定/动量天花板**消失**(感知而非预测)。
  4. 相对精度胜基线 64%、corr 0.9 ≫ 之前一切。**用户的"宏观轨迹+未来帧作输入"判断是决定性正确的**。
- **诚实 caveat**:
  - **过拟合(损失层面)**: train_action 0.0037 ≪ val_action 0.111; 但 **held-out 位姿 corr 仍 0.9**(轨迹形状泛化好), MAE 5.3cm 是真实泛化误差。latest.pt 位姿 MAE(5.31) 优于 best.pt(6.05, gripper-val选的) → 无真正位姿过拟合, 多训反而更好。
  - **★ 关键部署 gap**: 训练/评估都在**真实帧(gt.mp4)**; 部署帧=**生成的语义视频**(夹爪形变/大小/角度不准)。此模型**显式从像素读夹爪位姿**, 生成帧外观不同 → 不微调可能不迁移。这是这条路最大风险。→ 后续: 强增强 / 生成视频微调 / 加深度输入(用户计划)。
  - z 在真实帧上意外地好(0.886), 但部署生成帧的深度线索可能失真。
- **意义**: 把 gen2act 高层模块定位成"**视频→6DOF轨迹提取器**"是可行且强的; xyz+旋转全部高相关。下一步重心从"性能"转向"**真实帧→生成帧的迁移**"(跨视频/生成视频微调)。
- **鲁棒性探针(`scripts/robustness_eval.py`, 对 held-out 真实帧加扰动模拟生成外观 gap)**:
  | 扰动 | XYZ MAE | x/y/z corr |
  |---|---|---|
  | clean | 5.31 | 0.916/0.926/0.886 |
  | 模糊(σ2) | 6.11 | 0.906/0.918/0.853 |
  | 强变色 | 5.54 | 0.903/0.925/0.880 |
  | 降分辨率0.4× | 5.75 | 0.907/0.925/0.869 |
  | 遮挡cutout | 6.16 | 0.885/0.913/0.867 |
  | 重噪声 | 7.19 | 0.840/0.900/0.803 |
  - **模型对外观扰动相当鲁棒**: 模糊/变色/降分辨率/遮挡几乎不掉(corr>0.88), 重噪声才掉到 0.84; xy 尤其稳, z(深度线索)略脆。→ C7 靠**鲁棒的夹爪定位**而非脆弱精确像素, 对"生成视频外观 gap"是**好兆头**。
  - **caveat**: 这些只是光度扰动代理; 真实生成视频还有**结构性差异**(夹爪几何形变/朝向语义错), 光度扰动测不到 → 鲁棒性是必要非充分证据, 真正确认仍需真生成视频。

## Exp C8 — C7 video-to-trajectory + flow 头  ⚠️ 可用但不胜回归
- **动机**: C7 是 one-shot 回归, 8 个 waypoint 彼此只通过共享条件间接耦合。C8 把头换成 flow/DiT, 在去噪过程中让 8 个 waypoint 通过 self-attn 互相参照, 测试 joint iterative generation 是否能超过 C7。
- **config**: `droidex2000_C8_videotraj_flow`。输入/目标与 C7 相同: 当前帧 + 8 个未来帧 → 8 个相机系绝对位姿(pos+6D rot)+gripper。唯一核心差别: `fused_query_reg` 回归头换成 `fused_query_flow`, IDM式 flow DiT(1024/8层/16头, K=8 eval samples), lr 1.2e-4, 20ep。
- **训练曲线**: train velocity loss **0.6095 → 0.0148** 正常快速下降; infer_loss ep10 最低 **0.164** 后升到 ep20 **0.207**。但与 C7 一样, best.pt 受总 val loss/gripper 干扰, **pose 诊断 latest.pt 更好**。
- **终诊断(800 held-out, K=8 多采样)**:
  | ckpt | cam-x corr/MAE | cam-y corr/MAE | cam-z corr/MAE | XYZ MAE | 旋转测地角 |
  |---|---|---|---|---|---|
  | **C8 latest** | 0.898 / 6.86cm | **0.931 / 4.04cm** | 0.868 / 5.93cm | **5.61cm**(胜均值62.2%) | 26.68° |
  | C8 best | 0.903 / 7.33cm | 0.924 / 4.61cm | 0.859 / 6.60cm | 6.18cm(胜58.4%) | 28.63° |
  | **C7 latest(回归)** | **0.916 / 5.97cm** | 0.926 / 4.17cm | **0.886 / 5.79cm** | **5.31cm**(胜64.3%) | **23.15°** |
- **结论**:
  1. **flow 头能学会 video→trajectory 感知**: XYZ corr 0.87-0.93, 旋转角误差显著胜 mean-rotation baseline(58.7°→26.7°), 不是失败。
  2. **但不超过 C7 回归头**: XYZ MAE 5.61 > 5.31cm, 旋转 26.7° > 23.2°。joint generation 没带来 waypoint consistency 的收益, 反而更慢、更重(262M vs 174M)。
  3. **后续不应继续围绕 flow 头微调**: 在这个 well-determined "从给定帧读位姿"任务里, 直接回归已经足够且更优。下一阶段瓶颈不是 head, 而是 **真实帧训练 → 生成视频部署** 的 domain/structure gap, 以及如何把生成视频语义转成闭环控制。

## Exp C10 — C7 + 训练时结构增强（EE 针对性 blur/scale/cutout）  ⏳ 训练进行中（ep1 loss=0.6408，~392s/ep，20ep≈2.2h）
- **动机**: C7 在真实帧上 corr ~0.9，但部署时 source 是**生成视频**（夹爪外观不同：形变/尺度/纹理错误）。Robustness eval 只测光度扰动（blur/color/noise），测不到结构性差异。C10 在训练时针对 EE 区域加结构扰动，让模型从「夹爪精确像素」转向「场景 context + EE 大致位置」读取位姿。
- **架构**: 与 C7 完全相同（174M fused_query_reg，cross-attn 回归+深监督）。唯一区别=训练时 source_video 的增强。
- **4 类增强**（仅对 source_video 即 8 个未来帧，仅 train split）:
  - **A. 逐帧独立光度扰动**: 每帧独立随机化 brightness/contrast/saturation/noise（比 C7 per-clip 幅度更大），模拟生成视频帧间外观不一致。
  - **B. EE 区域局部 blur**: 以 GT 未来 EE 2D 投影坐标为中心，对 ~30%×30% 的 patch 做 Gaussian blur（prob 0.6），模拟夹爪纹理/外形模糊。
  - **C. EE 区域局部缩放**: EE 中心 patch ±20% 缩放后贴回（prob 0.5），模拟生成视频夹爪尺度/深度错误。
  - **D. EE 矩形 mask**: EE 中心 ~25% patch 填均值色（prob 0.3），最激进：强迫模型完全不依赖夹爪像素。
  - EE 2D 位置由 `_future_traj_ee_image_fracs`（新增于 openx_droid.py）从 GT 未来相机系 XYZ + 相机内参投影得到（训练时可得，部署时不需要）。
- **config**: `droidex2000_C10_videotraj_structaug`。ep1 train_loss=0.6408，未出错，输出目录有 latest.pt/loss_history.csv。
- **代码改动**:
  - `r2r_gen2act/data/transforms.py`: 新增 `apply_structural_augmentation`、辅助函数 `_gaussian_kernel`/`_blur_patch`/`_ee_patch_box`
  - `r2r_gen2act/data/adapters/base.py`: 新增 `struct_aug_cfg`/`struct_aug_enabled` + 虚方法 `_future_traj_ee_image_fracs(→None)` + sample_window 调用
  - `r2r_gen2act/data/adapters/openx_droid.py`: 实现 `_future_traj_ee_image_fracs`（内参投影 + resize_center_crop 几何映射到 224×224）
- **结果（20 epoch 完整训练，latest.pt，800 held-out）**:
  - **Diagnose**: dx 0.866 / dy 0.924 / dz 0.889；XYZ MAE **5.87 cm**（胜 baseline 60.5%）。对比 C7 latest (5.31 cm，64.3%)：clean eval 略低 0.56 cm，符合预期（增强让模型「练难题」，clean 上有微小代价）。pred_std/tgt 更接近 1（dy=1.04），说明模型更「敢预测」，没因增强而缩水。
  - **Robustness eval 对比（C7 vs C10）**:
    | perturb | C7 MAE | C10 MAE | Δ | 结论 |
    |---|---|---|---|---|
    | clean | 5.31 | 5.87 | +0.56 | 微降（增强代价） |
    | blur | 6.11 | 6.54 | +0.43 | 略降 |
    | color | 5.54 | 6.04 | +0.50 | 略降 |
    | downscale | 5.75 | 6.08 | +0.33 | 略降 |
    | **noise** | **7.19** | **6.05** | **−1.14 ✓** | **改善！** |
    | **cutout** | **6.16** | **11.79** | **+5.63 ✗** | **严重退化（x corr 0.885→0.412）** |
  - **分析（两个核心发现）**:
    1. **Noise 改善**（−1.14 cm）：per-frame 独立光度扰动让模型对帧间噪声更鲁棒——符合目标，这对生成视频有益。
    2. **Cutout 严重退化**（+5.63 cm，x corr 0.412）：训练时 EE-centered cutout（mask 夹爪区域）让模型减少对 EE 像素依赖、转向周围 context。但 robustness_eval 的 cutout 是 **随机** 3 块（各 56×56），可能正好打中模型转移依赖的 context 区域，导致 dx 塌缩。是**训练分布 ≠ 测试分布**导致的反效果。结论：EE-centered cutout（D 类增强）有害，应去掉；blur（B）和 scale（C）无此问题。
  - **val_loss 曲线**：ep16 最低（0.129），ep20 也接近（0.128）；latest.pt(ep20) 用于诊断。
  - **后续建议**：去掉 ee_cutout（D），只保留 per_frame_photometric（A）+ ee_blur（B）+ ee_scale（C），重跑 C10-v2，预期 noise 改善保留、cutout 退化消失。有真实生成视频后替换 source_video 直接跑 diagnose 才是最终评估。见 Exp C10v2。

## Exp C10v2 — C10 去掉 ee_cutout，低 LR 微调 10 epoch  ✅ noise 改善保留，cutout 部分收回但未消除
- **config**: `droidex2000_C10v2_videotraj_structaug`。从 C10 latest.pt(ep20) 热启动，lr=3e-5（C10 的 1/4），10 epoch，去掉 `ee_cutout`（D），保留 A+B+C。
- **训练曲线**: train 0.0150→0.0091，val_action ep10=0.1242（val_xyz 5.72cm），全程平稳无崩溃。
- **终诊断 (latest.pt, 800 held-out)**: dx 0.866 / dy 0.925 / dz 0.889，XYZ MAE **5.75 cm**（胜 baseline 61.3%）。
- **三模型 Robustness 对比（MAE cm，越低越好）**:
  | perturb | C7 | C10 | C10v2 | vs C7 |
  |---|---|---|---|---|
  | clean | 5.31 | 5.87 | **5.75** | +0.44 |
  | blur | 6.11 | 6.54 | **6.31** | +0.20 |
  | color | 5.54 | 6.04 | **5.90** | +0.36 |
  | downscale | 5.75 | 6.08 | **5.82** | +0.07 ≈持平 |
  | **noise** | **7.19** | **6.05** | **5.78** | **−1.41 ✓✓** |
  | cutout | 6.16 | 11.79 | 11.28 | +5.12 ✗ 仍退化 |
- **结论**:
  1. **Noise 改善稳定保留**（7.19→5.78，−1.41 cm）：per-frame 独立光度扰动（A）是有效且可部署的增益，去掉 cutout 后依然保持。
  2. **Blur/color/downscale 接近 C7**：微调后 clean eval 代价从 +0.56 收窄到 +0.44，各光度扰动也收窄。
  3. **Cutout 退化未消除**（11.28 vs C7 的 6.16）：退化来自 ee_blur + ee_scale 造成的注意力模式改变，不只是 ee_cutout。去掉 D 后小幅收回（11.79→11.28），但结构增强（B+C）本身就让模型在 random-cutout 下表现更差。这是 clean-eval 与 random-cutout-eval 之间的内在 trade-off。
  4. **目标对齐**：robustness_eval 的 random cutout 不是真实生成视频 gap 的代理。真实 gap 是夹爪外观不同（B+C 直接针对），不是随机遮挡。cutout 数字退化不影响 C10v2 对生成视频的实际价值。
- **当前最佳（有真实生成视频前的参考选择）**: noise 需要鲁棒→用 C10v2；clean eval 最优→用 C7。
- **下一步**: 获取真实生成视频后，替换 source_video_name，直接用 `diagnose_actions.py` 对比 C7 vs C10v2，才是终极评估。

## Exp C11 — PointWorld-DROID + Depth 3D Lifting（3D Diffuser Actor 风格）  ✅ 深度无显著增益，但整体极强

### 数据集：PointWorld-DROID 3000 episodes
- **数据**: `/mnt/pfs/share/shentingrui/dataset/pointworld-droid-3000/`（3000 episodes，1904 train + 62 val + 934 跳过）
  - 来源: PointWorld-DROID_restored（42935 episode 中的前 3000 匹配到本地 DROID-1.0.1 原始视频的）
  - 每个 episode: `gt.mp4`（RGB，symlink 到 DROID-1.0.1 exterior_image_1）+ `depth_frames/`（uint16 PNG，180×320，mm）+ `data.json`（兼容 droid-ex 格式）
  - **预处理脚本**: `scripts/preprocess_pointworld_droid.py`（32 workers，约 30 分钟完成全部 3000 eps）
  - **100% 匹配率**: PointWorld `scene_path` 的 `{lab}/success/{date}/{timestamp}` 可唯一映射到本地 DROID 1.0.1 episode

### 新增代码（depth 3D lifting 基础设施）
- `r2r_gen2act/modeling/depth_lifting.py` — `DepthTo3DPatchPositions`：纯几何，depth(uint16 mm)+内参(fx,fy,cx,cy@224) → 每个 DINOv2 patch 的 3D 相机系坐标 [B, 256, 3]
- `r2r_gen2act/modeling/fused_query_reg_policy.py` — 可选 depth token 组：256 patch 3D 位置 → MLP → attention pool → n_depth_tok 个条件 token，加入 self-attention mixer
- `r2r_gen2act/data/adapters/pointworld_droid.py` — 新 adapter：读 PointWorld 预处理结果，覆写 `_camera_serial`（ext1 优先）、`_camera_abs_pose_at`（注入 ext1 serial）、`_read_depth_at`（PIL 读 16-bit PNG）、`_get_camera_K_224`（内参缩放到 224×224 resize_center_crop 几何）
- `r2r_gen2act/data/adapters/base.py` — depth 加载钩子、`_frames_dir` 空目录回退修复

### 实验对比（20 epoch，batch_size=12，同 GPU 并行）

| 指标 | **C11-depth** (with depth) | **C11-nodepth** (RGB only) |
|---|---|---|
| dx corr | 0.997 | 0.997 |
| dy corr | 0.996 | 0.997 |
| dz corr | 0.982 | 0.981 |
| 旋转 r0-r5 corr | 0.962–0.983 | 0.954–0.979 |
| **XYZ MAE** | **1.193 cm** | **1.092 cm** |
| 胜 baseline | 91.1% | 91.8% |
| 参数量 | 176.8M | 173.8M |

### 核心结论
1. **两者都极强**（corr 0.997，XYZ MAE ~1.1 cm）——远超 droid-ex 上 C7 的 5.31 cm。PointWorld 数据质量更高，且 future_traj 任务本身 well-determined（从给定帧感知，非预测）。
2. **Depth 在此任务无显著增益**（+0.1 cm 劣势）：在 video-to-trajectory 感知任务里，RGB 像素已经 well-determined（corr 0.997），depth 的 3D 几何信息属冗余条件，反而引入轻微的拟合负担。
3. **Depth 更可能在欠定预测任务中有效**（C0-C5 系列）：深度能补充"相机系 dx/dz"等弱信号轴；但对从已知帧读位姿的感知任务，它没有帮助。
4. **旋转微弱优势**：有 depth 的 r0 corr 0.980 vs 无 depth 0.964，差距不显著。
- **config**: `pointworld3000_C11_videotraj_depth3d.yaml` / `pointworld3000_C11nodepth_videotraj.yaml`
- **val 曲线（infer_loss）**: C11-depth ep14 最低 0.0773 → ep20 0.0877；C11-nodepth ep14 最低 0.0696 → ep20 0.0887

## ⚠️ PointWorld-DROID 数据集问题（C11/C12/C13 结论存疑）
3000 个 episode 仅来自 **1 个实验室（AUTOLab）、5 台机器人、36 个 session**（同一天同一台机器人的连续录制）。
val split 按 episode 目录名随机，100 个 val episode 全部落在有 train episode 的 session 内（30/30 session 重叠）。
同一 session 内相机外参固定、场景几乎相同，"val" 实际是 within-session 插值，不是真正的泛化评估。
**C11/C12/C13 的 PointWorld 结论（corr ~0.997, MAE 1.1 cm）严重高估泛化能力，不可信。**
→ 后续 C12/C13 改用 droid-ex 重跑（见下方对应条目）。

---

## Exp C12 — Full-Patch Current Obs + Query-in-Mixer（组会后新方向实验1）  ⏳ 训练中

- **动机（组会讨论）**:
  1. 将 CrossAttn decoder 替换为 **Query-in-Mixer**：条件 token 和可学习 action query 共同进入 self-attention mixer，action token 直接从 mixer 输出读取。
  2. 当前帧使用**全部 256 个 DINOv2 patch token**（去掉 readout queries），保留空间细节；source video 仍用 32 query/帧的 readout 压缩。
- **架构变动（vs C11-nodepth）**:
  | 模块 | C11-nodepth | C12 |
  |---|---|---|
  | current obs 编码 | 32 readout queries | 256 patch tokens（全部） |
  | Mixer 输入 | 256(src)+32(cur)+8(ee) = 296 tok | 256(src)+256(cur)+8(ee)+8(act q) = 528 tok |
  | Action decoder | CrossAttnRegressionDecoder（6层，深监督，~57M） | Mixer 直接输出 → act_out_norm → Linear → tanh |
  | 参数量 | 173.8M | **117.1M**（去掉57M CrossAttn decoder） |
  | 深监督 | ✓（每层 aux loss） | ✗ |
- **任务**: 与 C11 相同，video-to-trajectory（future_traj 提供 8 帧 → 绝对相机系 EE 位姿）
- **数据**: PointWorld-DROID 3000 episodes（同 C11），batch=12，20 epoch
- **新增代码**:
  - `fused_query_reg_policy.py`: 新增 `current_full_patch`（`_encode_current_full` 方法，运行全12层 DINOv2、取 patch[:,1:]）+ `joint_action`（`act_query/type_action/act_out_norm/act_pose_head` + mixer 中追加 action token、从末尾读出）
  - `factory.py`: 读取 `current_full_patch`/`joint_action` flag，`joint_action=True` 时跳过 CrossAttnRegressionDecoder 构建
  - `configs/pointworld3000_C12_fullpatch_jointtok.yaml`: 新配置
- **config**: `pointworld3000_C12_fullpatch_jointtok.yaml`
- **训练曲线**: train_loss ep1 0.2154 → ep20 0.0010；infer_loss ep8 **0.0541**（最低）→ ep20 0.0822（过拟合；比 C11-nodepth 更早见底 ep8 vs ep14，因为无深监督）
- **终诊断**：
  | ckpt | dx corr | dy corr | dz corr | XYZ MAE | 胜基线 |
  |---|---|---|---|---|---|
  | best.pt (ep8) | 0.993 | 0.988 | 0.958 | 2.280 cm | 82.9% |
  | **latest.pt (ep20)** | **0.997** | **0.996** | **0.978** | **1.129 cm** | **91.5%** |
  旋转 corr: r0–r5 在 latest.pt 均为 0.950–0.974，pred_std/tgt ≈ 1.0（无塌缩）
- **★结论**:
  1. **C12 (1.129 cm) ≈ C11-nodepth (1.092 cm)**：差距仅 0.037 cm (3.4%)，但 C12 参数量少 57M（无 CrossAttn decoder）、无深监督。说明 **Query-in-Mixer 架构与 CrossAttn decoder 在此任务性能相当**。
  2. **256 patch token 的 current obs** 未带来明显收益（vs C11 的 32 readout token）：感知任务 well-determined，32 token 已足够；full patch 更多空间细节反而增加 mixer 容量负担（528 vs 296 token）。
  3. **过拟合更早出现**（ep8 vs ep14）：无深监督 + 118M vs 174M 参数 → 正则化弱，过拟合更快。latest.pt 仍好于 best.pt 说明 val_loss 受 gripper 干扰，同 C11 规律。
  4. **架构可行，可作为 Exp 2（行为克隆）基础**：joint-mixer + full-patch 方向被验证，下一步将任务改为 action chunk 预测（非 per-frame 位姿感知）。

## Exp C13 — Video-to-DeltaActionChunk（组会后新方向实验2）  ⏳ 训练中

- **动机（组会方向 2）**：在 C12 验证 Query-in-Mixer 架构之后，尝试将任务从 "future_traj 感知" 改成 "行为克隆"：
  - **C11/C12**：`future_traj.enabled=true` → source_video = 未来真实帧（泄漏），目标 = 绝对相机系 EE 位姿（感知任务，well-determined）
  - **C13**：`future_traj.enabled=false` → source_video = linspace demo 视频（可部署），目标 = 8 步 Delta action chunk（行为克隆，欠定）
  - C13 是 "Video-to-ActionChunk"：给定整段 demo 视频 + 当前观测，预测 8 个未来 delta 动作（ΔXY Z + 6D 旋转 delta + gripper）

- **架构**: 与 C12 完全相同（current_full_patch=True, joint_action=True, 117.1M 参数）
  | 项目 | C12 | C13 |
  |---|---|---|
  | source_video | 未来 8 帧（future_traj） | linspace demo（可部署） |
  | 动作目标 | 绝对相机系 EE 位姿 × 8 | 8 步 delta chunk（ΔXY Z+6D rot+grip） |
  | future_traj.enabled | true | **false** |
  | mapping | droid_observation_cartesian_future_delta_pose6d_camera | 同左（但走 `_action_at`） |
  | bounds | xyz±0.60 / rot±1 | **xyz±0.30 / rot±1** |
  | 参数量 | 117.1M | 117.1M（完全相同） |

- **代码新增**:
  - `r2r_gen2act/data/adapters/pointworld_droid.py`：新增 `_action_at` 覆写，注入 `ext1_cam_serial` 到 `mapping_cfg`，避免 `droid_action` 因 PointWorld 有3个 camera serial 而报 `ValueError: Expected exactly one 6D camera extrinsic`
  - `configs/pointworld3000_C13_deltaactionchunk.yaml`：新配置

- **config**: `pointworld3000_C13_deltaactionchunk.yaml`
- **数据**: PointWorld-DROID 3000 episodes（1904 train / 62 val，同 C11/C12）
- **训练**: batch=12, 20 epoch, lr=1.2e-4（同 C12），于 2026-07-02 启动
- **训练曲线**: train_loss ep1 0.215 → ep20 0.027；infer_loss ep8 **0.1437**（最低）→ ep20 0.2232（过拟合，规律同 C12）
- **终诊断（latest.pt @ep20，800 held-out windows × 8-step chunk = 6400 samples）**:

  | dim | corr | MAE | pred_std/tgt_std |
  |---|---|---|---|
  | dx | **0.829** | 3.214 cm | 0.894 |
  | dy | **0.809** | 3.266 cm | 0.860 |
  | dz | **0.782** | 3.512 cm | 0.868 |
  | r0–r5 | **0.643–0.725** | — | 0.597–0.936 |
  | **XYZ MAE** | — | **3.331 cm** | vs baseline **5.762 cm → 胜 42.2%** |

  best.pt: XYZ MAE 3.488 cm（胜 39.5%）；latest.pt 更好（同 C11/C12 规律，val_loss 含 gripper）

- **★结论**:
  1. **delta 预测三轴全部学到（corr 0.78–0.83），且不塌缩（pred_std/tgt ≈ 0.86–0.89）**——对比 droid-ex 上最好的 delta 实验（C5 dx 0.62、C1 dx 0.48），C13 在 PointWorld 上显著更强。原因：PointWorld 实验室数据标定质量更高、外参更一致，相机系 delta 更可学。
  2. **旋转也学到（corr 0.64–0.73）**——比 droid-ex 系列所有 delta 实验（旋转 ≈ 0）大幅提升；与 C7/C11 的"从给定帧读"感知结果（旋转 corr 0.78–0.85）相比仅有差距，考虑到 C13 是真正的预测任务，结果合理。
  3. **对比 C11-nodepth / C12（同数据，abs 位姿）**: 绝对位姿感知 corr ~0.997 / MAE ~1.1 cm，delta 预测 corr 0.78 / MAE 3.33 cm——任务难度差距合理（感知 well-determined vs 预测欠定）。
  4. **过拟合仍存在**（ep8 见底）：C13 的任务比 C11/C12 更欠定（delta 不像绝对位姿那样 well-determined），过拟合比 C12 更明显。latest.pt 仍好于 best.pt（gripper 污染 val_loss），同之前规律。
  5. **架构可行**：C12 的 Query-in-Mixer（joint_action=True）同样能有效处理行为克隆任务，无需 CrossAttn decoder。
  - ⚠️ **以上结论受 PointWorld session 泄漏影响，参考意义有限。** 见 droid-ex 重跑版本（下方）。

## Exp C12-droidex — Full-Patch + Query-in-Mixer，droid-ex 重跑  ✅ 完成

- **动机**: C12 PointWorld 结果存 session-level 数据泄漏，改用 droid-ex（episode-level 独立 split）重评估。
- **架构**: current_full_patch=True, joint_action=True，117.1M，无 CrossAttn decoder
- **任务**: video-to-trajectory（future_traj=True，abs 相机系 EE 位姿），与 C7 任务一致
- **对照**: C7（CrossAttn+深监督，174M）→ XYZ MAE 5.31 cm，corr 0.886–0.926
- **数据**: droid-ex（3096 train / 93 val），batch=12，20 epoch（ep1–11 直训，ep12–20 断点续训）
- **config**: `droidex2000_C12_fullpatch_jointtok.yaml`
- **训练曲线**: train ep1 0.360 → ep20 0.0025；val ep8 **0.196**（最低）→ ep20 0.254（过拟合）
- **诊断（800 windows × 8 chunk = 6400 samples）**:

  | ckpt | dx | dy | dz | 旋转 r0–r5 | XYZ MAE | 胜基线 |
  |---|---|---|---|---|---|---|
  | best.pt (ep8) | 0.898 | 0.916 | 0.904 | 0.818–0.879 | 5.363 cm | 63.9% |
  | **latest.pt (ep20)** | **0.909** | **0.916** | **0.900** | **0.816–0.888** | **5.190 cm** | **65.1%** |
  | C7 参照 | 0.886 | 0.926 | — | — | 5.31 cm | — |
  pred_std/tgt ≈ 0.84–1.04（无塌缩）

- **★结论**: **C12 ≈ C7 (5.19 vs 5.31 cm，+2%)**，参数量少 57M（无 CrossAttn decoder、无深监督）。full-patch current obs + joint mixer 架构在此任务上与 CrossAttn 持平，但更轻量。

## Exp C13-droidex — Video-to-DeltaActionChunk，droid-ex  ✅ 完成

- **动机**: C13 PointWorld 结论受数据泄漏影响，改用 droid-ex，与 C1/C5 等历史 delta 实验可比
- **架构**: 与 C12-droidex 相同（current_full_patch=True, joint_action=True, 117.1M）
- **任务**: delta action chunk（future_traj=False, linspace demo，ΔXY Z + 6D-rot-delta + gripper）
- **对照**: C5（droid-ex，EE 投影，dx 0.623，XYZ MAE 2.873 cm）；C1（双路因果，dx 0.482，3.027 cm）
- **数据**: droid-ex（同上），batch=12，20 epoch
- **config**: `droidex2000_C13_deltaactionchunk.yaml`
- **训练曲线**: train ep1 0.229 → ep20 0.025；val ep2 **0.180**（最低）→ ep20 0.313（严重过拟合）
- **诊断（800 windows × 8 chunk = 6400 samples）**:

  | ckpt | dx | dy | dz | 旋转 r0–r5 | XYZ MAE | 胜基线(5.32cm) |
  |---|---|---|---|---|---|---|
  | best.pt (ep2) | 0.649 | 0.762 | 0.667 | 0.443–0.584 | 4.036 cm | 24.2% |
  | **latest.pt (ep20)** | **0.674** | **0.766** | **0.664** | **0.455–0.579** | **3.945 cm** | **25.9%** |
  | C5 参照 | 0.623 | 0.655 | 0.454 | ~0 | 2.873 cm（基线 3.47cm） | — |
  pred_std/tgt ≈ 0.66–0.87（dz 略塌）

- **★结论**:
  1. **dx/dy/dz corr 0.65–0.77，比 C5（dx 0.62）有提升，旋转首次学到（0.44–0.58，C5 ≈ 0）**。
  2. **XYZ MAE 绝对值（3.95 cm）比 C5（2.87 cm）高**，但 baseline 也不同（5.32 vs 3.47 cm），droid-ex delta 任务更难（3000+ 多样 episode vs 当时 2000）。
  3. **严重过拟合**：val 在 ep2 就见底，可能因为 delta 任务在 diverse droid-ex 上高度欠定，joint mixer 无深监督正则更弱。
  4. **下一步**：换 flow 头（C14），看能否用生成式建模缓解 delta 预测的欠定问题。

## Exp C14-droidex — Flow 头 + Delta ActionChunk，droid-ex（暂停）

- **动机**: C13 val 在 ep2 就见底，严重过拟合；换 FlowMatchingDiT 头看能否改善。
- **架构**: fused_query_flow（262.5M；C13 的 117.1M 外加 FlowDiT 8层 1024dim）；current_full_patch=False（flow head 用标准 32-query readout）
- **config**: `droidex2000_C14_flow_deltaactionchunk.yaml`
- **训练**（仅跑 2 epoch，之后暂停转 C15 方向）：ep1 train 0.3992；ep2 train 0.2625 / val 0.1687。
- **备注**: 暂停，后续可接续跑完 20 epoch。

## Exp C15-droidex — DeltaActionChunk + Auxiliary Trajectory Loss  ✅ 完成

- **动机**: C13 在 droid-ex 上严重过拟合（val ep2 见底，ep20 回升至 0.313），delta BC 任务欠定。
  加辅助损失：对 source video token 预测各帧的绝对 EE 位姿（C12 信号，well-determined），强迫 source token 编码轨迹信息作为正则。
- **架构**: 同 C13（117.1M）+ aux_traj head：
  mean-pool source mixer tokens `[B, 8×32, dim]→[B, 8, dim]` → LayerNorm → Linear(768→9) → tanh → `traj_pred [B, 8, 9]`；`total = action_loss + 0.5×aux_traj_loss`
- **config**: `droidex2000_C15_delta_auxtraj.yaml`
- **训练曲线（20 epoch）**：train ep1→0.3548，ep20→0.0197；val 最低 ep4=**0.2321**（含 aux 贡献，不可与 C13 直接比）→ ep20=0.3461
- **诊断（best.pt @ep4，800 windows）**：
  - dx corr 0.558 / dy 0.722 / dz 0.650；旋转 corr 0.48–0.56
  - **XYZ MAE = 4.359 cm**，基线 5.324 cm（胜 18.1%）
  - Per-chunk-step: k=0 (+333ms) 1.70cm → k=7 (+2666ms) 6.36cm
- **对比 C13（best.pt）**：

  | 指标 | C13 (ep2 best) | C15 (ep4 best) |
  |---|---|---|
  | XYZ MAE | **3.945 cm** | 4.359 cm（+10%，更差）|
  | dx corr | **0.674** | 0.558 |
  | dy corr | **0.766** | 0.722 |
  | dz corr | **0.664** | 0.650 |
  | 胜基线 % | **25.9%** | 18.1% |

- **★结论**:
  1. **aux traj loss（weight=0.5）没有改善 delta 预测质量，反而略有下降（比 C13 差 10%）**。
  2. val 最优点从 ep2 推到 ep4（轻微推迟过拟合），但整体曲线形状相似。
  3. 可能原因：① aux_traj_weight=0.5 过大，与 action loss 争梯度；② linspace demo 帧的 abs pose 和当前 step 的 delta 预测在时序上不对齐（demo 帧分布全集 episode，delta 只关心当前位置附近）；③ best.pt 按总 loss 选出，action-only 视角非最优点。
  4. **下一步候选**：① 接续 C14 flow head 跑完；② 降低 aux_traj_weight（0.1–0.2）再跑 C15v2；③ 调整正则策略。

## Exp C15v2-droidex — C15 续训，aux_traj_weight=0.1  ✅ 完成

- **动机**: C15（w=0.5）比 C13 差 10%；降 weight 减少梯度竞争。
- **设置**: 从 C15 ep20 latest.pt 接续，再跑 10 epoch（ep21-30），aux_traj_weight=0.1，其余同 C15。
- **config**: `droidex2000_C15v2_delta_auxtraj_w01.yaml`（epochs=30，resume 从 C15 latest.pt）
- **训练曲线**: train ep21→0.0249，ep30→0.0137；val 最低 ep22=**0.2553** → ep30=0.3703（持续过拟合）
- **诊断（best.pt @ep22，800 windows）**：
  | 维度 | corr | MAE |
  |---|---|---|
  | dx | 0.714 | 3.736 cm |
  | dy | 0.761 | 3.809 cm |
  | dz | 0.709 | 3.853 cm |
  | r0–r5 | **0.645–0.709** | 0.022–0.105 |
  - **XYZ MAE = 3.799 cm**，基线 5.324 cm（**胜 28.6%**）
  - pred_std/tgt: 0.69–0.95（几乎无塌缩）
  - Per-chunk-step: k=0 (+333ms) 1.31cm → k=7 (+2666ms) 5.73cm

- **三实验对比（droid-ex delta chunk）**：

  | 指标 | C13（无 aux）| C15（w=0.5）| **C15v2（w=0.1）** |
  |---|---|---|---|
  | XYZ MAE | 3.945 cm | 4.359 cm | **3.799 cm** |
  | dx corr | 0.674 | 0.558 | **0.714** |
  | dy corr | **0.766** | 0.722 | 0.761 |
  | dz corr | 0.664 | 0.650 | **0.709** |
  | 旋转 corr | 0.44–0.58 | 0.48–0.56 | **0.64–0.71** |
  | 胜基线 % | 25.9% | 18.1% | **28.6%** |

- **★结论**:
  1. **aux_traj_weight=0.1 有效**：XYZ MAE 3.945→3.799 cm（-3.7%），超过 C13。
  2. **旋转大幅改善（corr 0.44–0.58 → 0.64–0.71）**：aux traj 正则确实帮助 source token 编码轨迹方向信息，旋转预测显著受益。
  3. **pred_std/tgt 接近 1（0.69–0.95）**：动作分布几乎不再塌缩，对比 C13 的 0.66–0.87。
  4. w=0.5 是副作用（与 action loss 争梯度），w=0.1 是合适的平衡点。

## Exp C13ext-droidex — C13 续训 10 epoch（对照实验，隔离 epoch 因素）  ✅ 完成

- **动机**: C15v2 跑到 ep30，C13 只到 ep20。为排除"单纯多训 10 epoch"的混淆，从 C13 ep20 接续再跑 ep21-30（**无 aux loss**），与 C15v2 同起点、同轮数、同 schedule，唯一差异是 aux loss。
- **config**: `droidex2000_C13ext_ep30.yaml`（epochs=30，resume 从 C13 latest.pt，输出到独立目录不覆盖 C13）
- **训练曲线**: train ep21→0.0296，ep30→0.0190；val 最低 ep22=**0.2549** → ep30=0.3688（与 C15v2 曲线几乎重合）
- **诊断（best.pt @ep22，800 windows）**：dx corr 0.677 / dy 0.766 / dz 0.668；旋转 corr 0.46–0.59；**XYZ MAE = 3.922 cm**（胜基线 26.3%）；pred_std/tgt 0.72–0.84

- **★三方对照（同起点 ep20，同 best 点 ep22）**：

  | 指标 | C13 (原, best ep2) | C13ext (续训, best ep22) | **C15v2 (aux w=0.1, best ep22)** |
  |---|---|---|---|
  | XYZ MAE | 3.945 cm | 3.922 cm | **3.799 cm** |
  | dx corr | 0.674 | 0.677 | **0.714** |
  | dy corr | 0.766 | 0.766 | 0.761 |
  | dz corr | 0.664 | 0.668 | **0.709** |
  | 旋转 corr | 0.44–0.58 | 0.46–0.59 | **0.64–0.71** |
  | xyz pred_std/tgt | 0.66–0.87 | 0.72–0.84 | **0.88–0.95** |
  | 胜基线 % | 25.9% | 26.3% | **28.6%** |

- **★★结论（关键对照）**: **C15v2 的改善来自 aux loss，不是多训练的 epoch。**
  1. **多训 10 epoch（C13→C13ext）几乎无变化**：XYZ MAE 3.945→3.922（-0.6%），旋转 corr 原地踏步 → C13 在 ep2 已饱和，之后纯过拟合。
  2. **加 aux loss（C13ext→C15v2）才有真提升**：同 ep30、同 ep22 best，XYZ MAE -3.1%，**旋转 corr 0.46–0.59 → 0.64–0.71**，xyz pred_std/tgt ~0.78 → ~0.92。
  3. 干净地证明 aux traj loss 的正则效果真实存在，尤其显著改善旋转维度。

## Exp C16-droidex — Flow 头 + Delta ActionChunk，完整 30 epoch  ✅ 完成

- **动机**: C14 只跑 2 epoch 暂停；完整评估 flow 头能否缓解 delta 任务过拟合。每 epoch 都评估（eval_every_epochs=1）以看清收敛曲线。
- **架构**: fused_query_flow（262.5M，FlowDiT 头 176.1M，8 层 1024dim，16 步 Euler 采样，8 eval samples）；current obs 用标准 32-query readout（flow policy 不支持 full_patch）
- **config**: `droidex2000_C16_flow_delta_ep30.yaml`（30 epoch，从头训练）
- **训练曲线**（train = velocity MSE；infer = 采样后 action MSE）：

  | epoch | train | infer | | epoch | train | infer |
  |---|---|---|---|---|---|---|
  | 1 | 0.4145 | 0.2068 | | 16 | 0.0840 | 0.1611 |
  | **4** | 0.2059 | **0.1341**(min) | | 20 | 0.0628 | 0.1727 |
  | 7 | 0.1535 | 0.1377 | | 24 | 0.0480 | 0.2120 |
  | 9 | 0.1348 | 0.1413 | | 28 | 0.0396 | 0.2335 |
  | 14 | 0.0962 | 0.1473 | | 30 | 0.0388 | 0.2452 |

  train 单调降（0.41→0.039）；infer ep4 见底（0.1341）后 ep8-14 平台（0.14–0.15），ep15 起单调升到 ep30=0.2452。

- **★★诊断的坑：flow 的 val loss 曲线严重误导 checkpoint 选择**：
  - **best.pt @ep4（infer 最低 0.1341）**: XYZ MAE **5.460 cm（输基线 2.6%）**；dx corr 0.013，旋转 corr≈0，**pred_std/tgt 仅 0.34–0.56（分布严重塌缩）**。
  - **latest.pt @ep30（infer 最高 0.2452）**: XYZ MAE **3.944 cm（胜基线 25.9%）**；dx 0.661 / dy 0.794 / dz 0.663；旋转 corr 0.37–0.58；pred_std/tgt 0.45–0.89。
  - **原因**: ep4 模型分布仍塌缩，采样动作接近均值 → 采样 MSE 看着低但 action 质量差；ep30 学到真实多模态分布，MAE 才真正下降。**flow 的 infer_loss(采样MSE) 低 ≠ action 准**，不能用它选 best。

- **★横向对比（各取实际最优 checkpoint）**：

  | 指标 | C13(回归) | C15v2(回归+aux) | C16 flow(ep30) |
  |---|---|---|---|
  | XYZ MAE | 3.945 | **3.799** | 3.944 |
  | dx corr | 0.674 | **0.714** | 0.661 |
  | dy corr | 0.766 | 0.761 | **0.794** |
  | dz corr | 0.664 | **0.709** | 0.663 |
  | 旋转 corr | 0.44–0.58 | **0.64–0.71** | 0.37–0.58 |
  | pred_std/tgt | 0.66–0.87 | **0.88–0.95** | 0.45–0.89 |
  | 参数量 | 117M | 117M | 262M |
  | 胜基线 % | 25.9% | **28.6%** | 25.9% |

- **★结论**:
  1. **flow 头（ep30）≈ C13 回归头**（XYZ MAE 3.944 vs 3.945），但用 2.2× 参数、更长训练。**未兑现"生成式建模缓解欠定"的预期。**
  2. flow 唯一亮点 dy corr 0.794（三者最高），但旋转反而比 C15v2 差。
  3. **目前最优仍是 C15v2（回归 + aux traj w=0.1）**：MAE 最低、旋转最好、分布最不塌缩、参数最少。
  4. **教训**: flow 头的 best.pt 选择需改用 diagnose 的 XYZ MAE 而非 infer_loss，否则会选到严重欠拟合的早期 checkpoint。

## Exp C17-droidex — Flow 头 + Aux Traj Loss（当前最优）  ✅ 完成

- **动机**: C16（flow 单用）≈ C13（无提升），C15v2（回归+aux）证明 aux 有效。把 aux traj loss 加到 flow 头上，看两者能否叠加。
- **架构**: fused_query_flow（262.5M）+ aux_traj head（挂在 cond 的 source token 上，与 C15v2 同款）；aux_traj_weight=0.1
- **实现**: `fused_query_flow_policy.py` 加 `_aux_traj_pred`（cond 前 8×32 source token → mean-pool → LN → Linear(768→9) → tanh），训练/采样两条路径都注入 `traj_pred`；factory 传 `aux_traj_cfg`；losses.py 无需改（aux 块 mode 无关）
- **config**: `droidex2000_C17_flow_delta_auxtraj.yaml`（ep5 处暂停后 resume 续训至 ep30）
- **训练曲线**: train ep1→0.44、ep30→0.025；infer ep7 见底 **0.1470** → ep30 升至 0.2631（同 C16 的过拟合形状，val loss 曲线误导）
- **诊断（latest.pt @ep30，800 windows）**：
  - dx corr 0.749 / dy 0.818 / dz 0.745；旋转 corr 0.63–0.69
  - **XYZ MAE = 3.423 cm（胜基线 35.7%）**；pred_std/tgt 0.85–0.92
  - Per-chunk-step: k=0 (+333ms) 1.23cm → k=7 (+2666ms) 5.20cm

- **★四方对比（各取实际最优 checkpoint）**：

  | 指标 | C13(回归) | C15v2(回归+aux) | C16(flow) | **C17(flow+aux)** |
  |---|---|---|---|---|
  | XYZ MAE | 3.945 | 3.799 | 3.944 | **3.423** |
  | dx corr | 0.674 | 0.714 | 0.661 | **0.749** |
  | dy corr | 0.766 | 0.761 | 0.794 | **0.818** |
  | dz corr | 0.664 | 0.709 | 0.663 | **0.745** |
  | 旋转 corr | 0.44–0.58 | 0.64–0.71 | 0.37–0.58 | **0.63–0.69** |
  | pred_std/tgt | 0.66–0.87 | 0.88–0.95 | 0.45–0.89 | **0.85–0.92** |
  | 胜基线 % | 25.9 | 28.6 | 25.9 | **35.7** |

- **★★结论（关键）**: **flow 和 aux 是叠加增益，不是替代关系。**
  1. flow 单用（C16 3.944）无提升、aux 单用（C15v2 3.799）小提升，**两者结合（C17 3.423）超过各自**——aux 提供了 flow 头缺的正则/轨迹表征，flow 的分布建模能力才真正兑现。
  2. C17 是目前所有 delta-chunk 实验的最优：XYZ MAE 最低、xyz corr 最高（0.75–0.82）、分布几乎不塌缩。
  3. 再次验证 checkpoint 选择：latest.pt(ep30, val 最高) 才是最优，早期 val 低点欠拟合。

## Exp C18-droidex — C17 + 时序进度 aux + source 帧浮动采样  ✅ 完成（略逊 C17）

- **动机**: 在 C17（当前最优）上叠加两个改进：① 时序进度对齐 aux（从全部 cond token mean-pool 预测 current 在 demo 的归一化进度 ∈[0,1]，弱 demo-current 对齐信号）；② source 帧浮动采样（linspace 窗口起止各浮动 ±20% clip 长度，train only，增加 demo 多样性）。
- **实现**:
  - data: `aux_progress`（progress_target = target_step/(num_steps-1)）+ `source_float`（float_frac=0.2）
  - flow policy: `aux_progress_head`（mean-pool 全 cond → LN → Linear→1 → sigmoid → progress_pred）
  - losses: `aux_progress_weight=0.1`，smooth_l1
- **config**: `droidex2000_C18_flow_aux_progress_float.yaml`（30 epoch，从头训练）
- **训练曲线**: train ep1→0.46、ep30→0.043；infer ep7 见底 **0.1658** → ep30 升至 0.2823（同 flow 过拟合形状）
- **诊断（latest.pt @ep30）**: dx corr 0.749 / dy 0.796 / dz 0.751；旋转 0.61–0.70；**XYZ MAE = 3.585 cm（胜基线 32.7%）**；pred_std/tgt 0.89–0.97

- **对比 C17**：

  | 指标 | C17(flow+aux) | C18(+progress+float) |
  |---|---|---|
  | XYZ MAE | **3.423** | 3.585（+4.7%）|
  | dy corr | **0.818** | 0.796 |
  | pred_std/tgt | 0.85–0.92 | 0.89–0.97 |
  | 胜基线 % | **35.7** | 32.7 |

- **★结论**:
  1. **两个改进合在一起是净负面（MAE +4.7%）**，但很轻微。
  2. pred_std/tgt 反而更接近 1（分布更不塌缩），但 MAE 没跟着降 → 预测方差变大但没更准。
  3. **无法归因**（两改动放一起）；最可疑是 source_float 改变了 source 帧覆盖分布，可能让 demo↔current 对齐更难，与 progress aux 打架。
  4. **下一步**: 消融拆开 → C19 只保留 source_float（关 progress），从 C17 ep30 微调 10 epoch，隔离 float 的边际效果。

## Exp C19-droidex — C17 + source_float 微调（消融：隔离 float）  ✅ 完成

- **动机**: C18 变差但拆不开归因。C19 只保留 source_float（关掉 progress aux），从 C17 ep30 **微调 10 epoch**（ep31-40，cosine 尾部低 lr ~2.7e-5），隔离 source_float 的边际效果。结构与 C17 一致（无 progress head）→ strict resume。
- **config**: `droidex2000_C19_flow_aux_float_ft10.yaml`（resume 从 C17 latest.pt，epochs 30→40，data.source_float.enabled=true）
- **训练曲线**: 微调初期 val 从 C17 ep30 的 0.2631 略降到 **ep32 0.2188**，之后缓升到 ep40 0.2635；train 0.041→0.029
- **诊断（latest.pt @ep40）**: dx corr 0.756 / dy 0.808 / dz 0.754；旋转 0.62–0.69；**XYZ MAE = 3.431 cm（胜基线 35.6%）**；pred_std/tgt 0.85–0.93

- **★三方消融**：

  | 指标 | C17(基线) | C18(progress+float) | C19(只float,ft) |
  |---|---|---|---|
  | XYZ MAE | 3.423 | 3.585 | **3.431** |
  | dx corr | 0.749 | 0.749 | 0.756 |
  | dy corr | 0.818 | 0.796 | 0.808 |
  | 胜基线 % | 35.7 | 32.7 | 35.6 |

- **★★结论（归因成功）**: **拖后腿的是 progress aux，不是 source_float。**
  1. **C19 ≈ C17（3.431 vs 3.423，持平）**：source_float 单独加入几乎无害也无明显收益。
  2. **去掉 progress 后从 C18 的 3.585 回到 C17 水平** → C18 变差的锅在 progress aux（全局 pool 预测标量进度，干扰了 cond token 的动作表征）。
  3. **progress aux 方向放弃**：弱对齐信号的这种设计对主任务是干扰。
  4. **source_float 中性偏正**（微调 ep32 val 短暂降到 0.219 < C17 的 0.263），但微调 10 epoch 从已过拟合起点发挥不出多样性正则；**应从头完整训练才能看真实价值**（体现在更晚过拟合 + 更低 val 底）。

## ★★★ Exp C19-new — 换数据集 droid-ex-3000-out，从头训 30 epoch  ✅✅✅ 历史最优，-28% MAE

- **动机**: 前面所有 delta 实验都在旧 droid-ex（2000 源，27634 windows）上过拟合，判断瓶颈是**数据量/欠定**而非模型。换更大更干净的 droid-ex-3000-out（VLM+运动筛选，train=4148/val=290，**43524 windows，+57%**）验证。
- **数据集**: 见 memory `droid-ex-3000-out-dataset`。格式全变（parquet+meta.json+rgb.mp4+分离 extrinsics.json），写了新 `droid_ex_out` adapter 组装 legacy payload；预抽 jpg 帧（GPU 91%）。
- **配置**: `droidex3000out_C19_flow_aux_float.yaml`——**与 C17/C19 建模完全相同**（flow 头 + aux_traj w=0.1 + source_float），唯一变量是数据集。从头训 30 epoch。
- **训练曲线**: train ep1→0.396、ep30→0.031；infer **ep8 见底 0.1257**（旧 C17 是 ep7=0.1470）→ ep30 升至 0.217。**val 底更低、过拟合更晚**（ep8-15 才明显 vs 旧 ep7）。
- **诊断（latest.pt @ep30，800 windows）**：

  | 指标 | 旧 droid-ex C17 | **新 C19** | 提升 |
  |---|---|---|---|
  | XYZ MAE | 3.423 | **2.479 cm** | **-27.6%** |
  | dx corr | 0.749 | **0.854** | +0.11 |
  | dy corr | 0.818 | **0.890** | +0.07 |
  | dz corr | 0.745 | **0.888** | +0.14 |
  | 旋转 corr | 0.63–0.69 | **0.76–0.83** | +0.13 |
  | pred_std/tgt | 0.85–0.92 | **0.91–0.96** | 几乎不塌缩 |
  | 胜基线 % | 35.7 | **52.5** | +17pt |
  - Per-chunk: k=0 (+333ms) 1.18cm → k=7 (+2667ms) 3.56cm（基线 8.05cm）

- **★★★ 结论（决定性）**:
  1. **XYZ MAE 3.42→2.48 cm（-28%），所有 delta-chunk 实验历史最优，首次胜基线 >50%**。
  2. **三平移轴全部 corr≥0.85，dz(0.888) 追平 dx/dy**——**z 轴（相机光轴/深度方向）之前一直最弱最塌缩，+57% 数据直接救活 → 证明 z 欠定的主因是数据量不足，非单目深度歧义**。（这也降低了硬上真实深度/Pow3R 的优先级。）
  3. 旋转 corr 0.76–0.83，逼近 C7"从帧读位姿"感知任务水平（0.78–0.85），但这是真·预测任务，含金量更高。
  4. **同一 C19 配置、唯一变量是数据 → 验证了长期判断：瓶颈是数据量/欠定，不是模型/头/loss**。换头（flow）、加 aux、加 float 都是小增益，**换更多更干净的数据才是最大杠杆**。
  5. checkpoint 选择再次验证：latest.pt(ep30, val 最高) 最优，早期 val 低点欠拟合（flow 头通病）。

## ★ Exp C20 — 显式 Δt 时间条件（当前 SOTA）  ✅ 有效，-5.9% MAE

- **动机**: 固定采 8 帧，但 clip 长度 31–597 帧（中位 116）跨度极大。同样 8 帧，短 clip 相邻帧真实间隔 ~0.4s、长 clip ~2s，模型不知道"每帧代表多少真实时间"→ 难把 demo 运动节奏对应到"下一步动作快慢"。
- **方案（做法 B）**: 给每帧显式喂 **真实 Δt（秒）= 相邻采样帧的帧差 / fps**（首帧=0）。不是归一化时间（那丢了绝对尺度），是真实秒数。sinusoid(秒)→MLP 编码，**叠加**到 source readout token 上（保留原 source_time_embed）。固定 8 帧不变。
- **实现**: base.py `sample_window` 产出 `source_dt`；flow policy `_dt_embed`（16 freq sinusoid, max_sec=5）+ `_readout` 叠加；trainer/diagnose 透传。+0.6M 参数（263.1M）。
- **训练**: 从 C19-new latest.pt **部分加载微调**（resume_checkpoint strict=False，Δt MLP 新初始化），10 epoch，LR 3e-5。config `droidex3000out_C20_dt_ft10.yaml`。
- **诊断（latest.pt @ep10）**:

  | 指标 | C19-new(无Δt) | **C20(+Δt)** |
  |---|---|---|
  | XYZ MAE | 2.479 | **2.333 cm (-5.9%)** |
  | dx corr | 0.854 | 0.864 |
  | dy corr | 0.890 | 0.904 |
  | dz corr | 0.888 | 0.901 |
  | 胜基线 % | 52.5 | **55.3** |
  - Per-chunk 全面下移: k=0 1.18→1.10cm, k=7 3.56→3.29cm

- **★结论**: **显式 Δt 有效**。10 epoch 微调 + 0.6M 参数换来 XYZ MAE -5.9%、三平移轴 corr 全升。验证"让模型知道每帧多少秒"能帮它把不同时长 demo 的运动速率对应到动作预测。注意 fps=15、action 预测 +333ms~+2667ms（chunk 8 步 × future_horizon 5）。

## Exp C21 — current 帧全 patch（256 token vs 32 readout）  ✅ 完成，无增益（负结果）

- **动机**: C12（回归头时代）验证过 current_full_patch 有效，但换 flow policy 后这个设计遗漏了——current 帧一直只用 32 readout query。补回：current 帧改用全 256 DINOv2 patch token（cond 序列 296→520）。
- **实现**: flow policy 加 `current_full_patch` 开关 + `_encode_current_full`（复用回归 policy 实现）。参数量不变（复用 backbone + type_current），从 C20 无缝续训（missing keys 为空）。
- **诊断（latest.pt @ep10）**: XYZ MAE **2.326 cm**（C20 是 2.333）→ **-0.3%，噪声范围内，无实质提升**。各维 corr 与 C20 几乎完全一致。
- **★结论（负结果）**: **full-patch current 在 flow + 大数据配置下无增益**。原因：① C12 有效是在回归头+小数据（2000）上，现在 flow+大数据（43524 windows）下 32 readout query 已能提取足够的当前状态信息；② current 帧不是当前瓶颈。**侧面印证瓶颈在动作生成表达力（DiT 容量）或数据量，非视觉输入** → 下一步扩容 DiT/vl_mixer 更对症。**决定回退到 32 readout current**。

## ★★★ Exp C22 — 扩容 DiT/vl_mixer（当前 SOTA）  ✅✅✅ -20.5% MAE，扩容大幅有效

- **动机**: C21 证明视觉输入不是瓶颈 → 猜测瓶颈在动作生成表达力（DiT 容量）。扩容验证。
- **改动**: flow DiT `num_layers` 8→12、vl_mixer_layers 4→6。current 回退 readout（32 query），保留 Δt+aux_traj+source_float。参数 **263M → 347.1M（+32%）**（DiT 117.6→176.3M、vl_mixer 50.4→75.6M）。
- **训练**: **从头 30 epoch**（扩容层是新容量，必须充分训练，微调不公平），LR 1.2e-4。config `droidex3000out_C22_bigdit_scratch.yaml`。~17.5min/epoch。
- **训练曲线**: infer_loss 明显低于 C19-new/C20（ep16 到 0.107 vs 0.126）；train ep30→0.029。
- **诊断（latest.pt @ep30）**:

  | 指标 | C20(263M) | **C22(347M)** | 变化 |
  |---|---|---|---|
  | XYZ MAE | 2.333 | **1.855 cm** | **-20.5%** |
  | dx corr | 0.864 | 0.929 | +0.065 |
  | dy corr | 0.904 | 0.951 | +0.047 |
  | dz corr | 0.901 | 0.941 | +0.040 |
  | 旋转 corr | 0.78–0.84 | **0.88–0.90** | +0.10 |
  | pred_std/tgt | 0.74–0.94 | 0.88–0.94 | 更不塌缩 |
  | 胜基线 % | 55.3 | **67.2** | +12pt |
  - Per-chunk 暴跌: k=7(+2667ms) 3.29→**2.26cm**（远未来长程质量大涨）
  - （注: C22 diagnose 抽到的 val baseline 5.658 vs C20 的 5.216，窗口不完全相同，但绝对 MAE 1.855<<2.333 是真实提升）

- **★★★ 结论（重要，修正认知）**: **大数据下模型容量重新成为瓶颈——数据和容量是互补的。**
  1. 扩容 +32% 参数 → XYZ MAE -20.5%，继换数据集后最大单次提升。
  2. **推翻"容量不是瓶颈"的旧判断**：C16（flow 头翻倍参数）在小数据（2000）上没用，让我们误以为容量无关。但 C22 证明——**数据不够时扩容没用（C16），数据够了（43524 windows）扩容才发挥威力**。二者互补，不是二选一。
  3. **远未来（k=7）受益最大**（3.29→2.26cm）：最难/最欠定的长程预测，扩容后 DiT 表达力明显帮到。旋转 corr 首次全 0.88+、pred_std/tgt 全 0.88+（几乎不塌缩）。
  4. 8→12 层就涨 20%，**可能还没到容量天花板** → 继续扩容（16层/更宽）值得试（见 C23，结论：其实已到顶）。
  5. 注: 数据集在后台持续写入，C22 启动后从 4579 一路涨到 ~28000（新 clip 04579+，mtime 07-08）；C22/C23 均用启动快照/max_episodes 锁定旧 4579，保证对照干净。

### C22 附加发现：采样精度是免费的杠杆（eval-only）
拿 C22 现成 checkpoint，只改 eval 采样 `num_inference_steps 16→32 + num_eval_samples 8→16`（零训练）：
XYZ MAE **1.855→1.672 cm（-9.9%）**，dx/dy corr 微升。ODE 积分更细 + 多样本平均降方差都有效。
→ 采样精度和模型容量是**两个独立杠杆**；16步/8样本是欠采样，拖累了模型真实能力。代价：eval 慢 4×。
**教训**: 训练用 16步/8样本省时，最终 diagnose 用 32步/16样本榨干真实能力。config `droidex3000out_C22_eval_steps32.yaml`。

## Exp C23 — 继续扩容 DiT16+mixer8 + source_len 12  ❌ 未涨点，触到容量天花板

- **动机**: C22（DiT12）猜"可能没到天花板"，继续推：DiT 12→16、vl_mixer 6→8、source_len 8→12。参数 347M→**431M（+24%）**，cond 序列 296→424 tokens。
- **数据**: `max_episodes: 4579` 锁定旧数据（00000-04578），与 C22 逐 episode 一致（train=4148/val=290），对照干净。从头训练，ep25 时用户手动停止（还差 5 ep）。
- **训练**: ~24min/epoch（比 C22 的 17.5min 慢，更大模型 + 12帧）。infer_loss 底 ep17-18=0.101（略低于 C22 的 0.107），但 ep25 已过拟合上升到 0.123。
- **诊断（32步/16样本，baseline 5.216 与 C22 可比）**:

  | checkpoint | XYZ MAE | dx/dy/dz corr |
  |---|---|---|
  | **C22 ep30** | **1.672** | 0.936/0.958/0.942 |
  | C23 ep25 (latest) | 1.829 | 0.910/0.949/0.934 |
  | C23 ep18 (best, val最低) | 2.091 | 0.889/0.934/0.921 |

- **★结论（容量天花板 + 负结果）**:
  1. **C23（431M）< C22（347M）**：继续扩容 + 加帧**没有涨点，反而退步**。即使补上少跑的 5 epoch，ep25 已在过拟合上升段，追不回 1.672。
  2. **天花板就在 C22 附近**：C22（8→12层）涨 20%，但 C23（12→16层）退步 → 这个数据量（43524 windows）下 **DiT ~12 层已够用**，再加层只增过拟合、不增表达力。
  3. **source_len 8→12 可能是主要拖累**：linspace 8 帧已覆盖操作关键阶段，多 4 帧信息量没增加却稀释注意力、加剧过拟合。（两改动同时上，未能单独归因，但方向层面确认无效。）
  4. 再次验证 flow checkpoint 选择：best.pt(ep18, val最低) 反而最差（2.091），latest(ep25) 才好——val loss 低≠action 好。
  5. **要再上一层的正确方向**：不是堆参数/加帧，而是**扩数据**（用后台新增的 ~23000 clip，容量+数据一起推）或换采样/架构。

## Exp C24 — 动态帧数（恒定Δt）+ 删除显式Δt输入  ❌ 退步，显式Δt更优

- **动机（用户假设）**: 用"按固定帧步长采样使 Δt 恒定"替代 C20/C22 的"固定8帧+显式Δt"——让节奏隐式编码进"每帧≈1秒"，从而可以删掉 Δt 输入。
- **方案**: k=clamp(round(num_frames/15),4,16)、linspace（目标 Δt=1.0s）；删 dt_time_embed；变长用桶采样（KBucketBatchSampler，同k一批）。架构回退 C22（DiT12/mixer6，346.5M）。数据锁 4579（同 C22/C23）。
- **实现**: base.py `_dynamic_source_len`+预计算 window_k；新建 bucket_sampler.py；trainer/diagnose 用桶采样；flow policy source_time_embed 尺寸到 k_max=16、aux_traj 用运行时 k。分布：k 覆盖 4-16，中位 8，长视频多为 16。
- **训练**: 从头 30 epoch，~21.7min/epoch（比 C22 慢，大量 k=16 长序列）。val 底 ep18=0.116（略高于 C22 的 0.107）。
- **诊断（latest.pt @ep30，32步/16样本；注意 baseline 5.701 与 C22 的 5.216 不同口径）**:

  | 指标 | C22(固定8帧+显式Δt) | C24(动态k+恒定Δt+删Δt) |
  |---|---|---|
  | XYZ MAE | **1.672** | 2.132 cm |
  | dx/dy/dz corr | 0.936/0.958/0.942 | 0.916/0.932/0.930 |
  | 旋转 corr | 0.86–0.93 | 0.78–0.83 |

- **★结论（负结果，用户假设不成立）**: **C24 < C22，"固定8帧+显式Δt"是更好设计。**
  1. **删显式Δt是主要损失**：C20 已证明显式Δt带来-5.9%；C24删掉它、靠恒定采样隐式编码替代，但恒定性不完美（只66%视频精确1.0s，34%因clamp偏离），丢失那34%的节奏信息又无显式Δt兜底 → 吐回C20的增益。
  2. **动态k本身中性偏负**：呼应C23（source_len 8→12无益）——linspace采样下"更多/可变帧"信息量没增加，长视频k=16的多余帧稀释注意力。
  3. **核心教训**: 显式Δt（对所有视频精确、不受clamp影响）比"靠采样恒定性隐式编码节奏"（做不到100%恒定）更可靠。**保留C22的固定8帧+显式Δt**。
  4. 当前SOTA仍是 **C22（347M，固定8帧+显式Δt）@32步采样 = 1.672cm**。

## Exp C25 — 动态帧数 + 加回Δt（归因 C24 掉分）  ✅ 归因成功：变长和删Δt各占一半

- **动机**: 用户质疑 C24 掉分主因不是删Δt（C20证明Δt仅-5.9%，量级对不上C24的-27%），而是**变长本身**。做对照：C24基础上加回Δt（其余不变），隔离两个因素。
- **设置**: 从 C24 latest.pt 续训（resume strict=False，dt_mlp新初始化），dt_time_embed开，动态k+桶采样+DiT12/mixer6不变，10 epoch，LR 3e-5。数据锁4579。
- **诊断（latest.pt @ep10，32步/16样本，baseline 5.701）**:

  | 指标 | C22(固定8+Δt) | C24(动态k,删Δt) | C25(动态k+Δt) |
  |---|---|---|---|
  | XYZ MAE | **1.672** | 2.132 | 1.906 cm |
  | dx/dy/dz corr | .936/.958/.942 | .916/.932/.930 | .923/.947/.943 |

- **★结论（归因成功，用户和我各对一半）**:
  1. **加回Δt有帮助**: C24→C25，2.132→1.906，补回约一半差距 → 删Δt确实是部分原因。
  2. **但Δt补不满**: C25(1.906)仍明显差于C22(1.672) → **变长本身也是问题**（用户判断对）。
  3. **C24掉分=两因素叠加**: ①删Δt(~0.23cm) + ②变长/训练不均(~0.23cm)，各占一半。
  4. **变长的机制**（代码核实）: `source_time_embed[max_source_len]` 是每帧一个可学习位置嵌入，`time_embed[:k]` 只取前k个 → **第12-15号位置只有k≥13的样本训练到，训练极不均衡**。这就是那~0.23cm的来源。
  5. **下一步（步骤2）**: 用户方案——所有case填充到16帧 + 独立可学习pad embedding + attention mask，让source_time_embed的16个位置均匀训练、消除变长。针对性解决剩余的0.23cm。

## Exp C26 — 填充到16帧 + 可学习pad embedding（无mask）  ⚠️ 方向对但幅度小，仍追不上C22

- **动机（步骤2）**: 解决 C25 诊断出的"位置嵌入训练不均衡"。所有case填充到16帧，不足用**可学习 pad_frame_embed**（非全0）。关键讨论：**不加 attention mask**——因为加mask会让pad位不参与反向，source_time_embed高位置仍拿不到梯度＝白做；纯填充（pad参与attention）才能让16个位置全部均匀训练。pad帧用独立可学习embedding标记，模型自学区分。
- **实现**: flow policy 加 `pad_source` + `pad_frame_embed[1,1,dim]`；`_readout` 加 `pad_to` 参数（真实k帧过backbone后补(16-k)个pad embedding帧，全16位置加source_time_embed）；aux_traj仍用真实k（pad帧不监督）。cond序列统一552（source16×32+current32+ee8）。从C25续训10ep。
- **诊断（latest.pt @ep10，32步/16样本，baseline 5.701）**:

  | 实验 | 设计 | XYZ MAE |
  |---|---|---|
  | C22 | 固定8帧+Δt | **1.672**（baseline 5.216）|
  | C24 | 动态k，删Δt | 2.132 |
  | C25 | 动态k+Δt | 1.906 |
  | **C26** | 动态k+Δt+填充16+pad embed | **1.840** |

- **★结论（动态帧这条线收尾）**:
  1. **填充方向对但幅度小**: C25→C26，1.906→1.840，补回0.066cm → 证明"位置嵌入训练不均衡"确是问题之一（用户诊断对），但只是次要因素。
  2. **仍追不上C22（1.672）**: 解决训练不均后，动态帧(1.840)还差固定8帧0.17cm → **变长还有其他损失**（pad参与attention的轻微污染 / 长视频16帧linspace信息密度变化 / 短视频pad占比高稀释信号）。
  3. **动态帧系列（C24→C25→C26）逐步逼近但天花板≈1.84，始终跨不过C22**。三次迭代（删Δt→加Δt→填充）证明：**固定8帧+显式Δt（C22）就是source编码的最优方案**。动态帧的理论优势（Δt恒定、覆盖长视频）被其代价抵消。
  4. **当前SOTA仍是C22（固定8帧+Δt，347M）@32步采样=1.672cm**。source编码/帧数策略这条线到头，下一杠杆是数据量。

## Exp C27 — 固定12帧 / 固定16帧（排除变长，单独验证帧数）  ❌ 更多帧更差，8帧是最优

- **动机**: 动态帧（C24-26）失败后，怀疑是"变长"的锅。C27做干净对照：**固定帧数**（所有视频都12或都16，linspace，无变长/无桶采样/无填充），Δt保留。从C22（固定8帧）微调10ep，隔离"帧数"这一个变量。
- **实现**: 两个config并行跑。source_len 8→12/16（data+model）。新增 `load_checkpoint` **shape-tolerant加载**：C22的source_time_embed[8]前缀拷贝进新[12]/[16]（前8位置保留、8-15新初始化）。数据锁4579（同C22；注：全量数据有新clip缺exterior_1 extrinsics会崩，故必须max_episodes）。
- **诊断（latest.pt @ep10，32步/16样本，baseline 5.216 同C22口径）**:

  | 实验 | source帧 | XYZ MAE | dx/dy/dz corr |
  |---|---|---|---|
  | **C22** | 固定8 | **1.672** | .936/.958/.942 |
  | C27-f12 | 固定12 | 1.935 | .905/.926/.925 |
  | C27-f16 | 固定16 | 1.902 | .902/.912/.919 |

- **★★结论（帧数策略最终定论）**:
  1. **固定12/16帧都明显差于固定8帧**（1.93/1.90 vs 1.67），且12≈16（几乎无差）。
  2. **彻底排除"变长是唯一问题"**：固定帧（无变长）下 8→12→16 依然单调变差 → 问题不在变长，在**帧数本身，8帧就是最优，更多帧有害**。
  3. **和C23完全呼应**（当时8→12无益）：**8帧linspace已覆盖操作关键阶段，更多帧只是稀释注意力/增加冗余token/加剧过拟合，信息量没增加**。source是"高层意图/参考"，粗粒度8帧足够表达demo意图。
  4. **帧数策略这条线彻底走到头**：综合C22-C27六个实验，**固定8帧+显式Δt（C22）是source编码的全局最优**。动态帧（C24-26）和固定更多帧（C27）都比不上它。
  5. **当前SOTA仍是C22 = 1.672cm**。下一杠杆只剩：扩数据（后台已攒~28k clip，需先修全量extrinsics缺失bug）或换任务范式。


## Exp C25v2/C26v2 — 排除"续训链污染"重跑（都从 C22 起）  ✅ 结论稳固：变长确实更差

- **动机（用户质疑）**: C25/C26 是接在 C24（动态k+删Δt，已跑坏到2.13）后面微调的，怀疑掉分是"续训链污染"——从一个差起点微调，而非动态帧本身的问题。重跑：**都直接从 C22（1.672 SOTA）微调**，用 shape-tolerant 加载（source_time_embed[8]→[16]前缀拷贝），隔离续训起点这个混淆变量。
- **C26v2**（动态k + Δt + pad-to-16可学习embed，从C22微调10ep，数据锁4579）诊断（latest.pt，32步/16样本，baseline 5.701）:
  - dx/dy/dz corr 0.926/0.940/0.941；**XYZ MAE = 1.934 cm**
- **对比**:

  | 实验 | 起点 | XYZ MAE |
  |---|---|---|
  | C22（固定8+Δt） | scratch | **1.672** |
  | C26（动态k+Δt+pad） | 从C24续训 | 1.840 |
  | **C26v2（动态k+Δt+pad）** | **从C22续训** | 1.934 |

- **★结论（用户假设不成立，动态帧定论加固）**:
  1. **换成从C22微调，动态帧不但没变好，反而更差（1.840→1.934）**——续训链污染不是主因，**动态帧本身就是比固定8帧差**。
  2. C26（从C24）1.840 反而略好于 C26v2（从C22）1.934：可能因为C24已在动态帧模式下收敛（source_time_embed 16位置训过），而C26v2从固定8帧起点，高位置嵌入是前缀拷贝的冷启动、10ep微调没训好。但两者都明显劣于C22。
  3. **彻底钉死结论**：无论从哪个checkpoint微调，动态帧/更多帧都追不上固定8帧+Δt。C22 是 source 编码的全局最优。**这条线到此为止，不再尝试帧数变体**。
  4. **当前SOTA仍是 C22 = 1.672cm**。唯一未尝试的大杠杆：全量数据（C28，~35.7k clip，已修 extrinsics 缺失bug，待启动）。

### C26v2 best.pt vs latest.pt（以后 best/latest 都测）
- C26v2 **best.pt**（ep2, val最低0.1456）: XYZ MAE **2.063 cm** — 比 latest.pt(1.934) 更差 0.13cm。
- 再次印证 flow 通病：best.pt(val最低) 欠拟合，latest.pt 更好。**以后 diagnose 固定 best 和 latest 都跑**。
- 但 best/latest 都 > C22(1.672) → 动态帧+pad 结论不受 checkpoint 选择影响。

## ★★★ Exp C28 — 全量数据 (~34k train clip, 8×) + C22架构  ✅ 数据scaling有效 (-8.5%, 干净对比)

- **配置**: C22架构(固定8帧+Δt, DiT12/mixer6, 347M)不变, 数据 4579→全量35696 (train 34008, val 500, windows 355461=8.2×). 修extrinsics缺exterior_1 bug. 从头12ep(~118min/ep), ep11手动停.
- **训练曲线**: val infer_loss 底 ep9=0.0584 (C22是0.107); train/val贴紧, 过拟合更晚 → 8×数据强正则.
- **⚠️ 对比口径大坑 (用户抓出泄漏)**:
  1. C28自己val(500): latest 1.757cm / best 1.868 (baseline 4.594). **别跟C22的1.672(baseline5.216)直接比 — val难度不同**.
  2. **错误尝试**: 用"max_episodes=4579+val300"划分测C28得1.389cm → **97.7%泄漏!** 那300 val里293个是C28的训练数据(C28全量训练只留500 val, 其余35196全train). 1.389作废.
  3. **正确干净对比**: 取 C28的val(500) ∩ C22没训练过的 = **437 clip 公共集** (两模型都没见过), 同脚本同采样(32/16)测:

  | 模型 | 数据 | XYZ MAE | dx/dy/dz corr |
  |---|---|---|---|
  | C22 | 4579 | 3.675 | .847/.880/.851 |
  | **C28** | 34008 (8×) | **3.363** | **.919/.934/.934** |

  (注: 该437-clip集baseline算法与标准diagnose略异, 绝对值偏高; 但两模型同脚本同数据, 相对比较可信)
- **★★★ 结论**: **数据scaling真实有效但幅度是 −8.5% (非泄漏版虚高的−17%)**. 三轴corr全面提升(dx.85→.92, dy.88→.93, dz.85→.93). 数据仍是最大杠杆但边际在递减: 2000→4579(+57%)给−28%, 4579→34008(8×)只给−8.5% → **接近数据饱和, 或新增的~30k clip质量/多样性不如前4579**.
- **★方法论教训 (关键)**: ① 不同val集(不同baseline)绝对MAE不可比; ② 跨训练规模比较必须构造"两模型都没训练过的公共val", 否则泄漏; ③ val_count小(500/35696)时, 换划分极易让旧val落入新train. flow latest>best再次成立.

## Exp C29 — C28 backbone 替换为 DINOv3-L/16  ⚠️ 未超过 C28

- **目的**: 单独验证视觉 backbone 扩容。保持 C28 的固定8帧+显式Δt、DiT12/mixer6、aux_traj、source_float 与训练配方不变，只把 DINOv2-B/14（85.7M）替换为 DINOv3-L/16（303.1M，LVD-1689M distilled）。完整 policy 参数从 347.1M 增至 564.6M。
- **实现细节**: DINOv3-L 有24个 blocks，`query_readout.segmenter_start=21`，保持与 C28 一样只让 readout query 经过最后3层；适配 DINOv3 RoPE 和5个 prefix tokens。预训练权重使用共享 HF cache 中的 `timm/vit_large_patch16_dinov3.lvd1689m`。
- **数据冻结**: `max_episodes=35696`，固定目录 `00000..35695`；继承 `val_count=500, split_seed=42`。过滤后 train=34008 episodes/355461 windows，val=488 episodes/5218 windows，与 C28 相同。
- **训练**: 从头12 epoch，latest=ep12。config `droidexFULL_C29_dinov3l_c28base.yaml`。
- **正式诊断**: latest.pt，冻结 val 前800 windows，32 ODE steps/16 samples：

  | 指标 | C29 ep12 latest |
  |---|---|
  | XYZ MAE | **1.874 cm** |
  | dx/dy/dz corr | 0.886 / 0.919 / 0.929 |
  | dx/dy/dz MAE | 2.103 / 1.753 / 1.765 cm |
  | pred_std/tgt | 0.928 / 0.894 / 0.954 |
  | 均值基线 | 4.594 cm（胜 59.2%） |

- **结论**: 在当前训练预算下，单独将 DINOv2-B 换成 DINOv3-L 没有超过 C28 ep17（1.703cm）。更大视觉 backbone 不是当前主要瓶颈；也可能需要不同的 backbone LR/更长训练才能兑现容量，但现有结果不支持直接替换。

## Exp C30 — 全量数据上单独扩容 DiT16/mixer8  ⏳ 低LR续训中

- **目的**: 干净拆开 C23 的混淆变量。保持 C28 的 DINOv2-B、固定8帧、Δt和全部数据设置，只将 flow DiT `12→16` 层、VL mixer `6→8` 层；source_len 不再同时改成12。完整 policy 347.1M→431.0M。
- **数据冻结**: 与 C29/C28 相同，候选35696、train=34008、实际val=488。
- **第一阶段**: 从头12 epoch，LR=1.2e-4。原始 latest=ep12；val flow loss 最低的 best=ep7。config `droidexFULL_C30_dit16_mixer8_c28base.yaml`。
- **第二阶段**: 从原始 ep12 latest 做 weights-only warm restart，新 optimizer/cosine，LR=3e-5，计划额外12 epoch；当前已有续训 ep5（等效总 ep17）。config `droidexFULL_C30_cont12_lr3e5.yaml`。
- **正式诊断**: 续训 ep5 latest，冻结 val 前800 windows，32 ODE steps/16 samples：

  | 指标 | C30 总ep17 latest |
  |---|---|
  | XYZ MAE | **1.765 cm** |
  | dx/dy/dz corr | 0.900 / 0.925 / 0.938 |
  | dx/dy/dz MAE | 1.990 / 1.656 / 1.648 cm |
  | pred_std/tgt | 0.911 / 0.894 / 0.956 |
  | 均值基线 | 4.594 cm（胜 61.6%） |

- **当前结论**:
  1. C30 在相同正式口径下优于 C29（1.765 vs 1.874cm，−5.8%），说明增加 action generator 容量比扩大视觉 backbone 更有效。
  2. 但当前 latest 仍略差于 C28 ep17 的1.703cm（+3.6%），尚未证明 DiT16/mixer8 胜出。
  3. 早期64-window快速诊断曾得到 ep4 best=1.444cm、ep6 latest=1.539cm，但该结果使用16 steps/8 samples且只测64 windows，**不能与800-window 32/16正式结果直接比较**。
  4. 原始 ep7 best 的800-window 32/16正式评估尚未完成；必须补测，因为 flow 的 val-loss best/latest 关系不稳定。低LR续训也只完成5/12 epoch。

## Exp C31 — corrected C30 + 当前腕部 RGB  ✅ 12 epoch完成，正式结果与C30持平

- **目的**: 验证近距离腕部视角能否降低夹取阶段几厘米位置误差。以 corrected C30 为唯一基线，保留8帧外部demo、当前外部RGB和EE状态，只新增与当前时刻同步的腕部RGB；不加入深度或未来腕部帧。
- **结构**: 当前外部帧和腕部帧共享同一个DINOv2-B、同一组32个readout query及最后3层query注入，仅增加独立视角类型向量 `type_wrist_current`（768参数），没有复制视觉backbone。
- **数据**: 仍冻结 `max_episodes=35696` 及原train/val划分。腕部原视频来自DROID `steps_observation_wrist_image_left.mp4`，按每个clip的 `source_frame_range=[f0,f1)` 离线截取为 `wrist_frames/<clip_relative_idx>.jpg`。已完成35696/35696 clip、5,028,323帧，零错误。
- **初始化/训练**: 从 corrected C30 ep11 checkpoint做weights-only warm restart；旧权重全部精确加载，仅新增 `type_wrist_current` 随机初始化。DiT16/mixer8不变，训练12 epoch，主LR `3e-5`。
- **配置与入口**: `configs/droidexFULL_C31_wrist_current_cont12_lr3e5.yaml`；训练 `bash scripts/run_train_c31_wrist.sh`；正式32-step/16-sample诊断 `bash scripts/run_eval_c31_wrist.sh <checkpoint>`。
- **实施验证**: 07970的首/中/末缓存帧与原视频 `f0+idx` 对齐（JPEG MAE 1.60–1.81/255）；dataset单样本 `[3,224,224]`、batch `[B,3,224,224]`；真实C30权重加载仅缺新增类型向量；完整loss前反向后所有431M可训练参数均有梯度。
- **训练完成**: 12 epoch全部完成；`latest.pt`=ep12，训练内置XYZ MAE约1.660cm。`best.pt`=ep6，但它按包含gripper/terminate的总val loss选择，不代表位置指标最优。
- **正式诊断**: 为消除flow随机采样噪声，给`diagnose_actions.py`增加固定seed，并用相同seed=0在冻结val前800 windows、32 ODE steps/16 samples下重跑corrected C30 ep11和C31 ep12 latest：

  | 指标 | corrected C30 ep11 | C31 wrist ep12 | C31-C30 |
  |---|---:|---:|---:|
  | XYZ MAE | **1.583 cm** | **1.582 cm** | -0.001 cm（实际持平） |
  | dx MAE | 1.502 cm | **1.485 cm** | -0.017 cm |
  | dy MAE | 1.555 cm | **1.547 cm** | -0.008 cm |
  | dz MAE | **1.691 cm** | 1.716 cm | +0.025 cm |
  | dx/dy/dz corr | .918/.937/.938 | .923/.936/.938 | dx略升，其余持平 |

- **结论**: 单独加入当前腕部RGB没有降低总体XYZ误差；它对x/y有毫米级改善，但被z方向约0.25mm退化抵消。当前离线全窗口平均指标不支持“腕部RGB整体更好”，但也不能排除它只在夹取附近、遮挡或近距离子集有效，下一步应按gripper闭合前后的窗口做分段评估。

## Exp C32 — corrected C30 + 当前腕部RGB + 当前正面FoundationStereo深度  🛠️ 实现/预处理中

- **目的**: 在C31腕部视角基础上增加当前正面metric depth，重点验证深度是否改善夹取所需的z轴和近距离定位；仍只使用部署时可获得的当前观测，不输入未来深度。
- **公平起点**: 与C31一样直接从corrected C30 ep11做weights-only warm restart，额外训练12 epoch、主LR `3e-5`。不从C31 ep12继续，以免把额外12 epoch训练预算混入深度收益。
- **数据冻结**: 继续使用 `max_episodes=35696, val_count=500, split_seed=42`。冻结候选中33333/35696有FoundationStereo深度，32664成功生成geometry；另669个有深度但缺少可用内参，不能可信反投影。所有缺失clip都不删除，返回全零几何和`valid_ratio=0`，保证train/val集合不变。
- **三路readout**: source demo、当前正面、当前腕部仍共享DINOv2-B及后续block，但各自使用独立的32个readout query。加载旧C30时，将旧共享query精确复制到三路，随后允许它们独立更新。
- **3D lifting**: 不沿用C11的“XYZ单独MLP→32 depth tokens”。当前正面深度经过与RGB完全一致的resize+center crop和valid-aware插值；每个14×14 DINO patch生成相机系`[X,Y,Z,valid_ratio]`，与对应RGB patch逐点融合，再由当前正面的32 query readout。XYZ不使用`LayerNorm(3)`。
- **预处理/I/O**: 原始`depth.npz`整段压缩，不在每个训练window中反复解压。`scripts/preprocess_c32_patch_geometry.py`一次性生成每clip的`patch_geometry_v1.npy`（float16，可mmap随机读取当前帧）。原始FoundationStereo结果保持只读。
- **帧/几何验证**: clip 00000 local frame 0严格对应双目source frame 12；RGB、深度的桌面/机械臂/栏杆边缘对齐，EE投影落在夹爪区域。图保存在`artifacts/c32_depth_alignment/00000_frame000.png`。
- **配置与入口**: 训练config `configs/droidexFULL_C32_wrist_frontdepth_cont12_lr3e5.yaml`；预处理 `bash scripts/run_preprocess_c32_patch_geometry.sh`；训练 `bash scripts/run_train_c32_wrist_frontdepth.sh`；正式评估 `bash scripts/run_eval_c32_wrist_frontdepth.sh <checkpoint>`。
- **训练完成**: 12 epoch全部完成。`latest.pt`=ep12；按总val loss选择的`best.pt`=ep4，主要受gripper loss先降后升影响，不代表位置最佳。
- **正式诊断**（冻结val前800 windows，32 ODE steps，16 samples，seed=0）:

  | checkpoint | XYZ MAE | dx / dy / dz MAE | dx / dy / dz corr |
  |---|---:|---:|---:|
  | C32 ep12 latest | **1.562 cm** | 1.494 / 1.509 / 1.682 cm | .918 / .939 / .939 |
  | C32 ep4 best | 1.634 cm | 1.547 / 1.568 / 1.787 cm | .917 / .940 / .937 |
  | corrected C30 ep11 | 1.583 cm | 1.502 / 1.555 / 1.691 cm | .918 / .937 / .938 |
  | C31 wrist ep12 | 1.582 cm | 1.485 / 1.547 / 1.716 cm | .923 / .936 / .938 |

- **结论**: C32 latest相对corrected C30改善0.021cm（约1.3%），相对C31改善0.020cm，属于很小但同seed正式口径下可测的增益。改善主要来自dy（C30 1.555→C32 1.509cm），dz仅小幅优于C30（1.691→1.682cm）；尚不能说明深度解决了夹取时的厘米级z误差。ep4 best明显差于ep12 latest，再次说明位置模型应按正式action eval而不是总val loss选checkpoint。

## Exp C33 — C32 + 0.66s/0.33s 当前正面与腕部RGB历史  🛠️ 训练中

- **动机**: RoboLab闭环诊断显示C30/C31/C32经常把高置信闭合放在chunk的`k=1`或更后，但部署只执行`k=0`后立即重规划，使离散闭合事件持续被推迟。引入短时观测历史，让策略可从真实运动状态分辨“仍在接近”与“上一个控制周期已到位”。
- **输入**: 15 FPS下固定帧偏移`[-10,-5,0]`，即过去约0.67s、0.33s和当前。正面RGB和腕部RGB均使用三帧；每个流有独立的三位置learnable时间embedding。source demo、source_dt、EE投影proprio、8步动作chunk和数据划分均不变。
- **深度范围**: FoundationStereo geometry仍只输入当前`0s`前面帧；历史两帧的geometry显式标为无效。故本实验只隔离测试视觉历史，不额外引入历史深度。
- **目标对齐**: 不改变`target_history_len=1`或`target_step`，所以动作标签仍对应当前时刻；episode开头不足历史的索引安全地复制frame 0。
- **初始化/训练**: 从C32 ep12 `latest.pt` weights-only warm start，新增`current_history_time_embed`与`wrist_history_time_embed`随机初始化；计划训练10 epoch，主LR `3e-5`。本机已完成ep1（val action loss `0.007896`，val gripper loss `0.286241`）并于2026-07-20暂停；后续通过C33 `latest.pt`完整恢复模型、optimizer和epoch计数，从ep2继续。集群续训每卡batch由3提高到12。
- **配置与入口**: `configs/droidexFULL_C33_front_wrist_history_cont10_lr3e5.yaml`；训练`bash scripts/run_train_c33_front_wrist_history.sh`；正式评估`bash scripts/run_eval_c33_front_wrist_history.sh <checkpoint>`。
- **部署契约**: 部署必须按同样的`[-10,-5,0]`偏移维护两路相机环形缓冲区，传入front/wrist `[1,3,3,224,224]`，而非继续只传单帧；当前深度仅对应第三帧。
- **训练完成**: 10 epoch全部完成。按总val loss选择的`best.pt`=ep2（主要受gripper loss影响）；`latest.pt`=ep10，其val action loss/MAE继续改善，不能用`best.pt`替代正式action eval选点。
- **正式action评估**（冻结val前800 windows，32 ODE steps，16 samples，seed=0）:

  | checkpoint | XYZ MAE | dx / dy / dz MAE | dx / dy / dz corr | k0 XYZ MAE |
  |---|---:|---:|---:|---:|
  | C33 ep10 latest | **1.442 cm** | 1.339 / 1.412 / 1.575 cm | .935 / .947 / .945 | **0.64 cm** |
  | C33 ep2 best | 1.533 cm | 1.455 / 1.468 / 1.675 cm | .935 / .946 / .941 | 0.72 cm |
  | C32 ep12 latest | 1.562 cm | 1.494 / 1.509 / 1.682 cm | .918 / .939 / .939 | — |

- **离线结论**: C33 ep10相对C32 ep12的XYZ MAE降低`0.120 cm`（约`7.7%`），三轴均改善，其中dx改善最大；相对C33 ep2降低`0.091 cm`。部署应优先测试ep10，但本实验的核心假设仍须在RoboLab闭环检查gripper闭合事件能否从`k=1`推进到`k=0`，普通action评估不能回答这一点。

## Exp C34 — C33 + current gripper state + full training pool  🛠️ 待训练

- **动机**: C33实际闭环能学会夹住但不稳定松开。C34只增加数据中当前时刻的真实夹爪二值状态，避免策略必须从RGB猜测“当前已经闭合/仍然打开”；不增加事件头、不改变gripper loss或部署策略。
- **输入变化**: `proprioception`由C33的`[u_norm, v_norm, z]`扩展为`[u_norm, v_norm, z, current_gripper_state]`，状态定义与标签相同：`gripper_position > 0.5`为1，否则为0。部署必须同步传入当前RoboLab夹爪状态。
- **source采样**: C34将训练集demo的`source_float`改为非对称裁剪：随机从前端移除`0%..40%`，后端移除固定`0%`；因此不删除任何window/episode，只保证source始终包含视频后段。验证集仍不做该随机裁剪。
- **数据变化**: 数据目录当前共有43,415条clip。C34使用全部clip；验证集仍只从旧的排序前35,696条按`split_seed=42, val_count=500`生成，因此与C33严格相同（487有效episodes、5210windows）；新增7,719条全部为train-only。
- **初始化/训练**: 从C33 ep10 `latest.pt` weights-only warm start；`ee_mlp`前三维权重前缀复制，新增夹爪维度随机初始化；其余模型、历史RGB、当前depth、loss和训练超参保持C33不变，训练10 epoch、主LR `3e-5`、每卡batch 12。
- **预处理**: 新增clip的原始腕部视频全部存在，但需先执行`START_ID=35696 END_ID=43415 WORKERS=16 bash scripts/run_preprocess_wrist_frames.sh`生成缓存。新增clip已有6847/7719条FoundationStereo geometry；缺失条目按invalid geometry处理，仍纳入训练。
- **配置与入口**: `configs/droidexFULL_C34_current_gripper_fulltrain.yaml`；训练`bash scripts/run_train_c34_current_gripper.sh`；正式评估`bash scripts/run_eval_c34_current_gripper.sh <checkpoint>`。
