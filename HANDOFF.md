# HANDOFF — gen2act 当前状态与执行细节 (2026-07-15)

给后续 AI 的交接文档。记录**当前 SOTA、正在进行的实验、关键约定、避坑点**。
详细逐实验历史见 `EXPERIMENTS.md`；跨会话记忆见 memory `gen2act-diffusion-pointtrack-task`。

---

## 0. 一句话现状
- **任务**: gen2act = demo视频 + 当前观测 → 预测机器人相机系 8步 delta action chunk (ΔXYZ+6D旋转+gripper).
- **当前最优模型**: 仍是 **C28 ep17 = 1.703cm**（C28冻结val，800 windows，32/16）。C29/C30 使用同一冻结数据与val，可直接比较。
- **新实验**: C29(DINOv3-L) ep12 latest=1.874cm；C30(DiT16/mixer8)低LR续训至总ep17 latest=1.765cm。两者均未超过C28；C30原始ep7 best正式评估待补。

---

## 1. 环境 / 运行方式
- conda env: `/root/miniconda3/envs/gen2act/bin/python` (见 memory `gen2act-run-environment`).
- 工作目录: `/mnt/pfs/users/shentingrui/code/robo/video_gen/gen2act/gen2act/`
- 训练: `nohup <py> scripts/train.py --config <cfg> > /tmp/<name>.log 2>&1 &`
- 诊断: `<py> scripts/diagnose_actions.py --config <cfg> --checkpoint <ckpt> --split val --max-windows 800`
- ⚠️ auto-mode 安全分类器偶发不可用 → Bash 被拦, 稍等重试即可 (不是命令问题).

---

## 2. 数据集
### droid-ex-3000-out (主力, 相机系正确, 已验证)
- 路径: `/mnt/pfs/data/shentingrui/droid-ex-3000-out/NNNNN/` (5位数字), 目录已超过 **35696 clips** 且后台持续增长；C28/C29/C30 必须用下述上界冻结。
- 每 clip: `data.parquet`(DROID字段,数组是JSON字符串) + `meta.json` + `rgb.mp4` + `frames/`(预抽jpg) + extrinsics 在 `provenance.episode_dir/extrinsics.json`.
- adapter: `dataset_type: droid_ex_out` (`r2r_gen2act/data/adapters/droid_ex_out.py`).
- 内参: `/mnt/pfs/data/shentingrui/KarlP-droid/intrinsics.json` (episode_id→serial).
- 相机系约定 (已投影验证100%落夹爪上): EULER `xyz`, `camera_pose_in_base`, `p_cam=R^T(p_base-t)`, cameraMatrix=[fx,cx,fy,cy].
- **坏数据**: 2 clip (30711,30712) extrinsics 缺 exterior_1 (用裸serial键) + 1 clip 空frames → adapter 已 try/except 跳过. 全量可安全训练.
- **子集控制**: `data.max_episodes: N` → sorted后取前N个 (4579 = C22时代旧数据 00000-04578, mtime 07-07).
- **C28/C29/C30冻结集**: `max_episodes: 35696` → 候选目录00000-35695；过滤后train=34008/355461 windows，实际val=488/5218 windows。后续目录35696+不得进入这组消融。

### RoboLab banana (仿真, 10 clip, adapter已写但坐标系可视化仍有问题, 暂搁置)
- 路径: `/mnt/pfs/users/shentingrui/code/robo/video_gen/gen2act/banana/banana_in_bowl_00N/`
- adapter: `dataset_type: robolab_sim` (`r2r_gen2act/data/adapters/robolab_sim.py`), 已注册.
- **坐标系差异**: EE是**世界系**(x~15,y~-10, 各env网格偏移, dim0/1 std小=常数偏移, dim2=真实高度). 需: ①减env_origin(=EE的x,y均值,z=0)→base系; ②外参轴修正 `Rcb=Rot('xyz',euler)@diag(-1,-1,1)`, `p_cam=Rcb(p-t)`; ③内参在clip自己的intrinsics.json(fx=fy=524,cx=640,cy=360,1280×720).
- adapter把banana约定转成droid等价格式(存 euler'=eulerOf(Rcb.T)), 下游droid机制复用. smoke测: adapter相机系pos与直接约定 diff=0 ✅.
- ⚠️ **未解决**: 可视化Z符号/落点用户说"还有问题", 暂停. **不要接入训练**, 等同事生成更多+修坐标系. 10 clip太少无独立价值.
- 可视化: `gen2act/viz_ee_check/banana_all/` (10 overlay + 视频).

---

## 3. 模型 = fused_query_flow (flow-matching DiT)
- `r2r_gen2act/modeling/fused_query_flow_policy.py`. 条件: source视频8帧(query-in-DINOv2 readout) + current帧(32 readout query) + EE tokens, 拼成序列喂 flow head的 vl_mixer, 再 DiT 去噪采样.
- **C22/C28 架构 (SOTA)**: DiT `num_layers=12`, `vl_mixer_layers=6`, hidden 1024, 347M参数. 固定8帧 + 显式Δt.
- 关键组件开关 (config `model.` 下): `current_full_patch`(C21试过无用), `dt_time_embed.enabled`(Δt,✅有用), `pad_source`(动态帧填充,✅代码在但方案已放弃), `max_source_len`(source_time_embed尺寸).
- Δt编码: 每source帧真实秒数Δt=帧差/fps, sinusoid→MLP, 叠加在source_time_embed上. data侧 `data.dt_time_embed.enabled` + `data.fps`.

---

## 4. ⚠️⚠️ 关键避坑点 (最重要, 血泪教训)

### (A) flow的 val infer_loss 不可靠 → 必须用 diagnose 的 XYZ MAE 选 checkpoint
- flow的infer_loss=采样后MSE, 早期分布塌缩、采样≈均值→MSE假低但action差.
- **latest.pt (val最高) 普遍优于 best.pt (val最低)**. 例: C16 best 5.46 vs latest 3.94; C23 best 2.09 vs latest 1.83; C26v2 best 2.06 vs latest 1.93.
- **规矩: best和latest都跑diagnose**, 取XYZ MAE最低的.

### (B) 诊断采样精度: 训练16步/8样本, 最终eval用 32步/16样本
- eval改 `flow_dit.num_inference_steps: 16→32` + `num_eval_samples: 8→16` (零训练, 降MAE ~10%). 每个实验都有 `*_eval32.yaml`.
- 采样精度与模型容量是独立杠杆.

### (C) ★★ 不同val集(不同baseline)的绝对MAE不可直接比!
- diagnose打印的 `predict-mean baseline` 因val集而异. C22的val baseline=5.216, C28自己的val=4.594.
- **1.703(C28自己val) 不能跟 1.672(C22 val) 直接比** — C28的val更简单(baseline低).
- 要么看**胜基线%** (相对指标, 跨val可比), 要么在**同一val集**上测两个模型.

### (D) ★★★ 跨训练规模比较必须构造"两模型都没训练过的公共val" — 否则泄漏
- C28全量训练只留500 val, 其余35196全train. 用"max_episodes=4579+val300"划分测C28 → 那300 val里**293个(97.7%)是C28的训练数据** → 得1.389cm虚高, 作废.
- **正确做法**: 取 C28的val ∩ C22没训练过的 = 437 clip公共集. 脚本 `/tmp/clean_compare.py` (限定ds._samples到指定episode列表, 两模型同脚本同采样).
- val_count小时换划分极易让旧val落入新train.

---

## 5. 核心结论 (已验证, 不要重复踩)
1. **数据是最大杠杆**: 2000→4579(+57%数据) 给 −28% (C19-new); 4579→34008(8×) 干净对比(437集)给 −8.5%. 边际递减但仍有效.
2. **容量与数据互补**: C16(小数据翻倍参数)没用 ≠ 容量无关; C22(数据够了 DiT8→12)给 −20%. 但 C23(DiT12→16)在4579上退步=容量到顶, 需更多数据才可能再扩.
3. **帧数策略到头**: 固定8帧+Δt(C22)是source编码全局最优. 动态帧(C24-26)/固定12/16帧(C27)/从C22微调动态帧(C25v2/C26v2) 全部更差. 8帧linspace已够表达demo高层意图.
4. **current全patch(C21)/progress aux(C18) 无用**. aux_traj(w=0.1) 有用(救旋转+防塌缩). source_float 中性.
5. **换头(flow vs 回归)不是杠杆**: flow≤回归当欠定; flow+aux才叠加增益(C17). 但大数据下flow工作良好.
6. **全量数据扩容消融**: C29扩大视觉backbone(DINOv3-L)得1.874cm；C30扩大DiT/mixer得1.765cm。动作头扩容更有效，但当前都未胜C28的1.703cm。

---

## 6. 关键实验结果表 (XYZ MAE, cm, 32步/16样本)
| 实验 | 架构 | 数据 | val集(baseline) | XYZ MAE | 备注 |
|---|---|---|---|---|---|
| **C22** | flow DiT12/mixer6, 固定8+Δt | 4579 | C22-val(5.216) | **1.672** | 4579时代SOTA |
| C28 ep11 | =C22架构 | 34008(8×) | C28-val(4.594) | 1.757 | 全量, 别跟C22直接比(val更简单) |
| **C28 ep17** | =C22架构, 续训 | 34008 | C28-val(4.594) | **1.703** | 续训小提升 −3% |
| C28 ep17 | 同上 | 34008 | 437公共集 | 待测 | ← 唯一能干净比C22的 |
| C22 | — | 4579 | 437公共集 | 3.675* | *该集baseline算法异, 仅相对可比 |
| C28 ep11 | — | 34008 | 437公共集 | 3.363* | 比C22好 −8.5% |
| C29 ep12 latest | DINOv3-L + DiT12/mixer6 | 34008冻结集 | C28-val(4.594) | 1.874 | 800 windows, 32/16 |
| C30 总ep17 latest | DINOv2-B + DiT16/mixer8 | 34008冻结集 | C28-val(4.594) | 1.765 | 低LR续训ep5, 800 windows, 32/16 |

---

## 7. 当前待办 (接手直接做)
1. **测 C30 原始 best.pt (ep7)**：用 `droidexFULL_C30_dit16_mixer8_eval32.yaml` 在冻结val前800 windows跑32/16，判断早期模型是否优于当前1.765。64-window快速值1.444不可作正式结论。
2. **继续 C30 低LR续训**：当前额外5/12 epoch；入口 `scripts/run_train_c30_dit16_mixer8.sh`，输出 `outputs/droidexFULL_C30_cont12_lr3e5/`。
3. **测 C28 best.pt (ep9, 全程val最低)** 在C28自己val(500)上, 对比 ep17 latest(1.703). config `droidexFULL_C28_eval32.yaml`.
4. **437公共集上测 C28 ep17** (续训后), 干净对比C22. 用 `/tmp/clean_compare.py` 改checkpoint为 ep17.
   - 该脚本: 构造 C28val(500) ∩ C22没训过 = 437 clip, 两模型同脚本测. ⚠️ 它的baseline算法与标准diagnose略异, 只做相对比较.
5. (可选) 决定下一步: 继续扩数据(后台增长中) / 修banana坐标系接入.

---

## 8. 关键文件
- 训练脚本: `scripts/train.py`; 诊断: `scripts/diagnose_actions.py`
- SOTA config: `configs/droidexFULL_C28_bigdit_scratch.yaml` (全量) + `_continue.yaml`(续训) + `_eval32.yaml`(诊断)
- checkpoint: `outputs/droidexFULL_C28_bigdit_scratch/{best.pt(ep9), latest.pt(ep17)}` (各~4.2G)
- C22 baseline: `outputs/droidex3000out_C22_bigdit_scratch/latest.pt` + `configs/droidex3000out_C22_eval_steps32.yaml`
- resume机制: `train.resume_full_checkpoint`(恢复model+optimizer+快进scheduler, 见trainer.py:341) — ⚠️ 改epochs会重建cosine, LR被"拉高"(18-ep曲线@ep11 = 4.7e-5 vs 原12-ep@ep11 = 8e-6). `resume_checkpoint`则是strict=False部分加载+全新optimizer.
- shape-tolerant加载: `load_checkpoint(strict=False)` 对形状不匹配参数前缀拷贝(source_time_embed[8]→[16]), 支持扩帧微调.
- EE坐标系可视化脚本: `/tmp/viz_droid_reproject.py`(droid) `/tmp/viz_banana_all.py`(banana). 结果在 `gen2act/viz_ee_check/`.
- 干净对比脚本: `/tmp/clean_compare.py`.
- C29: `configs/droidexFULL_C29_dinov3l_c28base.yaml` + `_eval32.yaml`; checkpoint `outputs/droidexFULL_C29_dinov3l_c28base/latest.pt`.
- C30: `configs/droidexFULL_C30_dit16_mixer8_c28base.yaml`; 低LR续训 `configs/droidexFULL_C30_cont12_lr3e5.yaml` + `_eval32.yaml`.
