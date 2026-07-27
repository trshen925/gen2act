# C37 后续实验决策

本文档记录实施前已达成一致的设计决策。它不修改任何实验配置，也不会启动新的训练。

## 背景

C37 以原始 DROID 的 15 Hz 频率预测稠密的 15-step action chunk：
`t+1, ..., t+15`。它保持 `action_stride: 10`，当前训练时的 window jitter
为 `+/-3` 帧。

名义 anchor 间隔为 10 帧时，`+/-3` 每个 anchor 仅能覆盖 7 个候选当前时刻。
例如，anchor 10 和 20 分别覆盖 `7..13` 与 `17..23`，所以 `14..16` 永远
不能作为当前时刻被采样。

## 已确认的修改

下一个从 C37 派生的实验，仅修改训练 window jitter：

```yaml
data:
  window_jitter: {enabled: true, max_offset: 5}
```

这样相邻 anchor 的候选区间刚好相接：

```text
anchor 10:  5..15
anchor 20: 15..25
```

因此，跨 epoch 来看每个内部 15 Hz 当前时刻都可被覆盖；anchor 边界处的重叠
可以接受。

## 明确不修改的部分

- 保持 `action_stride: 10`，不改为 1。
- 保持 C37 的 15-step、相邻间隔一帧的 action target，以及 1 秒 horizon。
- 保持当前 gripper 输入和单次 flow 推理。
- 不修改 C37 本身，以保持已完成训练和评估的可复现性。实施时创建新的派生实验。

## 已确认的光学增强

下一个从 C37 派生的实验，将当前全局光学增强替换为以下仅 ColorJitter 的设置，
并应用到每个相机流（source、front history、wrist history）：

```yaml
data:
  augmentation:
    enabled: true
    p: 1.0
    brightness: 0.30
    contrast: 0.40
    saturation: 0.50
    noise_std: 0.0
```

每个训练 clip 采样一组 brightness、contrast、saturation 参数，并一致应用到该
clip 的所有帧，以避免人造的时间闪烁。其含义分别为：brightness 加性偏移位于
`[-0.30, +0.30]`，contrast 系数位于 `[0.60, 1.40]`，saturation 系数位于
`[0.50, 1.50]`。此版本不添加高斯像素噪声。

已确认的单参数可视化扫描位于
`artifacts/c37_paper_colorjitter_parameter_sweep_v2/index.html`。其中每次只启用
一种参数，其他 ColorJitter 分量关闭：brightness 为
`-30%, -15%, 0, +15%, +30%`；contrast 为 `-40%, -20%, 0, +20%, +40%`；
saturation 为 `-50%, -25%, 0, +25%, +50%`。

## C38 已确认的 Wrist 预处理

C37 对 front 和 wrist RGB 都使用共享的 `resize_center_crop` 预处理。常见的
320x180 wrist 帧仅保留中间 180x180 原始像素区域（`x=70..250`），两侧各丢弃约
22% 的宽度。这不利于 wrist 相机，因为更大的视野对操作上下文有价值。

C38 将 wrist 专用预处理改为保持长宽比的 letterbox：将完整 wrist 图等比缩放到
224x224 内，再对剩余空间 padding，不 crop，也不做几何形变。因此，320x180 的
wrist 输入会变为 224x126，并在上下各 padding 49 像素。padding 值使用全黑
（`0`）。除非另行决定，C38 保持 front 相机现有的预处理不变。

这需要专用的 wrist 图像变换；不要修改共享的 `image_to_tensor` 函数，因为
front/source 输入应保留当前 C37 行为。训练前须可视化原始 wrist 帧及其完整的
224x224 letterbox tensor，以验证实现。

## C38 已确认的当前观测输入

C38 不再输入当前观测历史。front 和 wrist 均只保留当前时刻，即：

```yaml
data:
  current_history_offsets: [0]
```

因此，C38 的 front 和 wrist 输入各为一张当前 RGB 图，而不再是 C37 的
`[-10, -5, 0]` 三帧历史。source demo video 的 8 帧仍然保留；它是任务示范条件，
不属于当前机器人观测历史。实施时应将模型的 current-history 长度设为 1，并调整
或重新初始化与历史帧位置编码有关的参数，不能直接保留 C37 三帧时间位置编码的语义。

## C38 已确认的 Front 平移

在原生 320x180 front 相机分辨率上做平移，水平和竖直偏移均最大为 +/-10%
（分别最多 32 和 18 像素）。每个偏移在完整区间内独立均匀采样，因此包含接近
零的取值。对新露出的边界使用反射 padding，以保持原生帧尺寸；随后运行未改动的
C37 `resize_center_crop` 预处理。反射只用于边界填充，不是对整张观测进行水平或
竖直翻转。

每个训练样本内，source video 和单张当前 front 图共享完全相同的一次 front 平移采样。
将相同的原始图像平移应用到当前 EE 的 2D 投影，以及由 15 个
相机坐标系 action target 投影得到的未来 EE 端点。已确认的可视化检查为
`artifacts/c37_front_translation_native_geometry/00122_start00030_translation_sweep.png`。

保持相机坐标系下的 3D 平移及相对 3D 旋转 action label 不变。图像平移表示虚拟
主点的偏移，而不是物理相机朝向变化。不要添加整图水平/竖直翻转：翻转是反射，
并非物理上的 3D 相机旋转，会要求对 action rotation label 进行复杂且不合适的变换。

## C38 已确认的跨相机颜色一致性

每个训练样本采样一组 ColorJitter 参数（brightness、contrast、saturation），并应用
到同时观测到的当前 front 与当前 wrist 图。这用于模拟共享的场景光照变化，避免引入
人为的 front/wrist 曝光不一致。source demo video 是不同时间的示范图像序列，保持
独立采样一组 ColorJitter 参数。

## 理由

使用 `action_stride: 1` 确实会显式覆盖所有当前时刻，但会使高度重叠的训练 window
数量约增加 10 倍。这会增加成本并重复近乎相同的当前观测，从而在固定训练预算下
降低不同 episode、任务、场景和物体所占的比例。

`action_stride: 10` 配合 `+/-5` 保持每个 epoch 的 window 数量及跨场景多样性，同时
消除 `+/-3` 留下的永久三帧覆盖空洞。

## 评估计划

派生实验训练完成后，使用与 C37 相同的正式评估：

- 冻结验证集的前 800 个 window；
- 32 个 flow integration step、1 个 flow sample、seed 0；
- 使用 1 秒覆盖 horizon 的 15 Hz event-aware close/release 诊断。

对比整体 action 指标，以及 event-aware close 漏切换率/recall、时序 MAE、有符号
时序偏差和切换前后 XYZ MAE。
