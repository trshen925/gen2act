# C28 Dataset Camera / Extrinsics Issue

## 1. 结论摘要

在样例 `/root/autodl-tmp/data/07970` 上确认：`rgb.mp4` 对应的真实视频相机与
`DroidExOutDataset` 装配 calibration 时选择的相机不一致。

- `meta.json` 声明视频来源为 `exterior_image_2_left`。
- `extrinsics.json` 中该视频相机对应 `exterior_2`，serial 为 `26638268`。
- 当前 adapter 无条件使用 `exterior_1`，serial 为 `22246076`。
- 使用 `exterior_1` 投影时 EE 明显偏离夹爪；使用 `exterior_2` 时 EE 稳定落在夹爪上。

这不是帧索引偏移造成的。该样例 parquet 的 `t` 为 `7..155`，共 149 行，与裁剪后的
149 张图片逐行对应。

目前只能确认样例 `07970` 存在此问题。C28 全量数据中有多少 clip 使用
`exterior_image_2_left` 或其他相机，仍需扫描 `meta.json` 后统计。

## 2. 已确认的样例证据

样例目录：

```text
/root/autodl-tmp/data/07970
```

`meta.json`：

```json
{
  "clip_id": "07970",
  "episode_id": "episode_022352",
  "camera": "exterior_image_2_left",
  "source_frame_range": [7, 156],
  "num_frames": 149,
  "fps": 15
}
```

`extrinsics.json`：

| key | serial | 含义 |
|---|---:|---|
| `exterior_1` | `22246076` | 第一外部相机 |
| `exterior_2` | `26638268` | 第二外部相机，与本 clip 的视频来源匹配 |

内参来自：

```text
/root/autodl-tmp/data/intrinsics.json
```

本例两路相机的逐帧投影对照图：

```text
/root/autodl-tmp/data/07970/ee_visualization/camera_projection_comparison.png
```

对照结果：

- `exterior_1`：投影点整体位于真实夹爪右侧，轨迹与夹爪运动不一致。
- `exterior_2`：投影点持续落在夹爪/EE 附近，随夹爪运动。
- 两种投影的 149 个点都可能位于图像边界内，因此当前宽松的 projection quality
  filter 不能识别“使用了错误但仍能投进画面的相机”。

## 3. 当前代码中的根因

文件：`r2r_gen2act/data/adapters/droid_ex_out.py`

当前 `_read_action_payload()` 只读取 `extrinsics.json`，随后固定选择：

```python
cams = ext.get("cameras", {})
if "exterior_1" not in cams:
    raise ValueError(...)
cam = cams["exterior_1"]
```

这里没有读取或使用 clip 的：

```python
meta["camera"]
```

因此，无论 `rgb.mp4` 实际来自 `exterior_image_1_left` 还是
`exterior_image_2_left`，payload 的 `calibration` 都只会包含 `exterior_1` 的外参与内参。
下游 `_camera_serial()` 看到 calibration 中只有一个候选，也会正常接受它，不会报错。

## 4. C28 中受影响的训练信号

C28 配置 `configs/droidexFULL_C28_bigdit_scratch.yaml` 同时启用了三条依赖相机外参的路径。

### 4.1 Proprioception 的 EE 图像投影

配置：

```yaml
data:
  proprioception:
    enabled: true
    source: camera_projection
    dims: 3
```

`OpenXDroidDataset._project_ee_to_normalized_image()` 使用 payload 中唯一的 calibration，生成：

```text
[normalized_u, normalized_v, camera_depth]
```

若视频来自第二相机而 payload 使用第一相机，该 proprioception 与当前图像中的真实 EE
位置不一致。

### 4.2 Camera-frame action target

配置：

```yaml
action:
  mapping:
    type: droid_observation_cartesian_future_delta_pose6d_camera
    extrinsics_convention: camera_pose_in_base
```

`droid_action()` 将 base-frame 的平移和相对旋转变换到 payload 指定的相机系。如果选错
相机，监督目标会处于 `exterior_1` 坐标系，而模型看到的视频可能来自 `exterior_2`。

### 4.3 Aux trajectory target

配置：

```yaml
data:
  aux_traj:
    enabled: true
model:
  aux_traj:
    enabled: true
```

`OpenXDroidDataset._camera_abs_pose_at()` 同样使用 payload 的相机外参，将绝对 EE pose
转换到相机系。因此 aux trajectory 也可能与视频相机不一致。

## 5. 对 C28 checkpoint 的潜在影响

需要区分两种情况：

1. 如果 C28 数据绝大多数视频本来就是 `exterior_image_1_left`，问题只影响一部分
   `exterior_image_2_left` clip，表现为带系统性错误的训练样本。
2. 如果 C28 数据大量或全部来自 `exterior_image_2_left`，则 C28 可能是在“第二相机视频 +
   第一相机坐标系 action/proprio/aux target”的固定错配下训练。

即使错配在整个训练集中保持一致，也不能直接认为 checkpoint 可在修正后的 dataset 上
无缝使用。修复相机选择会同时改变：

- proprioception 输入；
- action translation/rotation target 的坐标系；
- aux trajectory target；
- 推理输出的物理解释。

因此，不应直接修改 adapter 后用原 C28 checkpoint 得出可比结论。至少需要同时保留
legacy 数据路径，分别评测并判断是否需要重训或坐标转换。

## 6. 建议的全量审计

在修改训练代码前，先对冻结的前 35696 个候选目录统计：

```text
meta.camera 的取值及数量
meta.camera -> extrinsics camera key 的映射成功率
每类相机的 clip 数和 window 数
缺少对应外参/内参的 clip 数
```

建议重点输出：

| meta.camera | 应选 extrinsics key | clip 数 | calibration 完整数 | 投影抽检结果 |
|---|---|---:|---:|---|
| `exterior_image_1_left` | `exterior_1` | 待统计 | 待统计 | 待抽检 |
| `exterior_image_2_left` | `exterior_2` | 待统计 | 待统计 | `07970` 已确认正确 |

另外应从每种相机随机抽样多个 clip，生成 EE overlay。只检查“投影是否在图内”不够，必须
检查投影是否真正跟随夹爪。

## 7. 建议的代码修复方向

### 7.1 根据 meta.camera 选择外参

建议在 `DroidExOutDataset._read_action_payload()` 中读取 `episode.metadata_path`，并建立显式
映射，例如：

```python
camera_name = str(meta["camera"])
camera_key_by_video = {
    "exterior_image_1_left": "exterior_1",
    "exterior_image_2_left": "exterior_2",
}
camera_key = camera_key_by_video[camera_name]
cam = cams[camera_key]
```

映射应根据全量 `meta.camera` 统计结果补全，未知值必须报错或跳过，不能静默回退到
`exterior_1`。

### 7.2 在 payload 中保留来源信息

建议增加：

```python
payload["_camera_name"] = camera_name
payload["_camera_key"] = camera_key
payload["_serial"] = serial
```

便于日志、可视化和问题追踪。

### 7.3 保留 legacy 兼容模式

为了复现 C28，建议提供明确配置开关，而不是直接改变旧实验语义：

```yaml
data:
  camera_selection: legacy_exterior_1  # 精确复现旧 C28
```

新实验使用：

```yaml
data:
  camera_selection: from_meta
```

这样可以避免修复后旧 checkpoint 的评测输入含义发生静默变化。

## 8. 修复后的最低验证要求

1. 扫描冻结数据边界 `max_episodes: 35696`，确认相机类型分布。
2. 对每类相机至少随机可视化 20 个 clip。
3. 验证 EE 投影在连续帧中跟随真实夹爪，而不只是位于画面内。
4. 验证 camera-frame action 经过逆变换后能恢复 base-frame delta。
5. 验证 aux trajectory 的绝对相机系位置与投影使用同一 serial。
6. legacy 与 corrected dataset 分开命名、分开输出目录、分开记录指标。
7. 不用修正后的 dataset 直接覆盖 C28 ep17 的 `1.703 cm` 结果；两者不是同一数据语义。

## 9. 当前状态

- `07970` 的错误已通过两路相机投影对照确认。
- 修正后的可视化已使用 `exterior_2` 重新生成。
- 尚未修改 `DroidExOutDataset`。
- 尚未统计 C28 冻结数据中各视频相机的分布。
- 尚未评估该问题对 C28 ep17 checkpoint 指标的实际影响。
