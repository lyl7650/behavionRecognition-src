# 模型辅助剪切+分类（auto_presort）—— 设计文档

> 目标：用**已微调的 teacher 模型**，把一批原始长视频**预剪切 + 预标注**成带时间线的建议，人工只需在网页上**审核/修正**，而不是从零逐帧拖动。
> 解决的核心痛点是 **temporal cutting（找一个动作在哪结束、下一个在哪开始）**，这是当前人工环节最耗时的部分。
> 本工具是 `run_crop.py` 之前的一个**独立前置步骤**，产出与现有 `sorted-<日期>/<类别>/` 完全一致的目录，下游流水线基本不变。

日期：2026-07-16
物种：pet（`eat/drink/other`）、rabbit（`resting/active/eating/drinking`），共用同一套代码。

---

## 1. 背景与动机

现状（见 `docs/SOP.en.md`、`docs/AUTOMATION.md`）：整条流水线已高度自动化，唯一无法自动化、且最耗时的人工节点是 **Step2：把原始长视频剪成单动作短片并分类**。一段长视频里可能 1–15 秒吃、15–37 秒 active，人工要反复拖进度条找边界、再分别丢进类别目录。

现在模型整体准确率约 80%，希望**借模型能力把这步从"从零剪切"降为"审核修正"**，同时——这是本设计的重中之重——**防止模型的标注误差进入训练集，造成"看起来在提升、其实在自我强化错误"的正反馈**。

**用户明确的两条硬性要求：**
1. **Review-all**：模型仍会输出不少错误，所以每个视频都必须人工确认后才能进训练集，模型绝不自动写入。
2. **宁可剪多，不可剪少**：clip 宁可短一点、干净一点，也不能跨越动作边界。例如真实 1–10 秒 eating、10–15 秒 active，宁可得到 0–8 秒的纯 eating clip，也不要 0–11 秒那种混入 1 秒 active 的脏 clip——那 1 秒错误动作会让训练学不到准确的动作判断。

## 2. 目标与非目标

**目标（MVP，最小闭环）**
- 一个独立前置工具 `auto_presort`，输入一批原始长视频，输出人工审核过的 `sorted-<日期>/<类别>/` clip。
- 三个命令按依赖轻重拆分：`label`（重）/ `review`（轻）/ `apply`（轻）。
- teacher 做预标注；人工在**一个网页**里批量审核整批视频，看每个视频的**动作时间线进度条**，拖边界、改标签、删除、批准。
- 把两条硬性要求做进机制：审核闸门 + 保守剪切。
- 与现有 `run_crop.py` 尽量兼容：产出目录命名满足 `run_crop.validate_sorted` / `assert_snapshot_empty`，接在其前面即可。

**非目标（本期不做）**
- 不接入现有 `.bat` / `scaffold_dirs.py` 自动化（**后期再 integrate**）。
- 不做 gap 上拖拽新建 segment（评审判定为 MVP 过度设计，延后）。
- 不做同事机器的 ffmpeg 便携打包（延后；MVP 先在本机跑通）。
- 不做浏览器不支持编码（AVI/MOV/特殊 MP4）的转码代理（延后；MVP 先限 H.264 MP4 审核）。
- 不改训练超参、模型结构、蒸馏逻辑。

## 3. 总体架构

三命令，按依赖轻重拆分（这是刻意设计，为后期同事跨机器协作留口子）：

| 步骤 | 脚本 | 依赖 | 运行位置 |
|------|------|------|----------|
| 1. label 预标注 | `auto_presort_label.py` | GPU / torch / teacher ckpt / YOLO | 仅训练机 |
| 2. review 审核 | `auto_presort_review.py` | 纯 Python 3 stdlib + 浏览器 | 本机（后期可给同事） |
| 3. apply 落地 | `auto_presort_apply.py` | 纯 Python 3 + ffmpeg | 本机（后期可给同事） |

```
一批原始长视频
  ⬇ [label] YOLO 抠图 + teacher 滑窗分类 → 每个视频一份 JSON（含 segments + gaps + 审计信息）
  ⛔人工：[review] 打开网页，逐视频看时间线，拖边界/改标签/删除/批准
  ⬇ [apply] 只切「已批准」视频的已批准 segment → sorted-<日期>/<类别>/（不抠图）
  ⬇ [现有 run_crop.py] YOLO 抠图 + 切 3s 片 → train/val 快照（split 逻辑本期需小改，见 4.1）
  ⬇ [现有 run_train_distill.py] 合并 + 体检 + teacher 训练 + student 蒸馏
```

**关键分工**：auto_presort 只决定 **何时（时间段）** 和 **什么标签**；`run_crop.py` 仍负责 **在哪抠图**。apply 保存的是**未抠图**的原始时间段，让下游 `run_crop` 照旧抠图，避免"抠两次 / 分布不一致"。

## 4. 引擎（label 步）设计

对每个原始长视频：

1. **YOLO 抠图（复用现有检测器）**：定位动物，同时记录无动物帧。
2. **滑窗分类**：以 ~3 秒窗口（= `MAX_CLIP_DURATION`）在视频上滑动。每个窗口 → 抠图 → 采样 16 帧 → ImageNet 归一化（**必须与训练完全一致**，见 4.2）→ teacher 分类 → `(start, end, label, confidence)`。
3. **置信度门（便利过滤，非安全保证，见 4.4）**：top-1 softmax 低于阈值（默认 0.90，可调、偏高）的窗口标为 **uncertain gap**，不分配类别。
4. **合并同类相邻窗口**为一个 segment；不同类之间、uncertain 之间留 **gap**。
5. **向内裁剪（"宁可剪多不可剪少"）**：每个 segment 两端各裁掉 margin（默认 1.0 秒，可调、偏大），只留高置信核心；裁完短于 `MIN_CLIP_DURATION`（2 秒）的整段丢弃。
6. **输出**：每视频一份 JSON —— `segments:[{start,end,label,confidence,source}]` + 可见的 gap（并区分 gap 原因：uncertain / no-animal / boundary）。

**默认参数偏保守**（用户要求：阈值高、trim 大，尽量只留高置信结果，其余交人工）：`confidence_threshold=0.90`、`trim_margin=1.0s`。两者都在脚本顶部易调。

### 4.1 【关键修复①】train/val 同源泄漏 —— 必须改 run_crop 的 split 逻辑

**问题**（两轮独立评审一致判定为最严重）：现有 `abnormal_crop_video.py::rebalance_val` 按**输入文件**粒度分 train/val。过去一个人工 clip = 一个动作实例，安全。但 auto_presort 会把**一条原始长视频切成很多 clip 文件**，`run_crop` 把它们当成互不相关的"视频"，于是同一条录像里**秒级相邻、几乎一样**的片段（同动物、同背景、同光照）会被分散到 train 和 val 两侧 → 验证准确率虚高、掩盖模型错误 —— 正是本设计要防的正反馈陷阱。而现有体检（`snapshot_merge.py::health_report.cross_split_leak`）**只查同名 basename**，发现不了。

**修复**：
- apply 输出的每个 clip 文件名**内嵌来源原视频 id**（如 `srcA1b2__eat__12.3-19.8.mp4`）。
- **小改** `rebalance_val`：改为按「来源 id」分组，保证**同一条原始视频的所有 clip 落在同一个 split**。
- **扩展体检**：新增"同一来源 id 同时出现在 train/val"的检测（现有 basename 检测补不了）。

这是本期唯一需要动 `run_crop` 的地方——一个函数的受控小改 + 一处体检增强，不是重写。（用户已批准将此纳入范围。）

### 4.2 【关键修复③】crop 与预处理必须与训练完全一致

**问题**：训练时的抠图**不是紧框**——`build_smoothed_crop_boxes` 用「检测框宽高的 80 分位数 × `EXPAND_RATIO(1.25)`」得到**每段固定大小**，中心点做 EMA 平滑（`smooth_alpha=0.35`），整段无检测时回退全画面；随后 `VideoDataset` 把非方形裁剪**拉伸**到 224×224。label 步在**选片时就要用 teacher 分类**，若那时的抠图/预处理与训练稍有差异，teacher 看到的就是分布外画面，置信度和标签都不可信——这也会**悄悄击穿 0.90 的门**。

**修复**：label 步**调用与训练完全相同的抠图 + 预处理代码**（`build_smoothed_crop_boxes` + `VideoDataset` 的 eval 变换，即 `is_train=False`，不含随机增强），**不得自行重写近似版**。新增一个 **golden test**：同一 clip 分别走「run_crop→VideoDataset」和「label 内存预处理」，比对抽帧位置、RGB、尺寸、归一化直到模型 logits **完全一致**；不一致即为 bug，须先修复再信任任何标签。

### 4.3 【关键修复②】teacher checkpoint 载入必须失败即停止

**问题**：现有 `models/teacher.py` 在 checkpoint 缺失 / 加载异常 / 参数不匹配（`strict=False`）时**只打印警告继续跑**；且 checkpoint **只存裸 state_dict，不记录类别顺序**。用于自动标注时，可能拿随机分类头或错误类别顺序去标全批数据，产出漂亮却全错的时间线——最危险的静默故障。

**修复**：`auto_presort_label` 必须**硬失败**，除非：
- checkpoint 文件存在且**无 missing/unexpected key**（不放过 `strict=False` 的漏网）；
- 分类头维度 **等于该物种类别数**；
- **显式确认该 checkpoint 对应的有序 class_names**（不能仅凭当前 config 猜测）。
并在 manifest 记录 **checkpoint 哈希 + 类别顺序 + 物种 + 预处理版本**；启动时对少量已知 clip 做 sanity check，不过则不产出结果。

### 4.4 【修正】置信度门的定位：便利，不是安全保证

**问题**：softmax **未校准**，网络对分布外输入常常**越错越自信**——真正危险的正反馈错误恰恰是"高置信但错"的预测，这类会被 0.90 的门**放行而非拦下**。pet 的 `other` 是很宽的兜底类，在真实 eat/drink 边界附近尤其容易"高置信却错"。

**结论**：0.90 阈值仅作为**降低垃圾量的便利过滤**，规格中明确写它**不是正确性保证**。**唯一真正的防反馈手段是人工审核（Review-all）**。阈值最好后续在真实滑窗数据上按类别校准，而非拍脑袋固定。

### 4.5 【已知局限，本期接受】下游会二次切分

apply 保存一个较长的纯行为段后，`run_crop` 会从该文件 0 秒起按**非重叠 3 秒**重新切片，并丢弃不足 2 秒的尾巴。例如审核通过的 4.5 秒纯段，最终可能只留前 3 秒、末尾 1.5 秒消失；叠加两端各裁 1 秒，会系统性偏向长、稳、易分类的行为，短动作类别损失更多。**本期文档化、不修复**；apply 报告里预估"下游预计生成多少 2–3s clip、多少秒因尾段不足被丢、哪些类别/来源几乎没保留"，让用户心里有数。

## 5. 审核（review 步）设计

**一个本地 server 网页，覆盖整批视频**（不是一个视频一个 html，避免逐个点开）：

- **左栏**：整批视频列表，每个带状态徽标（⬜未审 / ✅已批准）+ 小结（如 "3 eat, 2 active, 4 gaps"）。点击加载到右侧。
- **右栏**：视频播放器 + **彩色时间线进度条**（对齐视频时长）：
  - 彩色块 = 高置信 segment（每类一色 + 图例）；灰块 = gap。
  - 点击块 → 播放器跳到该处（一键核对）。
  - **拖 segment 边缘** → 调整 trim（实时、吸附到视频时间）。
  - **点击块** → 下拉改标签 / 删除。
  - **Approve** → 标记该视频 ✅ 并把 JSON 存回磁盘（经 server）。
- **任何编辑后 approved 自动失效**（防止改完忘了重新确认）。

**MVP 编辑能力**：看时间线 + 点击跳转 + 拖边界 trim + 改标签 + 删除 + 批准。（gap 上拖拽新建 segment 延后。）

**为何要 local server**：纯 `file://` html 能显示能拖，但浏览器沙箱**无法可靠写回磁盘**。要真正"拖完即存"，需要一个极小的本地 server（Python stdlib `http.server` 即可，无重依赖）。长视频拖动还需 server 支持 Range 请求。

## 6. 落地（apply 步）与文件布局

**工作区布局**（都在同一批次文件夹下，不散落）：
```
<raw_batch_dir>/                 # 你指向的一批原始长视频
  video_001.mp4 ...
  .autopresort/                  # 工具工作区（每批一个）
    video_001.json               # {segments:[{start,end,label,confidence,source}], approved:bool, source_id, ...}
    manifest.json                # 批级：物种、模型哈希、类别顺序、阈值、预处理版本、日期
```

每个 segment 带 **`source` 标签**（`model` / `human`）作审计线索。

**apply 步**：
- 读所有 JSON，**跳过 `approved=false` 的视频并打印跳过列表**（= Review-all 闸门）。
- 对每个已批准 segment，用 **ffmpeg 按精确时间戳（PTS，非关键帧吸附）重编码**切出该时间段（禁止会把切点吸到关键帧、抵消 inward trim 的粗糙 `-c copy`）；切完校验实际时长与请求区间误差。
- 输出到 `sorted-<日期>/<label>/<srcid>__<label>__<start>-<end>.mp4`（文件名内嵌来源 id，供 4.1 分组）。
- **不抠图**——下游 `run_crop` 照旧抠图，现有流水线不变（除 4.1 的 split 小改）。

**可移植性（为后期同事协作预留，MVP 不重点做）**：JSON 用**视频相对路径**（绝不写 `E:\...` 绝对路径）；每视频记录相对路径 + 文件大小 + 时长 + 内容指纹，防止同路径换视频、同名碰撞、ckpt/预处理变更后旧结果被误用；apply 遇身份不匹配 / 输出重名 / schema 不支持时**停止不覆盖**。

## 7. 防反馈闭环（本设计的根本目的）

| # | 安全措施 | 强度 |
|---|----------|------|
| 1 | **人工审核每个视频后才写入**（Review-all 闸门，apply 强制） | **强（唯一真正保证）** |
| 2 | 高阈值 + 大 inward trim → auto clip 是纯核心，不教模型模糊边界 | 中（便利，非保证，见 4.4） |
| 3 | 不确定区域显式呈现为灰 gap，绝不静默猜测填充 | 中 |
| 4 | 每 segment 标 `source=model/human`，可审计漂移来源 | 辅助 |
| 5 | 人工始终是最终标签权威，训练信号锚定人判断而非模型自我强化 | 强 |
| 6 | **同源 clip 不跨 split（4.1）**，防验证虚高掩盖错误 | **强** |

评审结论：真正有效的防线是 **1、5、6**；2、3、4 是辅助，规格中不得把它们当作正确性保证来依赖。

## 8. 分阶段交付（评审采纳：先验收引擎，再建 UI）

- **Phase 0 —— 引擎离线校准（先做，UI 之前）**：挑一批**已知真实边界**的视频，只跑引擎，输出 CSV/JSON，量化：**边界纯度**（clip 是否跨动作）、**每类召回**、**短动作丢失率**、**每类合理阈值**。附带 golden test（4.2）与 checkpoint 硬校验（4.3）。**引擎达标后**才建完整 UI，避免在不可靠引擎上堆漂亮界面。
- **Phase 1 —— review UI + apply**：本机跑通最小闭环（label → review → apply → run_crop）。
- **Phase 2（延后）**：接入 `.bat`/scaffold 自动化、同事跨机器打包、非 H.264 转码、gap 新建 segment。

## 9. 必须在实现前钉死的滑窗参数

- 窗口长度是否固定 3.0 秒；stride（0.25/0.5/1 秒）。
- 首尾不足完整窗口如何处理。
- 抽帧依据解码 PTS 还是平均 fps（VFR 视频不能用 `frame_index/avg_fps`）。
- 预测归到窗口**中心时间点**还是整段区间（评审建议：中心点 + 迟滞，避免把重叠窗口区间直接当动作占用区间）。
- uncertain 如何阻断合并；重叠但中间 uncertain 的窗口如何显示。
- 进入/退出某状态需连续几个窗口（迟滞规则，防单帧抖动切换）。
- segment confidence 取最小值 / 均值 / 低分位数。
- 无动物判定：连续缺失多久才算 gap（而非逐帧二值，参考现有"沿用上一个框"的抖动容忍）。
- 边界精度理论上受 stride + 窗口长度限制（MVP 不宣称精确切割，人工 trim 是承重步骤）。

## 10. 涉及/新增文件一览

**新增**：
- `auto_presort_label.py`、`auto_presort_review.py`、`auto_presort_apply.py`
- 引擎/预处理复用模块（滑窗、gap、trim、JSON schema）
- golden test 脚本

**改动（受控小改）**：
- `utils/abnormal_crop_video.py::rebalance_val`：按来源 id 分组 split（4.1）
- `snapshot_merge.py::health_report`：新增同源跨 split 检测（4.1）

**复用不改**：`pipeline_config.py`（物种查表）、`models/teacher.py` 的模型结构（但 label 步在其外层加硬校验）、`utils/video_dataset.py` 的 eval 预处理、`run_crop.py` 主流程。
