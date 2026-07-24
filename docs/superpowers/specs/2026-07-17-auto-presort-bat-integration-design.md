# auto_presort 并入自动化流程（.bat + AUTOMATION.md）设计

**日期：** 2026-07-17
**状态：** 已批准，待写实现计划

## 1. 背景与目标

`auto_presort` 三个 CLI（`auto_presort_label.py` / `auto_presort_review.py` / `auto_presort_apply.py`）已开发完成、测试通过。它们的作用：用微调好的 teacher 模型把**未剪辑的长视频**预切分预标注 → 网页人工审核批准 → 切片落地，替代大部分手工"剪辑分类"工作。

目标：把这三步"丝滑"并入现有自动化流程（`docs/AUTOMATION.md` + 根目录三个双击 `.bat`），做到**双击 .bat、只答三个问题（物种/机构/日期）、零手敲路径**，与现有 `1_scaffold` / `2_run_crop` / `3_run_train_distill` 体验一致。

## 2. 关键接缝（为什么天然吻合）

现有流程的目录约定已经把 auto_presort 的输入输出都框好了：

- `1_scaffold_dirs.py` 建出 `records/<date>/`（长视频下载区）和 `sorted-<date>/<类别>/`（剪辑分类区）。
- **label 的输入** = `records/<date>/`，正是 `pc.records_dir(species, site, date)`。
- **apply 的输出** = `sorted-<date>/<类别>/`，正是 `pc.sorted_dir(species, site, date)`；而这正是 `2_run_crop.py` 读取的输入（`validate_sorted` 期望 `<sorted>/<类别>/*.mp4`）。

因此 auto_presort 支线**替代的是人工节点"剪辑分类"**，插在 `1_scaffold`（下载长视频）之后、`2_run_crop`（切片）之前，前后严丝合缝。它是本流程标准的剪辑方式（手工剪辑分类仍可用作后备，但不作为默认路径宣传）。

唯一 `pipeline_config` 推不出的量是 **teacher checkpoint 路径**，需显式提供默认值（见 §3.1）。

## 3. 设计

### 3.1 pipeline_config：按物种写死默认 teacher

在 `SPECIES` 字典里给 `pet` 和 `rabbit` 各加一个 `teacher_checkpoint` 字段（默认教师模型路径），风格与现有 `yolo_model` 字段一致。这是"按物种路由的默认模型"。默认值：

- `pet`：`E:\ActionRecognition\behavionRecognition-src\outputs\pet\2026-07-10\teacher_finetuned_3cls_0710\pet_teacher_0710_ema-videos.pth`
- `rabbit`：`E:\ActionRecognition\behavionRecognition-src\outputs\rabbit\2026-07-06\teacher_finetuned_3cls_0706\rabbit_teacher_0706_ema-videos.pth`

运行时允许覆盖：wrapper 交互提示时可回车用默认、或粘贴一个临时新模型路径覆盖（用于试新训练出的 teacher）。

### 3.2 三个问答式 wrapper（根目录）

仿照 `run_crop.py` 包装 `abnormal_crop_video.py` 的模式，新增三个薄包装入口：

| 新脚本 | 问什么 | 推导 | 调用 |
| --- | --- | --- | --- |
| `run_presort_label.py` | 物种/机构/日期 + teacher（默认可回车/可粘贴覆盖） | `records_dir` 作 `--dir` | `auto_presort_label` 主逻辑 |
| `run_presort_review.py` | 物种/机构/日期 | 同一 `records_dir` | `auto_presort_review` 主逻辑（开浏览器） |
| `run_presort_apply.py` | 物种/机构/日期 | `records_dir` 作输入、`sorted_dir` 作 `--out` | `auto_presort_apply` 主逻辑 |

- 三件事问答复用现有 `pipeline_config.ask_context`（已有 species/site/date 交互，含"回车默认今天"）。
- teacher 覆盖提示形如 `Teacher checkpoint [default <SPECIES 默认路径>]:`，回车用默认。
- 现有 `auto_presort_*.py` 三个 CLI **保持不动**，继续作为"手动指定路径"的显式入口；wrapper 只在外面套"问三件事→推路径"，通过 import 调用其 `main`/核心函数，不复制逻辑。
- 控制台打印一律英文（cp1252 约定）。

### 3.3 三个双击 .bat

照搬现有 `.bat` 模板（`cd /d "%~dp0"` + `call conda activate AR` + 跑脚本 + `pause`）：

| .bat | 调用 | 备注 |
| --- | --- | --- |
| `2a_presort_label.bat` | `python run_presort_label.py` | 需 GPU、耗时长 |
| `2b_presort_review.bat` | `python run_presort_review.py` | 开浏览器审核，不需 GPU |
| `2c_presort_apply.bat` | `python run_presort_apply.py` | 切片落地，不需 GPU |

命名用 `2a/2b/2c` 表明"夹在 1_scaffold 与 2_run_crop 之间的可选支线"，不打乱主序号 1/2/3。

### 3.4 AUTOMATION.md 更新

- 双击表格里把 `2a/2b/2c` 作为**标准的剪辑步骤**列出（夹在 `1_scaffold` 与 `2_run_crop` 之间），即"下载完长视频后，用模型预标 → 网页审核 → 切片"。
- teacher 模型：默认按物种自动选，`2a` 运行时可回车用默认或粘贴临时路径。
- 保留底部原有纯命令行三步说明。
- 更新人工节点小节：原"剪辑分类"改为"在网页审核批准"（人只需在 `2b` 里审核，不再手工剪辑）；手工剪辑分类作为后备一句话带过。
- 新增一张顺序图：

  ```
  1_scaffold（建目录 + 下载长视频）
     → 2a_label（模型预标）→ 2b_review（网页审核批准）→ 2c_apply（切片落地）
     → 2_run_crop（裁目标区域、生成快照）
     → 3_run_train_distill（合并 + 训练 + 蒸馏）
  ```

## 4. 明确不做（YAGNI）

- 不改现有三个 `.bat` 的编号，不碰 `run_train_distill`。
- 不做跨机打包 / portable ffmpeg（spec 已延后 Phase 2）。
- 不在 wrapper 里重复实现 auto_presort 逻辑（只包装）。
- 不改 `auto_presort_*.py` 三个 CLI 的现有行为。

## 5. 成功标准

- 双击 `2a/2b/2c`，各自只问 物种/机构/日期（label 多问一次 teacher，可回车），全部路径自动推导、零手敲。
- `2c_apply` 产出的 `sorted-<date>/<类别>/` 能被现有 `2_run_crop` 直接读取，无需改名或改路径。
- `AUTOMATION.md` 让人一眼看懂两条路（人工 vs 模型）如何在 `2_run_crop` 前汇合。
- 现有 `auto_presort_*.py` CLI 与现有三个主 `.bat` 行为不变；相关测试仍全绿。
