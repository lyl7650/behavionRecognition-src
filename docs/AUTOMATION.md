# 宠物行为识别流水线 自动化使用说明

本流水线把「从原始视频到可部署模型」的过程收敛成一串双击入口脚本，覆盖建目录、切片、合并训练三个阶段。以下所有命令都在 `conda activate AR` 环境下、从**项目根目录**（`behavionRecognition-src`）启动。

```bash
conda activate AR
cd E:/ActionRecognition/behavionRecognition-src
```

## 最省事：直接双击 .bat（推荐）

项目根目录下有若干双击启动器，**双击即可运行**，不用开终端、不用敲 `conda activate AR`、不用记 python 命令。它们会自动切到项目目录、激活 `AR` 环境、跑对应脚本，跑完窗口停住让你看结果（按任意键关闭）。

| 双击文件 | 相当于 | 作用 |
|----------|--------|------|
| `1_scaffold_dirs.bat` | `python -m scripts.pipeline.scaffold_dirs` | 建目录 |
| `2a_presort_label.bat` | `python -m scripts.presort.run_presort_label` | 模型预标长视频（需 GPU） |
| `2b_presort_review.bat` | `python -m scripts.presort.run_presort_review` | 网页审核批准 |
| `2c_presort_apply.bat` | `python -m scripts.presort.run_presort_apply` | 切片落地到 sorted 目录 |
| `3_run_crop.bat` | `python -m scripts.pipeline.run_crop [--frames]` | 裁目标区域、生成快照 |
| `4_run_train_distill.bat` | `python -m scripts.pipeline.run_train_distill [--frames]` | 合并 + 训练 + 蒸馏 |
| `5_run_test.bat` | `python -m scripts.pipeline.run_test --interactive` | 离线评测 teacher 或 student checkpoint |

- `3` 和 `4` 双击后都会用编号选择格式：`1` 是默认的视频 clip 格式，`2` 是切帧格式。
- **`4` 的格式选择必须和 `3` 保持一致**（同一批数据要么都用视频 clip，要么都用切帧）。
- `5` 会依次用编号选择物种和模型类型（teacher/student），然后输入 checkpoint 路径。结果保存在 checkpoint 同级、带时间戳的 `test_results_<species>_<model_type>_<timestamp>/` 目录。
- 启动器用「自己所在的文件夹」当项目目录，所以这些 `.bat` 必须和脚本放在同一目录（即项目根目录）；整个项目挪到别的盘/路径也照常能用。
- 前置条件：已安装好名为 `AR` 的 conda 环境，且普通 cmd 里能直接调用 `conda`。

> 如果你更习惯命令行，下面仍保留手动方式。

## 各入口（手动命令行方式）

### 1. `python -m scripts.pipeline.scaffold_dirs` —— 建目录

按物种/机构/日期建好本批需要的目录骨架（下载区、剪辑分类区等），供人工往里放素材。

### 3. `python -m scripts.pipeline.run_crop [--frames]` —— 裁目标区域、生成快照

把剪辑好的原始视频用检测模型裁出目标，生成本批快照（数据集）。

- 不加参数：输出视频 clip 格式（默认）。
- 加 `--frames`：输出切帧格式。

### 4. `python -m scripts.pipeline.run_train_distill [--frames]` —— 合并 + 训练 + 蒸馏

合并历史旧快照 → 体检确认 → 改写 config → teacher 训练 → 回填 teacher 路径 → student 蒸馏。

- `--frames` 需与 `run_crop.py` 用的格式保持一致。

### 5. `python -m scripts.pipeline.run_test --interactive` —— 离线评测

选择 rabbit 或 pet，再选择 teacher 或 student，并输入要评测的 checkpoint 路径。评测会先用对应物种的 YOLO 模型裁出动物区域，再做行为识别；预测轨迹、每条视频的指标/时间线图，以及汇总 JSON/CSV 都保存在 checkpoint 同级的时间戳结果目录中。

建目录和裁剪脚本会交互询问：

- **物种**：`pet` 或 `rabbit`
- **机构（site）**：本批数据来源机构
- **日期**：见下方日期约定

训练和蒸馏只询问物种和日期；不再询问 site，因为它处理的是已经生成的数据集快照。

## 剪辑分类：模型预标 + 网页审核（2a/2b/2c）

下载完长视频后，用微调好的 teacher 模型自动预切分预标注，人只需在网页上审核批准，替代手工剪辑分类。双击 `2a → 2b → 2c` 依次跑：

- **2a 预标注**（需 GPU）：模型把长视频切成带标签的候选片段。teacher 模型按物种自动选默认；运行时提示 `Teacher checkpoint [default ...]:`，回车用默认、或粘贴一个临时新模型路径覆盖。
- **2b 网页审核**（不需 GPU）：自动开浏览器，一个页面审全批。可拖动调整片段边界、改标签、删除、在空白处拖拽新建片段。**每个视频必须点 Approve 才会被切片；任何编辑都会让批准失效**（防止模型错标混进训练集）。如提示端口被占用，先关掉上一个还开着的审核窗口再重试。
- **2c 切片落地**（不需 GPU）：只切"已批准"的视频，输出到 `sorted-<日期>/<类别>/`，正是下一步 `3_run_crop` 读取的目录，直接接着往下走。文件名内嵌来源信息用于防 train/val 泄漏，**不要改名**。

完整顺序：

```
1_scaffold（建目录 + 下载长视频）
   → 2a_label（模型预标）→ 2b_review（网页审核批准）→ 2c_apply（切片落地）
   → 3_run_crop（裁目标区域、生成快照）
   → 4_run_train_distill（合并 + 训练 + 蒸馏）
   → 5_run_test（选择 checkpoint，离线评测）
```

置信度阈值和边界内缩目前用脚本内默认值（0.90 / 1.0 秒）；如需按批次调整，可改用命令行方式（见下）直接传 `--threshold` / `--trim-margin`。

### 命令行方式（等价，供习惯终端的人）

```bash
python -m scripts.presort.auto_presort_label  --species rabbit --dir <长视频目录> --checkpoint <teacher.pth> [--threshold 0.90] [--trim-margin 1.0]
python -m scripts.presort.auto_presort_review --dir <同一目录>
python -m scripts.presort.auto_presort_apply  --dir <同一目录> --out <sorted-日期 目录>
```

手工剪辑分类仍可作为后备（把剪好的短片按类别放进 `sorted-<日期>/<类别>/`），与 2a/2b/2c 产出的目录结构一致。

### 分发审核给同事（可选，Windows / Mac 都行）

审核（2b）纯靠人工，视频多时可以分给同事一起审。审核这套工具是纯 Python 标准库，**不需要 conda、不需要显卡**，同事装个 Python 3.8+ 就能跑。

1. 本机双击 `pack_review.bat`（走 conda `AR`）→ 答物种/机构/日期 → 屏幕列出该批全部视频（带编号）→ 输入要发的编号（如 `0-19`，或 `all` 全发）。
2. 生成一个自包含文件夹 `review-pack-<机构>-<日期>/`，**根目录只放双击入口**（`review.bat` 给 Windows、`review.command` 给 Mac、`README.txt` 说明）；审核程序在 `app/` 子目录、视频和标注在 `data/` 子目录。
3. 把**整个文件夹**发给同事。同事装好 Python 3.8+ 后：Windows 双击 `review.bat`，Mac 双击 `review.command`（首次若被拦，右键→打开，或终端 `chmod +x review.command`）。浏览器会自动开审核页，逐个视频点 Approve。
4. 同事审完，把文件夹里 `data/` 下的 **`.autopresort` 整个发回来**；本机拷回 `records/<日期>/.autopresort/` **覆盖同名 JSON**，再正常双击 `2c_presort_apply.bat` 切片（之后接 `3_run_crop.bat`）。

每个视频对应一个 `.autopresort/<视频名>.mp4.json`，**一一对应**，所以能分工：发 `0-19` 给 A、`20-39` 给 B，各自审完只回传各自那部分覆盖，互不影响。

## 日期约定（重要）

- 输入的是**录制日**（视频实际录制那天），手动输入，直接回车默认今天。
- **同一批数据从建目录、切片到训练，必须使用同一个日期**，否则脚本会找不到上一步的产物。

## 人工节点

流水线不是全自动，以下环节需要人工介入：

1. **下载长视频**：把机构录制的长视频下载到 `scaffold_dirs.py` 建好的下载区。
2. **网页审核批准**：双击 `2a` 让模型预标，再双击 `2b` 在浏览器里逐个视频审核、调整、点 Approve（人不再手工剪辑）。手工剪辑分类作为后备仍可用。
3. **体检确认**：`run_train_distill.py` 打印体检报告后会停下（`回车继续，Ctrl+C 中止`），人工核对各类别数量、跨 split 泄漏、静默跳过等是否正常。
4. **看结果**：训练结束后查看输出模型和日志。

## 体检报告怎么看

`run_train_distill.py` 在合并完旧快照后会打印一份体检报告：

- **各类别样本数**：确认每个类别都有足够样本、比例合理。
- **泄漏（cross_split_leak）**：同名样本同时出现在 train 和 val，会导致评估虚高，应清理。
- **静默跳过（unrecognized）**：文件/目录名无法匹配到任何类别的样本，不会计入训练，需检查命名。
- **label_mismatch**：切帧模式下目录名与 `label.txt` 内的标签不一致。
- **同源跨 split（source_leak）**：同一原始视频切出的片段同时进入 train 和 val，会导致评估虚高，应先处理。
- 归类依赖文件/目录名里的类别关键字匹配。pet 当前固定为 `eat`、`drink`、`active`、`resting` 四类；样本不能再使用 `other` 代替 `active` 或 `resting`。

确认无误后回车继续，脚本才会真正开始改 config 和训练。

## 部署产物

- 部署要用的是 **student 的 `*_ema.pth`**，位于 `outputs/<物种>/<日期>/` 下。
- **不是 teacher**。teacher 只是训练中间产物，用于蒸馏出更小更快的 student。

## 注意事项

- `run_train_distill.py` 会改写 `config.yaml`（写入本批数据集路径、类别、teacher 路径等）。YAML 重写会**丢掉原有注释**，但每次改写前都会在 `config/backups/` 生成一个带时间戳的 `.bak` 备份（如 `config/backups/config.yaml.20260715-1930.bak`），需要时可从备份恢复。
- **合并不可逆**：当检测到无法确定日期的旧快照目录（ambiguous）时，脚本会打印自动选出的最新旧快照并要求输入 `yes` 确认后才合并，其它输入一律中止，避免误合并。

## 维护

- **物种规则查表**：全部集中在 `scripts/pipeline/pipeline_config.py` 的 `SPECIES` 字典里（类别名、label_map、数据集路径、检测模型路径等）。新增物种或调整类别改这里。
- **检测模型换代**：更换 YOLO 检测模型时，更新 `scripts/pipeline/pipeline_config.py::SPECIES` 里对应物种的 `yolo_model` 路径。
