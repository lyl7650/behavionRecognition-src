# 技术交接文档：宠物行为识别模型训练

*English version: [`handover.en.md`](./handover.en.md)*

端到端操作步骤见 [`docs/SOP.md`](./SOP.md)。本文档是参考手册：脚本清单、数据目录怎么摆、训练怎么改参数。

> 本文档所有脚本运行前，先执行 `conda activate AR` 激活对应虚拟环境。

## 1. 核心脚本清单（输入 → 输出）

| 脚本                                 | 作用                                     | 输入                                                                                                            | 输出                                                                       |
| ---------------------------------- | -------------------------------------- | ------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| `utils/abnormal_crop.py`           | 把已剪辑分类的短视频，用 YOLO 裁剪目标区域，输出为**逐帧图片**格式 | 脚本内 `INPUT_BASE_DIRS` 指定的分类好的短视频文件夹                                                                           | `OUTPUT_BASE_DIR` 下的 `train/val` 目录，每个样本是一个文件夹（16 张 JPG 帧 + `label.txt`） |
| `utils/abnormal_crop_video.py`     | 同上，输出为**裁剪后的短视频**格式                    | 同上（脚本内常量单独配置）                                                                                                 | `OUTPUT_BASE_DIR` 下的 `train/val` 目录，每个样本是一个按类别文件夹存放的 mp4 视频文件            |
| `train/train_finetune_normal_3.py` | 第一阶段：微调教师模型（Swin3D-B）                  | `config.yaml` → `task_configs.normal`（`dataset_path`、`class_names`、`finetune.*`）                              | `finetune.output_dir` 下的教师权重（EMA），文件名为 `finetune.output_model_name`      |
| `distill/distill_normal_3.py`      | 第二阶段：把教师模型蒸馏进轻量学生模型（RepViT + GRU）      | `config.yaml` → `distillation.teacher_checkpoint`（教师权重）+ `task_configs.normal.dataset_path`（训练数据，通常与教师用的是同一批） | `task_configs.normal.distillation.output_dir` 下的学生权重（原始 + EMA 两个文件）      |

`abnormal_crop.py` 和 `abnormal_crop_video.py` 是**两种可选的输出格式**，跟猫狗/兔子物种无关——两者都是"把分好类的长视频，用 YOLO 裁剪目标区域再切成固定长度片段"，只是一个存成逐帧图片、一个存成短视频文件。两种格式的样本都能被训练脚本直接读取。**后续新处理的数据建议优先用 `abnormal_crop_video.py`（视频片段格式）。**

以上两个训练脚本都不接受命令行参数，直接 `python <脚本路径>` 运行，配置全部来自根目录 `config.yaml`。

> 项目里还有其他类似命名的脚本（`train_finetune_ab_5.py`、`distill_abnormal_5.py`、`run_train.py`、`run_distill.py` 等），**当前训练流程不使用它们**，不在本文档范围内。

## 2. `dataset/` 目录摆放逻辑

### 2.1 按物种、按日期快照

```
dataset/
  pet/normal/<日期>/       # 猫狗训练数据快照，如 dataset/pet/normal/2026-07-10
    train/ val/            # 或者
    videos/train/ videos/val/   # 取决于当时用哪个处理脚本产出的
  rabbit/<日期(-videos)>/  # 兔子训练数据快照，如 dataset/rabbit/0703-videos
    train/
    val/
```

同一个日期快照下，`train/val` 到底是直接放着（逐帧格式）还是在一个 `videos/` 子目录下（视频片段格式），取决于当时用的是 `abnormal_crop.py` 还是 `abnormal_crop_video.py`，两种命名方式都存在，不完全统一，用之前先进目录确认一下实际结构。

**每个日期文件夹是一次"新旧数据合并"后的完整快照，不是当天新增的增量。** 判断哪个是当前有效数据的规则很简单：**同一物种下，日期最新的那个文件夹就是当前有效数据**，更早日期的文件夹是历史版本，仅作留存，不要往里面追加新数据，也不要拿旧日期的路径去训练。

新快照怎么产生、什么时候需要合并旧数据，见 [SOP 步骤 4-5](./SOP.md#步骤-4跑数据处理脚本产出训练样本)。

### 2.2 样本内部格式（两种物理格式，跟物种无关）

**逐帧格式**（`abnormal_crop.py` 产出）：每个样本是一个文件夹，帧图片 + `label.txt`：

```
train/
  0430-eat-CAM-04_..._00.00.18-00.00.27-0/
    frame_0000.jpg ... frame_0015.jpg
    label.txt          # 内容是一个整数，如 "0"
```

**视频片段格式**（`abnormal_crop_video.py` 产出）：按类别文件夹存放视频文件，没有 `label.txt`：

```
train/
  active/
    clip_001.mp4
  resting/
  eating/
  drinking/
```

两种格式都可能出现在猫狗或兔子任意一方的数据里，具体是哪种要看处理这批数据时用的是哪个脚本。两种格式都由同一个数据加载器 `utils/video_dataset.py::VideoDataset` 兼容读取：**优先读 `label.txt`，没有则用文件夹名/文件名里是否包含类别关键字做匹配**（大小写不敏感，子串匹配）。所以不管哪种格式，样本的文件夹名或文件名里必须包含类别关键字，否则会被判定为无法识别类别而被跳过（不会报错提示）。

### 2.3 其他目录（跟当前训练流程无关，忽略即可）

`dataset/pet/hierarchical-cls/`、`dataset/pet/abnormal/` 是废弃/未启用的猫狗异常行为（5类）数据，跟 `train_finetune_normal_3.py` / `distill_normal_3.py` 这条实际在用的流程无关。

## 3. 训练：涉及代码 & 怎么改参数

### 3.1 两阶段训练脚本

- **教师微调**：`python train/train_finetune_normal_3.py`
- **学生蒸馏**：`python distill/distill_normal_3.py`

猫狗和兔子共用这两个脚本，区别只在于 `config.yaml` 里 `task_configs.normal.dataset_path` / `class_names` 指向哪个物种的数据。

支撑这两个脚本的共享代码：

```
training/config.py               ← 读 config.yaml，拼出训练用的最终配置字典
training/teacher_finetune.py     ← 教师微调训练循环的具体实现
training/distillation.py         ← 蒸馏训练循环的具体实现
training/distillation_losses.py  ← 蒸馏 loss（KD loss、特征匹配 loss）构建
models/teacher.py                ← 教师模型定义（SwinTeacher）
models/student_factory.py        ← 根据配置构建学生模型
models/timm_gru_student.py       ← 学生模型结构（timm backbone + GRU）
dataset/__init__.py              ← get_dataset()，按配置构建 VideoDataset
utils/video_dataset.py           ← 数据集加载：扫描样本、读帧、数据增强、类别不均衡截断
```

### 3.2 `config.yaml` 关键字段

```yaml
task_configs:
  normal:
    dataset_path: "dataset/rabbit/0626-videos"   # 训练数据路径，切换物种/快照就改这里（路径里的 pet/rabbit 要跟着物种改，见下方⚠️）
    class_names: ["resting", "active", "eating", "drinking"]   # 类别列表，顺序即类别 index

    finetune:                      # 第一阶段：教师微调超参
      checkpoint_path: None        # None = 从预训练/随机初始化开始；接着练则填上次教师权重路径
      output_dir: "outputs/rabbit/{date}/teacher_finetuned_3cls_{date}"   # {date} 自动替换成当天日期
      output_model_name: "rabbit_teacher_{date}_ema-videos.pth"
      batch_size: 4
      accum_steps: 8               # 梯度累积步数，真实等效 batch = batch_size * accum_steps
      epochs: 20
      lr: 4e-5
      num_workers: 4

    recognizer:                    # 学生模型结构
      backbone: repvit_m1          # timm backbone 名字，可选值见 models/student_factory.py 的 BACKBONE_GRU_HIDDEN 表
      gru_hidden: auto             # auto = 按 backbone 名字自动查表
      temporal_window: 16          # 每个样本抽帧数，必须和数据处理脚本里的 FRAMES_PER_CLIP 保持一致
      imgsz: 224
      dropout: 0.2
      augment:                     # 数据增强，概率/幅度调小或设 0 即可减弱/关闭
        rotate: 15
        fliplr: 0.5
        brightness: 0.15
        reverse: 0.3              # 整段视频帧倒放的概率（仅训练时生效）

    distillation:                  # 第二阶段：本任务的输出路径（学习率/loss权重取顶层 distillation 块）
      output_dir: "outputs/rabbit/{date}"
      output_model_name: "repvit_best_distilled_{date}.pth"

distillation:                      # 蒸馏通用超参
  teacher_checkpoint: "outputs/rabbit/2026-07-06/teacher_finetuned_3cls_0706/rabbit_teacher_0706_ema-videos.pth"
  temperature: 4.0                # 蒸馏温度，越大教师概率分布越"软"
  loss_weights:
    gt: 1.0      # 真实标签 loss 权重
    kd: 4.0      # 蒸馏(KD) loss 权重
    feature: 1.0 # 中间特征匹配 loss 权重
  learning_rate: 3e-4
  batch_size: 16
  epochs: 15
```

`config.yaml` 里还有 `task_configs.abnormal` 一块，对应当前未启用的猫狗5类异常行为流程，不用管它。

### 3.3 常改参数速查表

| 想改什么                              | 改哪里                                                                                                                                                                        |
| --------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 训练数据路径                            | `task_configs.normal.dataset_path`                                                                                                                                         |
| 类别列表/顺序                           | `task_configs.normal.class_names`                                                                                                                                          |
| 教师微调的 batch size / epochs / 学习率   | `task_configs.normal.finetune`                                                                                                                                             |
| 蒸馏的 batch size / epochs / 学习率     | 顶层 `distillation` 块（或 `task_configs.normal.distillation` 覆盖）                                                                                                               |
| 蒸馏温度、loss 权重                      | `distillation.temperature` / `distillation.loss_weights`                                                                                                                   |
| 蒸馏用哪个教师权重                         | `distillation.teacher_checkpoint`                                                                                                                                          |
| 数据增强强弱                            | `task_configs.normal.recognizer.augment`                                                                                                                                   |
| 学生模型 backbone                     | `task_configs.normal.recognizer.backbone`                                                                                                                                  |
| 输出模型存哪、叫什么名字                      | `finetune.output_dir` / `output_model_name`（支持 `{date}` 占位符）                                                                                                               |
| 每个样本抽多少帧                          | `recognizer.temporal_window`，必须和数据处理脚本的 `FRAMES_PER_CLIP` 一致                                                                                                               |
| "背景类"（other/resting）在 loss 里的特殊处理 | 不在 `config.yaml` 里，需要改代码：`train/train_finetune_normal_3.py` 里 `HierarchicalPriorityLoss(other_idx=..., ...)` 的构造参数；`distill/distill_normal_3.py` 里的 `_resolve_other_idx()` |

⚠️ 切换猫狗/兔子时，`dataset_path`、`output_dir`、`output_model_name` 三个字段里的 `pet` / `rabbit` 字符串都要跟着改，不能只改 `class_names`，否则训练能跑，但产出的模型会存到错的物种目录下。

⚠️ `train/train_finetune_normal_3.py` 的 `other_idx` 是写死的整数（当前是 `2`），不会跟着 `class_names` 顺序自动变化。切换物种/改类别顺序时，务必确认"背景类"（猫狗的 `other`，兔子的 `resting`）在新的 `class_names` 里排第几位，对不上就要手动改这个数字。

## 4. 产出与交付

- 教师权重（步骤7产出）：仅用于监督蒸馏阶段，**不部署**。
- 学生权重（步骤8产出）：原始权重 + EMA 权重两个文件，**部署用 EMA 版本**。
- 拿到新的学生 EMA 权重后，用它更新线上部署的系统。
- 仓库里没有现成的、能直接跑通"权重 → 部署格式"的推理/导出脚本，`utils/export_onnx.py` 是相关的导出脚本，具体导出流程建议交接时单独确认一次。
