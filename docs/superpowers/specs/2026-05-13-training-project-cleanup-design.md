# 训练工程整理方案

## 背景

当前项目已经能完成 teacher 微调、学生模型蒸馏、QAT、导出和评估，但训练工程的组织方式比较松散：

- 参数有些写在 `config.yaml`，有些硬编码在训练脚本里。
- teacher 训练、蒸馏、评估脚本里都在重复指定 `mode`、`class_names`、`num_classes`、路径和超参数。
- `normal` 和 `abnormal` 被拆成多份脚本，导致逻辑重复，后续加新任务时容易继续复制脚本。
- 训练产物有时写入 `outputs/`，有时写入 `checkpoints/`，不容易判断哪个是实验产物、哪个是可复用权重。
- `train/`、`distill/`、`recognizers/`、`utils/` 的职责边界不够清楚。
- `recognizers/` 里既有当前主流程用到的 `RepViT_GRU`，也有暂时没有被主训练入口使用的通用模型模块。

这份文档只描述整理方案，不直接改代码。

## 整理目标

第一阶段目标不是追求“最标准”的深度学习模板，而是先让项目可维护：

1. `config.yaml` 成为训练、蒸馏、评估的单一配置入口。
2. teacher 训练和蒸馏各保留一个通用入口，不再按 `normal/abnormal` 复制脚本。
3. 明确 `outputs/` 和 `checkpoints/` 的分工。
4. 模型、数据集、训练逻辑、工具脚本的目录职责更清楚。
5. 旧脚本可以保留一段时间，避免一次性破坏已有训练流程。

## 推荐路线

推荐采用“小步重构”的路线，也就是后续的 B 方案。

不建议一开始就做完整 C 方案，把项目直接改成 `src/behavior_recognition/...` 的标准包结构。原因是当前脚本依赖本地路径、数据集目录和历史权重较多，直接大迁移会把“结构整理”和“训练行为变化”混在一起，验证成本偏高。

推荐先完成下面这些低风险改动：

- 保留现有目录基本形态。
- 新增统一入口脚本。
- 抽出公共训练逻辑。
- 统一配置 schema。
- 让旧脚本变成兼容 wrapper，或者标记为 legacy。

## 建议目录

第一阶段可以整理成这样：

```text
behavionRecognition-src/
  config.yaml
  train.py
  distill.py
  evaluate.py

  configs/
    normal.yaml
    abnormal.yaml

  training/
    config.py
    loops.py
    losses.py
    metrics.py
    checkpointing.py

  data/
    video_dataset.py
    build.py

  models/
    teacher.py
    student.py
    backbones.py
    heads.py

  tools/
    crop_dataset.py
    export_onnx.py

  legacy/
    train_finetune_normal_3.py
    train_finetune_ab_5.py
    distill_normal_3.py
    distill_abnormal_5.py

  outputs/
    experiments/

  checkpoints/
    pretrained/
    released/
```

这不是一次性必须达到的最终结构，而是方向。第一阶段可以先只新增 `training/`、`models/` 或统一入口，旧目录暂时保留。

## 配置约定

建议把配置拆成两层：

- `config.yaml`：全局默认值，例如设备、随机种子、输出根目录、默认 batch size。
- `configs/normal.yaml`、`configs/abnormal.yaml`：任务配置，例如数据集路径、类别、teacher checkpoint、student checkpoint、训练轮数。

训练命令可以变成：

```powershell
python train.py --config configs/normal.yaml
python train.py --config configs/abnormal.yaml
python distill.py --config configs/normal.yaml
python evaluate.py --config configs/normal.yaml --checkpoint outputs/experiments/xxx/best_ema.pth
```

配置里应该统一包含：

- `task.name`
- `task.class_names`
- `data.root`
- `data.temporal_window`
- `data.image_size`
- `teacher.checkpoint`
- `student.backbone`
- `train.epochs`
- `train.batch_size`
- `train.learning_rate`
- `output.experiment_name`

脚本里不应该再硬编码 `normal`、`abnormal`、`class_names`、`num_classes`、`outputs/rabbit/0508` 这类信息。

`num_classes` 可以由 `class_names` 自动推导，避免两处配置不一致。

## 产物目录约定

建议明确区分：

- `checkpoints/`：手动保留的权重、预训练权重、对外发布或复用的稳定模型。
- `outputs/`：每次训练、蒸馏、评估生成的实验产物。

推荐训练产物统一写到：

```text
outputs/experiments/{task}/{stage}/{timestamp-or-name}/
  config.resolved.yaml
  best.pth
  best_ema.pth
  last.pth
  metrics.json
  confusion_matrix.txt
```

当某个实验结果确认可复用，再手动复制或导出到 `checkpoints/released/`。

这样可以避免训练脚本直接把新实验覆盖进 `checkpoints/`。

## 脚本合并方式

teacher 训练建议合并成一个入口：

```text
train.py
```

内部根据配置决定：

- 当前任务类别。
- 数据集路径。
- 是否加载 teacher checkpoint。
- 是否冻结 backbone。
- 使用哪种 loss。
- 输出目录。

蒸馏也合并成一个入口：

```text
distill.py
```

内部根据配置决定：

- teacher 权重路径。
- student 模型结构。
- 蒸馏温度、KD loss 权重、feature loss 权重。
- 输出目录。

`normal` 和 `abnormal` 的差异应该体现在配置中，而不是拆成两份几乎一样的 Python 文件。

## 模块命名建议

`recognizers/` 可以改名为 `models/`，因为里面实际放的是模型结构，而不只是识别器。

建议对应关系：

- `distill/swin_wrapper.py` -> `models/teacher.py`
- `recognizers/gru_model.py` -> `models/student.py`
- `recognizers/backbones.py` -> `models/backbones.py`
- `recognizers/behavior_head.py` -> `models/heads.py`
- `recognizers/full_model.py` 如果当前不用，先放到 `legacy/` 或保留但标注 experimental。

数据集相关建议从 `utils/` 移到 `data/`：

- `utils/video_dataset.py` -> `data/video_dataset.py`
- `utils/dataset_normal_3.py`、`utils/dataset_abnormal_5.py` 如果只是 wrapper，可以合并到 `data/build.py`

工具脚本建议从 `utils/` 移到 `tools/`：

- `utils/abnormal_crop.py` -> `tools/crop_dataset.py`
- `utils/export_onnx.py` -> `tools/export_onnx.py`

## 迁移步骤

建议分 4 步做，每一步都能单独验证：

1. 统一配置读取

   新增配置加载模块，负责读取 YAML、补默认值、校验必填字段、自动推导 `num_classes`。

2. 合并训练入口

   新增 `train.py`，先复用当前 teacher 训练逻辑，只把 `normal/abnormal` 差异移到配置里。

3. 合并蒸馏入口

   新增 `distill.py`，把 `distill_normal_3.py` 和 `distill_abnormal_5.py` 的公共逻辑抽出来。

4. 整理目录和 legacy

   在新入口稳定后，再移动模型、数据集和工具模块。旧脚本保留到 `legacy/`，并在文件头写清楚推荐使用新入口。

## 暂不处理的内容

为了控制复杂度，第一阶段不建议同时处理这些事：

- 不重写模型结构。
- 不改变 loss 的数学定义。
- 不改变数据集采样和类别均衡策略。
- 不引入新的训练框架，例如 Lightning、Hydra。
- 不强制改成 Python package 安装模式。
- 不清理历史权重文件。

这些可以在统一入口稳定后再做。

## 验收标准

第一阶段完成后，应该满足：

- 修改类别、数据路径、输出目录时，只改 YAML，不改 Python。
- teacher 训练只有一个主要入口。
- 蒸馏只有一个主要入口。
- 训练产物默认进入 `outputs/experiments/`。
- `checkpoints/` 不再被普通训练过程直接写入。
- 旧脚本仍能找到，必要时可以回退。
- README 或文档里有明确的训练、蒸馏、评估命令。

## 结论

推荐先做 B 方案：小步重构，统一配置和入口，保留旧脚本兼容。

C 方案可以作为第二阶段，在训练入口稳定后再做完整 `src/` 包结构迁移。这样风险更低，也更容易判断每一步是否真的改善了项目，而不是只改变目录外观。
