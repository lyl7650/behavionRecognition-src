# 兔子行为识别训练与蒸馏速查

本文基于当前重构后的项目结构，示例任务为：

- 模式：`normal`
- 数据集路径：`dataset\rabbit\0507`

## 1. 先确认配置

主配置文件是：

- [config.yaml](E:\ActionRecognition\behavionRecognition-src\config.yaml)

当前这条链路主要看两段配置：

```yaml
task_configs:
  normal:
    dataset_path: "E:/ActionRecognition/behavionRecognition-src/dataset/rabbit/2026-05-07"
    class_names: ["eating", "moving", "resting", "drinking"]

    finetune:
      checkpoint_path: "outputs/pretrained/best_pretrain_mamba.pth"
      output_dir: "outputs/rabbit/2026-05-07/teacher_finetuned_4cls_0507"
      output_model_name: "teacher_best_0223_ema.pth"

    distillation:
      teacher_checkpoint: "outputs/rabbit/2026-05-07/teacher_finetuned_4cls_0507/teacher_best_0223_ema.pth"
      output_dir: "outputs/distilled_final/rabbit/2026-05-08"
```

如果你的数据就在 `dataset\rabbit\2026-05-07`，这部分通常不用再改。

## 2. 训练 teacher

命令：

```powershell
conda run -n AR python run_train.py --mode normal
```

说明：

- 入口脚本：`run_train.py`
- 实际读取：`config.yaml -> task_configs.normal.finetune`
- 输出目录：`outputs/rabbit/2026-05-07/teacher_finetuned_4cls_0507`
- 最优 teacher 权重文件名：`teacher_best_0223_ema.pth`

训练完成后，关键产物是：

```text
outputs/rabbit/2026-05-07/teacher_finetuned_4cls_0507/teacher_best_0223_ema.pth
```

## 3. 运行蒸馏

命令：

```powershell
conda run -n AR python run_distill.py --mode normal
```

说明：

- 入口脚本：`run_distill.py`
- 实际读取：`config.yaml -> task_configs.normal.distillation`
- teacher 权重路径来自：

```yaml
task_configs.normal.distillation.teacher_checkpoint
```

- 输出目录：`outputs/distilled_final/rabbit/2026-05-08`

蒸馏完成后，关键产物是：

```text
outputs/distilled_final/rabbit/2026-05-08/repvit_best_distilled.pth
outputs/distilled_final/rabbit/2026-05-08/repvit_best_distilled_ema.pth
```

## 4. 最常改的地方

如果你只是在同一条兔子链路上重复训练，通常只改这几个字段：

```yaml
task_configs:
  normal:
    dataset_path: "..."

    finetune:
      checkpoint_path: "..."
      output_dir: "..."

    distillation:
      teacher_checkpoint: "..."
      output_dir: "..."
```

## 5. 一条完整流程

先训练 teacher：

```powershell
conda run -n AR python run_train.py --mode normal
```

再蒸馏 student：

```powershell
conda run -n AR python run_distill.py --mode normal
```

## 6. 结构对应关系

当前结构里，这条链路对应的是：

- `run_train.py` / `run_distill.py`：统一入口
- `training/config.py`：配置解析
- `training/teacher_finetune.py`：teacher 公共训练循环
- `training/distillation.py`：蒸馏公共训练循环
- `train/train_finetune_normal_3.py`：normal teacher 薄入口
- `distill/distill_normal_3.py`：normal distill 薄入口
