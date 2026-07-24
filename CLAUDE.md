# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Background

- 目标：训练宠物行为识别模型，用于对视频片段中的宠物行为进行分类。
- 训练对象分两类数据集：
  1. **猫狗行为识别**：包含正常行为（3类：`eat` / `drink` / `other`）和异常行为（5类：`normal` / `convulsion` / `limp` / `sneeze` / `vomit`）。
  2. **兔子行为识别**：独立数据集，类别体系与猫狗不同（具体类别待补充）。
- 训练策略：两阶段教师-学生蒸馏流程。
  1. **第一阶段**：在目标数据集上微调预训练教师模型（`swin3d_b`），使其适配具体类别。
  2. **第二阶段**：以微调后的教师为监督信号，蒸馏训练轻量学生模型（`RepViT_GRU`），以便部署。
- 最终交付的是学生模型，教师只在训练过程中提供知识监督。

## Commands

- There is no checked-in build system, linter config, dependency manifest, or automated test suite in this repo.
- Main entrypoints are direct Python scripts:
  - `python train_finetune_normal_3.py`
  - `python train_finetune_ab_5.py`
  - `python distill/distill_normal_3.py`
  - `python distill/distill_abnormal_5.py`
  - `python utils/abnormal_crop.py`
- There is currently no single-test command because no tests are checked in.
- These scripts are edit-the-file entrypoints, not parameterized CLIs; most runtime settings are hardcoded in script-level constants or dictionaries.
- Training and distillation scripts assume CUDA (`.cuda()` / `torch.device("cuda")`). Only `utils/abnormal_crop.py` has a CPU fallback.
- Runtime dependencies must be inferred from imports: `torch`, `torchvision`, `timm`, `opencv-python`, `Pillow`, `numpy`, `scikit-learn`, `pyyaml`, `ultralytics`; `transformers` is additionally required for the Mamba/VMamba heads.

## Architecture Overview

- This is a video action-recognition project for pet-behavior clips.
- The practical pipeline in the checked-in scripts is teacher-student distillation:
  - Teacher: Torchvision `swin3d_b` wrapped by `distill/swin_wrapper.py`
  - Student: `recognizers/gru_model.py` (`RepViT_GRU`)
- `distill/swin_wrapper.py` adapts `swin3d_b` to arbitrary class counts, optionally loads a fine-tuned checkpoint, and registers feature hooks so distillation scripts can match intermediate teacher features.
- `recognizers/gru_model.py` is the lightweight deployment-oriented model: RepViT-M1 backbone -> global average pooling -> GRU over frame features -> linear classifier.
- The repo also contains a more generic recognizer stack in `recognizers/full_model.py`, but the current training/distillation entrypoints do not use it directly.
- `recognizers/full_model.py` composes a timm backbone from `recognizers/backbones.py` with temporal heads from `recognizers/behavior_head.py` and, for VMamba mode, `recognizers/vmamba_head.py`.
- `recognizers/backbones.py` can inject TSM blocks into deeper timm stages when `use_tsm=True`.
- Temporal modeling options in `recognizers/behavior_head.py` are `lstm`, `transformer`, `mamba`, and `tsm_head`; `recognizers/vmamba_head.py` adds a spatial-VMamba + temporal-Mamba head for 5D video tensors.

## Data Layout And Loading

- Dataset loading is task-specific:
  - `utils/dataset_normal_3.py` for 3-class normal behavior (`eat`, `drink`, `other`)
  - `utils/dataset_abnormal_5.py` for 5-class abnormal behavior (`normal`, `convulsion`, `limp`, `sneeze`, `vomit`)
- Loaders expect a dataset root with `train/` and `val/` subdirectories.
- Each sample can be either a folder of extracted frames or a raw video file (`.mp4`, `.avi`, `.mov`).
- Labels are inferred from directory/file names by substring matching against `class_names`; the loaders do not read `label.txt`.
- Both dataset loaders apply hardcoded balancing heuristics that truncate the dominant class by keeping higher-numbered samples and dropping earlier ones.
- Frame sampling is uniform: loaders resize frames to `imgsz`, normalize with ImageNet stats, and return a fixed `temporal_window` tensor.

## Data Preparation

- `utils/abnormal_crop.py` is the dataset-generation script.
- It uses Ultralytics YOLO to crop cat/dog regions, splits clips into `train`/`val` by a fixed modulo-10 rule, writes 16 frames per sample, and saves a `label.txt`.
- This script is driven by hardcoded Windows paths and constants at the top of the file; it is not wired to `config.yaml`.

## Configuration Gotchas

- `config.yaml` is not the single source of truth.
- `train_finetune_normal_3.py` and `train_finetune_ab_5.py` use hardcoded `TEACHER_SETTINGS` dictionaries and inline class lists instead of reading `config.yaml`.
- `distill/distill_normal_3.py` and `distill/distill_abnormal_5.py` do read `config.yaml`, but both force `mode = "normal"` inside the script and then overwrite `class_names` / `num_classes` in code. Always inspect the script constants before assuming YAML changes will take effect.
- The checked-in YAML task config for `abnormal` is out of sync with the 5-class abnormal dataset/model code.
- Scripts import `from dataset import get_dataset`, but the checked-in dataset implementations live in `utils/dataset_normal_3.py` and `utils/dataset_abnormal_5.py`. If execution fails on that import, look for or create the missing shim rather than assuming the utils modules are unused.
- `distill/swin_wrapper.py` downloads Torchvision Kinetics-400 weights when no valid checkpoint is provided.

## Outputs And Checkpoints

- Teacher fine-tuning scripts save best EMA teacher weights into `outputs/...`.
- Distillation scripts save both raw student and EMA student weights into `outputs/...`.
- `checkpoints/` stores checked-in model assets; runtime training outputs are written to `outputs/`.
- `config.yaml` mentions ONNX output names, but there is no checked-in inference or export script in this repo.
- Evaluation is embedded inside the training/distillation loops via validation accuracy and confusion-matrix logging rather than a separate evaluation CLI.
