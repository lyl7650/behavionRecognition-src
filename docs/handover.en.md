# Handover Doc: Pet Behavior Recognition Model Training

For the end-to-end operating steps, see [`docs/SOP.md`](./SOP.en.md). This document is a reference manual: script inventory, how `dataset/` is laid out, how to change training parameters.

> Run `conda activate AR` to activate the environment before running any script in this document.

## 1. Core script inventory (input → output)

| Script | Purpose | Input | Output |
| --- | --- | --- | --- |
| `utils/abnormal_crop.py` | Takes already-clipped-and-classified short videos, crops the target region with YOLO, and outputs **per-frame images** | Clipped/classified short-video folders specified by `INPUT_BASE_DIRS` in the script | `train/val` directories under `OUTPUT_BASE_DIR`; each sample is a folder (16 JPG frames + `label.txt`) |
| `utils/abnormal_crop_video.py` | Same as above, but outputs **cropped short video clips** | Same as above (constants configured separately in the script) | `train/val` directories under `OUTPUT_BASE_DIR`; each sample is an mp4 video file stored under a category folder |
| `train/train_finetune_normal_3.py` | Stage 1: fine-tune the teacher model (Swin3D-B) | `config.yaml` → `task_configs.normal` (`dataset_path`, `class_names`, `finetune.*`) | Teacher checkpoint (EMA) under `finetune.output_dir`, filename `finetune.output_model_name` |
| `distill/distill_normal_3.py` | Stage 2: distill the teacher into a lightweight student model (RepViT + GRU) | `config.yaml` → `distillation.teacher_checkpoint` (teacher weights) + `task_configs.normal.dataset_path` (training data, usually the same batch used for the teacher) | Student checkpoints (raw + EMA files) under `task_configs.normal.distillation.output_dir` |

`abnormal_crop.py` and `abnormal_crop_video.py` are **two optional output formats**, unrelated to the cat/dog vs. rabbit species — both do "crop the target region out of a classified long video with YOLO, then slice it into fixed-length clips," just saving as per-frame images vs. short video files respectively. Samples in either format can be read directly by the training scripts. **For newly processed data going forward, prefer `abnormal_crop_video.py` (video-clip format).**

Neither of the two training scripts accepts command-line arguments — run them directly with `python <script path>`; all configuration comes from the root `config.yaml`.

> The repo also has other similarly-named scripts (`train_finetune_ab_5.py`, `distill_abnormal_5.py`, `run_train.py`, `run_distill.py`, etc.) — **the current training pipeline does not use them**, and they are out of scope for this document.

## 2. `dataset/` layout logic

### 2.1 Category systems (fixed per species)

The behavior categories are fixed per species, defined in `pipeline_config.py` → `SPECIES` (order = class index):

| Species | `class_names` (in order) |
| --- | --- |
| Cat/dog (`pet`) | `eat`, `drink`, `other` (3 classes) |
| Rabbit (`rabbit`) | `resting`, `active`, `eating`, `drinking` (4 classes) |

`config.yaml`'s `task_configs.normal.class_names` must match this list exactly (including order) when training the corresponding species. Tools that take a `--species` argument (e.g. `auto_presort_label.py`) resolve the category list from `pipeline_config.py` automatically.

### 2.2 Snapshots by species and date

```
dataset/
  pet/normal/<date>/       # cat/dog training data snapshot, e.g. dataset/pet/normal/2026-07-10
    train/ val/            # or
    videos/train/ videos/val/   # depending on which processing script produced it
  rabbit/<date(-videos)>/  # rabbit training data snapshot, e.g. dataset/rabbit/0703-videos
    train/
    val/
```

Within a given date snapshot, whether `train/val` sits directly under the date folder (per-frame format) or under a `videos/` subdirectory (video-clip format) depends on whether `abnormal_crop.py` or `abnormal_crop_video.py` was used at the time — both naming conventions exist and aren't fully consistent, so check the actual structure before using a given snapshot.

**Each date folder is a complete snapshot after a "merge old + new data" pass, not just that day's incremental additions.** The rule for deciding which one is currently valid is simple: **within the same species, the folder with the latest date is the currently valid data** — earlier date folders are historical versions kept for reference only; don't append new data to them, and don't train against an old date's path.

For how a new snapshot is produced and when old data needs to be merged in, see [SOP Steps 4-5](./SOP.en.md#step-4-run-the-data-processing-script-to-produce-training-samples).

### 2.3 Sample-level format (two physical formats, unrelated to species)

**Per-frame format** (produced by `abnormal_crop.py`): each sample is a folder of frame images + `label.txt`:

```
train/
  0430-eat-CAM-04_..._00.00.18-00.00.27-0/
    frame_0000.jpg ... frame_0015.jpg
    label.txt          # contents are a single integer, e.g. "0"
```

**Video-clip format** (produced by `abnormal_crop_video.py`): video files stored under category folders, no `label.txt`:

```
train/
  active/
    clip_001.mp4
  resting/
  eating/
  drinking/
```

Either format can show up in either species' data — which one you get depends on which script was used to process that batch. Both formats are read by the same data loader, `utils/video_dataset.py::VideoDataset`: **it reads `label.txt` first if present, otherwise it matches the category keyword as a substring against the folder/file name** (case-insensitive). So regardless of format, the sample's folder or file name must contain the category keyword, or it will be silently skipped as unrecognized (no error is raised).

### 2.4 Other directories (unrelated to the current training pipeline — ignore)

`dataset/pet/hierarchical-cls/` and `dataset/pet/abnormal/` are deprecated/unused cat-dog abnormal-behavior (5-class) data, unrelated to the actually-used `train_finetune_normal_3.py` / `distill_normal_3.py` pipeline.

## 3. Training: code involved & how to change parameters

### 3.1 The two training-stage scripts

- **Teacher fine-tuning**: `python train/train_finetune_normal_3.py`
- **Student distillation**: `python distill/distill_normal_3.py`

Cat/dog and rabbit share these two scripts — the only difference is which species' data `config.yaml`'s `task_configs.normal.dataset_path` / `class_names` point to.

Shared code backing these two scripts:

```
training/config.py               ← reads config.yaml, assembles the final training config dict
training/teacher_finetune.py     ← the actual teacher fine-tuning training loop
training/distillation.py         ← the actual distillation training loop
training/distillation_losses.py  ← builds the distillation loss (KD loss, feature-matching loss)
models/teacher.py                ← teacher model definition (SwinTeacher)
models/student_factory.py        ← builds the student model based on config
models/timm_gru_student.py       ← student model architecture (timm backbone + GRU)
dataset/__init__.py              ← get_dataset(), builds a VideoDataset from config
utils/video_dataset.py           ← dataset loading: sample scanning, frame reading, augmentation, class-imbalance truncation
```

### 3.2 Key `config.yaml` fields

```yaml
task_configs:
  normal:
    dataset_path: "dataset/rabbit/0626-videos"   # training data path; change this to switch species/snapshot (the pet/rabbit substring must match the species — see ⚠️ below)
    class_names: ["resting", "active", "eating", "drinking"]   # category list; order = class index

    finetune:                      # Stage 1: teacher fine-tuning hyperparameters
      checkpoint_path: None        # None = start from pretrained/random init; to continue training, fill in the previous teacher checkpoint path
      output_dir: "outputs/rabbit/{date}/teacher_finetuned_3cls_{date}"   # {date} is auto-substituted with today's date
      output_model_name: "rabbit_teacher_{date}_ema-videos.pth"
      batch_size: 4
      accum_steps: 8               # gradient accumulation steps; effective batch = batch_size * accum_steps
      epochs: 20
      lr: 4e-5
      num_workers: 4

    recognizer:                    # student model architecture
      backbone: repvit_m1          # timm backbone name; see the BACKBONE_GRU_HIDDEN table in models/student_factory.py for valid values
      gru_hidden: auto             # auto = looked up automatically from the backbone name
      temporal_window: 16          # frames sampled per sample; must match FRAMES_PER_CLIP in the data processing script
      imgsz: 224
      dropout: 0.2
      augment:                     # data augmentation; lower the probability/magnitude or set to 0 to weaken/disable
        rotate: 15
        fliplr: 0.5
        brightness: 0.15
        reverse: 0.3              # probability of reversing the whole frame sequence (train-only)

    distillation:                  # Stage 2: this task's output paths (learning rate/loss weights come from the top-level distillation block)
      output_dir: "outputs/rabbit/{date}"
      output_model_name: "repvit_best_distilled_{date}.pth"

distillation:                      # shared distillation hyperparameters
  teacher_checkpoint: "outputs/rabbit/2026-07-06/teacher_finetuned_3cls_0706/rabbit_teacher_0706_ema-videos.pth"
  temperature: 4.0                # distillation temperature; higher = softer teacher probability distribution
  loss_weights:
    gt: 1.0      # ground-truth label loss weight
    kd: 4.0      # KD (distillation) loss weight
    feature: 1.0 # intermediate feature-matching loss weight
  learning_rate: 3e-4
  batch_size: 16
  epochs: 15
```

`config.yaml` also has a `task_configs.abnormal` block, corresponding to the currently-unused cat/dog 5-class abnormal-behavior pipeline — ignore it.

### 3.3 Quick-reference: common parameter changes

| What to change | Where |
| --- | --- |
| Training data path | `task_configs.normal.dataset_path` |
| Category list/order | `task_configs.normal.class_names` |
| Teacher fine-tuning batch size / epochs / learning rate | `task_configs.normal.finetune` |
| Distillation batch size / epochs / learning rate | top-level `distillation` block (or overridden by `task_configs.normal.distillation`) |
| Distillation temperature, loss weights | `distillation.temperature` / `distillation.loss_weights` |
| Which teacher checkpoint distillation uses | `distillation.teacher_checkpoint` |
| Data augmentation strength | `task_configs.normal.recognizer.augment` |
| Student model backbone | `task_configs.normal.recognizer.backbone` |
| Where output models are saved, what they're named | `finetune.output_dir` / `output_model_name` (supports the `{date}` placeholder) |
| Frames sampled per sample | `recognizer.temporal_window`, must match `FRAMES_PER_CLIP` in the data processing script |
| Special handling of the "background class" (other/resting) in the loss | Not in `config.yaml` — requires a code change: the constructor args of `HierarchicalPriorityLoss(other_idx=..., ...)` in `train/train_finetune_normal_3.py`; `_resolve_other_idx()` in `distill/distill_normal_3.py` |

⚠️ When switching between cat/dog and rabbit, the `pet` / `rabbit` substrings in `dataset_path`, `output_dir`, and `output_model_name` must all be updated together — not just `class_names` — otherwise training will still run, but the resulting model will be saved under the wrong species directory.

⚠️ The `other_idx` in `train/train_finetune_normal_3.py` is a hardcoded integer (currently `2`) and does not automatically follow the `class_names` order. When switching species or changing the category order, make sure to check which position the "background class" (`other` for cat/dog, `resting` for rabbit) occupies in the new `class_names` — if it's not in that position, this number needs to be changed by hand.

## 4. Output & delivery

- Teacher checkpoint (produced in Step 7): only used to supervise the distillation stage, **not deployed**.
- Student checkpoint (produced in Step 8): two files — raw weights + EMA weights — **deployment uses the EMA version**.
- Once the new student EMA weights are ready, use them to update the deployed system.
- The repo has no ready-to-run inference/export script that goes straight from "checkpoint → deployment format." `utils/export_onnx.py` is the relevant export script — confirm the exact export process separately during handover.
