# Handover Doc: Pet Behavior Recognition Model Training

For the end-to-end operating steps, see [`docs/SOP.en.md`](./SOP.en.md). This document is a reference manual: script inventory, how `dataset/` is laid out, how to change training parameters.

> Run `conda activate AR` to activate the environment before running any script in this document.

## 1. Recommended operating entrypoints (input → output)

Run the numbered launchers from the project root. They activate the `AR` environment and call the matching module; `5_run_test.bat` evaluates an existing checkpoint rather than creating a training snapshot.

| Launcher | Module | Purpose |
| --- | --- | --- |
| `1_scaffold_dirs.bat` | `scripts.pipeline.scaffold_dirs` | Create the raw-video and working-directory structure for a species/site/date batch. |
| `2a_presort_label.bat` → `2c_presort_apply.bat` | `scripts.presort.run_presort_label` → `run_presort_apply` | Pre-label long videos, review them in the browser, then write approved clips into `sorted-<date>/`. |
| `3_run_crop.bat` | `scripts.pipeline.run_crop` | YOLO-crop approved clips into a new training snapshot. Choose video clips (default) or per-frame samples. |
| `4_run_train_distill.bat` | `scripts.pipeline.run_train_distill` | Merge the previous snapshot, print a health report, update `config.yaml`, fine-tune the teacher, then distill the student. |
| `5_run_test.bat` | `scripts.pipeline.run_test --interactive` | Offline-evaluate a selected teacher or student checkpoint. |

`run_crop` configures `utils/abnormal_crop.py` (per-frame format) or `utils/abnormal_crop_video.py` (video-clip format) from the chosen species, site, date, and format. These crop scripts are pipeline internals, not the recommended manual entrypoints. Video clips are the default format for new data.

The thin task entries `train/train_finetune_normal_3.py` and `distill/distill_normal_3.py` remain usable for focused training work, but `run_train_distill` is the normal end-to-end entrypoint. It updates `config.yaml` and its teacher checkpoint automatically after the health-check confirmation.

## 2. `dataset/` layout logic

### 2.1 Category systems (fixed per species)

The behavior categories are fixed per species, defined in `pipeline_config.py` → `SPECIES` (order = class index):

| Species | `class_names` (in order) |
| --- | --- |
| Cat/dog (`pet`) | `eat`, `drink`, `active`, `resting` (4 classes) |
| Rabbit (`rabbit`) | `resting`, `active`, `eating`, `drinking` (4 classes) |

`config.yaml`'s `task_configs.normal.class_names` must match this list exactly (including order) when training the corresponding species. Tools that take a `--species` argument (e.g. `auto_presort_label.py`) resolve the category list from `pipeline_config.py` automatically.

### 2.2 Snapshots by species and date

```
dataset/
  pet/normal/<date>/       # cat/dog training data snapshot, e.g. dataset/pet/normal/2026-07-10
    videos/train/ videos/val/   # video-clip format (default)
    # or train/ val/            # per-frame format
  rabbit/<YYYY-MM-DD>/
    videos/train/ videos/val/   # video-clip format (default)
    # or train/ val/            # per-frame format
```

Within a date snapshot, `train/val` directly under the date folder means per-frame format; `videos/train` and `videos/val` mean video-clip format. The crop and training launchers must use the same format.

**Each date folder becomes a complete snapshot after the merge-old-plus-new pass, not just that day's incremental additions.** `run_train_distill` uses the date selected for the current batch and merges the latest older ISO-date snapshot. If it finds older directory names it cannot parse, it shows the auto-selected directory and requires typing `yes` before merging.

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
models/student.py                ← current student model definition
models/student_factory.py        ← builds the student model based on config
models/timm_gru_student.py       ← timm backbone + GRU implementation used by the factory
dataset/__init__.py              ← get_dataset(), builds a VideoDataset from config
utils/video_dataset.py           ← dataset loading: sample scanning, frame reading, augmentation, class-imbalance truncation
```

### 3.2 Key `config.yaml` fields

```yaml
task_configs:
  normal:
    dataset_path: "dataset/pet/normal/2026-07-23/videos"   # example video-clip snapshot; choose the current species/date batch
    class_names: ["eat", "drink", "active", "resting"]   # category list; order = class index

    finetune:                      # Stage 1: teacher fine-tuning hyperparameters
      checkpoint_path: None        # None = start from pretrained/random init; to continue training, fill in the previous teacher checkpoint path
      output_dir: "outputs/pet/normal/{date}/teacher_finetuned_{date}"   # {date} is auto-substituted with today's date
      output_model_name: "pet_teacher_{date}_ema-videos.pth"
      batch_size: 4
      accum_steps: 8               # gradient accumulation steps; effective batch = batch_size * accum_steps
      epochs: 20
      lr: 4e-5
      num_workers: 4

    recognizer:                    # student model architecture
      backbone: repvit_m1          # timm backbone name; see models/student_factory.py for valid values
      gru_hidden: auto             # auto = looked up automatically from the backbone name
      temporal_window: 16          # frames sampled per sample; must match FRAMES_PER_CLIP in the data processing script
      imgsz: 224
      dropout: 0.2
      augment:                     # data augmentation; lower the probability/magnitude or set to 0 to weaken/disable
        scale_min: 0.8
        scale_max: 1.0
        rotate: 15
        translate: 0.1
        fliplr: 0.5
        brightness: 0.15
        contrast: 0.15
        reverse: 0.3              # probability of reversing the whole frame sequence (train-only)

    distillation:                  # Stage 2: this task's output paths (learning rate/loss weights come from the top-level distillation block)
      output_dir: "outputs/pet/{date}"
      output_model_name: "repvit_best_distilled_{date}.pth"

distillation:                      # shared distillation hyperparameters
  teacher_checkpoint: "outputs/pet/normal/<date>/teacher_finetuned_<date>/pet_teacher_<date>_ema-videos.pth"
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
| Where output models are saved, what they're named | Teacher: `task_configs.normal.finetune.output_dir` / `output_model_name`; student: `task_configs.normal.distillation.output_dir` / `output_model_name` (supports `{date}`) |
| Frames sampled per sample | `recognizer.temporal_window`, must match `FRAMES_PER_CLIP` in the data processing script |
| Priority-loss anchor class | Teacher training resolves `resting` from `class_names` only when `loss_type: hierarchical` is used. Student distillation always uses this priority loss, so its normal-behavior labels must include `resting`. |

`run_train_distill` updates the normal dataset path, class names, species-specific output paths, and teacher checkpoint together. For manual changes, keep the species token in those paths consistent with `class_names`.

### 3.4 How training configuration actually takes effect

#### Teacher fine-tuning

`task_configs.normal.finetune` controls the teacher. `batch_size` is the number of samples loaded at once; `accum_steps` accumulates that many small batches before one optimizer update, so the effective batch size is roughly `batch_size × accum_steps`. This is the usual first adjustment when GPU memory is insufficient.

The teacher uses AdamW, cosine learning-rate decay, gradient clipping, and EMA (an averaged copy of the model used for validation and saving). It saves only the best EMA teacher checkpoint, selected first by validation accuracy and then by lower validation loss. Training stops after seven consecutive epochs without improvement. Teacher fine-tuning calls `.cuda()` directly, so it requires a CUDA-capable PyTorch environment.

`loss_type: ce` is the default and uses ordinary cross-entropy. `loss_type: hierarchical` changes the teacher loss to first distinguish `resting` from all other labels, then distinguish the remaining labels; it fails clearly if `resting` is absent from `class_names`.

#### Student distillation

For distillation, a setting in `task_configs.normal.distillation` overrides the same setting in top-level `distillation`; if neither is present, the code default is used. This applies to the teacher checkpoint, output paths, temperature, loss weights, learning rate, batch size, epochs, and worker count. Put a batch-specific override in the task block only when that difference is intentional.

The total student loss has three parts: `gt` keeps the student aligned with the manual labels; `kd` makes it imitate the teacher's predicted probabilities; `feature` compares intermediate features. `temperature` softens both models' probability distributions for the KD calculation. The feature-loss weight gradually decreases to half of its configured value over the run. The student saves raw and EMA checkpoints when validation accuracy reaches a new best value; deployment normally uses the EMA file.

The distillation code always builds a teacher with `num_classes = len(class_names)`. Use a teacher checkpoint made for the same species and the same class order. The checkpoint loader is non-strict, so a mismatch may load partially rather than fail immediately; confirm the printed class list and checkpoint path before the run proceeds.

#### Student architecture and data settings

`recognizer.backbone` is the current student-model selector; it takes precedence over the older compatibility field `timm_model`. Most values create a timm backbone plus GRU. `movinet_a2`, `movinet_a3`, `movinet_a4`, and `movinet_a5` create a 3D MoViNet student instead; for those models, feature loss is disabled because there is no compatible 2D backbone feature hook.

For timm + GRU students, `gru_hidden: auto` selects the hidden size from `models/student_factory.py`; set it manually only when deliberately testing an architecture. `dropout` is passed to the student classifier. Changing the backbone or class order requires a newly trained student checkpoint rather than reusing an old one.

The main training path reads `imgsz`, `temporal_window`, and the `augment` fields `scale_min`, `scale_max`, `rotate`, `translate`, `fliplr`, `brightness`, `contrast`, and `reverse`. Augmentation is train-only and is applied consistently to the whole clip; `reverse` reverses the full temporal sequence with its configured probability. `temporal_window` must continue to match the clip length produced by the crop pipeline.

The current standard student factory does not use `head`, `use_tsm`, `lstm_hidden`, `lstm_layers`, `mamba_d_state`, `mamba_d_conv`, or `mamba_expand` from the normal recognizer block. Treat them as inactive compatibility/configuration residue, not as knobs for the regular pipeline.

## 4. Output & delivery

- Teacher checkpoint (produced in Step 7): only used to supervise the distillation stage, **not deployed**.
- Student checkpoint (produced in Step 8): two files — raw weights + EMA weights — **deployment uses the EMA version**.
- Once the new student EMA weights are ready, use them to update the deployed system.
- `5_run_test.bat` provides ready-to-run offline inference and evaluation for teacher or student checkpoints. It YOLO-crops the animal first and writes traces, JSON/CSV summaries, and timeline PNGs beside the selected checkpoint.
- Deployment-format export remains separate: `utils/export_onnx.py` is the relevant script, and its exact deployment contract should be confirmed during handover.
