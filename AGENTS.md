# 

## Model Layout

## Root Layout

- Keep only user-facing numbered `.bat` launchers, `config.yaml`, project rules, and root metadata in the project root.

- Python pipeline entry points and helpers live in `scripts/pipeline/`; pre-sort entry points live in `scripts/presort/`; review package tools live in `scripts/review_tools/`.

- Launch moved Python entry points from the root with `python -m scripts.<group>.<module>` so root-level packages and `config.yaml` remain discoverable.

- `config_editor` saves timestamped `config.yaml` backups in `config/backups/`, not in the project root.

- Current teacher model definition lives in `models/teacher.py`.

- Current student model definition lives in `models/student.py`.

- RKNN quantization-friendly copies are opt-in only: use
  `models/timm_gru_student_rknn.py`, `models/student_factory_rknn.py`,
  `training/distillation_rknn.py`, and `distill/distill_normal_3_rknn.py`.
  They add feature LayerNorm, reparameterization validation, and post-fold
  fine-tuning without changing the standard student or distillation path.

- The matching fresh-teacher BN-gamma-regularized copy is
  `train/train_finetune_normal_3_rknn.py` with
  `training/teacher_finetune_rknn.py`; shared RKNN options are read by
  `training/config_rknn.py` from optional `rknn` mapping fields. Its entry
  script supports both `python -m train.train_finetune_normal_3_rknn` and
  direct `python train/train_finetune_normal_3_rknn.py` execution.

- `distill/swin_wrapper.py` and `recognizers/gru_model.py` are compatibility
  shims and should only forward exports unless a migration explicitly changes
  that policy.

- Shared teacher finetune loop now lives in `training/teacher_finetune.py`.

- Shared distillation loop now lives in `training/distillation.py`.

- Shared distillation loss builders now live in `training/distillation_losses.py`.

- `train/train_finetune_normal_3.py`, `train/train_finetune_ab_5.py`,
  `distill/distill_normal_3.py`, and `distill/distill_abnormal_5.py` are task
  entry scripts and should stay thin.

- Finetune and distillation model output filenames should be configured through
  `config.yaml` (`finetune.output_model_name` and `distillation.output_model_name`)
  with `{date}` expansion handled by `training/config.py`.

- Clip-level training augmentation is implemented in `utils/video_dataset.py`
  with `torchvision.transforms.v2`; keep random transforms applied after frames
  are stacked into one `[T, C, H, W]` tensor, not independently per frame.
  Temporal reversal uses the dataset-local `augment.reverse` probability and
  must remain train-only.

- Offline evaluation is launched by `5_run_test.bat` / `scripts.pipeline.run_test`. It always
  YOLO-crops the animal first (expand_ratio 1.25 via
  `utils.abnormal_crop_video.build_smoothed_crop_boxes`) before behavior inference:
  rabbit uses `detection-src/rabbit-detect/.../rabbit-yolo11n-0603.pt`, pet uses
  `detection-src/cat_dog_detect/.../yolo11n-ssb-0710.pt`. Keep its output compatible
  with `utils/metrics.py`; it saves traces, JSON/CSV summaries, and GT-vs-system
  timeline PNGs beside the selected checkpoint.

- Interactive binary prompts (species, teacher/student, crop/train clip-vs-frame)
  use numbered 1/2 menus in `pipeline_config._ask_choice`, `run_test._interactive_args`,
  and the matching `.bat` files. Free-text fields (site, date, checkpoint path) stay typed.

- Pet normal behavior uses exactly four labels in this order: `eat`, `drink`,
  `active`, `resting`. New data, checkpoints, and offline evaluation must not
  use an `other` class. The former `other` re-split launchers/tools were retired
  after the historical data migration.
