## Superpowers

For complex development tasks, follow the relevant skills in
`C:\Users\PPCUSER\.codex\superpowers\skills`.

Use superpowers for feature work, behavior changes, bug fixes, debugging,
multi-file refactors, implementation plans, code review, and completion
verification.

When using superpowers skills for development execution, treat
`superpowers:subagent-driven-development` as required workflow and follow its
current `SKILL.md` process instead of doing the implementation directly in the
main agent, unless the user explicitly instructs otherwise.

Skip superpowers for simple Q&A, code reading, trivial commands, formatting,
small documentation edits, and one-line changes with no behavior impact.

When using a superpowers skill, read its current `SKILL.md`, briefly mention
which skill is being used, and adapt Claude-specific tool names to Codex tools.
Do not require `superpowers:using-superpowers` on every conversation startup.

After modifying project code or performing project-level workflow changes such
as creating a new branch, update this `AGENTS.md` to keep its instructions in
sync with the current state and workflow expectations.

## Model Layout

## Root Layout

- Keep only user-facing numbered `.bat` launchers, `config.yaml`, project rules, and root metadata in the project root.
- Python pipeline entry points and helpers live in `scripts/pipeline/`; pre-sort entry points live in `scripts/presort/`; review package tools live in `scripts/review_tools/`.
- Launch moved Python entry points from the root with `python -m scripts.<group>.<module>` so root-level packages and `config.yaml` remain discoverable.
- `config_editor` saves timestamped `config.yaml` backups in `config/backups/`, not in the project root.

- Current teacher model definition lives in `models/teacher.py`.
- Current student model definition lives in `models/student.py`.
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
