# Output Model Name Config Design

## Goal

Remove hardcoded model save filenames from the normal finetune and distillation entry scripts. Both flows should resolve output filenames from `config.yaml`, with `{date}` expansion preserved.

## Scope

- `train/train_finetune_normal_3.py`
- `distill/distill_normal_3.py`
- shared config loader in `training/config.py`
- `config.yaml`
- focused tests covering config resolution and thin entrypoint delegation

## Approach

### Finetune

`build_train_config()` already resolves `finetune.output_model_name`, including `{date}` replacement. The normal finetune entry script should stop using its local constant and pass `cfg["output_model_name"]` into `run_teacher_finetune()`.

### Distillation

`build_distill_config()` should start resolving:

- `distillation.output_model_name`
- derived `distillation.output_ema_model_name`

The source of truth remains `config.yaml`. The EMA filename is derived automatically from the main filename by inserting `_ema` before the extension. `{date}` replacement should match the existing finetune behavior.

### Compatibility

- Keep `DEFAULT_MODE` in thin scripts; it still defines the script's default task selection.
- Preserve support for task-specific distillation config overriding root-level distillation config.
- Do not change training loop behavior or checkpoint save conditions.

## Validation

- Unit test config resolution for distillation output names and `{date}` expansion.
- Update thin-script tests to verify the scripts pass resolved config filenames instead of hardcoded constants.
