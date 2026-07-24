# Output Model Name Config Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make finetune and distillation model output filenames come from `config.yaml` instead of hardcoded constants, while preserving `{date}` expansion.

**Architecture:** Extend shared config resolution so distillation output filenames are produced in one place, then make the thin entry scripts pass those resolved names through unchanged. Keep the training loops intact and verify the change with focused config and entrypoint tests.

**Tech Stack:** Python, `unittest`, YAML config loading

---

### Task 1: Extend shared config resolution for distillation output filenames

**Files:**
- Modify: `training/config.py`
- Test: `tests/test_training_config.py`

- [ ] **Step 1: Add failing config expectations**

Add assertions in `tests/test_training_config.py` for:
- `build_distill_config()` returning `output_model_name`
- automatic derivation of `output_ema_model_name`
- `{date}` replacement in the resolved names

- [ ] **Step 2: Implement minimal shared config support**

Update `training/config.py` so `build_distill_config()` resolves:
- task-specific or global `distillation.output_model_name`
- fallback defaults per mode
- derived EMA filename by inserting `_ema` before the extension

- [ ] **Step 3: Run focused config tests**

Run:
```bash
python -m pytest tests/test_training_config.py -q
```

Expected: config tests pass with the new output name fields.

### Task 2: Remove hardcoded filenames from thin entry scripts

**Files:**
- Modify: `train/train_finetune_normal_3.py`
- Modify: `distill/distill_normal_3.py`
- Test: `tests/test_training_entrypoints.py`

- [ ] **Step 1: Add failing delegation expectations**

Update `tests/test_training_entrypoints.py` so the thin-script tests expect:
- normal finetune script passes `cfg["output_model_name"]`
- normal distill script passes runtime config derived names

- [ ] **Step 2: Implement minimal script changes**

Update the scripts to:
- remove local hardcoded save-name constants
- pass resolved config fields through to the shared trainers

- [ ] **Step 3: Run focused thin-script tests**

Run:
```bash
python -m pytest tests/test_training_entrypoints.py -q
```

Expected: entrypoint delegation tests pass without hardcoded filename constants.

### Task 3: Wire config.yaml and sync project instructions

**Files:**
- Modify: `config.yaml`
- Modify: `AGENTS.md`

- [ ] **Step 1: Add config source-of-truth fields**

Set `task_configs.*.distillation.output_model_name` in `config.yaml`, using `{date}` where appropriate for dated outputs.

- [ ] **Step 2: Sync AGENTS guidance**

Update `AGENTS.md` to note that distillation output filenames are now configured through `config.yaml` alongside finetune output filenames.

- [ ] **Step 3: Run final targeted verification**

Run:
```bash
python -m pytest tests/test_training_config.py tests/test_training_entrypoints.py -q
```

Expected: all targeted tests pass and confirm output filenames are configuration-driven.
