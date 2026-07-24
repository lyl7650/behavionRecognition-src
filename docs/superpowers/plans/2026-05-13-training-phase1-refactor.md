# Training Phase 1 Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add shared config resolution and unified train/distill entrypoints without rewriting the underlying training loops.

**Architecture:** Introduce a small `training` package that resolves task-specific runtime config from `config.yaml`, including distillation defaults and per-task overrides. New root entrypoints dispatch into the existing `train/` and `distill/` implementations, which are updated to accept resolved config objects instead of hardcoded values.

**Tech Stack:** Python, `argparse`, `yaml`, `unittest`

---

### Task 1: Shared config resolution

**Files:**
- Create: `training/__init__.py`
- Create: `training/config.py`
- Test: `tests/test_training_config.py`

- [ ] **Step 1: Write the failing test**

```python
def test_build_distill_config_merges_global_defaults_and_task_overrides():
    cfg = build_distill_config("abnormal", config_dict={
        "distillation": {
            "temperature": 4.0,
            "alpha": 1.0,
            "loss_weights": {"kd": 8.0},
        },
        "task_configs": {
            "abnormal": {
                "dataset_path": "dataset/abnormal/5_cls",
                "class_names": ["normal", "convulsion", "limp", "sneeze", "vomit"],
                "recognizer": {"temporal_window": 16, "imgsz": 224},
                "distillation": {
                    "temperature": 6.0,
                    "loss_weights": {"feature": 2.0},
                },
            },
        },
    })
    assert cfg["temperature"] == 6.0
    assert cfg["loss_weights"] == {"gt": 1.0, "kd": 8.0, "feature": 2.0}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_training_config -v`
Expected: FAIL with import error for `training.config` or missing `build_distill_config`

- [ ] **Step 3: Write minimal implementation**

```python
def build_distill_config(mode, config_dict=None, config_path=None):
    root_cfg = load_root_config(config_dict=config_dict, config_path=config_path)
    task_cfg = root_cfg["task_configs"][mode]
    weights = {"gt": 1.0, "kd": 8.0, "feature": 1.0}
    # merge global old/new schema, then task overrides
    return {
        "mode": mode,
        "temperature": ...,
        "loss_weights": weights,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_training_config -v`
Expected: PASS

- [ ] **Step 5: Commit**

Skip commit in this workspace because the project is not currently a git repository.

### Task 2: Unified entrypoint dispatch

**Files:**
- Create: `training/entrypoints.py`
- Create: `train.py`
- Create: `distill.py`
- Test: `tests/test_training_entrypoints.py`

- [ ] **Step 1: Write the failing test**

```python
def test_run_train_dispatches_resolved_config_to_mode_module():
    with patch("training.entrypoints.build_train_config", return_value={"mode": "normal"}) as build_cfg:
        fake_module = SimpleNamespace(train=Mock())
        with patch("training.entrypoints.importlib.import_module", return_value=fake_module) as importer:
            run_train("normal")
    build_cfg.assert_called_once()
    importer.assert_called_once_with("train.train_finetune_normal_3")
    fake_module.train.assert_called_once_with(config_override={"mode": "normal"})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_training_entrypoints -v`
Expected: FAIL because `training.entrypoints` does not exist or does not export `run_train`

- [ ] **Step 3: Write minimal implementation**

```python
TRAIN_MODULES = {"normal": "train.train_finetune_normal_3", "abnormal": "train.train_finetune_ab_5"}

def run_train(mode=None, config_path=None):
    resolved = build_train_config(mode=mode, config_path=config_path)
    module = importlib.import_module(TRAIN_MODULES[resolved["mode"]])
    return module.train(config_override=resolved)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_training_entrypoints -v`
Expected: PASS

- [ ] **Step 5: Commit**

Skip commit in this workspace because the project is not currently a git repository.

### Task 3: Adapt existing train scripts

**Files:**
- Modify: `train/train_finetune_normal_3.py`
- Modify: `train/train_finetune_ab_5.py`
- Test: `tests/test_training_entrypoints.py`

- [ ] **Step 1: Write the failing test**

```python
def test_run_train_calls_train_with_config_override():
    fake_module = SimpleNamespace(train=Mock())
    with patch("training.entrypoints.importlib.import_module", return_value=fake_module):
        with patch("training.entrypoints.build_train_config", return_value={"mode": "abnormal", "batch_size": 4}):
            run_train("abnormal")
    fake_module.train.assert_called_once_with(config_override={"mode": "abnormal", "batch_size": 4})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_training_entrypoints -v`
Expected: FAIL if the dispatcher still calls a no-arg `train()`

- [ ] **Step 3: Write minimal implementation**

```python
def train(config_override=None):
    cfg = config_override or build_train_config("abnormal")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_training_entrypoints -v`
Expected: PASS

- [ ] **Step 5: Commit**

Skip commit in this workspace because the project is not currently a git repository.

### Task 4: Adapt existing distill scripts

**Files:**
- Modify: `distill/distill_normal_3.py`
- Modify: `distill/distill_abnormal_5.py`
- Test: `tests/test_training_config.py`

- [ ] **Step 1: Write the failing test**

```python
def test_build_distill_config_preserves_task_specific_output_dir():
    cfg = build_distill_config("normal", config_dict={
        "distillation": {"output_dir": "outputs/default", "temperature": 4.0},
        "task_configs": {
            "normal": {
                "dataset_path": "dataset/rabbit/0507",
                "class_names": ["eating", "moving", "resting", "drinking"],
                "recognizer": {"temporal_window": 16, "imgsz": 224},
                "distillation": {"output_dir": "outputs/distilled_final/rabbit/0508"},
            }
        },
    })
    assert cfg["output_dir"] == "outputs/distilled_final/rabbit/0508"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_training_config -v`
Expected: FAIL because task-specific output_dir is ignored

- [ ] **Step 3: Write minimal implementation**

```python
distill_cfg = deep_merge(global_distill_cfg, task_cfg.get("distillation", {}))
resolved["output_dir"] = sanitize_path(distill_cfg["output_dir"])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_training_config -v`
Expected: PASS

- [ ] **Step 5: Commit**

Skip commit in this workspace because the project is not currently a git repository.

### Task 5: Verify end-to-end entrypoint behavior

**Files:**
- Modify: `train.py`
- Modify: `distill.py`
- Test: `tests/test_training_config.py`
- Test: `tests/test_training_entrypoints.py`

- [ ] **Step 1: Run the focused test suite**

Run: `python -m unittest tests.test_training_config tests.test_training_entrypoints -v`
Expected: PASS

- [ ] **Step 2: Smoke-check the new CLI help**

Run: `python train.py --help`
Expected: exits 0 and shows `--mode` and `--config`

- [ ] **Step 3: Smoke-check the distill CLI help**

Run: `python distill.py --help`
Expected: exits 0 and shows `--mode` and `--config`

- [ ] **Step 4: Review plan coverage**

Confirm phase 1 only introduced shared config resolution and unified entrypoints, without changing model math or dataset semantics.

- [ ] **Step 5: Commit**

Skip commit in this workspace because the project is not currently a git repository.
