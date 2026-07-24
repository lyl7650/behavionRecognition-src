# 行为识别流水线自动化 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把行为识别流水线（crop → 合并快照 → teacher 训练 → student 蒸馏）收敛成一个脚手架 + 两个连续脚本，只交互输入"物种/机构/日期"，其余靠物种查表+路径推导自动得出。

**Architecture:** 新增 3 个内部模块（`pipeline_config` 物种查表与路径推导、`config_editor` 改写 config.yaml、`snapshot_merge` 合并旧快照与体检）+ 3 个入口（`scaffold_dirs`/`run_crop`/`run_train_distill`）+ 1 份文档。改造 2 个 crop 脚本为可传 config。训练/蒸馏脚本不改（已支持 config_override/config_path）。

**Tech Stack:** Python 3、PyYAML、pytest（项目现有 `tests/` + monkeypatch 惯例）、ultralytics YOLO、conda 环境 `AR`。

**重要约束（来自 spec 与两轮审查）：**
- 项目**非 git 仓库**，所以本计划**不含 git commit 步骤**；每个任务以"跑通测试"作为完成标志。
- 写 config.yaml：反斜杠路径**必须用单引号/正斜杠**，禁双引号（`\n` 陷阱）；改前**时间戳备份 + 原子替换**。
- 脚本B **必须先改写 config.yaml 再 `build_train_config()`**（builder 从磁盘读，反了会用旧值训练）。
- teacher 回填路径 = `train_cfg["output_dir"]/train_cfg["output_model_name"]`（**不套** `resolve_ema_model_name`）。
- crop 改造：运行配置**显式传给辅助函数**，不改模块全局。

**运行测试：** 项目根 `E:\ActionRecognition\behavionRecognition-src` 下 `conda activate AR && python -m pytest tests/<file> -v`。

---

## 文件结构

| 文件 | 职责 |
|------|------|
| `pipeline_config.py` [新增] | 物种查表 `SPECIES` + 路径推导函数 + `ask_context()` 交互问答 |
| `config_editor.py` [新增] | 读/改写 `config.yaml`：改字段、规整 `r"..."`、时间戳备份、原子替换、子串替换断言 |
| `snapshot_merge.py` [新增] | 找最新旧快照、合并到新快照、生成体检报告（跨 split 泄漏 / 静默跳过 / label.txt 比对 / 双层布局警示） |
| `scaffold_dirs.py` [新增·入口①] | 建 records/<date> + sorted-<date>/<各类别> 空目录 |
| `run_crop.py` [新增·入口②] | 校验 sorted → 调 crop 脚本 run(config) → 打印产物 |
| `run_train_distill.py` [新增·入口③] | 合并 → 体检闸门 → 改 config → teacher → 回填 → student |
| `docs/AUTOMATION.md` [新增] | 三入口使用说明 + 日期约定 + 人工节点 |
| `utils/abnormal_crop_video.py` [改造] | 加 `run(config)`，辅助函数收 cfg 参数；`main()` 用默认 cfg 保持单跑 |
| `utils/abnormal_crop.py` [改造] | 同上 |
| `tests/test_pipeline_config.py` 等 [新增] | 各模块单测 |

**实现顺序**：先地基（Task 1-3 pipeline_config），再无依赖的内部模块（Task 4-5 config_editor、Task 6-8 snapshot_merge），再入口（Task 9 scaffold、Task 11-12 crop 改造 + Task 13 run_crop、Task 14 run_train_distill），最后文档（Task 15）。

---

## Task 1: pipeline_config —— 物种查表 SPECIES

**Files:**
- Create: `pipeline_config.py`
- Test: `tests/test_pipeline_config.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_pipeline_config.py
import pipeline_config as pc


def test_species_table_has_both_species():
    assert set(pc.SPECIES.keys()) == {"pet", "rabbit"}


def test_pet_species_fields():
    pet = pc.SPECIES["pet"]
    assert pet["class_names"] == ["eat", "drink", "other"]
    assert pet["label_map"] == {"eat": 0, "drink": 1, "other": 2}
    assert pet["target_classes"] == [1, 2]
    assert pet["path_token"] == "pet"
    assert pet["raw_root"] == r"dataset\raw_videos\pet\normal"
    assert pet["dataset_root"] == r"dataset\pet\normal"
    assert pet["yolo_model"].endswith("yolo11n-ssb-0710.pt")


def test_rabbit_species_fields():
    rb = pc.SPECIES["rabbit"]
    assert rb["class_names"] == ["resting", "active", "eating", "drinking"]
    assert rb["label_map"] == {"resting": 0, "active": 1, "eating": 2, "drinking": 3}
    assert rb["target_classes"] == [0]
    assert rb["path_token"] == "rabbit"
    assert rb["raw_root"] == r"dataset\raw_videos\rabbit"
    assert rb["dataset_root"] == r"dataset\rabbit"
    assert rb["yolo_model"].endswith("rabbit-yolo11n-0603.pt")
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_pipeline_config.py -v`
Expected: FAIL —— `ModuleNotFoundError: No module named 'pipeline_config'`

- [ ] **Step 3: 写最小实现**

```python
# pipeline_config.py
SPECIES = {
    "pet": {
        "class_names": ["eat", "drink", "other"],
        "label_map": {"eat": 0, "drink": 1, "other": 2},
        "yolo_model": r"E:\ActionRecognition\detection-src\cat_dog_detect\runs\yolo11n_scale0.6_fliplr0.42\weights\yolo11n-ssb-0710.pt",
        "target_classes": [1, 2],
        "raw_root": r"dataset\raw_videos\pet\normal",
        "dataset_root": r"dataset\pet\normal",
        "path_token": "pet",
    },
    "rabbit": {
        "class_names": ["resting", "active", "eating", "drinking"],
        "label_map": {"resting": 0, "active": 1, "eating": 2, "drinking": 3},
        "yolo_model": r"E:\ActionRecognition\detection-src\rabbit-detect\runs\rabbit-yolo11n-0603\weights\rabbit-yolo11n-0603.pt",
        "target_classes": [0],
        "raw_root": r"dataset\raw_videos\rabbit",
        "dataset_root": r"dataset\rabbit",
        "path_token": "rabbit",
    },
}
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_pipeline_config.py -v`
Expected: PASS（3 passed）

---

## Task 2: pipeline_config —— 路径推导

**Files:**
- Modify: `pipeline_config.py`
- Test: `tests/test_pipeline_config.py`

路径约定（用 `os.path.join`，返回相对 `dataset_root`/`raw_root` 的路径）：
- records：`<raw_root>\<site>-record\records\<date>`
- sorted：`<raw_root>\<site>-record\sorted-<date>`（各类别子目录 = `sorted-<date>\<class>`）
- 快照（视频格式，默认）：`<dataset_root>\<date>\videos`
- 快照（帧格式）：`<dataset_root>\<date>`

- [ ] **Step 1: 写失败测试**

```python
# 追加到 tests/test_pipeline_config.py
import os


def test_records_dir():
    got = pc.records_dir("rabbit", "hkbr", "2026-07-15")
    assert got == os.path.join(r"dataset\raw_videos\rabbit", "hkbr-record", "records", "2026-07-15")


def test_sorted_dir_and_class_dirs():
    base = pc.sorted_dir("pet", "hkbr", "2026-07-15")
    assert base == os.path.join(r"dataset\raw_videos\pet\normal", "hkbr-record", "sorted-2026-07-15")
    class_dirs = pc.sorted_class_dirs("pet", "hkbr", "2026-07-15")
    assert class_dirs == [os.path.join(base, c) for c in ["eat", "drink", "other"]]


def test_snapshot_dir_video_default_includes_videos_layer():
    got = pc.snapshot_dir("rabbit", "2026-07-15", frames=False)
    assert got == os.path.join(r"dataset\rabbit", "2026-07-15", "videos")


def test_snapshot_dir_frames_has_no_videos_layer():
    got = pc.snapshot_dir("rabbit", "2026-07-15", frames=True)
    assert got == os.path.join(r"dataset\rabbit", "2026-07-15")
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_pipeline_config.py -k "records or sorted or snapshot" -v`
Expected: FAIL —— `AttributeError: module 'pipeline_config' has no attribute 'records_dir'`

- [ ] **Step 3: 写最小实现**

```python
# 追加到 pipeline_config.py 顶部
import os


def records_dir(species, site, date):
    root = SPECIES[species]["raw_root"]
    return os.path.join(root, f"{site}-record", "records", date)


def sorted_dir(species, site, date):
    root = SPECIES[species]["raw_root"]
    return os.path.join(root, f"{site}-record", f"sorted-{date}")


def sorted_class_dirs(species, site, date):
    base = sorted_dir(species, site, date)
    return [os.path.join(base, c) for c in SPECIES[species]["class_names"]]


def snapshot_dir(species, date, frames=False):
    root = SPECIES[species]["dataset_root"]
    if frames:
        return os.path.join(root, date)
    return os.path.join(root, date, "videos")
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_pipeline_config.py -v`
Expected: PASS（全部）

---

## Task 3: pipeline_config —— ask_context 交互问答

**Files:**
- Modify: `pipeline_config.py`
- Test: `tests/test_pipeline_config.py`

`ask_context(input_fn=input, today_fn=...)` 依赖注入以便测试。返回 dict `{species, site, date}`。日期默认今天（`YYYY-MM-DD`），语义=录制日。物种只接受 pet/rabbit，非法重问。

- [ ] **Step 1: 写失败测试**

```python
# 追加到 tests/test_pipeline_config.py
def test_ask_context_collects_three_fields():
    answers = iter(["rabbit", "hkbr", "2026-07-14"])
    ctx = pc.ask_context(input_fn=lambda prompt: next(answers), today_fn=lambda: "2026-07-15")
    assert ctx == {"species": "rabbit", "site": "hkbr", "date": "2026-07-14"}


def test_ask_context_date_defaults_to_today_when_blank():
    answers = iter(["pet", "fotan", ""])
    ctx = pc.ask_context(input_fn=lambda prompt: next(answers), today_fn=lambda: "2026-07-15")
    assert ctx["date"] == "2026-07-15"


def test_ask_context_rejects_bad_species_then_retries():
    answers = iter(["dog", "rabbit", "hkbr", ""])
    ctx = pc.ask_context(input_fn=lambda prompt: next(answers), today_fn=lambda: "2026-07-15")
    assert ctx["species"] == "rabbit"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_pipeline_config.py -k ask_context -v`
Expected: FAIL —— `AttributeError: ... 'ask_context'`

- [ ] **Step 3: 写最小实现**

```python
# 追加到 pipeline_config.py
from datetime import datetime


def _today_str():
    return datetime.now().strftime("%Y-%m-%d")


def ask_context(input_fn=input, today_fn=_today_str):
    while True:
        species = input_fn("物种 (pet/rabbit): ").strip()
        if species in SPECIES:
            break
        print(f"未知物种 '{species}'，只能是 pet 或 rabbit，请重输。")
    site = input_fn("机构 (如 hkbr / fotan): ").strip()
    default_date = today_fn()
    date = input_fn(f"录制日期 [默认 {default_date}]: ").strip() or default_date
    return {"species": species, "site": site, "date": date}
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_pipeline_config.py -v`
Expected: PASS（全部）

---

## Task 4: config_editor —— 路径规整 sanitize + YAML 安全引号

**Files:**
- Create: `config_editor.py`
- Test: `tests/test_config_editor.py`

先做两个纯函数：`clean_path`（剥 `r"..."` 前缀、规范化）和 `yaml_safe_path`（把反斜杠路径转成 YAML 安全形式——正斜杠）。避免双引号 `\n` 陷阱。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_config_editor.py
import config_editor as ce


def test_clean_path_strips_raw_prefix():
    assert ce.clean_path('r"E:\\ActionRecognition\\x\\videos"') == r"E:\ActionRecognition\x\videos"


def test_clean_path_passes_through_plain():
    assert ce.clean_path(r"dataset\rabbit\2026-07-15\videos") == r"dataset\rabbit\2026-07-15\videos"


def test_yaml_safe_path_uses_forward_slashes():
    # 防止 "E:\new" 里的 \n 被 YAML 解析成换行
    assert ce.yaml_safe_path(r"E:\new\videos") == "E:/new/videos"


def test_yaml_safe_path_roundtrips_through_yaml_and_sanitize():
    import yaml, os
    from training.config import sanitize_path
    safe = ce.yaml_safe_path(r"E:\new\videos")
    dumped = yaml.safe_dump({"p": safe})
    loaded = yaml.safe_load(dumped)["p"]
    assert sanitize_path(loaded) == os.path.normpath(r"E:\new\videos")
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_config_editor.py -v`
Expected: FAIL —— `ModuleNotFoundError: No module named 'config_editor'`

- [ ] **Step 3: 写最小实现**

```python
# config_editor.py
import os


def clean_path(value):
    """剥掉 r"..." / '...' 包装并规范化，等价于 training.config.sanitize_path 的清洗部分。"""
    if value is None:
        return value
    text = str(value).strip()
    if len(text) >= 3 and text[0] in ("r", "R") and text[1] in ("'", '"'):
        text = text[1:].strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in ("'", '"'):
        text = text[1:-1]
    return os.path.normpath(text)


def yaml_safe_path(path):
    """把 Windows 反斜杠路径转成 YAML 安全的正斜杠形式，避免双引号 \\n 陷阱。"""
    return clean_path(path).replace("\\", "/")
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_config_editor.py -v`
Expected: PASS（4 passed）

---

## Task 5: config_editor —— 改写 config.yaml（备份 + 原子替换 + 子串替换断言）

**Files:**
- Modify: `config_editor.py`
- Test: `tests/test_config_editor.py`

`apply_species_config(config_path, species, dataset_path, backup_ts, now_fn)`：
1. 读 YAML。
2. `task_configs.normal.dataset_path` = `yaml_safe_path(dataset_path)`；`class_names` = 物种类别。
3. `finetune.output_dir` / `output_model_name` / `task_configs.normal.distillation.output_dir` 里把另一物种 token 替换成本物种 token（pet↔rabbit）。
4. 替换后**断言**这三个字段确含目标 token，否则 raise。
5. `loss_type`、`other_idx`、`{date}`、超参一律不碰。
6. 时间戳备份 `config.yaml.<YYYYMMDD-HHMM>.bak` → 写临时文件 → `os.replace` 原子替换。

- [ ] **Step 1: 写失败测试**

```python
# 追加到 tests/test_config_editor.py
import os
import yaml
import textwrap
import pipeline_config as pc


def _write_sample_config(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(textwrap.dedent('''
    task_configs:
      normal:
        dataset_path: r"E:\\old\\pet\\videos"
        class_names: ['eat','drink','other']
        finetune:
          loss_type: "ce"
          output_dir: "outputs/pet/{date}/teacher_finetuned_{date}"
          output_model_name: "pet_teacher_{date}_ema-videos.pth"
        distillation:
          output_dir: "outputs/pet/{date}"
    distillation:
      teacher_checkpoint: r"outputs\\pet\\old.pth"
    ''').lstrip(), encoding="utf-8")
    return cfg


def test_apply_switches_species_and_dataset(tmp_path):
    cfg = _write_sample_config(tmp_path)
    ce.apply_species_config(str(cfg), "rabbit", r"E:\new\rabbit\2026-07-15\videos",
                            now_fn=lambda: "20260715-2130")
    data = yaml.safe_load(cfg.read_text(encoding="utf-8"))
    assert data["task_configs"]["normal"]["class_names"] == ["resting", "active", "eating", "drinking"]
    assert "rabbit" in data["task_configs"]["normal"]["finetune"]["output_dir"]
    assert "rabbit" in data["task_configs"]["normal"]["finetune"]["output_model_name"]
    assert "rabbit" in data["task_configs"]["normal"]["distillation"]["output_dir"]
    assert "pet" not in data["task_configs"]["normal"]["finetune"]["output_dir"]
    # loss_type 未被碰
    assert data["task_configs"]["normal"]["finetune"]["loss_type"] == "ce"


def test_apply_writes_timestamped_backup(tmp_path):
    cfg = _write_sample_config(tmp_path)
    ce.apply_species_config(str(cfg), "rabbit", r"E:\new\rabbit\2026-07-15\videos",
                            now_fn=lambda: "20260715-2130")
    assert (tmp_path / "config.yaml.20260715-2130.bak").is_file()


def test_apply_dataset_path_yaml_safe_and_parsable(tmp_path):
    cfg = _write_sample_config(tmp_path)
    ce.apply_species_config(str(cfg), "rabbit", r"E:\new\rabbit\2026-07-15\videos",
                            now_fn=lambda: "20260715-2130")
    from training.config import sanitize_path
    data = yaml.safe_load(cfg.read_text(encoding="utf-8"))
    got = sanitize_path(data["task_configs"]["normal"]["dataset_path"])
    assert got == os.path.normpath(r"E:\new\rabbit\2026-07-15\videos")


def test_apply_raises_if_token_missing(tmp_path):
    # output_dir 里既无 pet 也无 rabbit → 替换无处落地，应报错
    cfg = tmp_path / "config.yaml"
    cfg.write_text(textwrap.dedent('''
    task_configs:
      normal:
        dataset_path: "x"
        class_names: ['eat']
        finetune:
          output_dir: "outputs/UNKNOWN/{date}"
          output_model_name: "UNKNOWN_{date}.pth"
        distillation:
          output_dir: "outputs/UNKNOWN/{date}"
    distillation:
      teacher_checkpoint: "y"
    ''').lstrip(), encoding="utf-8")
    import pytest
    with pytest.raises(ValueError):
        ce.apply_species_config(str(cfg), "rabbit", r"E:\a\b",
                                now_fn=lambda: "20260715-2130")
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_config_editor.py -k apply -v`
Expected: FAIL —— `AttributeError: ... 'apply_species_config'`

- [ ] **Step 3: 写最小实现**

```python
# 追加到 config_editor.py
import shutil
import tempfile
from datetime import datetime

import yaml

import pipeline_config as pc

_TOKENS = ("pet", "rabbit")


def _now_str():
    return datetime.now().strftime("%Y%m%d-%H%M")


def _swap_token(text, target_token):
    other = "rabbit" if target_token == "pet" else "pet"
    return text.replace(other, target_token)


def apply_species_config(config_path, species, dataset_path, now_fn=_now_str):
    with open(config_path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    normal = data["task_configs"]["normal"]
    normal["dataset_path"] = yaml_safe_path(dataset_path)
    normal["class_names"] = list(pc.SPECIES[species]["class_names"])
    token = pc.SPECIES[species]["path_token"]

    ft = normal["finetune"]
    ft["output_dir"] = _swap_token(ft["output_dir"], token)
    ft["output_model_name"] = _swap_token(ft["output_model_name"], token)
    normal["distillation"]["output_dir"] = _swap_token(normal["distillation"]["output_dir"], token)

    for label, val in [
        ("finetune.output_dir", ft["output_dir"]),
        ("finetune.output_model_name", ft["output_model_name"]),
        ("distillation.output_dir", normal["distillation"]["output_dir"]),
    ]:
        if token not in val:
            raise ValueError(
                f"{label} 替换后仍不含物种 token '{token}'（原值可能被手工改动）: {val!r}"
            )

    # 时间戳备份
    backup = f"{config_path}.{now_fn()}.bak"
    shutil.copy2(config_path, backup)

    # 写临时文件 + 原子替换
    dir_name = os.path.dirname(os.path.abspath(config_path))
    fd, tmp = tempfile.mkstemp(suffix=".yaml", dir=dir_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            yaml.safe_dump(data, fh, allow_unicode=True, sort_keys=False, default_flow_style=False)
        os.replace(tmp, config_path)
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        print(f"[config_editor] 写入失败，已保留备份，可手动恢复： copy \"{backup}\" \"{config_path}\"")
        raise
    return backup
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_config_editor.py -v`
Expected: PASS（全部）

---

## Task 6: snapshot_merge —— 找最新旧快照（日期解析 + 歧义要求人工指定）

**Files:**
- Create: `snapshot_merge.py`
- Test: `tests/test_snapshot_merge.py`

`find_latest_old_snapshot(dataset_root, current_date, dir_lister)`：在 `dataset_root` 下找日期早于 `current_date`、且能无歧义解析为 `YYYY-MM-DD` 的最新目录（排除本批）。遇到无年份的 `MMDD` / `MMDD-videos` 命名 → 收集进 `ambiguous` 列表。返回 `(latest_or_None, ambiguous_list)`。`dir_lister` 依赖注入便于测试。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_snapshot_merge.py
import snapshot_merge as sm


def test_find_latest_picks_newest_before_current():
    dirs = ["2026-06-24", "2026-06-26", "2026-07-02", "2026-07-15"]
    latest, ambiguous = sm.find_latest_old_snapshot(
        "dataset/rabbit", "2026-07-15", dir_lister=lambda root: dirs)
    assert latest == "2026-07-02"
    assert ambiguous == []


def test_find_latest_excludes_current_and_future():
    dirs = ["2026-07-15", "2026-07-20"]
    latest, ambiguous = sm.find_latest_old_snapshot(
        "dataset/rabbit", "2026-07-15", dir_lister=lambda root: dirs)
    assert latest is None


def test_find_latest_collects_ambiguous_names():
    dirs = ["2026-07-02", "0703-videos", "0626"]
    latest, ambiguous = sm.find_latest_old_snapshot(
        "dataset/rabbit", "2026-07-15", dir_lister=lambda root: dirs)
    assert latest == "2026-07-02"
    assert set(ambiguous) == {"0703-videos", "0626"}
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_snapshot_merge.py -v`
Expected: FAIL —— `ModuleNotFoundError: No module named 'snapshot_merge'`

- [ ] **Step 3: 写最小实现**

```python
# snapshot_merge.py
import os
import re
from datetime import date

_ISO = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")


def _parse_iso(name):
    m = _ISO.match(name)
    if not m:
        return None
    try:
        return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    except ValueError:
        return None


def _os_dir_lister(root):
    return [n for n in os.listdir(root) if os.path.isdir(os.path.join(root, n))]


def find_latest_old_snapshot(dataset_root, current_date, dir_lister=_os_dir_lister):
    current = _parse_iso(current_date)
    latest_name, latest_date, ambiguous = None, None, []
    for name in dir_lister(dataset_root):
        d = _parse_iso(name)
        if d is None:
            # 无法无歧义解析（如 0703-videos / 0626）
            if re.search(r"\d", name):
                ambiguous.append(name)
            continue
        if current is not None and d >= current:
            continue
        if latest_date is None or d > latest_date:
            latest_name, latest_date = name, d
    return latest_name, ambiguous
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_snapshot_merge.py -v`
Expected: PASS（3 passed）

---

## Task 7: snapshot_merge —— 合并样本（整体拷贝、同名跳过、双层布局只读同格式层）

**Files:**
- Modify: `snapshot_merge.py`
- Test: `tests/test_snapshot_merge.py`

`merge_snapshot(old_root, new_root, frames)`：把 `old_root/{train,val}` 下样本整体拷进 `new_root/{train,val}`，同名跳过，返回统计 `{copied, skipped}`。
- 视频格式（frames=False）：样本 = `train/<类别>/*.mp4` 文件，拷贝时保留类别子目录。
- 帧格式（frames=True）：样本 = `train/<样本文件夹>/`，整目录拷。
- 双层布局：`old_root` 传入时已由调用方选定正确层（见 Task 8 的 `resolve_old_layer`），此函数只按 frames 决定样本单位。

- [ ] **Step 1: 写失败测试**

```python
# 追加到 tests/test_snapshot_merge.py
def _touch(p):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("x", encoding="utf-8")


def test_merge_video_format_copies_and_skips_dupes(tmp_path):
    old = tmp_path / "old"; new = tmp_path / "new"
    _touch(old / "train" / "eat" / "a.mp4")
    _touch(old / "train" / "eat" / "b.mp4")
    _touch(old / "val" / "drink" / "c.mp4")
    _touch(new / "train" / "eat" / "a.mp4")   # 已存在，应跳过
    stats = sm.merge_snapshot(str(old), str(new), frames=False)
    assert (new / "train" / "eat" / "b.mp4").is_file()
    assert (new / "val" / "drink" / "c.mp4").is_file()
    assert stats == {"copied": 2, "skipped": 1}


def test_merge_frame_format_copies_whole_sample_folder(tmp_path):
    old = tmp_path / "old"; new = tmp_path / "new"
    _touch(old / "train" / "0626-eat-x-0" / "frame_0000.jpg")
    _touch(old / "train" / "0626-eat-x-0" / "label.txt")
    stats = sm.merge_snapshot(str(old), str(new), frames=True)
    assert (new / "train" / "0626-eat-x-0" / "frame_0000.jpg").is_file()
    assert (new / "train" / "0626-eat-x-0" / "label.txt").is_file()
    assert stats["copied"] == 1
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_snapshot_merge.py -k merge -v`
Expected: FAIL —— `AttributeError: ... 'merge_snapshot'`

- [ ] **Step 3: 写最小实现**

```python
# 追加到 snapshot_merge.py
import shutil


def _iter_samples(split_dir, frames):
    """产出 (相对 split 的路径, 绝对源路径)。"""
    if not os.path.isdir(split_dir):
        return
    if frames:
        for name in sorted(os.listdir(split_dir)):
            p = os.path.join(split_dir, name)
            if os.path.isdir(p):
                yield name, p
    else:
        for cls in sorted(os.listdir(split_dir)):
            cls_dir = os.path.join(split_dir, cls)
            if not os.path.isdir(cls_dir):
                continue
            for fn in sorted(os.listdir(cls_dir)):
                if fn.lower().endswith((".mp4", ".mov", ".avi", ".webm")):
                    yield os.path.join(cls, fn), os.path.join(cls_dir, fn)


def merge_snapshot(old_root, new_root, frames):
    copied, skipped = 0, 0
    for split in ("train", "val"):
        old_split = os.path.join(old_root, split)
        new_split = os.path.join(new_root, split)
        for rel, src in _iter_samples(old_split, frames):
            dst = os.path.join(new_split, rel)
            if os.path.exists(dst):
                skipped += 1
                continue
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if frames:
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)
            copied += 1
    return {"copied": copied, "skipped": skipped}
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_snapshot_merge.py -v`
Expected: PASS（全部）

---

## Task 8: snapshot_merge —— 体检报告 + 双层布局层选择

**Files:**
- Modify: `snapshot_merge.py`
- Test: `tests/test_snapshot_merge.py`

两个函数：
1. `resolve_old_layer(old_date_dir, frames)`：旧快照可能同时有顶层 `train/val`（帧）和 `videos/train,val`（视频）。返回 `(layer_root, other_layer_exists)`——frames=False→优先 `videos/` 层，frames=True→顶层；报告另一层是否存在。
2. `health_report(new_root, class_names, frames)`：扫描新快照，返回报告 dict：每类别样本数、跨 split 同名泄漏列表、名字不含类别关键词的样本列表（会被静默跳过）、（帧格式）label.txt 索引与文件夹名关键词不一致列表。

- [ ] **Step 1: 写失败测试**

```python
# 追加到 tests/test_snapshot_merge.py
def test_resolve_old_layer_video_prefers_videos_subdir(tmp_path):
    d = tmp_path / "2026-06-26"
    (d / "train").mkdir(parents=True)
    (d / "videos" / "train").mkdir(parents=True)
    layer, other = sm.resolve_old_layer(str(d), frames=False)
    assert layer == str(d / "videos")
    assert other is True   # 顶层 train 也存在 → 警示


def test_resolve_old_layer_frames_uses_top_level(tmp_path):
    d = tmp_path / "2026-06-26"
    (d / "train").mkdir(parents=True)
    layer, other = sm.resolve_old_layer(str(d), frames=True)
    assert layer == str(d)
    assert other is False


def test_health_report_flags_cross_split_leak_and_unrecognized(tmp_path):
    new = tmp_path / "new"
    _touch(new / "train" / "eat" / "dup.mp4")
    _touch(new / "val" / "eat" / "dup.mp4")          # 跨 split 泄漏
    _touch(new / "train" / "eat" / "mystery.mp4")    # 不含关键词? 含 'eat' 目录名→按目录识别
    _touch(new / "train" / "nocat" / "weird.mp4")    # 目录名不含任何类别关键词
    rep = sm.health_report(str(new), ["eat", "drink", "other"], frames=False)
    assert "dup.mp4" in rep["cross_split_leak"]
    assert any("nocat" in u or "weird" in u for u in rep["unrecognized"])
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_snapshot_merge.py -k "resolve_old_layer or health" -v`
Expected: FAIL —— `AttributeError: ... 'resolve_old_layer'`

- [ ] **Step 3: 写最小实现**

```python
# 追加到 snapshot_merge.py
def resolve_old_layer(old_date_dir, frames):
    videos_layer = os.path.join(old_date_dir, "videos")
    has_videos = os.path.isdir(os.path.join(videos_layer, "train")) or \
                 os.path.isdir(os.path.join(videos_layer, "val"))
    has_top = os.path.isdir(os.path.join(old_date_dir, "train")) or \
              os.path.isdir(os.path.join(old_date_dir, "val"))
    if frames:
        return old_date_dir, has_videos
    if has_videos:
        return videos_layer, has_top
    return old_date_dir, False


def _class_of(name, class_names):
    low = name.lower()
    for c in class_names:
        if c.lower() in low:
            return c
    return None


def health_report(new_root, class_names, frames):
    per_class = {c: 0 for c in class_names}
    unrecognized, seen_split, cross_split_leak, label_mismatch = [], {}, [], []
    for split in ("train", "val"):
        for rel, path in _iter_samples(os.path.join(new_root, split), frames):
            base = os.path.basename(rel)
            key = base
            if key in seen_split and seen_split[key] != split:
                cross_split_leak.append(key)
            seen_split[key] = split
            cls = _class_of(rel, class_names)
            if cls is None:
                unrecognized.append(os.path.join(split, rel))
            else:
                per_class[cls] += 1
            if frames:
                label_file = os.path.join(path, "label.txt")
                if os.path.isfile(label_file):
                    try:
                        idx = int(open(label_file).read().strip())
                        if cls is not None and 0 <= idx < len(class_names) and class_names[idx] != cls:
                            label_mismatch.append(os.path.join(split, rel))
                    except (ValueError, IOError):
                        pass
    return {
        "per_class": per_class,
        "cross_split_leak": cross_split_leak,
        "unrecognized": unrecognized,
        "label_mismatch": label_mismatch,
    }
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_snapshot_merge.py -v`
Expected: PASS（全部）

---

## Task 9: scaffold_dirs.py 入口①

**Files:**
- Create: `scaffold_dirs.py`
- Test: `tests/test_scaffold_dirs.py`

`build_dirs(ctx, root=".")`：按 ctx 建 records/<date> + sorted-<date>/<各类别> 空目录，返回建好的目录列表。`main()` 调 `ask_context()` 再 `build_dirs`。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_scaffold_dirs.py
import os
import scaffold_dirs as sd


def test_build_dirs_creates_records_and_sorted_classes(tmp_path):
    ctx = {"species": "pet", "site": "hkbr", "date": "2026-07-15"}
    created = sd.build_dirs(ctx, root=str(tmp_path))
    base = tmp_path / "dataset" / "raw_videos" / "pet" / "normal" / "hkbr-record"
    assert (base / "records" / "2026-07-15").is_dir()
    for c in ["eat", "drink", "other"]:
        assert (base / "sorted-2026-07-15" / c).is_dir()
    assert len(created) == 1 + 3   # records + 3 类别


def test_build_dirs_rabbit_has_four_classes(tmp_path):
    ctx = {"species": "rabbit", "site": "hkbr", "date": "2026-07-15"}
    sd.build_dirs(ctx, root=str(tmp_path))
    base = tmp_path / "dataset" / "raw_videos" / "rabbit" / "hkbr-record" / "sorted-2026-07-15"
    for c in ["resting", "active", "eating", "drinking"]:
        assert (base / c).is_dir()
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_scaffold_dirs.py -v`
Expected: FAIL —— `ModuleNotFoundError: No module named 'scaffold_dirs'`

- [ ] **Step 3: 写最小实现**

```python
# scaffold_dirs.py
import os

import pipeline_config as pc


def build_dirs(ctx, root="."):
    species, site, date = ctx["species"], ctx["site"], ctx["date"]
    created = []
    records = os.path.join(root, pc.records_dir(species, site, date))
    os.makedirs(records, exist_ok=True)
    created.append(records)
    for class_dir in pc.sorted_class_dirs(species, site, date):
        full = os.path.join(root, class_dir)
        os.makedirs(full, exist_ok=True)
        created.append(full)
    return created


def main():
    ctx = pc.ask_context()
    created = build_dirs(ctx)
    print("已建好以下目录：")
    for d in created:
        print("  ", d)
    print("\n下一步：把下载的长视频放进 records 目录；剪辑后的短片按行为丢进 sorted 的类别子文件夹。")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_scaffold_dirs.py -v`
Expected: PASS（2 passed）

---

## Task 10: crop 改造前置 —— 校验 crop 脚本导入无副作用

**Files:**
- Test: `tests/test_crop_importable.py`

先加一个"安全网"测试：确认两个 crop 脚本能被 import 而不触发 YOLO 加载/处理（改造过程中不能破坏这点）。

- [ ] **Step 1: 写测试**

```python
# tests/test_crop_importable.py
import importlib


def test_crop_video_imports_without_side_effects():
    m = importlib.import_module("utils.abnormal_crop_video")
    assert hasattr(m, "main")


def test_crop_frames_imports_without_side_effects():
    m = importlib.import_module("utils.abnormal_crop")
    assert hasattr(m, "main")
```

- [ ] **Step 2: 跑测试确认通过（改造前的基线）**

Run: `python -m pytest tests/test_crop_importable.py -v`
Expected: PASS（2 passed）—— 若 FAIL 说明有 import 副作用，需先排查再改造。

---

## Task 11: 改造 abnormal_crop_video.py —— run(config)

**Files:**
- Modify: `utils/abnormal_crop_video.py`
- Test: `tests/test_crop_video_run.py`

**改造原则（spec 4.3）**：运行配置显式传给辅助函数，`run()` 不改模块全局。做法：
1. 新增 `default_config()` 返回 dict，值取自现有模块常量。
2. 新增 `run(config=None, model_factory=None)`：合并默认 config，处理**单个** `input_base_dir`（config["input_base_dir"]），把 config 显式传给辅助函数。`model_factory` 依赖注入便于测试（默认 `lambda dev: YOLO(config["yolo_model_path"]).to(dev)`）。
3. 改造读全局的辅助函数签名，追加 `cfg` 参数：`process_clip_segment(..., cfg)`、`rebalance_val(video_records, cfg)`、`process_video_lowmem(..., cfg)`、`get_clip_segments(duration, cfg)`。函数体内 `TARGET_CLASSES`→`cfg["target_classes"]`、`MAX_DET`→`cfg["max_det"]`、`OUTPUT_BASE_DIR`→`cfg["output_base_dir"]`、`EXPAND_RATIO`→`cfg["expand_ratio"]`、`MAX_CLIP_DURATION`→`cfg["max_clip_duration"]` 等。
4. `main()` 改为：构造 `cfg = default_config()`，对 `INPUT_BASE_DIRS` 每个目录设 `cfg["input_base_dir"]` 调 `run(cfg)`——保持单独跑行为不变。
5. 保留 `if __name__=="__main__": main()`。

> ⚠️ 现有测试 `tests/test_abnormal_crop_video.py` monkeypatch `MAX_CLIP_DURATION` 调 `get_clip_segments(duration)`。改签名会破坏它——**给 `cfg` 参数默认值 `None`，为 None 时回退读模块全局**，保持旧测试通过。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_crop_video_run.py
import importlib


class _FakeModel:
    def __init__(self):
        self.calls = []
    def __call__(self, frame, classes=None, verbose=False, max_det=None):
        self.calls.append({"classes": classes, "max_det": max_det})
        class _R:
            boxes = []
        return [_R()]
    def to(self, dev):
        return self


def test_default_config_has_expected_keys():
    m = importlib.import_module("utils.abnormal_crop_video")
    cfg = m.default_config()
    for k in ["input_base_dir", "output_base_dir", "class_names", "yolo_model_path",
              "target_classes", "max_det", "expand_ratio", "max_clip_duration"]:
        assert k in cfg


def test_get_clip_segments_accepts_cfg_override():
    m = importlib.import_module("utils.abnormal_crop_video")
    cfg = m.default_config()
    cfg["max_clip_duration"] = 5
    cfg["min_clip_duration"] = 2.0
    segments = m.get_clip_segments(9.5, cfg)
    assert segments == [(0.0, 5.0)]


def test_get_clip_segments_backward_compatible_without_cfg(monkeypatch):
    m = importlib.import_module("utils.abnormal_crop_video")
    monkeypatch.setattr(m, "MAX_CLIP_DURATION", 4)
    monkeypatch.setattr(m, "MIN_CLIP_DURATION", 2.0)
    assert m.get_clip_segments(9.5) == [(0.0, 4.0), (4.0, 8.0)]


def test_run_uses_config_target_classes(tmp_path):
    m = importlib.import_module("utils.abnormal_crop_video")
    # 空 sorted 目录（无视频）→ run 应干净跑完、不加载真实 YOLO
    src = tmp_path / "sorted-2026-07-15"
    (src / "eat").mkdir(parents=True)
    cfg = m.default_config()
    cfg["input_base_dir"] = str(src)
    cfg["output_base_dir"] = str(tmp_path / "out")
    cfg["class_names"] = ["eat", "drink", "other"]
    fake = _FakeModel()
    m.run(cfg, model_factory=lambda dev: fake)   # 无视频，不会真跑检测，但不应报错
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_crop_video_run.py -v`
Expected: FAIL —— `AttributeError: ... 'default_config'`

- [ ] **Step 3: 实现改造**

按上述 5 点改造 `utils/abnormal_crop_video.py`。关键新增代码：

```python
# 新增到 utils/abnormal_crop_video.py（模块常量之后）
def default_config():
    return {
        "input_base_dir": INPUT_BASE_DIRS[0] if INPUT_BASE_DIRS else None,
        "output_base_dir": OUTPUT_BASE_DIR,
        "class_names": list(CLASS_NAMES),
        "yolo_model_path": YOLO_MODEL_PATH,
        "target_classes": list(TARGET_CLASSES),
        "max_det": MAX_DET,
        "expand_ratio": EXPAND_RATIO,
        "max_clip_duration": MAX_CLIP_DURATION,
        "min_clip_duration": MIN_CLIP_DURATION,
        "discard_on_extra_detections": DISCARD_ON_EXTRA_DETECTIONS,
        "max_clips_per_video": dict(MAX_CLIPS_PER_VIDEO),
    }


def run(config=None, model_factory=None):
    cfg = default_config()
    if config:
        cfg.update(config)
    device = check_gpu()
    if model_factory is None:
        model_factory = lambda dev: YOLO(cfg["yolo_model_path"]).to(dev)
    model = model_factory(device)

    input_base_dir = cfg["input_base_dir"]
    parent_dir_name = Path(input_base_dir).name
    class_counters, failed_paths, video_records = {}, [], {}
    videos = collect_class_videos(input_base_dir, cfg["class_names"])
    for file_path, class_name in tqdm(videos, desc=parent_dir_name, unit="video"):
        stats = process_video_lowmem(file_path, class_name, class_counters, model, parent_dir_name, cfg)
        if stats and stats.get("failed_paths"):
            failed_paths.extend(stats["failed_paths"])
        saved = stats.get("saved_clip_names", []) if stats else []
        if saved:
            video_records.setdefault(class_name, []).append((file_path.name, saved))
    rebalance_val(video_records, cfg)
    return {"video_records": video_records, "failed_paths": failed_paths}
```

改造 `get_clip_segments` 支持 cfg 且向后兼容：

```python
def get_clip_segments(duration, cfg=None):
    if duration <= 0:
        return []
    max_dur = float(cfg["max_clip_duration"]) if cfg else float(MAX_CLIP_DURATION)
    min_dur = float(cfg["min_clip_duration"]) if cfg else float(MIN_CLIP_DURATION)
    if max_dur <= 0:
        raise ValueError("max_clip_duration must be greater than 0")
    num_clips = math.ceil(duration / max_dur)
    segments = []
    for i in range(num_clips):
        start = i * max_dur
        end = min((i + 1) * max_dur, duration)
        if end - start < min_dur:
            continue
        segments.append((start, end))
    return segments
```

对 `rebalance_val`、`process_video_lowmem`、`process_clip_segment` 追加 `cfg` 参数并把函数体内的全局引用换成 `cfg[...]`（`OUTPUT_BASE_DIR`→`cfg["output_base_dir"]`、`TARGET_CLASSES`→`cfg["target_classes"]`、`MAX_DET`→`cfg["max_det"]`、`DISCARD_ON_EXTRA_DETECTIONS`→`cfg["discard_on_extra_detections"]`、`EXPAND_RATIO`→`cfg["expand_ratio"]`、`MAX_CLIPS_PER_VIDEO`→`cfg["max_clips_per_video"]`）。`main()` 改为循环调 `run()`：

```python
def main():
    base = default_config()
    for input_base_dir in INPUT_BASE_DIRS:
        cfg = dict(base)
        cfg["input_base_dir"] = input_base_dir
        run(cfg)
    print("🎉 全部批次处理完成！")
```

- [ ] **Step 4: 跑新测试确认通过**

Run: `python -m pytest tests/test_crop_video_run.py -v`
Expected: PASS（4 passed）

- [ ] **Step 5: 跑旧测试确认无回归**

Run: `python -m pytest tests/test_abnormal_crop_video.py -v`
Expected: PASS（全部旧测试仍通过——证明向后兼容）

---

## Task 12: 改造 abnormal_crop.py（帧格式）—— run(config)

**Files:**
- Modify: `utils/abnormal_crop.py`
- Test: `tests/test_crop_frames_run.py`

同 Task 11 的模式，但帧脚本多一个 `LABEL_MAP`（config 加 `label_map` 键）。样本输出到 `train/<folder>/`（含 `label.txt`）。改造 `default_config()`/`run()`，辅助函数收 cfg。保留 `main()` 循环调 `run()`、`if __name__`。同样给改签名的函数 `cfg=None` 回退。

> 🚫 **子代理审查采纳的两个阻塞项**（帧脚本与视频脚本结构不同，不能照抄 Task 11）：
> 1. 帧脚本的处理函数叫 **`process_video`**（`abnormal_crop.py:152`），**没有** `process_video_lowmem`。run() 必须调 `process_video`，否则 NameError。
> 2. 帧脚本 main（`abnormal_crop.py:334-347`）会收集 `stats["saved_folders"]` → `video_records` → 调 `rebalance_val(video_records)` 划分 val。run() **必须同样收集 saved_folders 并调 rebalance_val**，否则帧格式产物**没有 val split**。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_crop_frames_run.py
import importlib


def test_frames_default_config_has_label_map():
    m = importlib.import_module("utils.abnormal_crop")
    cfg = m.default_config()
    assert "label_map" in cfg
    assert "output_base_dir" in cfg
    assert "target_classes" in cfg
    assert "max_clips_per_video" in cfg


def test_frames_run_on_empty_sorted_is_noop(tmp_path):
    m = importlib.import_module("utils.abnormal_crop")
    src = tmp_path / "sorted-2026-07-15"
    (src / "eat").mkdir(parents=True)
    cfg = m.default_config()
    cfg["input_base_dir"] = str(src)
    cfg["output_base_dir"] = str(tmp_path / "out")
    cfg["class_names"] = ["eat", "drink", "other"]
    m.run(cfg, model_factory=lambda dev: object())   # 无视频，不加载真实模型


def test_frames_run_drives_video_calls_process_and_rebalance(tmp_path, monkeypatch):
    """驱动至少 1 个视频进循环，锁死：调 process_video（非 lowmem）+ 收集 saved_folders + 调 rebalance_val。"""
    m = importlib.import_module("utils.abnormal_crop")
    src = tmp_path / "sorted-2026-07-15"
    (src / "eat").mkdir(parents=True)
    (src / "eat" / "v.mp4").write_text("x", encoding="utf-8")

    # 假 process_video：返回一个 saved_folders，模拟成功产出
    calls = {"process_video": 0, "rebalance": 0}
    def fake_process_video(video_path, class_name, class_counters, model, parent_dir_name, cfg):
        calls["process_video"] += 1
        return {"saved_folders": [f"{parent_dir_name}-{class_name}-v"], "failed_paths": []}
    def fake_rebalance(video_records, cfg):
        calls["rebalance"] += 1
        assert "eat" in video_records   # 证明 saved_folders 被收集进 video_records
    monkeypatch.setattr(m, "process_video", fake_process_video)
    monkeypatch.setattr(m, "rebalance_val", fake_rebalance)

    cfg = m.default_config()
    cfg["input_base_dir"] = str(src)
    cfg["output_base_dir"] = str(tmp_path / "out")
    cfg["class_names"] = ["eat", "drink", "other"]
    m.run(cfg, model_factory=lambda dev: object())

    assert calls["process_video"] == 1
    assert calls["rebalance"] == 1
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_crop_frames_run.py -v`
Expected: FAIL —— `AttributeError: ... 'default_config'`

- [ ] **Step 3: 实现改造**

在 `utils/abnormal_crop.py` 新增（模仿 Task 11，追加 `label_map` 和 `max_clips_per_video`）：

```python
def default_config():
    return {
        "input_base_dir": INPUT_BASE_DIRS[0] if INPUT_BASE_DIRS else None,
        "output_base_dir": OUTPUT_BASE_DIR,
        "class_names": list(CLASS_NAMES),
        "label_map": dict(LABEL_MAP),
        "yolo_model_path": YOLO_MODEL_PATH,
        "target_classes": list(TARGET_CLASSES),
        "max_det": MAX_DET,
        "expand_ratio": EXPAND_RATIO,
        "frames_per_clip": FRAMES_PER_CLIP,
        "max_clip_duration": MAX_CLIP_DURATION,
        "discard_on_extra_detections": DISCARD_ON_EXTRA_DETECTIONS,
        "max_clips_per_video": dict(MAX_CLIPS_PER_VIDEO),
    }


def run(config=None, model_factory=None):
    cfg = default_config()
    if config:
        cfg.update(config)
    device = check_gpu()
    if model_factory is None:
        model_factory = lambda dev: YOLO(cfg["yolo_model_path"]).to(dev)
    model = model_factory(device)
    input_base_dir = cfg["input_base_dir"]
    parent_dir_name = Path(input_base_dir).name
    class_counters, failed_paths, video_records = {}, [], {}
    videos = collect_class_videos(input_base_dir, cfg["class_names"])
    for file_path, class_name in tqdm(videos, desc=parent_dir_name, unit="video"):
        stats = process_video(file_path, class_name, class_counters, model, parent_dir_name, cfg)
        if stats and stats.get("failed_paths"):
            failed_paths.extend(stats["failed_paths"])
        saved_folders = stats.get("saved_folders", []) if stats else []
        if saved_folders:
            video_records.setdefault(class_name, []).append((file_path.name, saved_folders))
    rebalance_val(video_records, cfg)
    return {"video_records": video_records, "failed_paths": failed_paths}
```

把 `process_video`（`:152`）、`rebalance_val`（`:118`）及它们调用的辅助函数追加 `cfg` 参数，函数体内替换全局：`OUTPUT_BASE_DIR`→`cfg["output_base_dir"]`、`LABEL_MAP`→`cfg["label_map"]`、`TARGET_CLASSES`→`cfg["target_classes"]`、`FRAMES_PER_CLIP`→`cfg["frames_per_clip"]`、`MAX_DET`→`cfg["max_det"]`、`EXPAND_RATIO`→`cfg["expand_ratio"]`、`MAX_CLIPS_PER_VIDEO`→`cfg["max_clips_per_video"]`、`DISCARD_ON_EXTRA_DETECTIONS`→`cfg["discard_on_extra_detections"]`。`main()` 改为循环调 `run()`。

> 注意：`collect_class_videos` 已收 `class_names` 参数，无需改。

- [ ] **Step 4: 跑新测试确认通过**

Run: `python -m pytest tests/test_crop_frames_run.py -v`
Expected: PASS（3 passed）

- [ ] **Step 5: 跑全量 crop 相关测试确认无回归**

Run: `python -m pytest tests/ -k crop -v`
Expected: PASS（全部）

---

## Task 13: run_crop.py 入口②

**Files:**
- Create: `run_crop.py`
- Test: `tests/test_run_crop.py`

`validate_sorted(sorted_dir, class_names)`：sorted 目录存在且至少一个类别下有视频，否则 raise。`assert_snapshot_empty(snapshot_dir)`：快照目录不存在或为空，否则 raise。`main(argv)`：解析 `--frames`，`ask_context`，校验，调对应 crop 脚本 `run(config)`。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_run_crop.py
import pytest
import run_crop


def test_validate_sorted_ok(tmp_path):
    (tmp_path / "eat").mkdir()
    (tmp_path / "eat" / "a.mp4").write_text("x", encoding="utf-8")
    run_crop.validate_sorted(str(tmp_path), ["eat", "drink", "other"])  # 不 raise


def test_validate_sorted_raises_when_no_videos(tmp_path):
    (tmp_path / "eat").mkdir()
    with pytest.raises(FileNotFoundError):
        run_crop.validate_sorted(str(tmp_path), ["eat", "drink", "other"])


def test_assert_snapshot_empty_raises_when_nonempty(tmp_path):
    (tmp_path / "train").mkdir()
    (tmp_path / "train" / "x.mp4").write_text("x", encoding="utf-8")
    with pytest.raises(FileExistsError):
        run_crop.assert_snapshot_empty(str(tmp_path))


def test_assert_snapshot_empty_ok_when_missing(tmp_path):
    run_crop.assert_snapshot_empty(str(tmp_path / "nope"))  # 不 raise
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_run_crop.py -v`
Expected: FAIL —— `ModuleNotFoundError: No module named 'run_crop'`

- [ ] **Step 3: 写最小实现**

```python
# run_crop.py
import argparse
import os
import sys

import pipeline_config as pc


def validate_sorted(sorted_dir, class_names):
    if not os.path.isdir(sorted_dir):
        raise FileNotFoundError(f"sorted 目录不存在: {sorted_dir}")
    exts = (".mp4", ".mov", ".avi", ".webm")
    for c in class_names:
        cdir = os.path.join(sorted_dir, c)
        if os.path.isdir(cdir):
            for _, _, files in os.walk(cdir):
                if any(f.lower().endswith(exts) for f in files):
                    return
    raise FileNotFoundError(f"sorted 目录下没有任何视频（是否还没剪辑？）: {sorted_dir}")


def assert_snapshot_empty(snapshot_dir):
    if os.path.isdir(snapshot_dir) and any(os.scandir(snapshot_dir)):
        raise FileExistsError(
            f"目标快照目录已非空，为避免旧产物混入本批，请先清空或换日期: {snapshot_dir}")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", action="store_true", help="切帧格式（默认视频 clip）")
    args = parser.parse_args(argv)
    ctx = pc.ask_context()
    species, site, date = ctx["species"], ctx["site"], ctx["date"]
    sp = pc.SPECIES[species]

    sorted_dir = pc.sorted_dir(species, site, date)
    validate_sorted(sorted_dir, sp["class_names"])
    snapshot = pc.snapshot_dir(species, date, frames=args.frames)
    assert_snapshot_empty(snapshot)

    if args.frames:
        from utils import abnormal_crop as crop
    else:
        from utils import abnormal_crop_video as crop
    cfg = crop.default_config()
    cfg.update({
        "input_base_dir": sorted_dir,
        "output_base_dir": snapshot,
        "class_names": sp["class_names"],
        "yolo_model_path": sp["yolo_model"],
        "target_classes": sp["target_classes"],
    })
    if args.frames:
        cfg["label_map"] = sp["label_map"]
    crop.run(cfg)
    print(f"\ncrop 完成，产物在: {snapshot}")


if __name__ == "__main__":
    main(sys.argv[1:])
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_run_crop.py -v`
Expected: PASS（4 passed）

---

## Task 14: run_train_distill.py 入口③

**Files:**
- Create: `run_train_distill.py`
- Test: `tests/test_run_train_distill.py`

编排（纯逻辑抽成可测函数，训练调用留在 `main`）：
- `teacher_output_path(train_cfg)`：返回 `os.path.join(train_cfg["output_dir"], train_cfg["output_model_name"])`。
- `main()` 流程：`ask_context` → 找旧快照（歧义则提示人工）→ 合并 → 打印体检报告 → **暂停等回车** → **先改 config.yaml** → **再 build_train_config** → teacher `train(config_override=train_cfg)` → 回填 teacher 路径（校验存在）→ student `train_distill()` → 打印产物。

关键顺序断言用测试锁住：`teacher_output_path` 不套 EMA 命名。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_run_train_distill.py
import os
import run_train_distill as rtd


def test_teacher_output_path_is_output_dir_plus_model_name():
    train_cfg = {
        "output_dir": os.path.normpath("outputs/rabbit/0715/teacher_finetuned_3cls_0715"),
        "output_model_name": "rabbit_teacher_0715_ema-videos.pth",
    }
    got = rtd.teacher_output_path(train_cfg)
    assert got == os.path.join(
        os.path.normpath("outputs/rabbit/0715/teacher_finetuned_3cls_0715"),
        "rabbit_teacher_0715_ema-videos.pth",
    )
    # 不得包含二次 _ema 叠加（证明没套 resolve_ema_model_name）
    assert got.count("_ema") == 1
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_run_train_distill.py -v`
Expected: FAIL —— `ModuleNotFoundError: No module named 'run_train_distill'`

- [ ] **Step 3: 写最小实现**

```python
# run_train_distill.py
import os
import sys

import pipeline_config as pc
import config_editor
import snapshot_merge


def teacher_output_path(train_cfg):
    return os.path.join(train_cfg["output_dir"], train_cfg["output_model_name"])


def _print_health_report(rep, old_layer_other_exists, ambiguous):
    print("\n===== 体检报告 =====")
    print("各类别样本数:", rep["per_class"])
    if rep["cross_split_leak"]:
        print("⚠️ 跨 split 泄漏（同名同时在 train 和 val）:", rep["cross_split_leak"])
    if rep["unrecognized"]:
        print("⚠️ 名字不含类别关键词、会被数据加载器静默跳过:", rep["unrecognized"])
    if rep.get("label_mismatch"):
        print("⚠️ label.txt 索引与文件夹名关键词不一致:", rep["label_mismatch"])
    if old_layer_other_exists:
        print("⚠️ 旧快照同时存在另一格式布局（videos/ 与顶层），本次只合并了同格式层，请确认没漏数据。")
    if ambiguous:
        print("⚠️ 存在无法解析日期的历史目录（未参与自动选择）:", ambiguous)
    print("提示: pet 三类训练时 other 会被截断到 1.5×max(eat,drink)，磁盘数≠训练实际用量。")
    print("====================\n")


def main(argv=None):
    frames = "--frames" in (argv or sys.argv[1:])
    ctx = pc.ask_context()
    species, site, date = ctx["species"], ctx["site"], ctx["date"]
    sp = pc.SPECIES[species]

    new_snapshot = pc.snapshot_dir(species, date, frames=frames)
    if not os.path.isdir(new_snapshot):
        raise FileNotFoundError(f"本批次快照不存在（先跑 run_crop）: {new_snapshot}")

    latest, ambiguous = snapshot_merge.find_latest_old_snapshot(sp["dataset_root"], date)
    # spec 4.4.a：存在无法解析日期的历史目录时，合并前要求人工确认（合并不可逆）
    if ambiguous:
        print(f"⚠️ 发现无法解析日期的历史快照目录: {ambiguous}")
        print(f"   自动选出的基准是: {latest}（可能不是真正最新）。")
        resp = input("这些目录未参与自动选择。确认用上面的基准继续合并？(yes/否则中止): ").strip()
        if resp.lower() != "yes":
            print("已中止。请手动确认基准快照后重跑。")
            return
    old_other = False
    if latest:
        old_date_dir = os.path.join(sp["dataset_root"], latest)
        old_layer, old_other = snapshot_merge.resolve_old_layer(old_date_dir, frames)
        stats = snapshot_merge.merge_snapshot(old_layer, new_snapshot, frames)
        print(f"合并旧快照 {latest}: 拷入 {stats['copied']}，跳过 {stats['skipped']}")
    else:
        print("未找到可自动合并的旧快照（可能是首批，或历史目录日期无法解析）。")

    rep = snapshot_merge.health_report(new_snapshot, sp["class_names"], frames)
    _print_health_report(rep, old_other, ambiguous)

    resp = input("体检确认无误？回车继续训练，Ctrl+C 中止： ")

    # 先改 config.yaml，再 build（顺序不可颠倒）
    config_editor.apply_species_config("config.yaml", species, new_snapshot)

    from training.config import build_train_config
    train_cfg = build_train_config("normal")   # 从改好的磁盘 config 读

    from train.train_finetune_normal_3 import train as teacher_train
    teacher_train(config_override=train_cfg)

    teacher_path = teacher_output_path(train_cfg)
    if not os.path.isfile(teacher_path):
        raise FileNotFoundError(f"teacher 训练产物不存在，无法蒸馏: {teacher_path}")
    config_editor.set_teacher_checkpoint("config.yaml", teacher_path)

    from distill.distill_normal_3 import train_distill
    train_distill()

    print(f"\n✅ 完成。teacher: {teacher_path}")
    print("部署用 student 的 *_ema.pth（见 outputs 目录）。")


if __name__ == "__main__":
    main(sys.argv[1:])
```

> 依赖新增 `config_editor.set_teacher_checkpoint(config_path, path)`——见 Step 4。

- [ ] **Step 4: 补 config_editor.set_teacher_checkpoint + 测试**

```python
# 追加到 tests/test_config_editor.py
def test_set_teacher_checkpoint_yaml_safe(tmp_path):
    cfg = _write_sample_config(tmp_path)
    ce.set_teacher_checkpoint(str(cfg), r"outputs\rabbit\0715\t_ema.pth")
    data = yaml.safe_load(cfg.read_text(encoding="utf-8"))
    assert data["distillation"]["teacher_checkpoint"] == "outputs/rabbit/0715/t_ema.pth"
```

```python
# 追加到 config_editor.py
def set_teacher_checkpoint(config_path, teacher_path):
    with open(config_path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    data["distillation"]["teacher_checkpoint"] = yaml_safe_path(teacher_path)
    dir_name = os.path.dirname(os.path.abspath(config_path))
    fd, tmp = tempfile.mkstemp(suffix=".yaml", dir=dir_name)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, allow_unicode=True, sort_keys=False, default_flow_style=False)
    os.replace(tmp, config_path)
```

- [ ] **Step 5: 跑测试确认通过**

Run: `python -m pytest tests/test_run_train_distill.py tests/test_config_editor.py -v`
Expected: PASS（全部）

---

## Task 15: docs/AUTOMATION.md

**Files:**
- Create: `docs/AUTOMATION.md`

- [ ] **Step 1: 写文档**

内容涵盖：三入口用法（`python scaffold_dirs.py` / `python run_crop.py [--frames]` / `python run_train_distill.py`）、每步交互输入什么、日期约定（**录制日，手输**）、人工节点（下载视频、剪辑分类、体检确认、看结果）、体检报告怎么看、部署用 student `*_ema.pth`、检测模型路径维护提醒、`conda activate AR`。

```markdown
# 行为识别流水线自动化使用说明

> 底层原理见 `SOP.en.md` / `handover.en.md`，本文只讲三个自动化入口怎么用。全部在 `conda activate AR` 下跑。

## 三个入口

1. `python scaffold_dirs.py` —— 建目录。交互输入 物种(pet/rabbit)/机构/日期。
   建好 `records/<日期>`（放下载的长视频）和 `sorted-<日期>/<各类别>`（放剪辑后的短片）。
2. 【人工】把长视频放进 records，剪成单动作短片按行为丢进 sorted 的类别文件夹。
3. `python run_crop.py`（默认视频格式；`--frames` 切帧格式）—— YOLO 抠图切片，产出本批快照。
4. `python run_train_distill.py` —— 合并旧快照 → 体检报告（**停下等你确认**）→ 改 config → teacher 训练 → student 蒸馏。

## 日期约定
输入的日期是**录制日**（视频录制/现场捞回的日子），默认今天可改。同一批数据从建目录到训练要**一直用同一个日期**，否则产物对不上。

## 人工节点
- 下载长视频、剪辑分类（脚本不做）。
- 体检确认：run_train_distill 合并后会打印各类别样本数、跨 split 泄漏、会被静默跳过的样本等，确认无误再回车。
- 看训练结果验收。

## 部署产物
部署用 **student 的 `*_ema.pth`**（在 `outputs/<物种>/<日期>/` 下），不是 teacher。

## 维护提醒
物种查表在 `pipeline_config.py::SPECIES`，检测模型换代时更新其中 `yolo_model` 路径。

## 注意事项
- 三个入口都从**项目根目录**启动（`run_train_distill.py` 用相对路径 `config.yaml` 和相对 snapshot 路径）。
- `run_train_distill.py` 改写 `config.yaml` 时会**丢掉原有注释**（YAML 重新序列化的代价），但每次改写前都有带时间戳的 `.bak` 备份可恢复。
```

- [ ] **Step 2: 跑全量测试做最终验证**

Run: `python -m pytest tests/ -v`
Expected: PASS（全部新旧测试通过）

---

## 自检记录（writing-plans self-review）

- **Spec 覆盖**：物种查表(T1)/路径推导(T2)/交互(T3)/config 规整+改写+断言(T4-5)/teacher_checkpoint 回填(T14)/找旧快照+歧义(T6)/合并(T7)/体检+双层布局+label.txt+泄漏(T8)/脚手架(T9)/crop 改造无副作用+显式传 cfg+向后兼容(T10-12)/crop 前置校验(T13)/步骤顺序阻塞项(T14)/文档(T15)——spec 各节均有任务覆盖。
- **占位符**：无 TBD；每个 code step 给了完整代码或精确的签名/替换指令。
- **类型/命名一致**：`default_config()`/`run(config, model_factory)`/`snapshot_dir(species,date,frames)`/`apply_species_config`/`set_teacher_checkpoint`/`teacher_output_path` 跨任务一致。
- **已知妥协**：crop 改造（T11-12）因原文件 ~600/330 行，用"精确签名变更 + 全局→cfg 替换清单"而非逐行重贴；实现子代理需读原文件。已用向后兼容测试（跑旧 test）兜底防回归。
