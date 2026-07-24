# Task 1 Brief: timeline 加档位常量、升 schema 版本、build_video_doc 支持 segment_levels

这是你的需求来源，使用其中的确切数值/命名（逐字照抄，勿改）。

## Files
- Modify: `auto_presort/timeline.py`
- Test: `tests/test_autopresort_timeline.py`

## Interfaces you must produce
- `timeline.CONF_LEVELS: list[float]` = `[0.5, 0.6, 0.7, 0.8, 0.9]`
- `timeline.SCHEMA_VERSION: str` = `"autopresort-2"`
- `timeline.build_video_doc(rel_path, duration, segments, boundary_spans, no_animal_spans, source_id, approved, meta, segment_levels=None)` → doc dict；当 `segment_levels` 非 None 时 doc 含 `"segment_levels"` 键，否则**不含**该键（向后兼容旧调用）。

## Step 1: 写失败测试

在 `tests/test_autopresort_timeline.py` 末尾追加：

```python
def test_conf_levels_constant():
    tl = importlib.import_module("auto_presort.timeline")
    assert tl.CONF_LEVELS == [0.5, 0.6, 0.7, 0.8, 0.9]


def test_schema_version_bumped_to_v2():
    tl = importlib.import_module("auto_presort.timeline")
    assert tl.SCHEMA_VERSION == "autopresort-2"


def test_build_video_doc_includes_segment_levels_when_given():
    tl = importlib.import_module("auto_presort.timeline")
    segs09 = [tl.make_segment(1.0, 3.0, "eat", 0.95, "model")]
    levels = {"0.5": segs09, "0.6": segs09, "0.7": segs09,
              "0.8": segs09, "0.9": segs09}
    doc = tl.build_video_doc(
        rel_path="v.mp4", duration=4.0, segments=segs09,
        boundary_spans=[], no_animal_spans=[], source_id="s1",
        approved=False, meta={"class_names": ["eat"]},
        segment_levels=levels,
    )
    assert set(doc["segment_levels"].keys()) == {"0.5", "0.6", "0.7", "0.8", "0.9"}
    assert doc["segment_levels"]["0.9"] == segs09


def test_build_video_doc_omits_segment_levels_when_none():
    tl = importlib.import_module("auto_presort.timeline")
    segs = [tl.make_segment(1.0, 3.0, "eat", 0.95, "model")]
    doc = tl.build_video_doc(
        rel_path="v.mp4", duration=4.0, segments=segs,
        boundary_spans=[], no_animal_spans=[], source_id="s1",
        approved=False, meta={"class_names": ["eat"]},
    )
    assert "segment_levels" not in doc
```

## Step 2: 运行确认失败
`conda run -n AR python -m pytest tests/test_autopresort_timeline.py -v -k "conf_levels or bumped or segment_levels"`
Expected: FAIL（CONF_LEVELS 不存在 / SCHEMA_VERSION 是旧值 / 无 segment_levels 参数）。

## Step 3: 实现

改 `auto_presort/timeline.py`：

第 3 行 `SCHEMA_VERSION = "autopresort-1"` 改为：

```python
SCHEMA_VERSION = "autopresort-2"

CONF_LEVELS = [0.5, 0.6, 0.7, 0.8, 0.9]
```

`build_video_doc`（当前第 148 行起）签名与 body 改为：

```python
def build_video_doc(rel_path, duration, segments, boundary_spans,
                    no_animal_spans, source_id, approved, meta,
                    segment_levels=None):
    doc = {
        "schema_version": SCHEMA_VERSION,
        "rel_path": rel_path,
        "source_id": source_id,
        "duration": float(duration),
        "approved": bool(approved),
        "segments": segments,
        "gaps": compute_gaps(duration, segments, boundary_spans, no_animal_spans),
        "meta": meta,
    }
    if segment_levels is not None:
        doc["segment_levels"] = segment_levels
    return doc
```

## Step 4: 运行确认通过
`conda run -n AR python -m pytest tests/test_autopresort_timeline.py -v`
Expected: PASS（全部，含旧测试）。

## 硬约束
- `timeline.py` 是纯逻辑，**不得** import torch/cv2/ultralytics。
- 测试命令固定：`conda run -n AR python -m pytest <path> -v`（从项目根运行）。
- 本仓库**不是 git 仓库**——不要执行任何 git 命令，不要 commit。"完成" = 上述测试全绿。
- 只改本 brief 列出的两个文件。现有 `test_schema_version_is_exposed` 只断言非空 str，升版后仍通过，勿改它。
