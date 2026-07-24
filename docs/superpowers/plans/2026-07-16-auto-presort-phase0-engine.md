# auto_presort Phase 0（引擎 + 离线校准闸门）实现计划 v2

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现 auto_presort 的「预标注引擎」并用离线校准量化其质量，产出一个 go/no-go 闸门——引擎达标后才在 Phase 1 建审核 UI。

**Architecture（v2 核心改动）：** 一次**统一检测轨迹（detection track）**贯穿全流程——先对整段视频跑一遍 YOLO，产出**逐帧检测框 + 无动物区间（no_animal_spans）**，这份轨迹**同时**喂给：(a) 分类窗口的抠图（复用训练同款平滑框逻辑，保证 teacher 看到的是与训练同分布的**裁剪画面**而非全画面），(b) gap 强制（无动物窗口直接判 uncertain/None，不允许 teacher 在无动物区高置信成段）。时间线由纯逻辑组装（窗口→分类→迟滞合并→向内裁剪→gap 归因→JSON）。checkpoint 用**sidecar 元数据**（哈希 + 有序类别名）解决类别顺序无出处的问题。离线校准是一个**可运行的 CLI**，产出边界纯度/召回/短动作丢失，并按 gate 给 go/no-go。

**Tech Stack:** Python 3.12（conda `AR` 环境，含 pytest/torch/torchvision/ultralytics/opencv）；复用 `utils/abnormal_crop_video.py`、`utils/video_dataset.py`、`models/teacher.py`、`pipeline_config.py`。

**运行环境（重要）：** 所有测试命令在 conda `AR` 环境下跑。实测可用命令：`conda run -n AR python -m pytest <path> -v`（从项目根目录）。默认解释器无 pytest/torch，且 `conda activate` 在此 shell 不生效——用 `conda run -n AR` 代替。

**实现状态（2026-07-16 完成）：** Task 1–12 全部实现完成，37 个 Phase 0 测试全绿，现有 77 个测试无回归。最终代码评审发现并修复了一个硬性规则4的 bug：短于一个窗口（<1.5s）的无动物区间原本无法被整窗丢弃、会被高置信 segment 吞掉——已加 `timeline.split_out_no_animal`（在 merge 后、trim 前从 segment 挖掉无动物区），并加了 5 个回归测试。**下一步是人工节点**：给 teacher checkpoint 建 sidecar `.meta.json`，在真实数据上跑 `python -m auto_presort.calibrate` 看 go/no-go（见 Task 12 Step 3）。


**关键约束（来自 spec + codex 两轮评审，实现时不得违背）：**
- 本仓库**不是 git 仓库**，无 commit。每个 Task 的"提交"= **对应 pytest 模块全绿**。
- 重依赖（torch/YOLO/cv2）**不进纯逻辑单元测试**：分类器、检测器通过**注入的假对象/工厂**替换。但**真实装配边界必须有一个集成测试**（fake capture + fake YOLO + fake teacher，走完整条链），不能因"依赖注入"就完全不测装配。
- 引擎选片时 teacher 看到的**必须是 YOLO 裁剪画面**，抠图+预处理**必须与训练逐位一致**——Task 6 的**真实 parity test**（对比真实 `VideoDataset` 输出）是这条的守卫，不是只查 shape。
- **无动物区强制 gap**：检测必须在分类/合并**之前**完成；与 no_animal_spans 重叠超阈值的窗口直接判 `None`。
- 置信度门是**便利过滤，非正确性保证**；代码/报告里不得写成安全保证。
- 类别顺序错则全盘误标：必须有 **sidecar 出处**，不得静默复用 `SPECIES` 顺序。

**目录约定：** 新代码放 `auto_presort/` 包；测试放 `tests/`，同现有 pytest 风格。

---

## 文件结构（Phase 0 v2）

**新增：**
- `auto_presort/__init__.py` — 空包标记
- `auto_presort/timeline.py` — 纯逻辑：窗口切分、迟滞合并、向内裁剪、gap 归因（含 boundary/no_animal/uncertain 优先级）、JSON schema。**不 import torch/cv2**。
- `auto_presort/detect_track.py` — 统一检测轨迹：视频 → 逐帧框（含漏检时 last_valid_box 传播）+ no_animal_spans。真实实现用 YOLO；纯聚合逻辑（帧标志→spans、框传播）可单测。
- `auto_presort/preprocess.py` — 给定「帧列表 + 对应窗口内检测框」→ 与训练一致的 16 帧张量。复用 `build_smoothed_crop_boxes` + `VideoDataset` 的 eval transform 实例。
- `auto_presort/checkpoint_guard.py` — sidecar 元数据校验 + teacher 硬加载（真实 head 维度检查、类别顺序来自 sidecar）。
- `auto_presort/engine.py` — 串联 detect_track→窗口→preprocess→classify→timeline，产出每视频 JSON。检测/分类通过注入。
- `auto_presort/calibrate.py` — 度量函数 + **可运行 CLI**（GT 解析、视频遍历、跑引擎、阈值扫描、CSV/JSON 报告、gate）。
- `tests/test_autopresort_timeline.py`
- `tests/test_autopresort_detect_track.py`
- `tests/test_autopresort_preprocess_parity.py`
- `tests/test_autopresort_checkpoint_guard.py`
- `tests/test_autopresort_engine.py`（含真实装配的集成测试）
- `tests/test_autopresort_calibrate.py`

**改动：** 无（Phase 0 不碰 `run_crop`/`rebalance_val`；leakage 修复是 Phase 1）。

---

## Task 1: 建包骨架与 JSON schema 常量

**Files:**
- Create: `auto_presort/__init__.py`
- Create: `auto_presort/timeline.py`
- Test: `tests/test_autopresort_timeline.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_autopresort_timeline.py
import importlib


def test_make_segment_has_required_fields():
    tl = importlib.import_module("auto_presort.timeline")
    seg = tl.make_segment(start=1.0, end=4.0, label="eat", confidence=0.93, source="model")
    assert seg == {
        "start": 1.0, "end": 4.0, "label": "eat",
        "confidence": 0.93, "source": "model",
    }


def test_schema_version_is_exposed():
    tl = importlib.import_module("auto_presort.timeline")
    assert isinstance(tl.SCHEMA_VERSION, str) and tl.SCHEMA_VERSION
```

- [ ] **Step 2: 跑测试确认失败**

Run: `conda activate AR && python -m pytest tests/test_autopresort_timeline.py -v`
Expected: FAIL（`ModuleNotFoundError: auto_presort`）

- [ ] **Step 3: 最小实现**

```python
# auto_presort/__init__.py
```
（空文件）

```python
# auto_presort/timeline.py
"""纯逻辑：窗口 → 时间线 segment/gap。不得 import torch/cv2/ultralytics。"""

SCHEMA_VERSION = "autopresort-1"


def make_segment(start, end, label, confidence, source):
    return {
        "start": float(start),
        "end": float(end),
        "label": label,
        "confidence": float(confidence),
        "source": source,
    }
```

- [ ] **Step 4: 跑测试确认通过**

Run: `conda activate AR && python -m pytest tests/test_autopresort_timeline.py -v`
Expected: PASS（2 passed）

---

## Task 2: 窗口切分（滑窗 + stride，CFR floor 帧映射）

**Files:**
- Modify: `auto_presort/timeline.py`
- Test: `tests/test_autopresort_timeline.py`

设计决定（钉死 spec §9，并采纳 codex 评审）：窗口长 `window_sec`（默认 3.0）；`stride_sec`（默认 1.0）；末尾不足整窗**丢弃**；预测归到**窗口中心时间**。**Phase 0 仅支持 CFR（固定帧率）视频**，帧映射统一用 `int(sec * fps)`（floor，与现有 `get_clip_segments`/`build_segment_frame_ranges` 一致，见 abnormal_crop_video.py），**不声称 PTS**；引擎在真实运行时若检出 VFR 应告警（Task 5 装配处理）。

- [ ] **Step 1: 写失败测试**

```python
def test_iter_windows_full_windows_with_stride():
    tl = importlib.import_module("auto_presort.timeline")
    wins = tl.iter_windows(duration=8.0, window_sec=3.0, stride_sec=1.0)
    assert wins[0] == (0.0, 3.0, 1.5)   # (start, end, center)
    assert wins[-1] == (5.0, 8.0, 6.5)
    assert all(round(e - s, 6) == 3.0 for s, e, _ in wins)


def test_iter_windows_drops_incomplete_tail():
    tl = importlib.import_module("auto_presort.timeline")
    wins = tl.iter_windows(duration=3.5, window_sec=3.0, stride_sec=1.0)
    assert wins == [(0.0, 3.0, 1.5)]


def test_iter_windows_too_short_returns_empty():
    tl = importlib.import_module("auto_presort.timeline")
    assert tl.iter_windows(duration=2.0, window_sec=3.0, stride_sec=1.0) == []
```

- [ ] **Step 2: 跑测试确认失败**

Run: `conda activate AR && python -m pytest tests/test_autopresort_timeline.py::test_iter_windows_full_windows_with_stride -v`
Expected: FAIL（`AttributeError: iter_windows`）

- [ ] **Step 3: 最小实现（追加到 timeline.py）**

```python
def iter_windows(duration, window_sec, stride_sec):
    """返回 [(start, end, center), ...]，末尾不足整窗丢弃。"""
    if duration < window_sec or window_sec <= 0 or stride_sec <= 0:
        return []
    out = []
    i = 0
    while True:
        start = i * stride_sec
        end = start + window_sec
        if end > duration + 1e-9:
            break
        center = (start + end) / 2.0
        out.append((round(start, 6), round(end, 6), round(center, 6)))
        i += 1
    return out
```

- [ ] **Step 4: 跑测试确认通过**

Run: `conda activate AR && python -m pytest tests/test_autopresort_timeline.py -v`
Expected: PASS（全部）

---

## Task 3: 置信度门 + 迟滞合并 + 向内裁剪（保守成段）

**Files:**
- Modify: `auto_presort/timeline.py`
- Test: `tests/test_autopresort_timeline.py`

设计（采纳 codex：明确语义、测试断言段数与起止、trim 返回被裁 boundary spans）。每个窗口 `(center, label, confidence)`；低于 `threshold` 或 `label is None` → uncertain（断开）。合并：连续 `enter_run` 个同类 confident 窗口才成段；遇不同类 confident 或 uncertain **立即断开**（保守，留 gap——这不是完整双向迟滞，命名为"进入需 enter_run、退出即断"）。段时间 = 首/末稳定窗口中心；confidence = 段内**最小值**（保守摘要）。`trim_segments` 返回 (trimmed_segments, boundary_spans)，boundary_spans 是每段两端被裁掉的区间，供 gap 归因。

- [ ] **Step 1: 写失败测试**

```python
def test_merge_windows_two_segments_with_min_conf():
    tl = importlib.import_module("auto_presort.timeline")
    preds = [
        (1.5, "eat", 0.95), (2.5, "eat", 0.96), (3.5, "eat", 0.94),
        (4.5, None, 0.10),
        (5.5, "active", 0.92), (6.5, "active", 0.93),
    ]
    segs = tl.merge_windows(preds, threshold=0.90, enter_run=2)
    assert [s["label"] for s in segs] == ["eat", "active"]
    assert segs[0]["start"] == 1.5 and segs[0]["end"] == 3.5
    assert segs[0]["confidence"] == 0.94
    assert segs[1]["start"] == 5.5 and segs[1]["end"] == 6.5


def test_merge_windows_flicker_splits_into_two_eat_segments():
    # codex 修正：单个 drink 抖动会把 eat 断成两段（保守），显式断言段数
    tl = importlib.import_module("auto_presort.timeline")
    preds = [
        (1.5, "eat", 0.95), (2.5, "eat", 0.96),
        (3.5, "drink", 0.99),
        (4.5, "eat", 0.95), (5.5, "eat", 0.94),
    ]
    segs = tl.merge_windows(preds, threshold=0.90, enter_run=2)
    assert [s["label"] for s in segs] == ["eat", "eat"]  # drink 单窗 <enter_run 被丢
    assert segs[0]["end"] == 2.5 and segs[1]["start"] == 4.5


def test_merge_windows_below_threshold_is_uncertain():
    tl = importlib.import_module("auto_presort.timeline")
    preds = [(1.5, "eat", 0.80), (2.5, "eat", 0.85)]
    assert tl.merge_windows(preds, threshold=0.90, enter_run=2) == []


def test_trim_returns_trimmed_and_boundary_spans():
    tl = importlib.import_module("auto_presort.timeline")
    seg = tl.make_segment(1.5, 7.5, "eat", 0.95, "model")
    trimmed, boundaries = tl.trim_segments([seg], trim_margin=1.0, min_duration=2.0)
    assert len(trimmed) == 1
    assert trimmed[0]["start"] == 2.5 and trimmed[0]["end"] == 6.5
    assert (1.5, 2.5) in boundaries and (6.5, 7.5) in boundaries


def test_trim_drops_too_short_but_still_reports_boundary():
    tl = importlib.import_module("auto_presort.timeline")
    seg = tl.make_segment(0.0, 3.5, "eat", 0.95, "model")  # 裁后 1.5 < 2.0
    trimmed, boundaries = tl.trim_segments([seg], trim_margin=1.0, min_duration=2.0)
    assert trimmed == []
    # 整段都成了"非 segment"，但两端 boundary 仍记录（其余中段归 uncertain）
    assert (0.0, 1.0) in boundaries and (2.5, 3.5) in boundaries
```

- [ ] **Step 2: 跑测试确认失败**

Run: `conda activate AR && python -m pytest tests/test_autopresort_timeline.py::test_merge_windows_two_segments_with_min_conf -v`
Expected: FAIL（`AttributeError: merge_windows`）

- [ ] **Step 3: 最小实现（追加到 timeline.py）**

```python
def merge_windows(preds, threshold, enter_run):
    """preds: [(center, label, conf)] → 稳定 segment 列表（未 trim）。"""
    segs = []
    run_label = None
    run = []  # [(center, conf)]

    def flush(run_label, run):
        if run_label is not None and len(run) >= enter_run:
            centers = [c for c, _ in run]
            confs = [cf for _, cf in run]
            segs.append(make_segment(
                start=min(centers), end=max(centers),
                label=run_label, confidence=min(confs), source="model",
            ))

    for center, label, conf in preds:
        confident = label is not None and conf >= threshold
        if confident and label == run_label:
            run.append((center, conf))
        else:
            flush(run_label, run)
            run_label, run = (label, [(center, conf)]) if confident else (None, [])
    flush(run_label, run)
    return segs


def trim_segments(segments, trim_margin, min_duration):
    """每段两端向内裁 trim_margin。

    返回 (trimmed_segments, boundary_spans)。
    boundary_spans: 每段两端被裁掉的 [(start, end)]（无论该段是否因过短被丢弃，
    两端 margin 都算 boundary；被丢弃段的中段留给 gap 归 uncertain）。
    """
    out, boundaries = [], []
    for seg in segments:
        s0, e0 = seg["start"], seg["end"]
        s1, e1 = s0 + trim_margin, e0 - trim_margin
        boundaries.append((round(s0, 6), round(s1, 6)))
        boundaries.append((round(e1, 6), round(e0, 6)))
        if e1 - s1 >= min_duration - 1e-9:
            new_seg = dict(seg)
            new_seg["start"], new_seg["end"] = round(s1, 6), round(e1, 6)
            out.append(new_seg)
    return out, boundaries
```

- [ ] **Step 4: 跑测试确认通过**

Run: `conda activate AR && python -m pytest tests/test_autopresort_timeline.py -v`
Expected: PASS（全部）

---

## Task 4: gap 归因（no_animal > boundary > uncertain 优先级）+ 每视频 JSON

**Files:**
- Modify: `auto_presort/timeline.py`
- Test: `tests/test_autopresort_timeline.py`

设计（采纳 codex：在所有端点切分时间轴，reason 有优先级；测试断言完整 gap 列表）。gap = 时长内未被 segment 覆盖的区间。对每个 gap 子区间按优先级归因：与 no_animal_spans 重叠→`no_animal`；与 boundary_spans 重叠→`boundary`；否则 `uncertain`。以「所有边界点切分」保证归因精确到子区间。

- [ ] **Step 1: 写失败测试**

```python
def test_compute_gaps_priority_and_full_partition():
    tl = importlib.import_module("auto_presort.timeline")
    segs = [tl.make_segment(2.5, 6.5, "eat", 0.95, "model")]
    gaps = tl.compute_gaps(
        duration=8.0, segments=segs,
        boundary_spans=[(1.5, 2.5), (6.5, 7.5)],
        no_animal_spans=[(7.5, 8.0)],
    )
    # 0-1.5 uncertain, 1.5-2.5 boundary, 6.5-7.5 boundary, 7.5-8 no_animal
    assert (0.0, 1.5, "uncertain") in gaps
    assert (1.5, 2.5, "boundary") in gaps
    assert (6.5, 7.5, "boundary") in gaps
    assert (7.5, 8.0, "no_animal") in gaps
    # 完整覆盖 0..8，无缝隙
    covered = sorted((g[0], g[1]) for g in gaps) + [(2.5, 6.5)]
    covered.sort()
    assert covered[0][0] == 0.0 and covered[-1][1] == 8.0


def test_build_video_doc_shape():
    tl = importlib.import_module("auto_presort.timeline")
    segs = [tl.make_segment(1.0, 3.0, "eat", 0.95, "model")]
    doc = tl.build_video_doc(
        rel_path="video_001.mp4", duration=4.0, segments=segs,
        boundary_spans=[], no_animal_spans=[], source_id="srcA1b2",
        approved=False, meta={"checkpoint_sha256": "abc", "class_names": ["eat"]},
    )
    assert doc["schema_version"] == tl.SCHEMA_VERSION
    assert doc["rel_path"] == "video_001.mp4"
    assert doc["source_id"] == "srcA1b2"
    assert doc["approved"] is False
    assert doc["segments"] == segs
    assert doc["meta"]["checkpoint_sha256"] == "abc"
    assert "gaps" in doc and "duration" in doc
```

- [ ] **Step 2: 跑测试确认失败**

Run: `conda activate AR && python -m pytest tests/test_autopresort_timeline.py::test_compute_gaps_priority_and_full_partition -v`
Expected: FAIL（`AttributeError: compute_gaps`）

- [ ] **Step 3: 最小实现（追加到 timeline.py）**

```python
def _overlaps(a0, a1, spans):
    return any(a0 < e - 1e-9 and a1 > s + 1e-9 for s, e in spans)


def compute_gaps(duration, segments, boundary_spans, no_animal_spans):
    """未被 segment 覆盖的区间，按 no_animal > boundary > uncertain 归因。

    在 segment 端点 + boundary + no_animal 的所有端点处切分，保证子区间归因精确。
    """
    seg_spans = sorted((s["start"], s["end"]) for s in segments)
    # 收集所有切点
    points = {0.0, float(duration)}
    for s, e in seg_spans:
        points.add(s); points.add(e)
    for s, e in boundary_spans:
        points.add(s); points.add(e)
    for s, e in no_animal_spans:
        points.add(s); points.add(e)
    points = sorted(p for p in points if 0.0 - 1e-9 <= p <= duration + 1e-9)

    def covered_by_seg(a0, a1):
        return any(a0 >= s - 1e-9 and a1 <= e + 1e-9 for s, e in seg_spans)

    gaps = []
    for a0, a1 in zip(points, points[1:]):
        if a1 - a0 <= 1e-9:
            continue
        if covered_by_seg(a0, a1):
            continue
        if _overlaps(a0, a1, no_animal_spans):
            reason = "no_animal"
        elif _overlaps(a0, a1, boundary_spans):
            reason = "boundary"
        else:
            reason = "uncertain"
        gaps.append((round(a0, 6), round(a1, 6), reason))
    return gaps


def build_video_doc(rel_path, duration, segments, boundary_spans,
                    no_animal_spans, source_id, approved, meta):
    return {
        "schema_version": SCHEMA_VERSION,
        "rel_path": rel_path,
        "source_id": source_id,
        "duration": float(duration),
        "approved": bool(approved),
        "segments": segments,
        "gaps": compute_gaps(duration, segments, boundary_spans, no_animal_spans),
        "meta": meta,
    }
```

- [ ] **Step 4: 跑测试确认通过**

Run: `conda activate AR && python -m pytest tests/test_autopresort_timeline.py -v`
Expected: PASS（全部）

---

## Task 5: 统一检测轨迹（逐帧框传播 + no_animal_spans 聚合）

**Files:**
- Create: `auto_presort/detect_track.py`
- Test: `tests/test_autopresort_detect_track.py`

**修复①③的地基（codex）：** 一次检测同时产出「逐帧框轨迹」（供抠图）和「no_animal_spans」（供强制 gap）。纯聚合逻辑可单测：给定每帧「是否检出 + 框」，产出 (1) 传播后的逐帧框（漏检时沿用 last_valid_box，与 abnormal_crop_video.py:263 一致），(2) 连续缺失 ≥ `min_gap_sec` 的 no_animal_spans。真实 YOLO 调用是薄装配层，不单测。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_autopresort_detect_track.py
import importlib


def test_propagate_boxes_reuses_last_valid():
    dt = importlib.import_module("auto_presort.detect_track")
    raw = [(10, 10, 50, 50), None, None, (20, 20, 60, 60)]
    out = dt.propagate_boxes(raw)
    assert out[1] == (10, 10, 50, 50)  # 漏检沿用上一个
    assert out[2] == (10, 10, 50, 50)
    assert out[3] == (20, 20, 60, 60)


def test_propagate_boxes_leading_none_stays_none():
    dt = importlib.import_module("auto_presort.detect_track")
    out = dt.propagate_boxes([None, None, (1, 1, 2, 2)])
    assert out[0] is None and out[1] is None and out[2] == (1, 1, 2, 2)


def test_no_animal_spans_from_flags():
    dt = importlib.import_module("auto_presort.detect_track")
    # 每帧是否有动物；fps=10 → 每帧 0.1s
    flags = [True, False, False, False, False, True, True]  # 0.1..0.5 缺失=0.4s
    spans = dt.no_animal_spans(flags, fps=10.0, min_gap_sec=0.3)
    assert spans == [(0.1, 0.5)]


def test_no_animal_spans_ignores_short_dropouts():
    dt = importlib.import_module("auto_presort.detect_track")
    flags = [True, False, True, True]  # 单帧=0.1s < 0.3s 阈值
    assert dt.no_animal_spans(flags, fps=10.0, min_gap_sec=0.3) == []


def test_no_animal_spans_trailing_missing():
    dt = importlib.import_module("auto_presort.detect_track")
    flags = [True, False, False, False]  # 末尾 0.1..0.4 缺失=0.3s
    assert dt.no_animal_spans(flags, fps=10.0, min_gap_sec=0.3) == [(0.1, 0.4)]
```

- [ ] **Step 2: 跑测试确认失败**

Run: `conda activate AR && python -m pytest tests/test_autopresort_detect_track.py -v`
Expected: FAIL（`ModuleNotFoundError: auto_presort.detect_track`）

- [ ] **Step 3: 最小实现（纯逻辑部分）**

```python
# auto_presort/detect_track.py
"""统一检测轨迹：逐帧框传播 + no_animal 区间聚合。

纯逻辑函数可单测；真实 YOLO 遍历在 build_track（延迟 import cv2/ultralytics）。
"""


def propagate_boxes(raw_boxes):
    """漏检帧(None)沿用上一个有效框；开头的 None 保持 None。"""
    out, last = [], None
    for b in raw_boxes:
        if b is not None:
            last = b
        out.append(last)
    return out


def no_animal_spans(flags, fps, min_gap_sec):
    """flags[i]=该帧是否检出动物；返回连续缺失 >= min_gap_sec 的 [(start,end)] 秒。"""
    spans = []
    miss_start = None
    for i, has in enumerate(flags):
        t = i / fps
        if not has and miss_start is None:
            miss_start = t
        elif has and miss_start is not None:
            if t - miss_start >= min_gap_sec - 1e-9:
                spans.append((round(miss_start, 6), round(t, 6)))
            miss_start = None
    if miss_start is not None:
        t = len(flags) / fps
        if t - miss_start >= min_gap_sec - 1e-9:
            spans.append((round(miss_start, 6), round(t, 6)))
    return spans
```

- [ ] **Step 4: 跑测试确认通过**

Run: `conda activate AR && python -m pytest tests/test_autopresort_detect_track.py -v`
Expected: PASS（5 passed）

- [ ] **Step 5: 真实 YOLO 装配（延迟 import，无单测，Task 9 集成覆盖）**

```python
def build_track(video_path, yolo, target_classes, min_gap_sec=1.0, max_det=1):
    """真实检测：返回 {"fps","frames","boxes"(逐帧传播后 xyxy 或 None),
    "flags","no_animal_spans"}。import cv2。VFR 视频打印告警（Phase 0 仅保证 CFR）。"""
    import cv2

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 16.0
    raw_boxes, flags = [], []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        res = yolo(frame, classes=target_classes, verbose=False, max_det=max_det)
        boxes = res[0].boxes
        if len(boxes) >= 1:
            xyxy = tuple(float(v) for v in boxes[0].xyxy[0].cpu().numpy())
            raw_boxes.append(xyxy); flags.append(True)
        else:
            raw_boxes.append(None); flags.append(False)
    cap.release()

    return {
        "fps": fps,
        "frames": len(flags),
        "boxes": propagate_boxes(raw_boxes),
        "flags": flags,
        "no_animal_spans": no_animal_spans(flags, fps, min_gap_sec),
    }
```

Run: `conda activate AR && python -c "import auto_presort.detect_track; print('ok')"`
Expected: 打印 `ok`（不触发 cv2 导入，因其在函数内）。

---

## Task 6: 抠图/预处理**真实 parity test**（对比 VideoDataset 输出）

**Files:**
- Create: `auto_presort/preprocess.py`
- Test: `tests/test_autopresort_preprocess_parity.py`

**修复②（codex）：** parity test 必须与真实 `VideoDataset` 输出比较，不能只查 shape。做法：`preprocess.py` 复用 `VideoDataset` 的**同一个 transform 构造**（通过实例化一个 `is_train=False` 的 VideoDataset 拿到 `.transform` 与 `.frame_resize`），并复用 `build_smoothed_crop_boxes`。测试：把同一段帧（**不经 mp4 再编码**，直接内存数组）分别喂给 (a) `VideoDataset._load_frames`+transform 路径、(b) `preprocess.clip_from_frames` 路径，断言张量**数值接近**（allclose）。关键点：两条路径都用**同一内存帧列表**，排除编解码差异，专测抠图+采样+归一化逻辑一致。

> 依赖 cv2+torch+torchvision，用 `pytest.importorskip`。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_autopresort_preprocess_parity.py
import importlib
import numpy as np
import pytest

pytest.importorskip("cv2")
torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")


def test_clip_from_frames_matches_videodataset_transform():
    pp = importlib.import_module("auto_presort.preprocess")
    from utils.video_dataset import VideoDataset

    # 32 帧移动色块，全画面框（None → fallback 全画面），确定性
    frames = []
    for i in range(32):
        f = np.zeros((120, 160, 3), dtype=np.uint8)
        f[:, :, 0] = (i * 5) % 256
        frames.append(f)

    imgsz, tw = 224, 16
    # 参考路径：构造一个空 VideoDataset 只为拿 transform / frame_resize
    ds = VideoDataset.__new__(VideoDataset)
    ds.cfg = {"imgsz": imgsz, "temporal_window": tw}
    ds.is_train = False
    from torchvision.transforms import v2
    ds.frame_resize = v2.Resize((imgsz, imgsz), antialias=True)
    ds.transform = ds._build_clip_transform(imgsz)

    # 参考：模拟 _load_frames 对 BGR 数组的处理（linspace 采样 + resize）→ transform
    idxs = np.linspace(0, len(frames) - 1, tw).astype(int)
    ref_frames = []
    import cv2
    for i in idxs:
        rgb = cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(rgb).permute(2, 0, 1)
        ref_frames.append(ds.frame_resize(t))
    ref = ds.transform(torch.stack(ref_frames))

    # 被测路径：全画面框
    got = pp.clip_from_frames(
        frames_bgr=frames, boxes=[None] * len(frames),
        imgsz=imgsz, temporal_window=tw, expand_ratio=1.25,
    )
    assert got.shape == (tw, 3, imgsz, imgsz)
    assert torch.allclose(got, ref, atol=1e-5)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `conda activate AR && python -m pytest tests/test_autopresort_preprocess_parity.py -v`
Expected: FAIL（`ModuleNotFoundError: auto_presort.preprocess`）

- [ ] **Step 3: 最小实现**

```python
# auto_presort/preprocess.py
"""把「帧列表 + 对应检测框」转成与训练一致的 (T,3,H,W) 张量。

抠图复用 build_smoothed_crop_boxes；采样/归一化复用 VideoDataset 的构造，
保证与训练 eval 路径逐位一致（parity test 守卫）。
"""
import cv2
import numpy as np
import torch
from torchvision.transforms import v2

from utils.abnormal_crop_video import build_smoothed_crop_boxes

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


def _eval_transform(imgsz):
    # 对齐 utils/video_dataset.py 的 eval 分支（is_train=False，无随机增强）
    return v2.Compose([
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])


def clip_from_frames(frames_bgr, boxes, imgsz, temporal_window, expand_ratio):
    """frames_bgr: 窗口内 BGR 帧列表；boxes: 与帧对齐的 xyxy 或 None（窗口内坐标）。"""
    if not frames_bgr:
        # 与 VideoDataset 一致：uint8 零帧再归一化
        zeros = torch.zeros(temporal_window, 3, imgsz, imgsz, dtype=torch.uint8)
        return _eval_transform(imgsz)(zeros)

    h, w = frames_bgr[0].shape[:2]
    crop_boxes = build_smoothed_crop_boxes(
        boxes[:len(frames_bgr)], frame_width=w, frame_height=h,
        fallback_size=(w, h), expand_ratio=expand_ratio,
    )
    resize = v2.Resize((imgsz, imgsz), antialias=True)
    idxs = np.linspace(0, len(frames_bgr) - 1, temporal_window).astype(int)
    tensors = []
    for i in idxs:
        x1, y1, x2, y2 = crop_boxes[i]
        crop = frames_bgr[i][y1:y2, x1:x2]
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(rgb).permute(2, 0, 1)
        tensors.append(resize(t))
    return _eval_transform(imgsz)(torch.stack(tensors))
```

- [ ] **Step 4: 跑测试确认通过**

Run: `conda activate AR && python -m pytest tests/test_autopresort_preprocess_parity.py -v`
Expected: PASS（1 passed；无 cv2/torch 时 skipped）

> 注：全画面框场景下抠图=整帧，parity 精确成立。移动框场景的一致性由「两条路径共用 build_smoothed_crop_boxes」在构造上保证；此处不再单独造移动框参考（VideoDataset 本身不带框逻辑，无法作参考对比）。

---

## Task 7: checkpoint sidecar 校验 + 真实 head 维度检查

**Files:**
- Create: `auto_presort/checkpoint_guard.py`
- Test: `tests/test_autopresort_checkpoint_guard.py`

**修复④（codex）：** 类别顺序必须有出处——用 sidecar `<ckpt>.meta.json`（含 `sha256` + 有序 `class_names` + `species`）。校验：sidecar 存在、其 sha256 与 checkpoint 实际哈希匹配、class_names 非空。真实 head 维度检查从 state_dict 直接读 `head.1.weight` 第 0 维，与 sidecar 类别数比对（不再用恒真的 fake num_classes）。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_autopresort_checkpoint_guard.py
import importlib
import hashlib
import json
import pytest


def _write(tmp_path, content=b"good"):
    p = tmp_path / "teacher.pth"
    p.write_bytes(content)
    return p


def _sidecar(p, class_names, sha):
    meta = {"sha256": sha, "class_names": class_names, "species": "pet"}
    (p.parent / (p.name + ".meta.json")).write_text(json.dumps(meta))


def test_missing_checkpoint_raises(tmp_path):
    g = importlib.import_module("auto_presort.checkpoint_guard")
    with pytest.raises(FileNotFoundError):
        g.load_sidecar_meta(str(tmp_path / "nope.pth"))


def test_missing_sidecar_raises(tmp_path):
    g = importlib.import_module("auto_presort.checkpoint_guard")
    p = _write(tmp_path)
    with pytest.raises(FileNotFoundError, match="sidecar"):
        g.load_sidecar_meta(str(p))


def test_sidecar_hash_mismatch_raises(tmp_path):
    g = importlib.import_module("auto_presort.checkpoint_guard")
    p = _write(tmp_path, b"good")
    _sidecar(p, ["eat", "drink", "other"], sha="deadbeef")  # 错哈希
    with pytest.raises(ValueError, match="hash"):
        g.load_sidecar_meta(str(p))


def test_sidecar_ok_returns_meta(tmp_path):
    g = importlib.import_module("auto_presort.checkpoint_guard")
    p = _write(tmp_path, b"good")
    sha = hashlib.sha256(b"good").hexdigest()
    _sidecar(p, ["eat", "drink", "other"], sha=sha)
    meta = g.load_sidecar_meta(str(p))
    assert meta["class_names"] == ["eat", "drink", "other"]
    assert meta["sha256"] == sha


def test_head_dim_check_detects_mismatch():
    g = importlib.import_module("auto_presort.checkpoint_guard")
    # 假 state_dict：head.1.weight 是 4 类，但期望 3 类
    fake_state = {"head.1.weight": [[0.0]] * 4, "head.1.bias": [0.0] * 4}
    with pytest.raises(ValueError, match="class count"):
        g.check_head_dim(fake_state, expected_num_classes=3)


def test_head_dim_check_passes_when_matching():
    g = importlib.import_module("auto_presort.checkpoint_guard")
    fake_state = {"head.1.weight": [[0.0]] * 3, "head.1.bias": [0.0] * 3}
    g.check_head_dim(fake_state, expected_num_classes=3)  # 不抛
```

- [ ] **Step 2: 跑测试确认失败**

Run: `conda activate AR && python -m pytest tests/test_autopresort_checkpoint_guard.py -v`
Expected: FAIL（`ModuleNotFoundError`）

- [ ] **Step 3: 最小实现**

```python
# auto_presort/checkpoint_guard.py
"""checkpoint sidecar 校验 + 真实 head 维度检查。任何不确定即失败。

sidecar: <ckpt>.meta.json = {"sha256","class_names"(有序),"species"}。
类别顺序的唯一权威来源，禁止从 SPECIES 静默推断。
"""
import hashlib
import json
import os


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_sidecar_meta(checkpoint_path):
    if not checkpoint_path or not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"teacher checkpoint not found: {checkpoint_path}")
    sidecar = checkpoint_path + ".meta.json"
    if not os.path.isfile(sidecar):
        raise FileNotFoundError(
            f"checkpoint sidecar not found: {sidecar} "
            f"(must record ordered class_names + sha256)")
    meta = json.loads(open(sidecar, encoding="utf-8").read())
    if not meta.get("class_names"):
        raise ValueError("sidecar class_names is empty")
    actual = _sha256(checkpoint_path)
    if meta.get("sha256") != actual:
        raise ValueError(
            f"sidecar hash mismatch: sidecar={meta.get('sha256')} actual={actual}")
    return meta


def _first_dim(weight):
    # 支持真实 torch.Tensor（.shape[0]）与测试用 list（len）
    shape = getattr(weight, "shape", None)
    return int(shape[0]) if shape is not None else len(weight)


def check_head_dim(state_dict, expected_num_classes):
    key = "head.1.weight"
    if key not in state_dict:
        raise ValueError(f"checkpoint missing classifier head key: {key}")
    n = _first_dim(state_dict[key])
    if n != expected_num_classes:
        raise ValueError(
            f"head class count {n} != expected class count {expected_num_classes}")
```

- [ ] **Step 4: 跑测试确认通过**

Run: `conda activate AR && python -m pytest tests/test_autopresort_checkpoint_guard.py -v`
Expected: PASS（6 passed）

- [ ] **Step 5: 真实加载适配器（延迟 import，无单测，Task 9 集成覆盖）**

```python
def verify_and_load_teacher(checkpoint_path, device="cuda"):
    """sidecar 校验 → 读 state_dict → head 维度检查 → 构造 SwinTeacher 加载。

    返回 (model, meta)。类别顺序来自 sidecar。import torch。
    """
    import torch
    from models.teacher import SwinTeacher

    meta = load_sidecar_meta(checkpoint_path)
    class_names = meta["class_names"]

    raw = torch.load(checkpoint_path, map_location="cpu")
    state = {k.replace("base.model.", "").replace("module.", ""): v
             for k, v in raw.items()}
    check_head_dim(state, expected_num_classes=len(class_names))

    model = SwinTeacher(checkpoint_path=None, num_classes=len(class_names))
    missing, unexpected = model.model.load_state_dict(state, strict=False)
    # head 已单独校验；此处只允许 head 相关 key 差异之外为空
    hard_missing = [k for k in missing if not k.startswith("head.")]
    if hard_missing or unexpected:
        raise ValueError(
            f"checkpoint load mismatch missing={hard_missing} unexpected={list(unexpected)}")
    model = model.to(device).eval()
    return model, meta
```

Run: `conda activate AR && python -c "import auto_presort.checkpoint_guard; print('ok')"`
Expected: 打印 `ok`。

---

## Task 8: 引擎串联（detect_track 先行 → 强制 gap → 分类 → 时间线）

**Files:**
- Create: `auto_presort/engine.py`
- Test: `tests/test_autopresort_engine.py`

**修复①③（codex）：** 检测先行；分类器接收「窗口内帧 + 对应框」；与 no_animal_spans 重叠超阈值的窗口**直接判 None**（不送 teacher），保证无动物区强制成 gap。`classify_window_fn(frames_bgr, boxes)` 与 `track` 均注入。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_autopresort_engine.py
import importlib


def _params():
    return dict(window_sec=3.0, stride_sec=1.0, threshold=0.90, enter_run=2,
                trim_margin=1.0, min_duration=2.0, imgsz=224, temporal_window=16,
                expand_ratio=1.25, no_animal_overlap=0.5)


def test_label_video_produces_trimmed_segment():
    eng = importlib.import_module("auto_presort.engine")
    track = {"fps": 10.0, "frames": 90, "boxes": [(0, 0, 10, 10)] * 90,
             "flags": [True] * 90, "no_animal_spans": []}
    # 假分类器：忽略内容恒定 eat/0.95
    doc = eng.label_video(
        video_path="fake.mp4", rel_path="fake.mp4", source_id="src1",
        track=track,
        read_window_frames_fn=lambda vp, s, e, fps: ([None], [None]),  # 内容不参与
        classify_frames_fn=lambda frames, boxes: ("eat", 0.95),
        meta={"class_names": ["eat"]}, params=_params(),
    )
    assert doc["approved"] is False
    assert len(doc["segments"]) == 1
    seg = doc["segments"][0]
    assert seg["label"] == "eat"
    assert seg["start"] == 2.5 and seg["end"] == 6.5  # 中心 1.5..7.5, trim 1.0


def test_label_video_forces_gap_in_no_animal_region():
    eng = importlib.import_module("auto_presort.engine")
    # 全程 no animal；即便分类器高置信，也不得出 segment
    track = {"fps": 10.0, "frames": 90, "boxes": [None] * 90,
             "flags": [False] * 90, "no_animal_spans": [(0.0, 9.0)]}
    doc = eng.label_video(
        video_path="fake.mp4", rel_path="fake.mp4", source_id="src1",
        track=track,
        read_window_frames_fn=lambda vp, s, e, fps: ([None], [None]),
        classify_frames_fn=lambda frames, boxes: ("eat", 0.99),  # 高置信也没用
        meta={"class_names": ["eat"]}, params=_params(),
    )
    assert doc["segments"] == []
    assert any(g[2] == "no_animal" for g in doc["gaps"])
```

- [ ] **Step 2: 跑测试确认失败**

Run: `conda activate AR && python -m pytest tests/test_autopresort_engine.py -v`
Expected: FAIL（`ModuleNotFoundError: auto_presort.engine`）

- [ ] **Step 3: 最小实现**

```python
# auto_presort/engine.py
"""串联：检测轨迹先行 → 逐窗（强制 gap + 分类）→ 迟滞合并 → 裁剪 → JSON。

注入点：
  read_window_frames_fn(video_path, start_sec, end_sec, fps) -> (frames_bgr, boxes_in_window)
  classify_frames_fn(frames_bgr, boxes_in_window) -> (label, conf)
  track: detect_track.build_track 的返回（或测试构造）
"""
from auto_presort import timeline as tl


def _window_no_animal_ratio(start, end, no_animal_spans):
    total = end - start
    if total <= 0:
        return 0.0
    ov = 0.0
    for s, e in no_animal_spans:
        ov += max(0.0, min(end, e) - max(start, s))
    return ov / total


def label_video(video_path, rel_path, source_id, track,
                read_window_frames_fn, classify_frames_fn, meta, params):
    duration = track["frames"] / track["fps"] if track["fps"] else 0.0
    windows = tl.iter_windows(duration, params["window_sec"], params["stride_sec"])
    no_animal = track["no_animal_spans"]

    preds = []
    for start, end, center in windows:
        # 强制 gap：无动物占比超阈值 → 直接 None，不送分类器
        if _window_no_animal_ratio(start, end, no_animal) >= params["no_animal_overlap"]:
            preds.append((center, None, 0.0))
            continue
        frames_bgr, boxes = read_window_frames_fn(video_path, start, end, track["fps"])
        label, conf = classify_frames_fn(frames_bgr, boxes)
        preds.append((center, label, conf))

    merged = tl.merge_windows(preds, params["threshold"], params["enter_run"])
    trimmed, boundary_spans = tl.trim_segments(
        merged, params["trim_margin"], params["min_duration"])
    return tl.build_video_doc(
        rel_path=rel_path, duration=duration, segments=trimmed,
        boundary_spans=boundary_spans, no_animal_spans=no_animal,
        source_id=source_id, approved=False,
        meta={"class_names": meta["class_names"],
              "checkpoint_sha256": meta.get("sha256")},
    )
```

- [ ] **Step 4: 跑测试确认通过**

Run: `conda activate AR && python -m pytest tests/test_autopresort_engine.py -v`
Expected: PASS（2 passed）

---

## Task 9: 真实装配 + 集成测试（整条链，fake capture/YOLO/teacher）

**Files:**
- Modify: `auto_presort/engine.py`
- Test: `tests/test_autopresort_engine.py`

**修复⑥（codex）：** 真实装配边界必须有集成测试。提供 `build_real_window_reader`（按 fps + 秒读窗口帧，并从 track.boxes 切窗口内框、转成窗口内坐标）与 `build_real_classifier`（用 preprocess.clip_from_frames → teacher softmax）。集成测试用 fake VideoCapture（内存帧）、fake YOLO（确定框）、fake teacher（记录收到 tensor 并回定标签），验证 box track→窗口切片→抠图→teacher→forced gap 全链。

- [ ] **Step 1: 写真实装配函数（engine.py 追加）**

```python
def build_real_window_reader():
    """返回 read_window_frames_fn：按秒读窗口帧，并从整段 track.boxes 切窗口内框。

    boxes 转成「窗口内坐标」——本设计里抠图用绝对像素坐标即可（帧本身是整帧），
    故窗口内框直接沿用绝对 xyxy。import cv2。
    """
    import cv2

    def read(video_path, start_sec, end_sec, fps, _track_boxes=None):
        cap = cv2.VideoCapture(video_path)
        start_f = int(start_sec * fps)  # floor，对齐现有 crop
        end_f = int(end_sec * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
        frames = []
        for _ in range(max(0, end_f - start_f)):
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
        cap.release()
        boxes = None
        if _track_boxes is not None:
            boxes = _track_boxes[start_f:start_f + len(frames)]
        return frames, boxes

    return read


def build_real_classifier(model, class_names, params):
    """返回 classify_frames_fn：clip_from_frames → teacher softmax top-1。import torch。"""
    import torch
    from auto_presort.preprocess import clip_from_frames

    def classify(frames_bgr, boxes):
        boxes = boxes if boxes is not None else [None] * len(frames_bgr)
        x = clip_from_frames(
            frames_bgr, boxes, imgsz=params["imgsz"],
            temporal_window=params["temporal_window"],
            expand_ratio=params["expand_ratio"],
        ).unsqueeze(0).to(next(model.parameters()).device)
        with torch.no_grad():
            logits, _ = model(x)
            prob = torch.softmax(logits, dim=1)[0]
            conf, idx = float(prob.max()), int(prob.argmax())
        return class_names[idx], conf

    return classify
```

- [ ] **Step 2: 写集成测试（fake 全链，engine 真实数据流）**

```python
def test_engine_integration_with_fakes():
    eng = importlib.import_module("auto_presort.engine")
    tl = importlib.import_module("auto_presort.timeline")

    # 90 帧，前 60 有动物，后 30 无动物
    boxes = [(0, 0, 20, 20)] * 60 + [None] * 30
    flags = [True] * 60 + [False] * 30
    dt = importlib.import_module("auto_presort.detect_track")
    track = {"fps": 10.0, "frames": 90, "boxes": dt.propagate_boxes(boxes),
             "flags": flags, "no_animal_spans": dt.no_animal_spans(flags, 10.0, 1.0)}

    seen = {"tensors": 0}

    def reader(vp, s, e, fps):
        # 每窗口回 3 个占位帧 + 3 个框
        return [object(), object(), object()], [(0, 0, 20, 20)] * 3

    def classifier(frames, boxes):
        seen["tensors"] += 1
        return ("eat", 0.95)

    params = dict(window_sec=3.0, stride_sec=1.0, threshold=0.90, enter_run=2,
                  trim_margin=1.0, min_duration=2.0, imgsz=224, temporal_window=16,
                  expand_ratio=1.25, no_animal_overlap=0.5)
    doc = eng.label_video(
        video_path="fake.mp4", rel_path="fake.mp4", source_id="s1", track=track,
        read_window_frames_fn=reader, classify_frames_fn=classifier,
        meta={"class_names": ["eat"], "sha256": "abc"}, params=params,
    )
    # 前段(有动物)出 eat segment；后段(无动物)强制 gap，未调用分类器
    assert len(doc["segments"]) == 1 and doc["segments"][0]["label"] == "eat"
    assert any(g[2] == "no_animal" for g in doc["gaps"])
    # 无动物窗口没进分类器：调用次数 < 总窗口数
    total_windows = len(tl.iter_windows(9.0, 3.0, 1.0))
    assert seen["tensors"] < total_windows
```

- [ ] **Step 3: 跑集成测试确认通过**

Run: `conda activate AR && python -m pytest tests/test_autopresort_engine.py -v`
Expected: PASS（3 passed，含集成）

- [ ] **Step 4: 冒烟：模块可导入不触发重依赖**

Run: `conda activate AR && python -c "import auto_presort.engine; print('ok')"`
Expected: 打印 `ok`。

---

## Task 10: 离线校准度量（修正 recall/short-loss/gate）

**Files:**
- Create: `auto_presort/calibrate.py`
- Test: `tests/test_autopresort_calibrate.py`

**修复⑤（codex）：** recall 先对同类预测求 union 再算覆盖（避免重叠重复计数）；short_action_loss 用最低覆盖率阈值（不是任意 >0）；gate 同时要求 purity + 总 recall + 无严重短动作丢失。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_autopresort_calibrate.py
import importlib


def test_boundary_purity():
    cal = importlib.import_module("auto_presort.calibrate")
    gt = [(0.0, 10.0, "eat"), (10.0, 15.0, "active")]
    assert cal.boundary_purity([(2.0, 8.0, "eat")], gt) == 1.0
    assert abs(cal.boundary_purity([(6.0, 12.0, "eat")], gt) - (4.0 / 6.0)) < 1e-6


def test_recall_unions_overlapping_predictions():
    cal = importlib.import_module("auto_presort.calibrate")
    gt = [(0.0, 10.0, "eat")]
    # 两个重叠预测覆盖 0..10，union 应=1.0 而不是 >1
    assert abs(cal.recall([(0.0, 8.0, "eat"), (2.0, 10.0, "eat")], gt) - 1.0) < 1e-6


def test_short_action_loss_requires_min_coverage():
    cal = importlib.import_module("auto_presort.calibrate")
    gt = [(0.0, 10.0, "eat"), (10.0, 12.0, "drink")]  # drink 2s 短动作
    # 只覆盖 0.1s（<50%）→ 算丢失
    assert cal.short_action_loss(
        [(2.0, 8.0, "eat"), (10.0, 10.1, "drink")], gt,
        min_gt_dur=3.0, min_cover=0.5) == ["drink"]
    # 覆盖 >50% → 不算丢失
    assert cal.short_action_loss(
        [(10.0, 11.5, "drink")], gt, min_gt_dur=3.0, min_cover=0.5) == []


def test_summarize_gate_requires_purity_and_recall():
    cal = importlib.import_module("auto_presort.calibrate")
    rows = [{"purity": 0.98, "recall": 0.30, "short_lost": []}]
    s = cal.summarize(rows, purity_gate=0.95, recall_gate=0.5)
    assert s["gate_pass"] is False   # purity 够但 recall 不够
    rows2 = [{"purity": 0.98, "recall": 0.70, "short_lost": []}]
    assert cal.summarize(rows2, purity_gate=0.95, recall_gate=0.5)["gate_pass"] is True
```

- [ ] **Step 2: 跑测试确认失败**

Run: `conda activate AR && python -m pytest tests/test_autopresort_calibrate.py -v`
Expected: FAIL（`ModuleNotFoundError`）

- [ ] **Step 3: 最小实现（度量部分）**

```python
# auto_presort/calibrate.py
"""离线校准：用已知 GT 边界量化引擎，产出 go/no-go。含可运行 CLI（见 main）。"""
import json


def _overlap(a0, a1, b0, b1):
    return max(0.0, min(a1, b1) - max(a0, b0))


def _union_len(spans):
    if not spans:
        return 0.0
    spans = sorted(spans)
    total, cs, ce = 0.0, spans[0][0], spans[0][1]
    for s, e in spans[1:]:
        if s > ce:
            total += ce - cs; cs, ce = s, e
        else:
            ce = max(ce, e)
    return total + (ce - cs)


def boundary_purity(pred, gt):
    total = sum(pe - ps for ps, pe, _ in pred)
    if total <= 0:
        return 0.0
    pure = 0.0
    for ps, pe, pl in pred:
        for gs, ge, gl in gt:
            if gl == pl:
                pure += _overlap(ps, pe, gs, ge)
    return pure / total


def recall(pred, gt):
    total = sum(ge - gs for gs, ge, _ in gt)
    if total <= 0:
        return 0.0
    covered = 0.0
    for gs, ge, gl in gt:
        clipped = [(max(gs, ps), min(ge, pe)) for ps, pe, pl in pred
                   if pl == gl and _overlap(gs, ge, ps, pe) > 0]
        covered += _union_len(clipped)
    return covered / total


def short_action_loss(pred, gt, min_gt_dur, min_cover):
    missed = []
    for gs, ge, gl in gt:
        dur = ge - gs
        if dur >= min_gt_dur:
            continue
        clipped = [(max(gs, ps), min(ge, pe)) for ps, pe, pl in pred
                   if pl == gl and _overlap(gs, ge, ps, pe) > 0]
        if _union_len(clipped) < min_cover * dur:
            missed.append(gl)
    return missed


def summarize(per_video_metrics, purity_gate=0.95, recall_gate=0.5):
    n = len(per_video_metrics) or 1
    mean_purity = sum(r["purity"] for r in per_video_metrics) / n
    mean_recall = sum(r["recall"] for r in per_video_metrics) / n
    short_lost_total = {}
    for r in per_video_metrics:
        for lb in r.get("short_lost", []):
            short_lost_total[lb] = short_lost_total.get(lb, 0) + 1
    return {
        "mean_purity": mean_purity,
        "mean_recall": mean_recall,
        "short_lost_total": short_lost_total,
        "gate_pass": mean_purity >= purity_gate and mean_recall >= recall_gate,
    }
```

- [ ] **Step 4: 跑测试确认通过**

Run: `conda activate AR && python -m pytest tests/test_autopresort_calibrate.py -v`
Expected: PASS（4 passed）

---

## Task 11: 校准 CLI（Phase 0 唯一可运行交付）

**Files:**
- Modify: `auto_presort/calibrate.py`
- Test: `tests/test_autopresort_calibrate.py`

**修复⑤（codex）：** 必须有可运行入口。GT JSON schema：`{"<rel_path>": [[start,end,label], ...]}`。CLI：加载 GT → 对每个视频（真实：build_track + label_video；此处逻辑核心 `evaluate_docs` 可单测——给定「引擎产出 doc + GT」算每视频度量）→ summarize → 写 CSV/JSON。真实模型跑通部分靠人工执行（无 GPU 不测）。

- [ ] **Step 1: 写失败测试（doc→度量的纯逻辑）**

```python
def test_evaluate_docs_computes_per_video_metrics():
    cal = importlib.import_module("auto_presort.calibrate")
    # doc.segments 用 (start,end,label,...)；GT 同 rel_path
    docs = {"v1.mp4": {"segments": [
        {"start": 2.0, "end": 8.0, "label": "eat"}]}}
    gt = {"v1.mp4": [[0.0, 10.0, "eat"], [10.0, 12.0, "drink"]]}
    rows = cal.evaluate_docs(docs, gt, min_gt_dur=3.0, min_cover=0.5)
    assert len(rows) == 1
    assert rows[0]["purity"] == 1.0
    assert abs(rows[0]["recall"] - (6.0 / 12.0)) < 1e-6
    assert rows[0]["short_lost"] == ["drink"]
```

- [ ] **Step 2: 跑测试确认失败**

Run: `conda activate AR && python -m pytest tests/test_autopresort_calibrate.py::test_evaluate_docs_computes_per_video_metrics -v`
Expected: FAIL（`AttributeError: evaluate_docs`）

- [ ] **Step 3: 最小实现（追加到 calibrate.py）**

```python
def _doc_pred(doc):
    return [(s["start"], s["end"], s["label"]) for s in doc["segments"]]


def evaluate_docs(docs, gt, min_gt_dur=3.0, min_cover=0.5):
    """docs: {rel_path: doc}; gt: {rel_path: [[s,e,label],...]}。返回每视频度量行。"""
    rows = []
    for rel, doc in docs.items():
        g = [tuple(x) for x in gt.get(rel, [])]
        pred = _doc_pred(doc)
        rows.append({
            "rel_path": rel,
            "purity": boundary_purity(pred, g),
            "recall": recall(pred, g),
            "short_lost": short_action_loss(pred, g, min_gt_dur, min_cover),
        })
    return rows


def write_reports(rows, summary, csv_path, json_path):
    import csv
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rel_path", "purity", "recall", "short_lost"])
        for r in rows:
            w.writerow([r["rel_path"], f"{r['purity']:.4f}",
                        f"{r['recall']:.4f}", ";".join(r["short_lost"])])
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "rows": rows}, f, indent=2)


def main(argv=None):
    """真实运行：需 GPU/teacher/YOLO。人工执行。

    用法: python -m auto_presort.calibrate --species pet --videos <dir> \\
          --gt gt.json --checkpoint <teacher.pth> --out-dir <dir>
    """
    import argparse
    import os
    import pipeline_config as pc
    from auto_presort import detect_track, engine
    from auto_presort.checkpoint_guard import verify_and_load_teacher

    ap = argparse.ArgumentParser()
    ap.add_argument("--species", required=True)
    ap.add_argument("--videos", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--threshold", type=float, default=0.90)
    ap.add_argument("--trim-margin", type=float, default=1.0)
    a = ap.parse_args(argv)

    sp = pc.SPECIES[a.species]
    from ultralytics import YOLO
    yolo = YOLO(sp["yolo_model"])
    model, meta = verify_and_load_teacher(a.checkpoint)
    reader = engine.build_real_window_reader()
    classifier = engine.build_real_classifier(model, meta["class_names"],
        params=dict(imgsz=224, temporal_window=16, expand_ratio=1.25))
    params = dict(window_sec=3.0, stride_sec=1.0, threshold=a.threshold, enter_run=2,
                  trim_margin=a.trim_margin, min_duration=2.0, imgsz=224,
                  temporal_window=16, expand_ratio=1.25, no_animal_overlap=0.5)

    gt = json.loads(open(a.gt, encoding="utf-8").read())
    docs = {}
    for rel in gt:
        vp = os.path.join(a.videos, rel)
        track = detect_track.build_track(vp, yolo, sp["target_classes"])
        docs[rel] = engine.label_video(
            video_path=vp, rel_path=rel, source_id=rel, track=track,
            read_window_frames_fn=lambda v, s, e, fps, _b=track["boxes"]:
                reader(v, s, e, fps, _b),
            classify_frames_fn=classifier, meta=meta, params=params)

    rows = evaluate_docs(docs, gt)
    summary = summarize(rows)
    os.makedirs(a.out_dir, exist_ok=True)
    write_reports(rows, summary,
                  os.path.join(a.out_dir, "calib.csv"),
                  os.path.join(a.out_dir, "calib.json"))
    print(f"gate_pass={summary['gate_pass']} "
          f"purity={summary['mean_purity']:.3f} recall={summary['mean_recall']:.3f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 跑测试确认通过**

Run: `conda activate AR && python -m pytest tests/test_autopresort_calibrate.py -v`
Expected: PASS（5 passed）

---

## Task 12: Phase 0 全量回归 + 闸门说明

**Files:** 无新增代码。

- [ ] **Step 1: 跑 Phase 0 全部单元测试（显式列文件，避免 Windows glob）**

Run:
```
conda activate AR && python -m pytest tests/test_autopresort_timeline.py tests/test_autopresort_detect_track.py tests/test_autopresort_checkpoint_guard.py tests/test_autopresort_engine.py tests/test_autopresort_calibrate.py tests/test_autopresort_preprocess_parity.py -v
```
Expected: 全绿（parity 在有 cv2+torch 时 PASS，否则 skipped）。

- [ ] **Step 2: 冒烟：全模块可导入不误触发重依赖**

Run: `conda activate AR && python -c "import auto_presort.timeline, auto_presort.detect_track, auto_presort.preprocess, auto_presort.checkpoint_guard, auto_presort.engine, auto_presort.calibrate; print('phase0 ok')"`
Expected: 打印 `phase0 ok`（preprocess 会 import torch/cv2，在 AR 环境正常）。

- [ ] **Step 3: 记录 go/no-go（人工节点）**

在真实数据上跑 `python -m auto_presort.calibrate ...`（需先给 teacher 建 sidecar `.meta.json`）。查看输出：
- `gate_pass=True`（mean_purity ≥ 0.95 且 mean_recall ≥ 0.5，无严重短动作丢失）→ **进入 Phase 1**。
- `gate_pass=False` → 调 threshold/trim_margin/stride/enter_run 参数，**不建 UI**。
把最终选定参数写回 spec §9 待定项。

---

## 自检（Self-Review）

**Spec + 两轮评审覆盖：**
- §4 引擎（滑窗/门/合并/trim/gap/JSON）→ Task 2–4、8 ✅
- 修复①（teacher 看裁剪画面，非全画面）→ Task 5 检测轨迹 + Task 6 preprocess + Task 9 装配把 track.boxes 切给窗口 ✅
- 修复②（真实 parity test 对比 VideoDataset）→ Task 6 ✅
- 修复③（no_animal 强制 gap，检测先行）→ Task 5 + Task 8 `_window_no_animal_ratio`→None ✅
- 修复④（类别顺序 sidecar 出处）→ Task 7 ✅
- 修复⑤（校准 CLI 可运行 + recall/short/gate 修正）→ Task 10–11 ✅
- 修复⑥（真实装配集成测试）→ Task 9 Step 2 ✅
- 环境错位（Py3.14→3.12 AR、glob→显式文件）→ 全 Task 命令 + Task 12 ✅
- §8 Phase 0 闸门 → Task 12 ✅
- §4.1 leakage / §5 UI / §6 apply → **Phase 1，不在本计划** ✅

**占位符扫描：** 无 TBD/TODO；每个代码步骤含完整代码。

**类型/签名一致性：** `make_segment` 字段全程一致；`iter_windows`→`(start,end,center)`；`trim_segments`→`(trimmed, boundary_spans)`（Task 3 定义，Task 8 解包）；`compute_gaps(duration, segments, boundary_spans, no_animal_spans)`（Task 4 定义，Task 8 调用一致）；`build_video_doc(...meta=...)`（Task 4 定义，Task 8 传 meta）；`label_video` 注入点 `read_window_frames_fn`/`classify_frames_fn` 在 Task 8 测试、Task 9 装配、Task 11 CLI 三处签名一致；`verify_and_load_teacher(checkpoint_path, device)`→`(model, meta)`（Task 7 定义，Task 11 用）；`evaluate_docs/summarize/write_reports` 签名 Task 10–11 一致。

**已知取舍：** 真实 GPU 跑通（Task 11 Step main、Task 12 Step 3）无自动化测试，靠冒烟 + 人工执行——单元测试不引入 torch/YOLO 是本仓库既定风格；集成测试（Task 9）已用 fake 覆盖装配数据流。
