# Task 2 Report: engine 循环 5 档产 segment_levels

## Status: DONE

实现（brief Step 3 生产代码）逐字落地且正确；`doc["segment_levels"]` 含 5 档字符串键 `"0.5"…"0.9"`，且 `doc["segments"] == segment_levels["0.9"]`。brief Step 1 原始 dip 测试的 fixture 数学上不可能通过（已由 coordinator 确认为测试数据笔误，非实现 bug），coordinator 提供了修正版 fixture（6 高 + 1 低 + 6 高 = 13 窗），已替换。最终 Step 4 (6 passed) 与 Step 5 (26 passed) 全绿。

## Test commands and results

运行环境说明：`conda run` 默认缓冲输出，在 Windows cp1252 下遇非 ASCII 会 `UnicodeEncodeError`；实际以 `conda run --no-capture-output` + `PYTHONIOENCODING=utf-8` 运行，命令语义与 brief 指定一致。

### Step 2 (确认失败) — 预期失败（实现前）
命令: `conda run -n AR python -m pytest tests/test_autopresort_engine.py -v -k "all_conf_levels or merges_across_dip"`
结果: `2 failed, 4 deselected` — 两个新测试均因 `KeyError: 'segment_levels'` 失败（实现前的预期失败）。

### Step 4: `conda run -n AR python -m pytest tests/test_autopresort_engine.py -v`
结果: **6 passed**
```
test_label_video_produces_trimmed_segment PASSED
test_label_video_forces_gap_in_no_animal_region PASSED
test_engine_integration_with_fakes PASSED
test_label_video_short_no_animal_span_still_forces_gap PASSED
test_label_video_produces_all_conf_levels PASSED
test_label_video_lower_threshold_merges_across_dip PASSED
============================== 6 passed in 0.02s ==============================
```

### Step 5: `conda run -n AR python -m pytest tests/test_autopresort_timeline.py tests/test_autopresort_engine.py -v`
结果: **26 passed**
```
============================= 26 passed in 0.03s ==============================
```
(timeline 20 + engine 6，全绿，无回归)

## Fixture 修正说明（dip 测试）

原 brief Step 1 的 dip 测试 fixture（7 窗：`[0.95,0.93,0.94,0.60,0.94,0.96,0.95]`, frames=90）：0.9 档 0.60 打断出的两段各仅跨 2.0s，trim(margin 1.0, min_duration 2.0) 后跨度归 0 被丢弃 → `segment_levels["0.9"] == []`，断言 `== 2` 不可能成立。

coordinator 确认为测试数据笔误，生产代码正确，提供修正 fixture 并已应用：
- 13 窗：`[0.95]*6 + [0.60] + [0.95]*6`，frames=150 → duration=15.0s → `iter_windows(15.0,3.0,1.0)` 产 13 窗（center 1.5..13.5）。
- 0.9 档：dip 在 center 7.5 打断 → run1 (center 1.5..6.5, 跨 5.0 → trim 3.0, 保留) + run2 (center 8.5..13.5, 跨 5.0 → trim 3.0, 保留) = **2 段** ✔
- 0.5 档：13 窗全过 → 单段 (跨 12.0 → trim 10.0) = **1 段** ✔

已实测通过，与上面数学推导一致。

## Diff

### auto_presort/engine.py (label_video 尾部替换，其余函数未动)
```diff
-    merged = tl.merge_windows(preds, params["threshold"], params["enter_run"])
-    # 硬性规则4：即便某无动物区间短于一个窗口、未能整窗丢弃，也必须从 segment 中挖掉，
-    # 保证没有 segment 跨越/包含无动物区。
-    merged = tl.split_out_no_animal(merged, no_animal)
-    trimmed, boundary_spans = tl.trim_segments(
-        merged, params["trim_margin"], params["min_duration"])
-    return tl.build_video_doc(
-        rel_path=rel_path, duration=duration, segments=trimmed,
-        boundary_spans=boundary_spans, no_animal_spans=no_animal,
-        source_id=source_id, approved=False,
-        meta={"class_names": meta["class_names"],
-              "checkpoint_sha256": meta.get("sha256")},
-    )
+    # 逐档切段：merge → split_out_no_animal → trim，复用现有纯函数。
+    segment_levels = {}
+    default_trimmed, default_boundary = [], []
+    for level in tl.CONF_LEVELS:
+        merged = tl.merge_windows(preds, level, params["enter_run"])
+        merged = tl.split_out_no_animal(merged, no_animal)
+        trimmed, boundary_spans = tl.trim_segments(
+            merged, params["trim_margin"], params["min_duration"])
+        key = f"{level:.1f}"
+        segment_levels[key] = trimmed
+        if level == 0.9:
+            default_trimmed, default_boundary = trimmed, boundary_spans
+
+    return tl.build_video_doc(
+        rel_path=rel_path, duration=duration, segments=default_trimmed,
+        boundary_spans=default_boundary, no_animal_spans=no_animal,
+        source_id=source_id, approved=False,
+        meta={"class_names": meta["class_names"],
+              "checkpoint_sha256": meta.get("sha256")},
+        segment_levels=segment_levels,
+    )
```

### tests/test_autopresort_engine.py (文件末尾追加两个测试)
```diff
+
+
+def test_label_video_produces_all_conf_levels():
+    eng = importlib.import_module("auto_presort.engine")
+    tl = importlib.import_module("auto_presort.timeline")
+    track = {"fps": 10.0, "frames": 90, "boxes": [(0, 0, 10, 10)] * 90,
+             "flags": [True] * 90, "no_animal_spans": []}
+    doc = eng.label_video(
+        video_path="fake.mp4", rel_path="fake.mp4", source_id="src1",
+        track=track,
+        read_window_frames_fn=lambda vp, s, e, fps: ([None], [None]),
+        classify_frames_fn=lambda frames, boxes: ("eat", 0.95),
+        meta={"class_names": ["eat"]}, params=_params(),
+    )
+    assert set(doc["segment_levels"].keys()) == {"0.5", "0.6", "0.7", "0.8", "0.9"}
+    assert doc["segments"] == doc["segment_levels"]["0.9"]
+
+
+def test_label_video_lower_threshold_merges_across_dip():
+    # 中间一个低分窗口(0.60)在 0.9 档打断成两段、在 0.5 档合成一段。
+    eng = importlib.import_module("auto_presort.engine")
+    # 6 高 + 1 低(0.60) + 6 高 = 13 窗；每侧 6 窗，trim(margin 1.0) 后仍 ≥ min_duration 2.0 得以保留。
+    confs = iter([0.95] * 6 + [0.60] + [0.95] * 6)
+
+    def classifier(frames, boxes):
+        return ("eat", next(confs))
+
+    track = {"fps": 10.0, "frames": 150, "boxes": [(0, 0, 10, 10)] * 150,
+             "flags": [True] * 150, "no_animal_spans": []}
+    doc = eng.label_video(
+        video_path="fake.mp4", rel_path="fake.mp4", source_id="src1",
+        track=track,
+        read_window_frames_fn=lambda vp, s, e, fps: ([None], [None]),
+        classify_frames_fn=classifier,
+        meta={"class_names": ["eat"]}, params=_params(),
+    )
+    # 0.9 档：0.60 打断 → 两段；0.5 档：0.60 够格 → 一段
+    assert len(doc["segment_levels"]["0.9"]) == 2
+    assert len(doc["segment_levels"]["0.5"]) == 1
```

## Concerns
1. **环境**：`conda run` 在本机默认缓冲 + cp1252 编码，遇非 ASCII 输出会 `UnicodeEncodeError`；须加 `--no-capture-output`（并设 `PYTHONIOENCODING=utf-8`）才能拿到真实 pytest 输出。后续同环境任务注意。此为环境注意事项，不影响本任务结果。
