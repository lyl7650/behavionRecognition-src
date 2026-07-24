# Task 2 Brief: engine 循环 5 档产 segment_levels，segments 取 0.9 档

这是你的需求来源，使用其中的确切数值/命名（逐字照抄，勿改）。

## Files
- Modify: `auto_presort/engine.py`（`label_video` 函数，当前第 21-48 行）
- Test: `tests/test_autopresort_engine.py`

## Interfaces
- Consumes（Task 1 已完成，已在 `auto_presort/timeline.py` 存在）:
  - `timeline.CONF_LEVELS` = `[0.5, 0.6, 0.7, 0.8, 0.9]`
  - `timeline.merge_windows(preds, threshold, enter_run)` → segment list
  - `timeline.split_out_no_animal(segments, no_animal_spans)` → segment list
  - `timeline.trim_segments(segments, trim_margin, min_duration)` → `(trimmed, boundary_spans)`
  - `timeline.build_video_doc(..., segment_levels=None)` → doc（segment_levels 非 None 时含该键）
- Produces: `engine.label_video(...)` 返回的 doc 含 `"segment_levels"`（5 档，key 为字符串 `"0.5"`…`"0.9"`），且 `doc["segments"] == doc["segment_levels"]["0.9"]`。函数签名不变。

## Step 1: 写失败测试

在 `tests/test_autopresort_engine.py` 末尾追加（文件顶部已有 `_params()`，`threshold=0.90`，直接复用）：

```python
def test_label_video_produces_all_conf_levels():
    eng = importlib.import_module("auto_presort.engine")
    tl = importlib.import_module("auto_presort.timeline")
    track = {"fps": 10.0, "frames": 90, "boxes": [(0, 0, 10, 10)] * 90,
             "flags": [True] * 90, "no_animal_spans": []}
    doc = eng.label_video(
        video_path="fake.mp4", rel_path="fake.mp4", source_id="src1",
        track=track,
        read_window_frames_fn=lambda vp, s, e, fps: ([None], [None]),
        classify_frames_fn=lambda frames, boxes: ("eat", 0.95),
        meta={"class_names": ["eat"]}, params=_params(),
    )
    assert set(doc["segment_levels"].keys()) == {"0.5", "0.6", "0.7", "0.8", "0.9"}
    assert doc["segments"] == doc["segment_levels"]["0.9"]


def test_label_video_lower_threshold_merges_across_dip():
    # 中间一个低分窗口(0.60)在 0.9 档打断成两段、在 0.5 档合成一段。
    eng = importlib.import_module("auto_presort.engine")
    confs = iter([0.95, 0.93, 0.94, 0.60, 0.94, 0.96, 0.95])

    def classifier(frames, boxes):
        return ("eat", next(confs))

    track = {"fps": 10.0, "frames": 90, "boxes": [(0, 0, 10, 10)] * 90,
             "flags": [True] * 90, "no_animal_spans": []}
    doc = eng.label_video(
        video_path="fake.mp4", rel_path="fake.mp4", source_id="src1",
        track=track,
        read_window_frames_fn=lambda vp, s, e, fps: ([None], [None]),
        classify_frames_fn=classifier,
        meta={"class_names": ["eat"]}, params=_params(),
    )
    # 0.9 档：0.60 打断 → 两段；0.5 档：0.60 够格 → 一段
    assert len(doc["segment_levels"]["0.9"]) == 2
    assert len(doc["segment_levels"]["0.5"]) == 1
```

说明：`duration = frames/fps = 90/10 = 9.0s`，`iter_windows(9.0, 3.0, 1.0)` 产 7 窗（start 0..6），与 `confs` 的 7 个值一一对应。

## Step 2: 运行确认失败
`conda run -n AR python -m pytest tests/test_autopresort_engine.py -v -k "all_conf_levels or merges_across_dip"`
Expected: FAIL（doc 无 `segment_levels` 键 → KeyError）。

## Step 3: 实现

把 `auto_presort/engine.py` 的 `label_video`（当前第 21-48 行）整个替换为：

```python
def label_video(video_path, rel_path, source_id, track,
                read_window_frames_fn, classify_frames_fn, meta, params):
    duration = track["frames"] / track["fps"] if track["fps"] else 0.0
    windows = tl.iter_windows(duration, params["window_sec"], params["stride_sec"])
    no_animal = track["no_animal_spans"]

    preds = []
    for start, end, center in windows:
        if _window_no_animal_ratio(start, end, no_animal) >= params["no_animal_overlap"]:
            preds.append((center, None, 0.0))
            continue
        frames_bgr, boxes = read_window_frames_fn(video_path, start, end, track["fps"])
        label, conf = classify_frames_fn(frames_bgr, boxes)
        preds.append((center, label, conf))

    # 逐档切段：merge → split_out_no_animal → trim，复用现有纯函数。
    segment_levels = {}
    default_trimmed, default_boundary = [], []
    for level in tl.CONF_LEVELS:
        merged = tl.merge_windows(preds, level, params["enter_run"])
        merged = tl.split_out_no_animal(merged, no_animal)
        trimmed, boundary_spans = tl.trim_segments(
            merged, params["trim_margin"], params["min_duration"])
        key = f"{level:.1f}"
        segment_levels[key] = trimmed
        if level == 0.9:
            default_trimmed, default_boundary = trimmed, boundary_spans

    return tl.build_video_doc(
        rel_path=rel_path, duration=duration, segments=default_trimmed,
        boundary_spans=default_boundary, no_animal_spans=no_animal,
        source_id=source_id, approved=False,
        meta={"class_names": meta["class_names"],
              "checkpoint_sha256": meta.get("sha256")},
        segment_levels=segment_levels,
    )
```

注意：不要改动 `label_video` 之外的函数（`_window_no_animal_ratio`、`build_real_window_reader`、`build_real_classifier` 保持原样）。文件顶部已有 `from auto_presort import timeline as tl`。

## Step 4: 运行确认通过
`conda run -n AR python -m pytest tests/test_autopresort_engine.py -v`
Expected: PASS（含旧测试。旧测试断言 `doc["segments"]` 与 gaps，取 0.9 档后与改动前 `threshold=0.90` 行为一致，应保持通过）。

## Step 5: 跑 timeline+engine 确认无回归
`conda run -n AR python -m pytest tests/test_autopresort_timeline.py tests/test_autopresort_engine.py -v`
Expected: PASS（全部）。

## 硬约束
- 本仓库**不是 git 仓库**——不要执行任何 git 命令，不要 commit。"完成" = 上述测试全绿。
- 测试命令固定：`conda run -n AR python -m pytest <path> -v`（从项目根运行）。
- 只改本 brief 列出的两个文件。
- `segment_levels` 的 key 必须是字符串一位小数（用 `f"{level:.1f}"` 生成 → `"0.5"`,`"0.6"`,…,`"0.9"`）。
