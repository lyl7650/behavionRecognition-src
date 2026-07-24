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
        track_boxes=track.get("boxes"), fps=track.get("fps"),
        frame_w=track.get("frame_w"), frame_h=track.get("frame_h"),
    )


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
