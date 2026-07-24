"""纯逻辑：窗口 → 时间线 segment/gap。不得 import torch/cv2/ultralytics。"""

SCHEMA_VERSION = "autopresort-2"

CONF_LEVELS = [0.5, 0.6, 0.7, 0.8, 0.9]


def make_segment(start, end, label, confidence, source):
    return {
        "start": float(start),
        "end": float(end),
        "label": label,
        "confidence": float(confidence),
        "source": source,
    }


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


def split_out_no_animal(segments, no_animal_spans):
    """从每个 segment 中挖掉与 no_animal_spans 重叠的部分，返回切分后的 segment 列表。

    保证任何 segment 都不会跨越/包含无动物区间（硬性规则4），
    无论 no_animal 区间是否短于一个窗口。挖空后剩下的每段保留原 label/confidence；
    完全被覆盖的 segment 被丢弃。
    """
    if not no_animal_spans:
        return list(segments)
    spans = sorted(no_animal_spans)
    out = []
    for seg in segments:
        pieces = [(seg["start"], seg["end"])]
        for ns, ne in spans:
            nxt = []
            for s, e in pieces:
                if ne <= s + 1e-9 or ns >= e - 1e-9:
                    nxt.append((s, e))  # 不重叠
                    continue
                if s < ns - 1e-9:
                    nxt.append((s, ns))  # 左侧残留
                if ne < e - 1e-9:
                    nxt.append((ne, e))  # 右侧残留
            pieces = nxt
        for s, e in pieces:
            if e - s > 1e-9:
                out.append(make_segment(round(s, 6), round(e, 6),
                                        seg["label"], seg["confidence"], seg["source"]))
    return out


def _overlaps(a0, a1, spans):
    return any(a0 < e - 1e-9 and a1 > s + 1e-9 for s, e in spans)


def compute_gaps(duration, segments, boundary_spans, no_animal_spans):
    """未被 segment 覆盖的区间，按 no_animal > boundary > uncertain 归因。

    在 segment 端点 + boundary + no_animal 的所有端点处切分，保证子区间归因精确。
    """
    seg_spans = sorted((s["start"], s["end"]) for s in segments)
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
                    no_animal_spans, source_id, approved, meta,
                    segment_levels=None, track_boxes=None, fps=None,
                    frame_w=None, frame_h=None):
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
    if track_boxes is not None:
        doc["track_boxes"] = [
            [round(float(v), 1) for v in b] if b is not None else None
            for b in track_boxes
        ]
        doc["fps"] = float(fps) if fps else 0.0
        doc["frame_w"] = int(frame_w) if frame_w else 0
        doc["frame_h"] = int(frame_h) if frame_h else 0
    return doc
