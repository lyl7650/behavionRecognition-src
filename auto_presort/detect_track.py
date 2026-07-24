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


def build_track(video_path, yolo, target_classes, min_gap_sec=1.0, max_det=1):
    """真实检测：返回 {"fps","frames","boxes"(逐帧传播后 xyxy 或 None),
    "flags","no_animal_spans"}。import cv2。VFR 视频打印告警（Phase 0 仅保证 CFR）。"""
    import cv2

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 16.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
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
        "frame_w": frame_w,
        "frame_h": frame_h,
        "boxes": propagate_boxes(raw_boxes),
        "flags": flags,
        "no_animal_spans": no_animal_spans(flags, fps, min_gap_sec),
    }
