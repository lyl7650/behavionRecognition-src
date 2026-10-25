import cv2
import math
import re
import shutil
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO
from scripts.pipeline.lineage.store import record_derivation, record_input_if_root

INPUT_BASE_DIRS = [
    r"E:\ActionRecognition\behavionRecognition-src\dataset\raw_videos\pet\normal\ft-record\sorted-0629",
    r"E:\ActionRecognition\behavionRecognition-src\dataset\raw_videos\pet\normal\ssp-record\sorted-0430",
    r"E:\ActionRecognition\behavionRecognition-src\dataset\raw_videos\pet\normal\ssp-record\sorted-0501",
    r"E:\ActionRecognition\behavionRecognition-src\dataset\raw_videos\pet\normal\ssp-record\sorted-0503",
    r"E:\ActionRecognition\behavionRecognition-src\dataset\raw_videos\pet\normal\ssp-record\sorted-0504",
    r"E:\ActionRecognition\behavionRecognition-src\dataset\raw_videos\pet\normal\ssp-record\sorted-0605",
    r"E:\ActionRecognition\behavionRecognition-src\dataset\raw_videos\pet\normal\ssp-record\sorted-0606",
    r"E:\ActionRecognition\behavionRecognition-src\dataset\raw_videos\pet\normal\ssp-record\sorted-0607",
    r"E:\ActionRecognition\behavionRecognition-src\dataset\raw_videos\pet\normal\ssp-record\sorted-0608",
    r"E:\ActionRecognition\behavionRecognition-src\dataset\raw_videos\pet\normal\ssp-record\sorted-0630",
    r"E:\ActionRecognition\behavionRecognition-src\dataset\raw_videos\pet\normal\ssp-record\sorted-0701",
    r"E:\ActionRecognition\behavionRecognition-src\dataset\raw_videos\pet\normal\ssp-record\sorted-0702",
    r"E:\ActionRecognition\behavionRecognition-src\dataset\raw_videos\pet\normal\ssp-record\sorted-0708",
    r"E:\ActionRecognition\behavionRecognition-src\dataset\raw_videos\pet\normal\ssp-record\sorted-0709",
    r"E:\ActionRecognition\behavionRecognition-src\dataset\raw_videos\pet\normal\ssp-record\sorted-0710",
    # r"E:\ActionRecognition\behavionRecognition-src\dataset\raw_videos\pet\normal\sort\0501",
    # r"E:\ActionRecognition\behavionRecognition-src\dataset\raw_videos\pet\normal\sort\0503",
    # r"E:\ActionRecognition\behavionRecognition-src\dataset\raw_videos\pet\normal\sort\0504",
    # # r"E:\ActionRecognition\behavionRecognition-src\dataset\raw_videos\pet\normal\sort\0626",
    # r"E:\ActionRecognition\behavionRecognition-src\dataset\raw_videos\pet\normal\sort\0629",
]
# CLASS_NAMES     = [
#                 "eating", 
#                 "active", 
#                 "resting", 
#                 "drinking", 
#                 "moving"
#                    ]      # rabbit dataset class names

CLASS_NAMES      = ["eat", "drink", "resting", "active"] # dog-cat dataset class names
OUTPUT_BASE_DIR  = r"E:\ActionRecognition\behavionRecognition-src\dataset\pet\normal\2026-07-23\videos"
YOLO_MODEL_PATH  = r"e:\ActionRecognition\detection-src\cat_dog_detect\runs\yolo11n_scale0.6_fliplr0.42\weights\yolo11n-ssb-0710.pt" # dog-cat detect model
# YOLO_MODEL_PATH = r'e:\ActionRecognition\detection-src\rabbit-detect\runs\checkpoints\yolo11n.pt3\weights\rabbit-yolo11n-0603.pt' # rabbit detect model
TARGET_CLASSES   = [1, 2]   # dog-cat labels in the model, 1: cat, 2: dog
# TARGET_CLASSES   = [0]    # rabbit label in the model, 0: rabbit

EXPAND_RATIO       = 1.25
MAX_CLIP_DURATION  = 3          # 每段 clip 的最大时长（秒）
MIN_CLIP_DURATION  = 2.0        # Skip short tail clips to avoid 0-second outputs.
MAX_DET            = 1
DISCARD_ON_EXTRA_DETECTIONS = False
# MAX_CLIPS_PER_VIDEO = {
#     'eating':   20,
#     'drinking': 40,
#     'active': 20,
#     'resting':20
# }
MAX_CLIPS_PER_VIDEO = {
    'eat': 40,
    'drink': 40,
    'active': 40,
    'resting': 40
}

# 输出视频编码与扩展名
VIDEO_FOURCC = cv2.VideoWriter_fourcc(*"mp4v")
VIDEO_EXT    = ".mp4"
# ============================================


def clamp_crop_box(cx, cy, crop_w, crop_h, frame_width, frame_height):
    crop_w = max(1, min(int(round(crop_w)), frame_width))
    crop_h = max(1, min(int(round(crop_h)), frame_height))
    x1 = int(round(cx - crop_w / 2))
    y1 = int(round(cy - crop_h / 2))
    x1 = max(0, min(x1, frame_width - crop_w))
    y1 = max(0, min(y1, frame_height - crop_h))
    return x1, y1, x1 + crop_w, y1 + crop_h


def build_smoothed_crop_boxes(
    boxes,
    frame_width,
    frame_height,
    fallback_size,
    expand_ratio,
    smooth_alpha=0.35,
):
    valid_boxes = [box for box in boxes if box is not None]
    if valid_boxes:
        widths = [float(x2 - x1) for x1, y1, x2, y2 in valid_boxes]
        heights = [float(y2 - y1) for x1, y1, x2, y2 in valid_boxes]
        crop_w = np.percentile(widths, 80) * expand_ratio
        crop_h = np.percentile(heights, 80) * expand_ratio
    else:
        crop_w, crop_h = fallback_size

    crop_w = min(max(1, int(round(crop_w))), frame_width)
    crop_h = min(max(1, int(round(crop_h))), frame_height)
    fallback_cx = frame_width / 2
    fallback_cy = frame_height / 2
    smoothed_cx = None
    smoothed_cy = None
    crop_boxes = []

    for box in boxes:
        if box is None:
            raw_cx = fallback_cx if smoothed_cx is None else smoothed_cx
            raw_cy = fallback_cy if smoothed_cy is None else smoothed_cy
        else:
            x1, y1, x2, y2 = box
            raw_cx = (float(x1) + float(x2)) / 2
            raw_cy = (float(y1) + float(y2)) / 2

        if smoothed_cx is None:
            smoothed_cx = raw_cx
            smoothed_cy = raw_cy
        else:
            smoothed_cx = smooth_alpha * raw_cx + (1 - smooth_alpha) * smoothed_cx
            smoothed_cy = smooth_alpha * raw_cy + (1 - smooth_alpha) * smoothed_cy

        crop_boxes.append(
            clamp_crop_box(smoothed_cx, smoothed_cy, crop_w, crop_h, frame_width, frame_height)
        )

    return crop_boxes


def check_gpu():
    print("-" * 40)
    if torch.cuda.is_available():
        print(f"[OK] GPU detected: {torch.cuda.get_device_name(0)}")
        return 'cuda:0'
    else:
        print("[WARNING] No GPU available; using CPU for inference.")
        return 'cpu'


def get_clip_segments(duration, cfg=None):
    """Split a video into fixed-duration clips."""
    max_clip_duration = MAX_CLIP_DURATION if cfg is None else cfg["max_clip_duration"]
    min_clip_duration = MIN_CLIP_DURATION if cfg is None else cfg["min_clip_duration"]
    if duration <= 0:
        return []
    clip_duration = float(max_clip_duration)
    if clip_duration <= 0:
        raise ValueError("MAX_CLIP_DURATION must be greater than 0")
    num_clips = math.ceil(duration / clip_duration)
    segments = []
    for i in range(num_clips):
        start = i * clip_duration
        end = min((i + 1) * clip_duration, duration)
        if end - start < min_clip_duration:
            continue
        segments.append((start, end))
    return segments


def make_clip_name(video_stem, clip_index):
    return f"{video_stem}_clip-{clip_index}{VIDEO_EXT}"


def get_train_clip_relative_path(class_name, video_stem, segment_index):
    clip_name = make_clip_name(video_stem, segment_index + 1)
    return Path(class_name) / clip_name


def build_segment_frame_ranges(segments, fps, total_frames):
    ranges = []
    for i, (start_time, end_time) in enumerate(segments):
        start_frame = min(int(start_time * fps), total_frames - 1)
        end_frame = min(int(end_time * fps), total_frames)
        if end_frame <= start_frame:
            continue
        ranges.append((i, start_frame, end_frame))
    return ranges


def seek_to_frame(cap, frame_index):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)


def collect_class_videos(input_base_dir, class_names):
    input_base_path = Path(input_base_dir)
    videos = []
    for class_name in class_names:
        class_dir = input_base_path / class_name
        if not class_dir.is_dir():
            print(f"Skipping non-existent class directory: {class_dir}")
            continue
        for file_path in sorted(class_dir.rglob('*')):
            if file_path.suffix.lower() in ['.mp4', '.mov', '.avi', '.webm']:
                videos.append((file_path, class_name))
    return videos


# ---------------------------------------------------------------------------
# Train/Val 重分配
# ---------------------------------------------------------------------------

def source_key(video_name):
    """从 auto_presort 命名 <原视频名>-<start>-<end>.mp4 剥掉尾部时间段还原来源；

    无法识别时间段后缀的旧命名回退整名。同一原视频的各切片共享同一来源 key，
    保证 rebalance_val 不把秒级相邻片段拆到 train/val 两侧造成泄漏。
    """
    stem = re.sub(r"\.(mp4|avi|mov|webm)$", "", video_name, flags=re.IGNORECASE)
    m = re.match(r"^(.*?)-\d+(?:\.\d+)?-\d+(?:\.\d+)?(?:_clip-\d+)?$", stem)
    return m.group(1) if m else video_name


def _group_records_by_source(records):
    """[(video_name, clips)] → 按 source_key 合并为 [(key, merged_clips)]，保持首见顺序。"""
    grouped, order = {}, []
    for video_name, clip_files in records:
        key = source_key(video_name)
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].extend(clip_files)
    return [(k, grouped[k]) for k in order]


def rebalance_val(video_records, cfg=None):
    """按视频粒度将约 20% 的片段从 train/ 移到 val/，均匀抽取视频避免泄露。"""
    output_base_dir = OUTPUT_BASE_DIR if cfg is None else cfg["output_base_dir"]
    for class_name, records in video_records.items():
        # 同源分组：auto_presort 会把一条原始视频切成多个输入文件（<srcid>__...），
        # 必须整组同 split，否则秒级相邻片段跨 train/val 造成泄漏。
        records = _group_records_by_source(records)
        total_segs = sum(len(clip_files) for _, clip_files in records)
        if total_segs == 0:
            continue
        n_videos = len(records)
        n_val = max(1, round(n_videos * 0.2))
        if n_val == 1:
            val_indices = {n_videos - 1}
        else:
            val_indices = {round(i * (n_videos - 1) / (n_val - 1)) for i in range(n_val)}

        val_seg_count = 0
        for idx, (video_name, clip_files) in enumerate(records):
            if idx not in val_indices:
                continue
            for clip_file in clip_files:
                src = Path(output_base_dir) / "train" / clip_file
                dst = Path(output_base_dir) / "val"   / clip_file
                if src.exists():
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(src), str(dst))
            val_seg_count += len(clip_files)

        train_seg_count = total_segs - val_seg_count
        ratio = val_seg_count / total_segs if total_segs else 0
        print(f"  [{class_name}] total_segments={total_segs}, train={train_seg_count}, val={val_seg_count} ({ratio:.1%})")


# ---------------------------------------------------------------------------
# 视频处理逻辑
# ---------------------------------------------------------------------------

def process_clip_segment(
    cap,
    model,
    start_frame,
    end_frame,
    out_path,
    fps,
    stats,
    cfg=None,
):
    target_classes = TARGET_CLASSES if cfg is None else cfg["target_classes"]
    max_det = MAX_DET if cfg is None else cfg["max_det"]
    discard_on_extra = DISCARD_ON_EXTRA_DETECTIONS if cfg is None else cfg["discard_on_extra_detections"]
    expand_ratio = EXPAND_RATIO if cfg is None else cfg["expand_ratio"]
    frames = []
    boxes_by_frame = []
    last_valid_box = None

    seek_to_frame(cap, start_frame)
    for frame_idx in range(start_frame, end_frame):
        try:
            ret, frame = cap.read()
        except cv2.error as e:
            stats.setdefault("failed_paths", []).append((str(out_path), f"read failed at frame {frame_idx}: {e}"))
            return False
        if not ret:
            break

        try:
            results = model(frame, classes=target_classes, verbose=False, max_det=max_det)
            boxes = results[0].boxes
        except Exception as e:
            stats.setdefault("failed_paths", []).append((str(out_path), f"detect failed at frame {frame_idx}: {e}"))
            return False

        if len(boxes) > max_det:
            if discard_on_extra:
                stats["segments_discarded_multi_det"] += 1
                return False
            crop_box = last_valid_box
        elif len(boxes) >= 1:
            crop_box = boxes[0].xyxy[0].cpu().numpy()
            last_valid_box = crop_box
        elif last_valid_box is not None:
            crop_box = last_valid_box
        else:
            crop_box = None
            stats["frames_with_no_detection"] += 1

        frames.append(frame)
        if crop_box is None:
            boxes_by_frame.append(None)
        else:
            boxes_by_frame.append(tuple(float(v) for v in crop_box))

    if not frames:
        stats["segments_no_valid_crop"] += 1
        return False

    first_h, first_w = frames[0].shape[:2]
    crop_boxes = build_smoothed_crop_boxes(
        boxes_by_frame,
        frame_width=first_w,
        frame_height=first_h,
        fallback_size=(first_w, first_h),
        expand_ratio=expand_ratio,
    )
    crop_w = crop_boxes[0][2] - crop_boxes[0][0]
    crop_h = crop_boxes[0][3] - crop_boxes[0][1]

    writer = None
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(out_path), VIDEO_FOURCC, fps, (crop_w, crop_h))
        if not writer.isOpened():
            raise RuntimeError(f"VideoWriter æ— æ³•æ‰“å¼€: {out_path}")

        for frm, crop_box in zip(frames, crop_boxes):
            x1, y1, x2, y2 = crop_box
            frm = frm[y1:y2, x1:x2]
            if frm.shape[1] != crop_w or frm.shape[0] != crop_h:
                frm = cv2.resize(frm, (crop_w, crop_h))
            writer.write(frm)
        writer.release()
    except Exception as e:
        stats.setdefault("failed_paths", []).append((str(out_path), str(e)))
        if writer is not None:
            writer.release()
        return False

    return True


def process_video(video_path, class_name, class_counters, model, parent_dir_name):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Cannot open video: {video_path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or total_frames <= 0:
        cap.release()
        return None

    # 读取原始视频的宽高，用于 VideoWriter 初始化
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    duration = total_frames / fps
    segments = get_clip_segments(duration)
    stats = {
        "segments_total": len(segments),
        "segments_saved": 0,
        "segments_discarded_multi_det": 0,
        "segments_no_valid_crop": 0,
        "frames_with_no_detection": 0,
    }

    # 每个 segment 存储原帧和检测框，保存时再生成稳定裁剪框。
    local_seqs = {}

    # 构建 segment → frame_range 映射，收集需要处理的帧号
    seg_frame_ranges = {}
    for i, (start_time, end_time) in enumerate(segments):
        start_frame = min(int(start_time * fps), total_frames - 1)
        end_frame   = min(int(end_time * fps),   total_frames - 1)
        if end_frame <= start_frame:
            continue
        seg_frame_ranges[i] = (start_frame, end_frame)
        local_seqs[i] = {
            'frames':  [],
            'boxes':   [],
            'discard': False,
        }

    if not seg_frame_ranges:
        cap.release()
        return None

    # 确定需要读取到的最大帧号
    max_target_frame = max(end for _, end in seg_frame_ranges.values())

    current_frame_idx = 0
    last_valid_box = None

    with tqdm(total=max_target_frame + 1, desc=video_path.name, unit="f", leave=False) as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # 判断当前帧属于哪些 segment
            active_segs = [
                i for i, (s, e) in seg_frame_ranges.items()
                if s <= current_frame_idx < e and not local_seqs[i]['discard']
            ]

            if active_segs:
                results = model(frame, classes=TARGET_CLASSES, verbose=False, max_det=MAX_DET)
                boxes = results[0].boxes

                if len(boxes) > MAX_DET:
                    if DISCARD_ON_EXTRA_DETECTIONS:
                        for seg_i in active_segs:
                            local_seqs[seg_i]['discard'] = True
                        stats["segments_discarded_multi_det"] += len(active_segs)
                else:
                    # 计算裁剪框
                    crop_box = None
                    if len(boxes) >= 1:
                        crop_box = boxes[0].xyxy[0].cpu().numpy()
                        last_valid_box = crop_box
                    elif last_valid_box is not None:
                        crop_box = last_valid_box
                    else:
                        stats["frames_with_no_detection"] += 1

                    for seg_i in active_segs:
                        if not local_seqs[seg_i]['discard']:
                            local_seqs[seg_i]['frames'].append(frame.copy())
                            if crop_box is None:
                                local_seqs[seg_i]['boxes'].append(None)
                            else:
                                local_seqs[seg_i]['boxes'].append(tuple(float(v) for v in crop_box))

            current_frame_idx += 1
            pbar.update(1)
            if current_frame_idx > max_target_frame:
                break

    cap.release()

    # ---- 延迟保存逻辑：将每个 segment 写成视频 clip ----
    max_clips  = MAX_CLIPS_PER_VIDEO.get(class_name.lower())
    video_stem = video_path.stem.strip()

    # 筛选合格 segment：未丢弃 且 至少有一帧
    valid_seqs = [
        (i, seq_data)
        for i, seq_data in local_seqs.items()
        if not seq_data['discard'] and len(seq_data['frames']) > 0
    ]

    # 均匀抽取：若合格数超过上限，则从全段均匀采样
    if max_clips is not None and len(valid_seqs) > max_clips:
        sample_indices = np.linspace(0, len(valid_seqs) - 1, max_clips, dtype=int)
        valid_seqs = [valid_seqs[j] for j in sample_indices]

    saved_clip_names = []
    for i, seq_data in valid_seqs:
        relative_clip_path = get_train_clip_relative_path(class_name, video_stem, i)

        out_path = Path(output_base_dir) / "train" / relative_clip_path
        try:
            out_path.parent.mkdir(parents=True, exist_ok=True)

            first_h, first_w = seq_data['frames'][0].shape[:2]
            crop_boxes = build_smoothed_crop_boxes(
                seq_data['boxes'],
                frame_width=first_w,
                frame_height=first_h,
                fallback_size=(first_w, first_h),
                expand_ratio=EXPAND_RATIO,
            )
            crop_w = crop_boxes[0][2] - crop_boxes[0][0]
            crop_h = crop_boxes[0][3] - crop_boxes[0][1]
            writer = cv2.VideoWriter(str(out_path), VIDEO_FOURCC, fps, (crop_w, crop_h))
            if not writer.isOpened():
                raise RuntimeError(f"VideoWriter 无法打开: {out_path}")

            for frm, crop_box in zip(seq_data['frames'], crop_boxes):
                x1, y1, x2, y2 = crop_box
                frm = frm[y1:y2, x1:x2]
                if frm.shape[1] != crop_w or frm.shape[0] != crop_h:
                    frm = cv2.resize(frm, (crop_w, crop_h))
                writer.write(frm)
            writer.release()

            record_input_if_root(str(video_path), species=(cfg or {}).get("lineage_species"))
            record_derivation(
                str(out_path), str(video_path), operation="yolo_crop_clip",
                label=class_name,
                parameters={
                    "segment_index": i,
                    "fps": fps,
                    "expand_ratio": cfg["expand_ratio"] if cfg else EXPAND_RATIO,
                    "max_clip_duration": cfg["max_clip_duration"] if cfg else MAX_CLIP_DURATION,
                },
            )

            class_counters[class_name] = class_counters.get(class_name, 0) + 1
            stats["segments_saved"] += 1
            saved_clip_names.append(str(relative_clip_path))

        except Exception as e:
            stats.setdefault("failed_paths", []).append((str(out_path), str(e)))
            tqdm.write(f"  [ERROR] 保存失败: {out_path} — {e}")

    tqdm.write(
        f"video_done [{class_name}] {video_path.name}: "
        f"segments={stats['segments_total']}, "
        f"saved={stats['segments_saved']}, "
        f"discarded_multi_det={stats['segments_discarded_multi_det']}, "
        f"frames_no_det={stats['frames_with_no_detection']}"
    )
    stats["saved_clip_names"] = saved_clip_names
    return stats


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def process_video_lowmem(video_path, class_name, class_counters, model, parent_dir_name, cfg=None):
    output_base_dir = OUTPUT_BASE_DIR if cfg is None else cfg["output_base_dir"]
    max_clips_per_video = MAX_CLIPS_PER_VIDEO if cfg is None else cfg["max_clips_per_video"]
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Cannot open video: {video_path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or total_frames <= 0:
        cap.release()
        return None

    duration = total_frames / fps
    segments = get_clip_segments(duration, cfg)
    segment_ranges = build_segment_frame_ranges(segments, fps, total_frames)
    stats = {
        "segments_total": len(segment_ranges),
        "segments_saved": 0,
        "segments_discarded_multi_det": 0,
        "segments_no_valid_crop": 0,
        "frames_with_no_detection": 0,
    }

    if not segment_ranges:
        cap.release()
        return None

    max_clips = max_clips_per_video.get(class_name.lower())
    selected_ranges = segment_ranges
    if max_clips is not None and len(selected_ranges) > max_clips:
        sample_indices = np.linspace(0, len(selected_ranges) - 1, max_clips, dtype=int)
        selected_ranges = [selected_ranges[j] for j in sample_indices]

    video_stem = video_path.stem.strip()
    saved_clip_names = []
    for segment_index, start_frame, end_frame in tqdm(
        selected_ranges,
        desc=video_path.name,
        unit="clip",
        leave=False,
    ):
        relative_clip_path = get_train_clip_relative_path(class_name, video_stem, segment_index)
        out_path = Path(output_base_dir) / "train" / relative_clip_path
        ok = process_clip_segment(
            cap=cap,
            model=model,
            start_frame=start_frame,
            end_frame=end_frame,
            out_path=out_path,
            fps=fps,
            stats=stats,
            cfg=cfg,
        )
        if ok:
            class_counters[class_name] = class_counters.get(class_name, 0) + 1
            stats["segments_saved"] += 1
            saved_clip_names.append(str(relative_clip_path))
        elif stats.get("failed_paths"):
            tqdm.write(f"  [WARN] 跳过片段: {out_path}")

    cap.release()

    tqdm.write(
        f"video_done [{class_name}] {video_path.name}: "
        f"segments={stats['segments_total']}, "
        f"saved={stats['segments_saved']}, "
        f"discarded_multi_det={stats['segments_discarded_multi_det']}, "
        f"frames_no_det={stats['frames_with_no_detection']}"
    )
    stats["saved_clip_names"] = saved_clip_names
    return stats


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
    print("Loading YOLO model...")
    model = model_factory(device)
    print("-" * 40)

    input_base_dir = cfg["input_base_dir"]
    parent_dir_name = Path(input_base_dir).name
    print(f"\n{'=' * 40}")
    print(f"Processing batch: {parent_dir_name}  ({input_base_dir})")
    print(f"{'=' * 40}")

    class_counters = {}
    failed_paths = []
    video_records = {}

    videos = collect_class_videos(input_base_dir, cfg["class_names"])
    for file_path, class_name in tqdm(videos, desc=parent_dir_name, unit="video"):
        tqdm.write(f"处理中 [{class_name}]: {file_path.name}")
        stats = process_video_lowmem(file_path, class_name, class_counters, model, parent_dir_name, cfg)
        if stats and stats.get("failed_paths"):
            failed_paths.extend(stats["failed_paths"])
        saved_clip_names = stats.get("saved_clip_names", []) if stats else []
        if saved_clip_names:
            video_records.setdefault(class_name, []).append((file_path.name, saved_clip_names))

    print(f"\nBatch [{parent_dir_name}] done; splitting into train/val ...")
    rebalance_val(video_records, cfg)
    if failed_paths:
        print(f"\n  [WARNING] The following {len(failed_paths)} paths failed to save:")
        for path, err in failed_paths:
            print(f"    {path}  —  {err}")

    return {"video_records": video_records, "failed_paths": failed_paths}


def main():
    for input_base_dir in INPUT_BASE_DIRS:
        cfg = default_config()
        cfg["input_base_dir"] = input_base_dir
        run(cfg)

    print("\n" + "=" * 40)
    print("[DONE] All batches processed!")


if __name__ == "__main__":
    main()
