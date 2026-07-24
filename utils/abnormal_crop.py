import cv2
import math
import shutil
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO

# ================= 配置参数 =================
INPUT_BASE_DIRS = [
    # r"E:\ActionRecognition\behavionRecognition-src\dataset\raw_videos\pet\normal\sort\sorted-0430",
    # r"E:\ActionRecognition\behavionRecognition-src\dataset\raw_videos\pet\normal\sort\sorted-0501",
    # r"E:\ActionRecognition\behavionRecognition-src\dataset\raw_videos\pet\normal\sort\sorted-0503",
    # r"E:\ActionRecognition\behavionRecognition-src\dataset\raw_videos\pet\normal\sort\sorted-0504",
    # r"E:\ActionRecognition\behavionRecognition-src\dataset\raw_videos\pet\normal\sort\sorted-0605",
    # r"E:\ActionRecognition\behavionRecognition-src\dataset\raw_videos\pet\normal\sort\sorted-0606",
    # r"E:\ActionRecognition\behavionRecognition-src\dataset\raw_videos\pet\normal\sort\sorted-0607",
    # r"E:\ActionRecognition\behavionRecognition-src\dataset\raw_videos\pet\normal\sort\sorted-0608",
    r"E:\ActionRecognition\behavionRecognition-src\dataset\raw_videos\pet\normal\sort\sorted-0629",
    # r"E:\ActionRecognition\behavionRecognition-src\dataset\raw_videos\pet\normal\sort\sorted-0630",
    # r"E:\ActionRecognition\behavionRecognition-src\dataset\raw_videos\pet\normal\sort\sorted-0701",
    # r"E:\ActionRecognition\behavionRecognition-src\dataset\raw_videos\pet\normal\sort\sorted-0702",
    # r"E:\ActionRecognition\behavionRecognition-src\dataset\raw_videos\pet\normal\sort\0430",
    # r"E:\ActionRecognition\behavionRecognition-src\dataset\raw_videos\pet\normal\sort\0501",
    # r"E:\ActionRecognition\behavionRecognition-src\dataset\raw_videos\pet\normal\sort\0503",
    # r"E:\ActionRecognition\behavionRecognition-src\dataset\raw_videos\pet\normal\sort\0504",
    # # r"E:\ActionRecognition\behavionRecognition-src\dataset\raw_videos\pet\normal\sort\0626",
    # r"E:\ActionRecognition\behavionRecognition-src\dataset\raw_videos\pet\normal\sort\0629",
    r"E:\ActionRecognition\behavionRecognition-src\dataset\raw_videos\pet\normal\sort\temp\sorted-0702"
]
# CLASS_NAMES     = [
#                 "eating", 
#                 "active", 
#                 "resting", 
#                 "drinking", 
#                 "moving"
#                    ]      # rabbit dataset class names

CLASS_NAMES      = ["eat", "drink", "other"] # dog-cat dataset class names
OUTPUT_BASE_DIR  = r"E:\ActionRecognition\behavionRecognition-src\dataset\pet\normal\2026-07-10\videos"
YOLO_MODEL_PATH  = r"E:\ActionRecognition\detection-src\cat_dog_detect\runs\yolo11n_scale0.6_fliplr0.4\weights\best.pt" # dog-cat detect model
# YOLO_MODEL_PATH = r'e:\ActionRecognition\detection-src\rabbit-detect\runs\checkpoints\yolo11n.pt3\weights\rabbit-yolo11n-0603.pt' # rabbit detect model
TARGET_CLASSES   = [1, 2]   # dog-cat labels in the model, 1: cat, 2: dog
# TARGET_CLASSES   = [0]    # rabbit label in the model, 0: rabbit
# LABEL_MAP = {
#     'resting': 0,
#     'active':  1,
#     'moving':  1,
#     'eating':  2,
#     'drinking': 3,
# }
LABEL_MAP = {
    'eat': 0,
    'drink':  1,
    'other': 2,
}


EXPAND_RATIO       = 1.25
FRAMES_PER_CLIP    = 16
MAX_CLIP_DURATION  = 4
MAX_DET            = 1
DISCARD_ON_EXTRA_DETECTIONS = False
MAX_CLIPS_PER_VIDEO = {
    'eat': 40,
    'drink': 40,
    'active': 10,
    'resting': 10,
}
# ============================================


def check_gpu():
    print("-" * 40)
    if torch.cuda.is_available():
        print(f"[OK] GPU detected: {torch.cuda.get_device_name(0)}")
        return 'cuda:0'
    else:
        print("[WARNING] No GPU available; using CPU for inference.")
        return 'cpu'


def get_clip_segments(duration, cfg=None):
    max_clip_duration = MAX_CLIP_DURATION if cfg is None else cfg["max_clip_duration"]
    if duration < 5.0:
        return [(0.0, duration)]
    num_clips = math.ceil(duration / max_clip_duration)
    clip_duration = duration / num_clips
    return [(i * clip_duration, (i + 1) * clip_duration) for i in range(num_clips)]


def write_image(image_path, image):
    image_path = Path(image_path)
    suffix = image_path.suffix or ".jpg"
    ok, encoded = cv2.imencode(suffix, image)
    if not ok:
        raise ValueError(f"无法编码图片: {image_path}")
    image_path.write_bytes(encoded.tobytes())


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

def rebalance_val(video_records, cfg=None):
    """按视频粒度将约 20% 的片段从 train/ 移到 val/，均匀抽取视频避免泄露。"""
    output_base_dir = OUTPUT_BASE_DIR if cfg is None else cfg["output_base_dir"]
    for class_name, records in video_records.items():
        total_segs = sum(len(folders) for _, folders in records)
        if total_segs == 0:
            continue
        n_videos = len(records)
        n_val = max(1, round(n_videos * 0.2))
        if n_val == 1:
            val_indices = {n_videos - 1}
        else:
            val_indices = {round(i * (n_videos - 1) / (n_val - 1)) for i in range(n_val)}

        val_seg_count = 0
        for idx, (video_name, folders) in enumerate(records):
            if idx not in val_indices:
                continue
            for folder_name in folders:
                src = Path(output_base_dir) / "train" / folder_name
                dst = Path(output_base_dir) / "val" / folder_name
                if src.exists():
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(src), str(dst))
            val_seg_count += len(folders)

        train_seg_count = total_segs - val_seg_count
        ratio = val_seg_count / total_segs if total_segs else 0
        print(f"  [{class_name}] total_segments={total_segs}, train={train_seg_count}, val={val_seg_count} ({ratio:.1%})")


# ---------------------------------------------------------------------------
# 视频处理逻辑
# ---------------------------------------------------------------------------

def process_video(video_path, class_name, class_counters, model, parent_dir_name, cfg=None):
    output_base_dir = OUTPUT_BASE_DIR if cfg is None else cfg["output_base_dir"]
    label_map = LABEL_MAP if cfg is None else cfg["label_map"]
    target_classes = TARGET_CLASSES if cfg is None else cfg["target_classes"]
    frames_per_clip = FRAMES_PER_CLIP if cfg is None else cfg["frames_per_clip"]
    max_det = MAX_DET if cfg is None else cfg["max_det"]
    expand_ratio = EXPAND_RATIO if cfg is None else cfg["expand_ratio"]
    discard_on_extra = DISCARD_ON_EXTRA_DETECTIONS if cfg is None else cfg["discard_on_extra_detections"]
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
    stats = {
        "segments_total": len(segments),
        "segments_saved": 0,
        "segments_discarded_multi_det": 0,
        "segments_incomplete": 0,
        "frames_with_no_detection": 0,
    }

    local_seqs = {}
    target_frames_dict = {}

    for i, (start_time, end_time) in enumerate(segments):
        start_frame = min(int(start_time * fps), total_frames - 1)
        end_frame   = min(int(end_time * fps), total_frames - 1)
        if end_frame <= start_frame:
            continue

        frame_indices = np.linspace(start_frame, end_frame - 1, frames_per_clip, dtype=int)
        local_seqs[i] = {
            'frames':  [None] * frames_per_clip,
            'discard': False,
            'filled':  0,
        }
        for idx, f_idx in enumerate(frame_indices):
            if f_idx not in target_frames_dict:
                target_frames_dict[f_idx] = []
            target_frames_dict[f_idx].append((i, idx))

    if not target_frames_dict:
        cap.release()
        return None

    max_target_frame = max(target_frames_dict.keys())
    current_frame_idx = 0
    last_valid_box = None

    with tqdm(total=max_target_frame + 1, desc=video_path.name, unit="f", leave=False) as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if current_frame_idx in target_frames_dict:
                active_tasks = [
                    (seq_i, idx)
                    for seq_i, idx in target_frames_dict[current_frame_idx]
                    if not local_seqs[seq_i]['discard']
                ]

                if active_tasks:
                    img_h, img_w = frame.shape[:2]
                    results = model(frame, classes=target_classes, verbose=False, max_det=max_det)
                    boxes = results[0].boxes

                    if len(boxes) > max_det:
                        if discard_on_extra:
                            for seq_i, _ in active_tasks:
                                local_seqs[seq_i]['discard'] = True
                            stats["segments_discarded_multi_det"] += len({seq_i for seq_i, _ in active_tasks})
                    else:
                        crop_box = None
                        if len(boxes) >= 1:
                            crop_box = boxes[0].xyxy[0].cpu().numpy()
                            last_valid_box = crop_box
                        elif last_valid_box is not None:
                            crop_box = last_valid_box
                        else:
                            stats["frames_with_no_detection"] += 1

                        if crop_box is not None:
                            x1, y1, x2, y2 = crop_box
                            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                            w, h   = x2 - x1, y2 - y1
                            new_w  = w * expand_ratio
                            new_h  = h * expand_ratio
                            nx1 = max(0, int(cx - new_w / 2))
                            ny1 = max(0, int(cy - new_h / 2))
                            nx2 = min(img_w, int(cx + new_w / 2))
                            ny2 = min(img_h, int(cy + new_h / 2))
                            cropped_frame = frame[ny1:ny2, nx1:nx2]
                        else:
                            cropped_frame = frame

                        for seq_i, idx in active_tasks:
                            if not local_seqs[seq_i]['discard']:
                                local_seqs[seq_i]['frames'][idx] = cropped_frame
                                local_seqs[seq_i]['filled'] += 1

            current_frame_idx += 1
            pbar.update(1)
            if current_frame_idx > max_target_frame:
                break

    cap.release()

    # ---- 延迟保存逻辑 ----
    label_val = label_map.get(class_name.lower(), 0)
    max_clips = max_clips_per_video.get(class_name.lower())
    video_saved_count = 0

    # 先统计所有合格的 segment（未丢弃且帧数完整）
    valid_seqs = [
        (i, seq_data)
        for i, seq_data in local_seqs.items()
        if not seq_data['discard'] and seq_data['filled'] == frames_per_clip
    ]
    stats["segments_incomplete"] += sum(
        1 for _, seq_data in local_seqs.items()
        if not seq_data['discard'] and seq_data['filled'] != frames_per_clip
    )

    # 均匀抽取：若合格数超过上限，则从全段均匀采样，而非只取开头
    if max_clips is not None and len(valid_seqs) > max_clips:
        sample_indices = np.linspace(0, len(valid_seqs) - 1, max_clips, dtype=int)
        valid_seqs = [valid_seqs[j] for j in sample_indices]

    video_stem = video_path.stem.strip()
    for i, seq_data in valid_seqs:
        seq_idx = class_counters.get(class_name, 0)

        if len(local_seqs) > 1:
            folder_name = f"{parent_dir_name}-{class_name}-{video_stem}-{i}"
        else:
            folder_name = f"{parent_dir_name}-{class_name}-{video_stem}"
        seq_dir = Path(output_base_dir) / "train" / folder_name
        try:
            seq_dir.mkdir(parents=True, exist_ok=True)
            for idx, img in enumerate(seq_data['frames']):
                write_image(seq_dir / f"frame_{idx:04d}.jpg", img)
            with open(seq_dir / "label.txt", "w", encoding="utf-8") as f:
                f.write(str(label_val))
            class_counters[class_name] = seq_idx + 1
            video_saved_count += 1
            stats["segments_saved"] += 1
            stats.setdefault("saved_folders", []).append(folder_name)
        except Exception as e:
            stats.setdefault("failed_paths", []).append((str(seq_dir), str(e)))
            tqdm.write(f"  [ERROR] 保存失败: {seq_dir} — {e}")

    tqdm.write(
        f"video_done [{class_name}] {video_path.name}: "
        f"segments={stats['segments_total']}, "
        f"saved={stats['segments_saved']}, "
        f"discarded_multi_det={stats['segments_discarded_multi_det']}, "
        f"incomplete={stats['segments_incomplete']}, "
        f"frames_no_det={stats['frames_with_no_detection']}"
    )
    return stats


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def default_config():
    return {
        "input_base_dir": INPUT_BASE_DIRS[0] if INPUT_BASE_DIRS else None,
        "output_base_dir": OUTPUT_BASE_DIR,
        "class_names": list(CLASS_NAMES),
        "yolo_model_path": YOLO_MODEL_PATH,
        "target_classes": list(TARGET_CLASSES),
        "label_map": dict(LABEL_MAP),
        "frames_per_clip": FRAMES_PER_CLIP,
        "max_det": MAX_DET,
        "expand_ratio": EXPAND_RATIO,
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
        stats = process_video(file_path, class_name, class_counters, model, parent_dir_name, cfg)
        if stats and stats.get("failed_paths"):
            failed_paths.extend(stats["failed_paths"])
        saved_folders = stats.get("saved_folders", []) if stats else []
        if saved_folders:
            video_records.setdefault(class_name, []).append((file_path.name, saved_folders))

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
