"""Interactive offline behavior evaluation for rabbit and pet checkpoints.

The runner writes prediction traces in the format consumed by utils.metrics so
that event-level F1, segment Time IoU, and the existing timeline plots remain
consistent with the rest of the project.

Inference always crops the animal first with YOLO (same expand_ratio path as
utils.abnormal_crop_video), matching how training clips are produced.
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import yaml
from torchvision.transforms import v2
from ultralytics import YOLO

from models.teacher import SwinTeacher
from utils.abnormal_crop_video import build_smoothed_crop_boxes
from utils.metrics import write_behavior_metrics


VIDEO_SUFFIXES = {".mp4", ".avi", ".mov"}
ROOT = Path(__file__).resolve().parents[2]
RABBIT_VIDEOS = ROOT / "dataset/raw_videos/rabbit/兔子评测数据集/兔子测试新评测集"
RABBIT_ANNOTATIONS = ROOT / "dataset/raw_videos/rabbit/兔子评测数据集/兔子annotation"
PET_VIDEOS = ROOT / "dataset/raw_videos/pet/猫狗评测集/猫狗评测数据集"
PET_STATE_MAP = {
    "EATING": "EATING",
    "DRINKING": "DRINKING",
    "RESTING": "RESTING",
    "ACTIVE": "ACTIVE",
}
SPECIES = {
    "rabbit": ["resting", "active", "eating", "drinking"],
    "pet": ["eat", "drink", "active", "resting"],
}
# Detection + crop settings aligned with training crop pipeline (abnormal_crop_video).
EXPAND_RATIO = 1.25
MAX_DET = 1
DETECTOR_CONFIG = {
    "rabbit": {
        "yolo_model": Path(r"E:\ActionRecognition\detection-src\rabbit-detect\runs\rabbit-yolo11n-0603\weights\rabbit-yolo11n-0603.pt"),
        "target_classes": [0],
    },
    "pet": {
        "yolo_model": Path(r"E:\ActionRecognition\detection-src\cat_dog_detect\runs\yolo11n_scale0.6_fliplr0.42\weights\yolo11n-ssb-0710.pt"),
        "target_classes": [1, 2],
    },
}


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _safe_name(path: Path) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", path.stem).strip("_") or "video"

DASHBOARD_DIR = Path(r"E:\ActionRecognition\handover\ppc_dashboard")
RABBIT_OFFLINE_SCRIPT = DASHBOARD_DIR / "offline_infer_rabbit.py"


def _dashboard_rabbit_result(video: Path, result_path: Path) -> None:
    """Run the dashboard offline rabbit fusion engine for one video."""
    if not RABBIT_OFFLINE_SCRIPT.is_file():
        raise FileNotFoundError(f"dashboard inference script not found: {RABBIT_OFFLINE_SCRIPT}")
    command = [
        sys.executable, str(RABBIT_OFFLINE_SCRIPT),
        "--video", str(video),
        "--output", str(result_path),
        "--no-display",
    ]
    subprocess.run(command, cwd=str(DASHBOARD_DIR), check=True)


def _trace_state(raw: str | None) -> str:
    """Map dashboard rabbit states to metrics-compatible trace states."""
    state = str(raw or "").strip().upper()
    if state == "MOVING":
        return "ACTIVE"
    if state not in {"RESTING", "ACTIVE", "EATING", "DRINKING"}:
        raise ValueError(f"dashboard rabbit engine returned unsupported state: {raw!r}")
    return state


def infer_rabbit_fusion_video(
    video_path: Path,
    trace_path: Path,
    summary_path: Path,
    raw_result_path: Path,
) -> None:
    """Evaluate one rabbit video with the dashboard fusion engine and save trace records."""
    started = time.perf_counter()
    _dashboard_rabbit_result(video_path, raw_result_path)
    with raw_result_path.open(encoding="utf-8") as handle:
        result = json.load(handle)

    frames = result.get("results")
    video_info = result.get("video") or {}
    fps = float(video_info.get("fps") or 0.0)
    if not isinstance(frames, list) or fps <= 0:
        raise ValueError(f"dashboard inference output is missing frames or valid FPS for {video_path}")

    records: list[dict[str, Any]] = []
    for frame_index, frame in enumerate(frames):
        detections = frame.get("detections") or []
        if not detections:
            continue
        detection = max(detections, key=lambda item: float(item.get("conf", 1.0)))
        bbox = detection.get("bbox")
        if bbox is None or len(bbox) != 4:
            raise ValueError(f"dashboard inference did not include a valid bbox at frame {frame_index}")
        probabilities = detection.get("probabilities") or {}
        records.append(
            {
                "frame_idx": frame_index,
                "video_time_s": round(float(frame.get("timestamp_sec", frame_index / fps)), 4),
                "track_id": detection.get("track_id", 0),
                "fusion": {"final_state": _trace_state(detection.get("final_state"))},
                "probabilities": {str(label).upper(): float(prob) for label, prob in probabilities.items()},
                "bbox": {"x1": float(bbox[0]), "y1": float(bbox[1]), "x2": float(bbox[2]), "y2": float(bbox[3])},
                "frame_pipeline_ms": 0.0,
                "action_ms": 0.0,
                "target_fps": fps,
            }
        )
    if not records:
        raise RuntimeError(f"dashboard inference produced no detections for {video_path}")

    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with trace_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "start_seconds": 0.0,
                "end_seconds": records[-1]["video_time_s"] + 1.0 / fps,
                "wall_time_seconds": round(time.perf_counter() - started, 3),
                "engine": "dashboard_rabbit_fusion",
                "engine_script": str(RABBIT_OFFLINE_SCRIPT),
                "detector": "dashboard_default_yolo11n-rabbit.pt.enc",
                "behavior_model": "dashboard_default_repvit_best_distilled.pth.enc",
            },
            handle,
            indent=2,
        )


def find_rabbit_pairs(video_root: Path = RABBIT_VIDEOS, annotation_root: Path = RABBIT_ANNOTATIONS) -> list[tuple[Path, Path]]:
    """Match annotations by their video_path field, ignoring redundant JSONs."""
    videos = {path.name: path for path in video_root.rglob("*") if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES}
    pairs: list[tuple[Path, Path]] = []
    for annotation in annotation_root.rglob("*.json"):
        try:
            video_name = Path(str(_read_json(annotation).get("video_path", ""))).name
        except (OSError, json.JSONDecodeError):
            continue
        if video_name in videos:
            pairs.append((videos[video_name], annotation))
    return sorted(pairs, key=lambda item: item[0].name.lower())


def pet_annotation_from_states(states_path: Path, output_path: Path) -> Path:
    """Convert an existing pet *_states.json to metrics.py's annotation schema."""
    payload = _read_json(states_path)
    frames = payload.get("frames", [])
    if not frames:
        raise ValueError(f"No frames found in annotation: {states_path}")
    points = sorted(frames, key=lambda item: float(item.get("timestamp_sec", 0)))
    segments: list[dict[str, Any]] = []
    for index, frame in enumerate(points):
        state = PET_STATE_MAP.get(str(frame.get("state", "")).upper(), "ACTIVE")
        start_ms = int(round(float(frame.get("timestamp_sec", 0)) * 1000))
        if index + 1 < len(points):
            end_ms = int(round(float(points[index + 1].get("timestamp_sec", 0)) * 1000))
        elif len(points) > 1:
            end_ms = start_ms + max(1, start_ms - int(round(float(points[index - 1].get("timestamp_sec", 0)) * 1000)))
        else:
            end_ms = start_ms + 1
        if end_ms <= start_ms:
            continue
        if segments and segments[-1]["state"] == state and segments[-1]["end_ms"] == start_ms:
            segments[-1]["end_ms"] = end_ms
        else:
            segments.append({"start_ms": start_ms, "end_ms": end_ms, "state": state})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump({"schema": "behavior_eval_annotations_v1", "video_path": Path(str(payload.get("video", ""))).name,
                   "time_unit": "ms", "segments": segments}, handle, indent=2)
    return output_path


def find_pet_pairs(video_root: Path = PET_VIDEOS, converted_dir: Path | None = None) -> list[tuple[Path, Path]]:
    """Return only pet videos with a *_states.json annotation."""
    converted_dir = converted_dir or video_root / ".generated_annotations"
    pairs: list[tuple[Path, Path]] = []
    for states_path in video_root.rglob("*_states.json"):
        try:
            video = Path(str(_read_json(states_path).get("video", "")))
        except (OSError, json.JSONDecodeError):
            continue
        if not video.is_file():
            # State files may retain an old absolute root; locate their sibling by name.
            candidates = list(states_path.parent.glob(video.name))
            if not candidates:
                continue
            video = candidates[0]
        annotation = converted_dir / f"{_safe_name(video)}.annotations.json"
        pairs.append((video, pet_annotation_from_states(states_path, annotation)))
    return sorted(pairs, key=lambda item: item[0].name.lower())


def _recognizer_config() -> dict[str, Any]:
    with (ROOT / "config.yaml").open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    return dict(config.get("task_configs", {}).get("normal", {}).get("recognizer", {}))


def _unwrap_state(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if isinstance(raw.get(key), dict):
                return raw[key]
    return raw if isinstance(raw, dict) else {}


def _infer_backbone_from_checkpoint(checkpoint: Path, cfg: dict[str, Any]) -> str:
    """Infer backbone type from checkpoint path or filename.

    Looks for repvit_m* pattern in path/filename. Falls back to config if not found.
    """
    import re
    # Check path components for repvit_m* pattern
    path_str = str(checkpoint)
    match = re.search(r'repvit_m\d+(?:_\d+)?', path_str, re.IGNORECASE)
    if match:
        backbone = match.group(0).lower()
        print(f"[load_model] Inferred backbone from checkpoint path: {backbone}")
        return backbone

    # Fall back to config
    backbone = cfg.get("backbone") or cfg.get("timm_model", "repvit_m1")
    print(f"[load_model] Using backbone from config: {backbone}")
    return backbone


def load_model(checkpoint: Path, model_kind: str, class_names: list[str], device: torch.device):
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    state = _unwrap_state(torch.load(checkpoint, map_location="cpu"))
    if model_kind == "teacher":
        model = SwinTeacher(checkpoint_path=None, num_classes=len(class_names))
        state = {key.replace("base.model.", "").replace("module.", ""): value for key, value in state.items()}
        target = model.model
    else:
        from models.student_factory import MOVINET_VARIANTS, build_student

        cfg = _recognizer_config()
        # Infer backbone from checkpoint path/filename first, fall back to config
        backbone = _infer_backbone_from_checkpoint(checkpoint, cfg)

        if backbone in MOVINET_VARIANTS:
            model = build_student(cfg, len(class_names))
        else:
            # Do not download pretrained weights when a complete checkpoint is supplied.
            from models.timm_gru_student import TimmGRUStudent
            hidden = cfg.get("gru_hidden", "auto")
            if hidden == "auto" or not hidden:
                from models.student_factory import BACKBONE_GRU_HIDDEN
                hidden = BACKBONE_GRU_HIDDEN.get(backbone, 256)
            model = TimmGRUStudent(backbone, len(class_names), int(hidden), pretrained=False, dropout=cfg.get("dropout", 0.2))
        state = {key.replace("module.", ""): value for key, value in state.items()}
        target = model
    missing, unexpected = target.load_state_dict(state, strict=False)
    if unexpected or any("head" in key or "fc" in key for key in missing):
        raise ValueError(f"Checkpoint does not match the selected {model_kind} model: missing={missing}, unexpected={unexpected}")
    return model.to(device).eval()


def load_detector(species: str, device: torch.device):
    """Load the species-specific YOLO detector used for animal crops."""
    if species not in DETECTOR_CONFIG:
        raise ValueError(f"Unsupported species for detector: {species}")
    cfg = DETECTOR_CONFIG[species]
    weights = Path(cfg["yolo_model"])
    if not weights.is_file():
        raise FileNotFoundError(f"YOLO checkpoint not found for {species}: {weights}")
    detector = YOLO(str(weights))
    # Ultralytics accepts device strings like "cpu" / "0".
    device_str = "cpu" if device.type == "cpu" else str(device.index if device.index is not None else 0)
    try:
        detector.to(device_str)
    except Exception:
        # Some ultralytics builds ignore .to(); predict(device=...) still works.
        pass
    return detector, {
        "target_classes": list(cfg["target_classes"]),
        "max_det": MAX_DET,
        "expand_ratio": EXPAND_RATIO,
        "device": device_str,
        "yolo_model": str(weights),
    }


def _clip_transform(img_size: int):
    return v2.Compose([v2.ToDtype(torch.float32, scale=True), v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


def _detect_box(detector, frame: np.ndarray, detect_cfg: dict[str, Any], last_valid_box: tuple[float, float, float, float] | None):
    """Return xyxy box with last-valid fallback, matching abnormal_crop_video."""
    results = detector.predict(
        frame,
        classes=detect_cfg["target_classes"],
        verbose=False,
        max_det=detect_cfg["max_det"],
        device=detect_cfg.get("device", "cpu"),
    )[0]
    boxes = results.boxes
    if boxes is not None and len(boxes) >= 1:
        return tuple(float(v) for v in boxes[0].xyxy[0].cpu().numpy())
    return last_valid_box


def _crop_frame(frame: np.ndarray, crop_box: tuple[int, int, int, int], target_wh: tuple[int, int] | None = None) -> np.ndarray:
    x1, y1, x2, y2 = crop_box
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        crop = frame
    if target_wh is not None and (crop.shape[1] != target_wh[0] or crop.shape[0] != target_wh[1]):
        crop = cv2.resize(crop, target_wh)
    return crop


def _video_frames(
    video_path: Path,
    sample_every_s: float,
    temporal_window: int,
    img_size: int,
    detector,
    detect_cfg: dict[str, Any],
):
    """Yield (index, time_s, clip_tensor, bbox) with YOLO crop + expand_ratio preprocessing.

    bbox is the detection box for the last frame in the window (xyxy format in original frame coordinates).
    """
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frame_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    frame_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    duration_s = count / fps if fps > 0 else 0.0
    sample_times = np.arange(0.0, max(duration_s, 0.001), sample_every_s)
    transform = _clip_transform(img_size)
    last_valid_box: tuple[float, float, float, float] | None = None
    missing_windows = 0

    for point_index, point_s in enumerate(sample_times):
        end_frame = min(max(0, int(round(point_s * fps))), max(0, count - 1))
        indices = np.linspace(max(0, end_frame - temporal_window + 1), end_frame, temporal_window).astype(int)
        frames_bgr: list[np.ndarray] = []
        boxes_by_frame: list[tuple[float, float, float, float] | None] = []

        for frame_index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            ok, frame = capture.read()
            if not ok or frame is None:
                frame = np.zeros((max(frame_h, 1), max(frame_w, 1), 3), dtype=np.uint8)
                box = last_valid_box
            else:
                if frame_w <= 0 or frame_h <= 0:
                    frame_h, frame_w = frame.shape[:2]
                box = _detect_box(detector, frame, detect_cfg, last_valid_box)
                if box is not None:
                    last_valid_box = box
            frames_bgr.append(frame)
            boxes_by_frame.append(box)

        if frame_w <= 0 or frame_h <= 0:
            frame_h, frame_w = frames_bgr[0].shape[:2]

        if all(box is None for box in boxes_by_frame):
            # No detection in this window and no history: fall back to full frame.
            missing_windows += 1
            cropped_frames = frames_bgr
        else:
            crop_boxes = build_smoothed_crop_boxes(
                boxes_by_frame,
                frame_width=frame_w,
                frame_height=frame_h,
                fallback_size=(frame_w, frame_h),
                expand_ratio=float(detect_cfg.get("expand_ratio", EXPAND_RATIO)),
            )
            crop_w = crop_boxes[0][2] - crop_boxes[0][0]
            crop_h = crop_boxes[0][3] - crop_boxes[0][1]
            cropped_frames = [
                _crop_frame(frame, crop_box, target_wh=(crop_w, crop_h))
                for frame, crop_box in zip(frames_bgr, crop_boxes)
            ]

        tensors = []
        for frame in cropped_frames:
            rgb = cv2.cvtColor(cv2.resize(frame, (img_size, img_size)), cv2.COLOR_BGR2RGB)
            tensors.append(torch.from_numpy(rgb).permute(2, 0, 1))

        # Return the detection box for the last frame (corresponding to video_time_s)
        final_bbox = boxes_by_frame[-1]
        yield point_index, float(point_s), transform(torch.stack(tensors)), final_bbox

    capture.release()
    if missing_windows:
        print(f"  [warn] {missing_windows} window(s) had no detection; used full-frame fallback.")


def infer_video(
    model,
    model_kind: str,
    video_path: Path,
    class_names: list[str],
    trace_path: Path,
    summary_path: Path,
    sample_every_s: float,
    temporal_window: int,
    img_size: int,
    device: torch.device,
    detector,
    detect_cfg: dict[str, Any],
) -> None:
    records: list[dict[str, Any]] = []
    started = time.perf_counter()
    with torch.no_grad():
        for frame_idx, video_time_s, clip, bbox in _video_frames(
            video_path, sample_every_s, temporal_window, img_size, detector, detect_cfg
        ):
            action_started = time.perf_counter()
            batch = clip.unsqueeze(0).to(device)
            if model_kind == "teacher":
                logits, _ = model(batch)
            elif hasattr(model, "forward_seq"):
                logits = model.forward_seq(batch)
            else:
                logits = model(batch)
            predicted = int(logits.argmax(dim=1).item())

            # Get probabilities via softmax
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
            probs_dict = {class_names[i].upper(): float(probs[i]) for i in range(len(class_names))}

            action_ms = (time.perf_counter() - action_started) * 1000.0

            # Prepare bbox data (None if no detection)
            bbox_data = None
            if bbox is not None:
                bbox_data = {
                    "x1": float(bbox[0]),
                    "y1": float(bbox[1]),
                    "x2": float(bbox[2]),
                    "y2": float(bbox[3]),
                }

            records.append(
                {
                    "frame_idx": frame_idx,
                    "video_time_s": round(video_time_s, 4),
                    "track_id": 0,
                    "fusion": {"final_state": class_names[predicted].upper()},
                    "probabilities": probs_dict,
                    "bbox": bbox_data,
                    "frame_pipeline_ms": 0.0,
                    "action_ms": round(action_ms, 4),
                    "target_fps": round(1.0 / sample_every_s, 4),
                }
            )
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with trace_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    end_s = records[-1]["video_time_s"] + sample_every_s if records else 0.0
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "start_seconds": 0.0,
                "end_seconds": end_s,
                "wall_time_seconds": round(time.perf_counter() - started, 3),
                "detector": detect_cfg.get("yolo_model"),
                "expand_ratio": detect_cfg.get("expand_ratio", EXPAND_RATIO),
            },
            handle,
            indent=2,
        )


def _scalar_metric(value: Any) -> float | None:
    """Normalize metrics.py fields that may be a bare number or a {f1:...} dict."""
    if value is None:
        return None
    if isinstance(value, dict):
        value = value.get("f1", value.get("macro"))
    if value is None:
        return None
    return float(value)


def _metric_row(video: Path, metric: dict[str, Any]) -> dict[str, Any]:
    quality = metric["p0_quality"]
    event = quality["event_detection_f1"]
    by_tiou = event.get("by_tiou") or {} if isinstance(event, dict) else {}
    agreement = quality["time_agreement"]
    event_counts = event.get("by_class", {})
    transition = quality["transition_f1"]
    return {
        "video": video.name,
        "segment_iou_macro": _scalar_metric(quality["segment_iou_macro"]),
        "time_accuracy": agreement["accuracy"],
        "unknown_rate": agreement["unknown_rate"],
        "time_iou_by_class": quality["segment_iou_by_class"],
        "time_iou_macro": _scalar_metric(quality["segment_iou_macro"]),
        "event_f1_tiou_0.5": _scalar_metric(event.get("macro") if isinstance(event, dict) else event),
        "event_f1_tiou_0.3": _scalar_metric(by_tiou.get("0.3")),
        "event_f1_tiou_0.7": _scalar_metric(by_tiou.get("0.7")),
        "transition_f1": _scalar_metric(quality["transition_f1"]),
        "event_tp": sum(item["tp"] for item in event_counts.values()),
        "event_fp": sum(item["fp"] for item in event_counts.values()),
        "event_fn": sum(item["fn"] for item in event_counts.values()),
        "transition_tp": transition["tp"],
        "transition_fp": transition["fp"],
        "transition_fn": transition["fn"],
        "false_transitions_per_minute": round(transition["fp"] * 60000 / agreement["gt_ms"], 4) if agreement["gt_ms"] else None,
        "timeline_plot": metric["artifacts"]["timeline_plot"],
    }


def _mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values = []
    for row in rows:
        value = _scalar_metric(row.get(key))
        if value is not None:
            values.append(value)
    return round(sum(values) / len(values), 4) if values else None


def _aggregate_quality(rows: list[dict[str, Any]], metrics: list[dict[str, Any]]) -> dict[str, Any]:
    aggregate = {
        key: _mean(rows, key)
        for key in ("time_iou_macro", "event_f1_tiou_0.3", "event_f1_tiou_0.5", "event_f1_tiou_0.7", "transition_f1")
    }
    gt_ms = correct_ms = unknown_ms = 0
    classes: dict[str, dict[str, int]] = {}
    event_counts: dict[str, dict[str, int]] = {}
    transition_counts = {"tp": 0, "fp": 0, "fn": 0}
    for metric in metrics:
        quality = metric["p0_quality"]
        agreement = quality["time_agreement"]
        gt_ms += agreement["gt_ms"]
        correct_ms += agreement["correct_ms"]
        unknown_ms += agreement["unknown_ms"]
        for state, counts in agreement["by_class"].items():
            total = classes.setdefault(state, {"gt_ms": 0, "pred_ms": 0, "intersection_ms": 0})
            for key in total:
                total[key] += counts[key]
        for state, counts in quality["event_detection_f1"]["by_class"].items():
            total = event_counts.setdefault(state, {"tp": 0, "fp": 0, "fn": 0})
            for key in total:
                total[key] += counts[key]
        for key in transition_counts:
            transition_counts[key] += quality["transition_f1"][key]
    by_class = {}
    for state, counts in sorted(classes.items()):
        if not counts["gt_ms"]:
            continue
        union = counts["gt_ms"] + counts["pred_ms"] - counts["intersection_ms"]
        by_class[state] = round(counts["intersection_ms"] / union, 4) if union else 0.0
    tp, fp, fn = (transition_counts[key] for key in ("tp", "fp", "fn"))
    aggregate.update({
        "time_accuracy": round(correct_ms / gt_ms, 4) if gt_ms else None,
        "unknown_rate": round(unknown_ms / gt_ms, 4) if gt_ms else None,
        "annotated_ms": gt_ms,
        "correct_ms": correct_ms,
        "time_iou_by_class": by_class,
        "pooled_time_iou_macro": round(sum(by_class.values()) / len(by_class), 4) if by_class else None,
        "event_counts_tiou_0.5": event_counts,
        "transition_counts": transition_counts,
        "pooled_transition_f1": round(2 * tp / (2 * tp + fp + fn), 4) if 2 * tp + fp + fn else None,
        "false_transitions_per_minute": round(fp * 60000 / gt_ms, 4) if gt_ms else None,
    })
    aggregate["segment_iou_macro"] = aggregate["pooled_time_iou_macro"]
    aggregate["time_iou_macro"] = aggregate["segment_iou_macro"]
    return aggregate


def _dominant_gt_state(annotation_path: Path) -> str:
    """Return the GT state with the longest total duration, e.g. 'resting'."""
    try:
        segments = _read_json(annotation_path).get("segments", [])
    except (OSError, json.JSONDecodeError):
        return ""
    durations: dict[str, float] = {}
    for segment in segments:
        state = str(segment.get("state", "")).lower()
        span = float(segment.get("end_ms", 0)) - float(segment.get("start_ms", 0))
        durations[state] = durations.get(state, 0.0) + max(0.0, span)
    return max(durations, key=durations.get) if durations else ""


def write_evaluation_report(output_dir: Path, summary: dict[str, Any]) -> Path:
    """Write evaluation_report.html: a single-file multi-tab report with Summary + per-video detail tabs."""
    def fmt(value: Any) -> str:
        value = _scalar_metric(value)
        return "-" if value is None else f"{value:.4f}"

    aggregate = summary["aggregate"]
    class_iou = aggregate.get("time_iou_by_class", {})
    classes = sorted(class_iou)
    summary_items = [
        ("Species", summary["species"]),
        ("Model kind", summary["model_kind"]),
        ("Checkpoint", summary["checkpoint"]),
        ("Detector", summary["detector"]),
        ("Expand ratio", summary["expand_ratio"]),
        ("Annotated videos", summary["annotated_videos"]),
        ("Sample every (s)", summary["sample_every_seconds"]),
        ("segment_iou_macro (Time IoU)", fmt(aggregate.get("segment_iou_macro"))),
    ]

    # HTML header with tab-switching CSS and JavaScript
    lines = [
        "<!DOCTYPE html>",
        "<html><head><meta charset='utf-8'><title>Evaluation Report</title><style>",
        "body{font-family:Segoe UI,Arial,sans-serif;margin:0;padding:0;}",
        ".tabs{display:flex;background:#f0f0f0;border-bottom:2px solid #ccc;overflow-x:auto;}",
        ".tab-button{padding:12px 20px;cursor:pointer;border:none;background:none;font-size:14px;white-space:nowrap;}",
        ".tab-button:hover{background:#e0e0e0;}",
        ".tab-button.active{background:#fff;border-bottom:2px solid #fff;margin-bottom:-2px;font-weight:bold;}",
        ".tab-content{display:none;padding:24px;}",
        ".tab-content.active{display:block;}",
        "table{border-collapse:collapse;margin-bottom:32px;}",
        "th,td{border:1px solid #bbb;padding:4px 10px;text-align:left;vertical-align:top;}",
        "th{background:#f0f0f0;}",
        ".video-link{color:#0066cc;cursor:pointer;text-decoration:underline;}",
        ".video-link:hover{color:#004499;}",
        ".video-container{margin:20px 0;max-width:900px;}",
        "video{width:100%;max-width:800px;border:1px solid #ccc;}",
        ".timeline-container{margin:20px 0;max-width:900px;}",
        ".timeline-container img{width:100%;max-width:800px;}",
        ".metrics-table{max-width:600px;}",
        ".back-link{display:inline-block;margin-bottom:16px;color:#0066cc;cursor:pointer;text-decoration:underline;}",
        ".back-link:hover{color:#004499;}",
        "</style></head><body>",
        "<div class='tabs'>",
        "<button class='tab-button active' onclick='showTab(\"summary\")'>Summary</button>",
    ]

    # Add tab buttons for each video
    for idx, row in enumerate(summary["videos"]):
        video_name = html.escape(str(row['video']))
        lines.append(f"<button class='tab-button' onclick='showTab(\"video{idx}\")'>{video_name}</button>")

    lines.append("</div>")

    # Summary tab content
    lines += [
        "<div id='summary' class='tab-content active'>",
        "<h1>Evaluation Report</h1>",
        "<h2>Summary</h2>",
        "<table>",
    ]
    for key, value in summary_items:
        lines.append(f"<tr><th>{html.escape(str(key))}</th><td>{html.escape(str(value))}</td></tr>")
    for state in classes:
        lines.append(f"<tr><th>{html.escape(state)} Time IoU</th><td>{fmt(class_iou[state])}</td></tr>")
    lines += [
        "</table>",
        "<p>Per-video: mean IoU of GT classes. Overall: sum intersection and union per GT class across videos, then average class IoU. A dash means that class has no GT in the video.</p>",
        "<h2>Timeline Comparison</h2>",
        "<table>",
        "<tr><th>Video</th><th>GT label</th><th>Eval s</th><th>segment_iou_macro (Time IoU)</th>"
        + "".join(f"<th>{html.escape(state)} Time IoU</th>" for state in classes)
        + "<th>GT/system timeline</th><th>Notes</th></tr>",
    ]

    # Table rows with clickable video names
    for idx, row in enumerate(summary["videos"]):
        video_name = html.escape(str(row['video']))
        timeline_plot = str(row.get('timeline_plot', ''))
        timeline_src = html.escape(os.path.relpath(timeline_plot, output_dir).replace(os.sep, "/")) if timeline_plot else ""
        timeline_img = f"<img src='{timeline_src}' alt='timeline' style='max-width:520px;'>" if timeline_src else "-"
        class_cells = "".join(f"<td>{fmt(row.get('time_iou_by_class', {}).get(state))}</td>" for state in classes)
        lines.append(
            "<tr>"
            f"<td><span class='video-link' onclick='showTab(\"video{idx}\")'>{video_name}</span></td>"
            f"<td>{html.escape(str(row.get('gt_label', '')))}</td>"
            f"<td>{fmt(row.get('eval_seconds'))}</td>"
            f"<td>{fmt(row.get('segment_iou_macro'))}</td>"
            f"{class_cells}"
            f"<td>{timeline_img}</td>"
            "<td></td>"
            "</tr>"
        )

    lines += ["</table>", "</div>"]

    # Generate individual video detail tabs
    for idx, row in enumerate(summary["videos"]):
        video_name = html.escape(str(row['video']))
        video_path = row.get('video_path', '')
        timeline_plot = str(row.get('timeline_plot', ''))
        video_dir = output_dir / _safe_name(Path(video_path))
        trace_file = video_dir / "track_trace.jsonl"

        # Convert paths to relative URLs
        video_src = html.escape(os.path.relpath(video_path, output_dir).replace(os.sep, "/")) if video_path else ""
        timeline_src = html.escape(os.path.relpath(timeline_plot, output_dir).replace(os.sep, "/")) if timeline_plot else ""
        trace_src = html.escape(os.path.relpath(trace_file, output_dir).replace(os.sep, "/")) if trace_file.exists() else ""

        lines += [
            f"<div id='video{idx}' class='tab-content'>",
            "<span class='back-link' onclick='showTab(\"summary\")'>&larr; Back to Summary</span>",
            f"<h1>{video_name}</h1>",
        ]

        # Video player section with Canvas overlay
        if video_src:
            lines += [
                "<div class='video-container'>",
                "<h2>Video Preview</h2>",
                "<div style='position:relative;display:inline-block;'>",
                f"<video id='video{idx}_player' controls style='display:block;'>",
                f"<source src='{video_src}' type='video/mp4'>",
                "Your browser does not support the video tag.",
                "</video>",
                f"<canvas id='video{idx}_canvas' style='position:absolute;top:0;left:0;pointer-events:none;'></canvas>",
                "</div>",
                f"<div id='video{idx}_info' style='margin-top:8px;font-size:14px;color:#333;'></div>",
                "</div>",
            ]

        # Timeline section (clickable)
        if timeline_src:
            lines += [
                "<div class='timeline-container'>",
                "<h2>GT vs Prediction Timeline (click to jump)</h2>",
                "<div style='position:relative;display:inline-block;cursor:pointer;'>",
                f"<img id='video{idx}_timeline' src='{timeline_src}' alt='timeline' style='display:block;'>",
                f"<div id='video{idx}_timeline_overlay' style='position:absolute;top:0;left:0;width:100%;height:100%;'></div>",
                "</div>",
                "</div>",
            ]

        # Metrics section
        lines += [
            "<div class='metrics-container'>",
            "<h2>Metrics</h2>",
            "<table class='metrics-table'>",
            "<tr><th>Metric</th><th>Value</th></tr>",
            f"<tr><td>GT Label</td><td>{html.escape(str(row.get('gt_label', '')))}</td></tr>",
            f"<tr><td>Evaluation Duration (s)</td><td>{fmt(row.get('eval_seconds'))}</td></tr>",
            f"<tr><td>segment_iou_macro (Time IoU)</td><td>{fmt(row.get('segment_iou_macro'))}</td></tr>",
        ]
        lines.extend(
            f"<tr><td>{html.escape(state)} Time IoU</td><td>{fmt(row.get('time_iou_by_class', {}).get(state))}</td></tr>"
            for state in classes
        )
        lines += [
            "</table>",
            "</div>",
            # Embed trace data and setup for this video
            f"<script>",
            f"(function() {{",
            f"  var videoId = 'video{idx}';",
            f"  var tracePath = '{trace_src}';",
            f"  var videoDuration = {row.get('eval_seconds', 0)};",
            f"  if (tracePath) {{",
            f"    setupVideoAnnotation(videoId, tracePath, videoDuration);",
            f"  }}",
            f"}})();",
            f"</script>",
            "</div>",
        ]

    # JavaScript for tab switching and video annotation
    lines += [
        "<script>",
        "// Tab switching",
        "function showTab(tabId) {",
        "  var contents = document.getElementsByClassName('tab-content');",
        "  for (var i = 0; i < contents.length; i++) {",
        "    contents[i].classList.remove('active');",
        "  }",
        "  var buttons = document.getElementsByClassName('tab-button');",
        "  for (var i = 0; i < buttons.length; i++) {",
        "    buttons[i].classList.remove('active');",
        "  }",
        "  document.getElementById(tabId).classList.add('active');",
        "  var btnIndex = tabId === 'summary' ? 0 : parseInt(tabId.replace('video', '')) + 1;",
        "  buttons[btnIndex].classList.add('active');",
        "}",
        "",
        "// Video annotation system",
        "var videoAnnotations = {};",
        "",
        "function setupVideoAnnotation(videoId, tracePath, videoDuration) {",
        "  var video = document.getElementById(videoId + '_player');",
        "  var canvas = document.getElementById(videoId + '_canvas');",
        "  var info = document.getElementById(videoId + '_info');",
        "  var timeline = document.getElementById(videoId + '_timeline');",
        "  var timelineOverlay = document.getElementById(videoId + '_timeline_overlay');",
        "  ",
        "  if (!video || !canvas) return;",
        "  ",
        "  // Load trace data",
        "  fetch(tracePath)",
        "    .then(response => response.text())",
        "    .then(text => {",
        "      var traces = text.trim().split('\\n').map(line => JSON.parse(line));",
        "      videoAnnotations[videoId] = {",
        "        traces: traces,",
        "        videoDuration: videoDuration",
        "      };",
        "      ",
        "      // Setup canvas size to match video",
        "      video.addEventListener('loadedmetadata', function() {",
        "        canvas.width = video.videoWidth;",
        "        canvas.height = video.videoHeight;",
        "        canvas.style.width = video.offsetWidth + 'px';",
        "        canvas.style.height = video.offsetHeight + 'px';",
        "      });",
        "      ",
        "      // Update canvas on video time update",
        "      video.addEventListener('timeupdate', function() {",
        "        updateVideoAnnotation(videoId);",
        "      });",
        "      ",
        "      // Setup timeline click handler",
        "      if (timeline && timelineOverlay) {",
        "        timelineOverlay.addEventListener('click', function(e) {",
        "          var rect = timeline.getBoundingClientRect();",
        "          var x = e.clientX - rect.left;",
        "          var clickRatio = x / rect.width;",
        "          var targetTime = clickRatio * videoDuration;",
        "          video.currentTime = Math.max(0, Math.min(targetTime, video.duration));",
        "        });",
        "      }",
        "    })",
        "    .catch(err => console.error('Failed to load trace:', err));",
        "}",
        "",
        "function updateVideoAnnotation(videoId) {",
        "  var video = document.getElementById(videoId + '_player');",
        "  var canvas = document.getElementById(videoId + '_canvas');",
        "  var info = document.getElementById(videoId + '_info');",
        "  var data = videoAnnotations[videoId];",
        "  ",
        "  if (!video || !canvas || !data) return;",
        "  ",
        "  var ctx = canvas.getContext('2d');",
        "  ctx.clearRect(0, 0, canvas.width, canvas.height);",
        "  ",
        "  // Find closest trace record to current video time",
        "  var currentTime = video.currentTime;",
        "  var closest = null;",
        "  var minDiff = Infinity;",
        "  ",
        "  for (var i = 0; i < data.traces.length; i++) {",
        "    var trace = data.traces[i];",
        "    var diff = Math.abs(trace.video_time_s - currentTime);",
        "    if (diff < minDiff) {",
        "      minDiff = diff;",
        "      closest = trace;",
        "    }",
        "  }",
        "  ",
        "  if (!closest) return;",
        "  ",
        "  // Draw bbox if available",
        "  if (closest.bbox) {",
        "    var bbox = closest.bbox;",
        "    var scaleX = canvas.width / video.videoWidth;",
        "    var scaleY = canvas.height / video.videoHeight;",
        "    ",
        "    ctx.strokeStyle = '#00ff00';",
        "    ctx.lineWidth = 3;",
        "    ctx.strokeRect(",
        "      bbox.x1 * scaleX,",
        "      bbox.y1 * scaleY,",
        "      (bbox.x2 - bbox.x1) * scaleX,",
        "      (bbox.y2 - bbox.y1) * scaleY",
        "    );",
        "    ",
        "    // Draw label and probability",
        "    var state = closest.fusion.final_state;",
        "    var probs = closest.probabilities || {};",
        "    var prob = probs[state] || 0;",
        "    ",
        "    var label = state + ' (' + (prob * 100).toFixed(1) + '%)';",
        "    ",
        "    ctx.font = 'bold 16px Arial';",
        "    ctx.fillStyle = '#00ff00';",
        "    ctx.strokeStyle = '#000000';",
        "    ctx.lineWidth = 3;",
        "    ",
        "    var textX = bbox.x1 * scaleX;",
        "    var textY = bbox.y1 * scaleY - 10;",
        "    if (textY < 20) textY = bbox.y1 * scaleY + 25;",
        "    ",
        "    ctx.strokeText(label, textX, textY);",
        "    ctx.fillText(label, textX, textY);",
        "  }",
        "  ",
        "  // Update info text",
        "  if (info) {",
        "    var state = closest.fusion.final_state;",
        "    var probs = closest.probabilities || {};",
        "    var probText = Object.keys(probs)",
        "      .map(key => key + ': ' + (probs[key] * 100).toFixed(1) + '%')",
        "      .join(', ');",
        "    info.innerHTML = '<strong>Time:</strong> ' + currentTime.toFixed(2) + 's | ' +",
        "                     '<strong>Prediction:</strong> ' + state + ' | ' +",
        "                     '<strong>Probabilities:</strong> ' + probText;",
        "  }",
        "}",
        "</script>",
        "</body></html>",
    ]

    report_path = output_dir / "evaluation_report.html"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def _write_evaluation_outputs(output_dir: Path, metadata: dict[str, Any], rows: list[dict[str, Any]], metrics: list[dict[str, Any]]) -> Path:
    summary = {
        **metadata,
        "annotated_videos": len(rows),
        "aggregate": _aggregate_quality(rows, metrics),
        "videos": rows,
    }
    with (output_dir / "evaluation_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    with (output_dir / "evaluation_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ("video", "gt_label", "eval_seconds", "segment_iou_macro", "timeline_plot", "video_path")
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in fields} for row in rows)
    report_path = write_evaluation_report(output_dir, summary)
    aggregate = summary["aggregate"]
    print("\nEvaluation complete")
    print(f"Annotated videos: {len(rows)}")
    print(f"segment_iou_macro (Time IoU): {aggregate['segment_iou_macro']}")
    print(f"Results saved to: {output_dir}")
    print(f"Report: {report_path}")
    return output_dir


def rescore_evaluation(source_dir: Path, output_dir: Path) -> Path:
    """Re-evaluate saved predictions without model inference, preserving source outputs."""
    source = _read_json(source_dir / "evaluation_summary.json")
    for row in source["videos"]:
        old_dir = source_dir / _safe_name(Path(row["video_path"]))
        for filename in ("track_trace.jsonl", "summary.json", "metrics.json"):
            if not (old_dir / filename).is_file():
                raise FileNotFoundError(old_dir / filename)
        annotation = Path(_read_json(old_dir / "metrics.json")["inputs"]["annotations_path"])
        if not annotation.is_file():
            raise FileNotFoundError(annotation)
    output_dir.mkdir(parents=True, exist_ok=False)
    rows: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    for source_row in source["videos"]:
        video = Path(source_row["video_path"])
        name = _safe_name(video)
        old_dir = source_dir / name
        new_dir = output_dir / name
        new_dir.mkdir()
        for filename in ("track_trace.jsonl", "summary.json"):
            shutil.copy2(old_dir / filename, new_dir / filename)
        old_metric = _read_json(old_dir / "metrics.json")
        annotation = Path(old_metric["inputs"]["annotations_path"])
        metric = write_behavior_metrics(
            trace_path=new_dir / "track_trace.jsonl",
            summary_path=new_dir / "summary.json",
            annotations_path=annotation,
            output_path=new_dir / "metrics.json",
            timeline_plot_path=new_dir / "gt_system_timeline.png",
            segment_iou_gt_classes_only=True,
        )
        row = _metric_row(video, metric)
        row.update(gt_label=_dominant_gt_state(annotation), eval_seconds=_read_json(new_dir / "summary.json")["end_seconds"], video_path=str(video))
        rows.append(row)
        metrics.append(metric)
    metadata = {key: value for key, value in source.items() if key not in {"aggregate", "videos", "annotated_videos"}}
    metadata["recomputed_from"] = str(source_dir.resolve())
    return _write_evaluation_outputs(output_dir, metadata, rows, metrics)


def run_evaluation(
    species: str,
    model_kind: str,
    checkpoint: Path | str,
    output_dir: Path | None = None,
    sample_every_s: float = 0.5,
) -> Path:
    class_names = SPECIES[species]
    output_dir = output_dir or Path(str(checkpoint)).parent / f"test_results_{species}_{model_kind}_{datetime.now():%Y%m%d_%H%M%S}"
    output_dir.mkdir(parents=True, exist_ok=True)
    if species == "rabbit" and model_kind == "fusion":
        pairs = find_rabbit_pairs()
        if not pairs:
            raise RuntimeError("No annotated rabbit evaluation videos were found.")
        rows: list[dict[str, Any]] = []
        metrics: list[dict[str, Any]] = []
        for index, (video, annotation) in enumerate(pairs, 1):
            video_dir = output_dir / _safe_name(video)
            print(f"[{index}/{len(pairs)}] {video.name}")
            infer_rabbit_fusion_video(
                video,
                video_dir / "track_trace.jsonl",
                video_dir / "summary.json",
                video_dir / "dashboard_result.json",
            )
            metric = write_behavior_metrics(
                trace_path=video_dir / "track_trace.jsonl",
                summary_path=video_dir / "summary.json",
                annotations_path=annotation,
                output_path=video_dir / "metrics.json",
                timeline_plot_path=video_dir / "gt_system_timeline.png",
                segment_iou_gt_classes_only=True,
            )
            row = _metric_row(video, metric)
            row["gt_label"] = _dominant_gt_state(annotation)
            rows.append(row)
            metrics.append(metric)
        metadata = {
            "species": species,
            "model_kind": model_kind,
            "checkpoint": "dashboard_default",
            "detector": "dashboard_default_yolo11n-rabbit.pt.enc",
            "expand_ratio": "dashboard_engine_default",
            "sample_every_seconds": "per_frame",
        }
        return _write_evaluation_outputs(output_dir, metadata, rows, metrics)
    pairs = find_rabbit_pairs() if species == "rabbit" else find_pet_pairs(converted_dir=output_dir / "annotations")
    if not pairs:
        raise RuntimeError(f"No annotated {species} evaluation videos were found.")
    cfg = _recognizer_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating {len(pairs)} annotated {species} video(s) on {device.type.upper()}.")
    detector, detect_cfg = load_detector(species, device)
    print(f"Detector: {detect_cfg['yolo_model']} (expand_ratio={detect_cfg['expand_ratio']})")
    model = load_model(checkpoint, model_kind, class_names, device)
    metric_class_names = [
        "eating", "drinking", "active", "resting"
    ] if species == "pet" else class_names
    rows: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    for index, (video, annotation) in enumerate(pairs, 1):
        video_dir = output_dir / _safe_name(video)
        print(f"[{index}/{len(pairs)}] {video.name}")
        infer_video(
            model,
            model_kind,
            video,
            metric_class_names,
            video_dir / "track_trace.jsonl",
            video_dir / "summary.json",
            sample_every_s,
            int(cfg.get("temporal_window", 16)),
            int(cfg.get("imgsz", 224)),
            device,
            detector,
            detect_cfg,
        )
        metric = write_behavior_metrics(
            trace_path=video_dir / "track_trace.jsonl",
            summary_path=video_dir / "summary.json",
            annotations_path=annotation,
            output_path=video_dir / "metrics.json",
            timeline_plot_path=video_dir / "gt_system_timeline.png",
            segment_iou_gt_classes_only=True,
        )
        row = _metric_row(video, metric)
        row["gt_label"] = _dominant_gt_state(annotation)
        row["eval_seconds"] = _read_json(video_dir / "summary.json").get("end_seconds")
        row["video_path"] = str(video)  # Add video path for HTML video player
        rows.append(row)
        metrics.append(metric)
    metadata = {
        "species": species,
        "model_kind": model_kind,
        "checkpoint": str(checkpoint),
        "detector": detect_cfg["yolo_model"],
        "expand_ratio": detect_cfg["expand_ratio"],
        "sample_every_seconds": sample_every_s,
    }
    return _write_evaluation_outputs(output_dir, metadata, rows, metrics)


def _interactive_args(input_fn=None) -> list[str]:
    """Ask species / model kind with 1/2 menus, then checkpoint path."""
    if input_fn is None:
        input_fn = input
    # Keep option order aligned with pipeline_config: 1=pet, 2=rabbit.
    from .pipeline_config import _ask_choice
    species = _ask_choice("Species", ["pet", "rabbit"], input_fn)
    if species == "rabbit":
        model_kind = _ask_choice("Model type", ["fusion", "teacher", "student"], input_fn)
    else:
        model_kind = _ask_choice("Model type", ["teacher", "student"], input_fn)
    if model_kind == "fusion":
        return ["--species", species, "--model-kind", model_kind, "--checkpoint", "dashboard_default"]
    checkpoint = input_fn("Checkpoint path: ").strip().strip('"')
    return ["--species", species, "--model-kind", model_kind, "--checkpoint", checkpoint]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run offline behavior evaluation and save metrics plus timeline plots.")
    parser.add_argument("--species", choices=sorted(SPECIES))
    parser.add_argument("--model-kind", choices=("teacher", "student", "fusion"))
    parser.add_argument("--checkpoint")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--recalculate-from", type=Path, help="Recompute metrics from an existing result directory without inference.")
    parser.add_argument("--sample-every-seconds", type=float, default=0.5)
    parser.add_argument("--interactive", action="store_true")
    args = parser.parse_args(argv)
    if args.interactive:
        args = parser.parse_args(_interactive_args())
    if args.recalculate_from:
        if args.interactive or args.species or args.model_kind or args.checkpoint:
            parser.error("--recalculate-from cannot be combined with inference options")
        return args
    if not args.species or not args.model_kind:
        parser.error("--species and --model-kind are required")
    if not args.checkpoint:
        if args.model_kind == "fusion" and args.species == "rabbit":
            args.checkpoint = "dashboard_default"
        else:
            parser.error("--checkpoint is required")
    if args.sample_every_seconds <= 0:
        parser.error("--sample-every-seconds must be positive")
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.recalculate_from:
        target = args.output_dir or args.recalculate_from.with_name(args.recalculate_from.name + "_recalculated")
        rescore_evaluation(args.recalculate_from, target)
    else:
        run_evaluation(args.species, args.model_kind, Path(args.checkpoint), args.output_dir, args.sample_every_seconds)


if __name__ == "__main__":
    main()
