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
        backbone = cfg.get("backbone") or cfg.get("timm_model", "repvit_m1")
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
    """Yield (index, time_s, clip_tensor) with YOLO crop + expand_ratio preprocessing."""
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
        yield point_index, float(point_s), transform(torch.stack(tensors))

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
        for frame_idx, video_time_s, clip in _video_frames(
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
            action_ms = (time.perf_counter() - action_started) * 1000.0
            records.append(
                {
                    "frame_idx": frame_idx,
                    "video_time_s": round(video_time_s, 4),
                    "track_id": 0,
                    "fusion": {"final_state": class_names[predicted].upper()},
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
    return {
        "video": video.name,
        "time_iou_macro": _scalar_metric(quality["segment_iou_macro"]),
        "event_f1_tiou_0.5": _scalar_metric(event.get("macro") if isinstance(event, dict) else event),
        "event_f1_tiou_0.3": _scalar_metric(by_tiou.get("0.3")),
        "event_f1_tiou_0.7": _scalar_metric(by_tiou.get("0.7")),
        "transition_f1": _scalar_metric(quality["transition_f1"]),
        "timeline_plot": metric["artifacts"]["timeline_plot"],
    }


def _mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values = []
    for row in rows:
        value = _scalar_metric(row.get(key))
        if value is not None:
            values.append(value)
    return round(sum(values) / len(values), 4) if values else None


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
    """Write evaluation_report.html: a Summary table plus one row per video with its timeline plot."""
    def fmt(value: Any) -> str:
        value = _scalar_metric(value)
        return "-" if value is None else f"{value:.4f}"

    aggregate = summary["aggregate"]
    summary_items = [
        ("Species", summary["species"]),
        ("Model kind", summary["model_kind"]),
        ("Checkpoint", summary["checkpoint"]),
        ("Detector", summary["detector"]),
        ("Expand ratio", summary["expand_ratio"]),
        ("Annotated videos", summary["annotated_videos"]),
        ("Sample every (s)", summary["sample_every_seconds"]),
        ("Macro Time IoU", fmt(aggregate["time_iou_macro"])),
        ("Event F1 (tIoU 0.3)", fmt(aggregate["event_f1_tiou_0.3"])),
        ("Event F1 (tIoU 0.5)", fmt(aggregate["event_f1_tiou_0.5"])),
        ("Event F1 (tIoU 0.7)", fmt(aggregate["event_f1_tiou_0.7"])),
        ("Transition F1", fmt(aggregate["transition_f1"])),
    ]
    lines = [
        "<!DOCTYPE html>",
        "<html><head><meta charset='utf-8'><title>Evaluation Report</title><style>",
        "body{font-family:Segoe UI,Arial,sans-serif;margin:24px;}",
        "table{border-collapse:collapse;margin-bottom:32px;}",
        "th,td{border:1px solid #bbb;padding:4px 10px;text-align:left;vertical-align:top;}",
        "th{background:#f0f0f0;}",
        "img{max-width:520px;display:block;}",
        "</style></head><body>",
        "<h1>Evaluation Report</h1>",
        "<h2>Summary</h2>",
        "<table>",
    ]
    for key, value in summary_items:
        lines.append(f"<tr><th>{html.escape(str(key))}</th><td>{html.escape(str(value))}</td></tr>")
    lines += [
        "</table>",
        "<h2>Timeline Comparison</h2>",
        "<table>",
        "<tr><th>Video</th><th>GT label</th><th>Eval s</th><th>Time IoU</th>"
        "<th>Event F1 (tIoU 0.3)</th><th>Event F1 (tIoU 0.5)</th><th>Event F1 (tIoU 0.7)</th>"
        "<th>Transition F1</th><th>GT/system timeline</th><th>Notes</th></tr>",
    ]
    for row in summary["videos"]:
        plot = str(row.get("timeline_plot", ""))
        src = html.escape(os.path.relpath(plot, output_dir).replace(os.sep, "/")) if plot else ""
        image = f"<img src='{src}' alt='timeline'>" if src else "-"
        lines.append(
            "<tr>"
            f"<td>{html.escape(str(row['video']))}</td>"
            f"<td>{html.escape(str(row.get('gt_label', '')))}</td>"
            f"<td>{fmt(row.get('eval_seconds'))}</td>"
            f"<td>{fmt(row.get('time_iou_macro'))}</td>"
            f"<td>{fmt(row.get('event_f1_tiou_0.3'))}</td>"
            f"<td>{fmt(row.get('event_f1_tiou_0.5'))}</td>"
            f"<td>{fmt(row.get('event_f1_tiou_0.7'))}</td>"
            f"<td>{fmt(row.get('transition_f1'))}</td>"
            f"<td>{image}</td>"
            "<td></td>"
            "</tr>"
        )
    lines += ["</table>", "</body></html>"]
    report_path = output_dir / "evaluation_report.html"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def run_evaluation(
    species: str,
    model_kind: str,
    checkpoint: Path,
    output_dir: Path | None = None,
    sample_every_s: float = 0.5,
) -> Path:
    class_names = SPECIES[species]
    output_dir = output_dir or checkpoint.parent / f"test_results_{species}_{model_kind}_{datetime.now():%Y%m%d_%H%M%S}"
    output_dir.mkdir(parents=True, exist_ok=True)
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
        )
        row = _metric_row(video, metric)
        row["gt_label"] = _dominant_gt_state(annotation)
        row["eval_seconds"] = _read_json(video_dir / "summary.json").get("end_seconds")
        rows.append(row)
    summary = {
        "species": species,
        "model_kind": model_kind,
        "checkpoint": str(checkpoint),
        "detector": detect_cfg["yolo_model"],
        "expand_ratio": detect_cfg["expand_ratio"],
        "annotated_videos": len(rows),
        "sample_every_seconds": sample_every_s,
        "aggregate": {
            key: _mean(rows, key)
            for key in ("time_iou_macro", "event_f1_tiou_0.3", "event_f1_tiou_0.5", "event_f1_tiou_0.7", "transition_f1")
        },
        "videos": rows,
    }
    with (output_dir / "evaluation_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    with (output_dir / "evaluation_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report_path = write_evaluation_report(output_dir, summary)
    aggregate = summary["aggregate"]
    print("\nEvaluation complete")
    print(f"Annotated videos: {len(rows)}")
    print(f"Macro Time IoU: {aggregate['time_iou_macro']}")
    print(f"Event F1 (tIoU 0.5): {aggregate['event_f1_tiou_0.5']}")
    print(f"Transition F1: {aggregate['transition_f1']}")
    print(f"Results saved to: {output_dir}")
    print(f"Report: {report_path}")
    return output_dir


def _interactive_args(input_fn=None) -> list[str]:
    """Ask species / model kind with 1/2 menus, then checkpoint path."""
    if input_fn is None:
        input_fn = input
    # Keep option order aligned with pipeline_config: 1=pet, 2=rabbit.
    from .pipeline_config import _ask_choice
    species = _ask_choice("Species", ["pet", "rabbit"], input_fn)
    model_kind = _ask_choice("Model type", ["teacher", "student"], input_fn)
    checkpoint = input_fn("Checkpoint path: ").strip().strip('"')
    return ["--species", species, "--model-kind", model_kind, "--checkpoint", checkpoint]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run offline behavior evaluation and save metrics plus timeline plots.")
    parser.add_argument("--species", choices=sorted(SPECIES))
    parser.add_argument("--model-kind", choices=("teacher", "student"))
    parser.add_argument("--checkpoint")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--sample-every-seconds", type=float, default=0.5)
    parser.add_argument("--interactive", action="store_true")
    args = parser.parse_args(argv)
    if args.interactive:
        args = parser.parse_args(_interactive_args())
    if not args.species or not args.model_kind or not args.checkpoint:
        parser.error("--species, --model-kind, and --checkpoint are required")
    if args.sample_every_seconds <= 0:
        parser.error("--sample-every-seconds must be positive")
    return args


def main(argv=None):
    args = parse_args(argv)
    run_evaluation(args.species, args.model_kind, Path(args.checkpoint), args.output_dir, args.sample_every_seconds)


if __name__ == "__main__":
    main()
