"""Run the dashboard offline entry points and adapt their fused states to auto-presort."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

from auto_presort import detect_track

DASHBOARD_DIR = Path(r"E:\ActionRecognition\handover\ppc_dashboard")
RABBIT_SCRIPT = DASHBOARD_DIR / "offline_infer_rabbit.py"
PET_SCRIPT = DASHBOARD_DIR / "offline_infer_dog_cat.py"

_STATE_LABELS = {
    "EATING": "eating",
    "EAT": "eat",
    "DRINKING": "drinking",
    "DRINK": "drink",
    "ACTIVE": "active",
    "MOVING": "active",
    "RESTING": "resting",
    "REST": "resting",
}


def _run_engine(species: str, video_path: str, result_path: str,
                beh_model: str | None = None) -> dict:
    script = RABBIT_SCRIPT if species == "rabbit" else PET_SCRIPT
    if not script.is_file():
        raise FileNotFoundError(f"dashboard inference script not found: {script}")
    command = [sys.executable, str(script), "--video", video_path,
               "--output", result_path, "--no-display"]
    if beh_model:
        command += ["--beh-model", beh_model]
    if species == "pet":
        # The dog/cat script defaults to the dashboard's configured behavior and YOLO models.
        pass
    subprocess.run(command, cwd=str(DASHBOARD_DIR), check=True)
    with open(result_path, encoding="utf-8") as stream:
        return json.load(stream)


def _frame_detections(species: str, frame_result: dict) -> list[dict]:
    if species == "rabbit":
        return frame_result.get("detections") or []
    return (frame_result.get("left") or {}).get("detections") or []


def _build_track(species: str, result: dict) -> tuple[dict, list[list[dict]]]:
    frames = result.get("results")
    video = result.get("video") or {}
    fps = float(video.get("fps") or 0)
    if not isinstance(frames, list) or fps <= 0:
        raise ValueError("dashboard inference output is missing frames or a valid FPS")

    all_detections = [_frame_detections(species, frame) for frame in frames]
    track_counts = Counter(
        det.get("track_id") for detections in all_detections for det in detections
        if det.get("track_id") is not None
    )
    target_track = track_counts.most_common(1)[0][0] if track_counts else None
    per_frame = [
        [det for det in detections if det.get("track_id") == target_track]
        if target_track is not None else []
        for detections in all_detections
    ]
    raw_boxes = []
    flags = []
    for detections in per_frame:
        if detections:
            best = max(detections, key=lambda item: float(item.get("conf", 1.0)))
            bbox = best.get("bbox")
            if bbox is None:
                raise ValueError("dashboard inference did not include the detected animal bbox")
            raw_boxes.append(tuple(float(value) for value in bbox))
            flags.append(True)
        else:
            raw_boxes.append(None)
            flags.append(False)

    frame_size = result.get("frame_size") or {}
    frame_w = int(frame_size.get("width", 0))
    frame_h = int(frame_size.get("height", 0))
    if frame_w <= 0 or frame_h <= 0:
        raise ValueError("dashboard inference output is missing valid frame dimensions")
    track = {
        "fps": fps,
        "frames": len(frames),
        "frame_w": frame_w,
        "frame_h": frame_h,
        "boxes": detect_track.propagate_boxes(raw_boxes),
        "flags": flags,
        "no_animal_spans": detect_track.no_animal_spans(flags, fps, 1.0),
    }
    return track, per_frame


def _canonical_label(species: str, final_state: str, class_names: list[str]) -> str | None:
    normalized = _STATE_LABELS.get(str(final_state or "").strip().upper())
    if normalized is None:
        return None
    aliases = {
        "eat": "eating", "eating": "eat",
        "drink": "drinking", "drinking": "drink",
    }
    if normalized in class_names:
        return normalized
    alternative = aliases.get(normalized)
    return alternative if alternative in class_names else None


def build_window_classifier(species: str, per_frame: list[list[dict]], fps: float,
                            class_names: list[str]):
    def classify(start: float, end: float) -> tuple[str | None, float]:
        first = max(0, int(start * fps))
        last = min(len(per_frame), int(end * fps))
        labels = []
        confidence_by_label: dict[str, list[float]] = {}
        for detections in per_frame[first:last]:
            if not detections:
                continue
            det = detections[0]
            label = _canonical_label(species, det.get("final_state"), class_names)
            if label is None:
                continue
            labels.append(label)
            prob_key = "probabilities" if species == "rabbit" else "model_probs"
            probs = det.get(prob_key) or {}
            source_label = label
            if species == "pet" and label in {"active", "resting"}:
                source_label = "other"
            elif species == "rabbit":
                source_label = {"eat": "eating", "drink": "drinking"}.get(label, label)
            try:
                confidence = float(probs.get(source_label, 0.0))
            except (TypeError, ValueError):
                confidence = 0.0
            confidence_by_label.setdefault(label, []).append(confidence)
        if not labels:
            return None, 0.0
        counts = Counter(labels)
        label = max(counts, key=lambda key: (counts[key], -labels.index(key)))
        values = confidence_by_label.get(label) or [0.0]
        return label, sum(values) / len(values)

    return classify


def infer_video(species: str, video_path: str, class_names: list[str],
                beh_model: str | None = None):
    if species not in {"pet", "rabbit"}:
        raise ValueError(f"unsupported species: {species}")
    with tempfile.TemporaryDirectory(prefix="autopresort-fusion-") as temp_dir:
        result_path = os.path.join(temp_dir, "result.json")
        result = _run_engine(species, video_path, result_path, beh_model)
    track, per_frame = _build_track(species, result)
    classifier = build_window_classifier(species, per_frame, track["fps"], class_names)
    model_info = result.get("models") or {}
    return track, classifier, model_info
