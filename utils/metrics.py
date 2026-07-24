from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator


EVENT_TIOU_THRESHOLDS = (0.3, 0.5, 0.7)
UNKNOWN_STATE = "UNKNOWN"
AMBIGUOUS_STATE = "AMBIGUOUS"
SYSTEM_STATES = {"RESTING", "ACTIVE", "EATING", "DRINKING", "OTHER"}
ANNOTATION_STATES = SYSTEM_STATES | {AMBIGUOUS_STATE}
MOTION_STATES = {"RESTING", "ACTIVE"}
TIMELINE_PLOT_NAME = "gt_system_timeline.png"
TIMELINE_STATE_COLORS = {
    "RESTING": "#4caf50",
    "ACTIVE": "#ff9800",
    "EATING": "#e53935",
    "DRINKING": "#1e88e5",
    "OTHER": "#8d6e63",
    "AMBIGUOUS": "#64748b",
    "GAP": "#eeeeee",
    "UNKNOWN": "#9e9e9e",
}
TIMELINE_STATE_ORDER = ["RESTING", "ACTIVE", "EATING", "DRINKING", "OTHER", "AMBIGUOUS", "GAP", "UNKNOWN"]
ANNOTATION_SUFFIXES = (".annotations.json", ".annotation.json")


def _round_float(value: Any, digits: int = 4) -> float:
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return 0.0


def _load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                records.append(json.loads(text))
    return records


def _annotation_output_stem(annotations_path: str | Path) -> str:
    name = Path(annotations_path).name
    lower_name = name.lower()
    for suffix in ANNOTATION_SUFFIXES:
        if lower_name.endswith(suffix):
            return name[: -len(suffix)]
    return Path(name).stem


def default_metrics_output_name(annotations_path: str | Path) -> str:
    return f"{_annotation_output_stem(annotations_path)}.metrics.json"


def default_timeline_plot_name(annotations_path: str | Path) -> str:
    return f"{_annotation_output_stem(annotations_path)}.gt_system_timeline.png"


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * (percentile / 100.0)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _f1(precision: float, recall: float) -> float:
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def _duration(segment: dict[str, Any]) -> int:
    return max(0, int(segment["end_ms"]) - int(segment["start_ms"]))


def _overlap_ms(a: dict[str, Any], b: dict[str, Any]) -> int:
    return max(0, min(int(a["end_ms"]), int(b["end_ms"])) - max(int(a["start_ms"]), int(b["start_ms"])))


def _tiou(a: dict[str, Any], b: dict[str, Any]) -> float:
    intersection = _overlap_ms(a, b)
    union = _duration(a) + _duration(b) - intersection
    return intersection / union if union > 0 else 0.0


def _interval(start_ms: int, end_ms: int) -> dict[str, int]:
    return {"start_ms": int(start_ms), "end_ms": int(end_ms)}


def _contains_time(intervals: list[dict[str, Any]], time_ms: int) -> bool:
    return any(int(interval["start_ms"]) <= time_ms < int(interval["end_ms"]) for interval in intervals)


def _merge_adjacent(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for segment in sorted(segments, key=lambda item: (int(item["start_ms"]), int(item["end_ms"]))):
        if _duration(segment) <= 0:
            continue
        if merged and merged[-1]["state"] == segment["state"] and merged[-1]["end_ms"] == segment["start_ms"]:
            merged[-1]["end_ms"] = segment["end_ms"]
        else:
            merged.append(dict(segment))
    return merged


def load_annotations(annotations_path: str | Path) -> list[dict[str, Any]]:
    payload = _load_json(annotations_path)
    if payload.get("schema") != "behavior_eval_annotations_v1":
        raise ValueError("annotation schema must be behavior_eval_annotations_v1")
    if payload.get("time_unit") != "ms":
        raise ValueError('annotation time_unit must be "ms"')

    segments = payload.get("segments")
    if not isinstance(segments, list):
        raise ValueError("annotation segments must be a list")

    normalized: list[dict[str, Any]] = []
    previous_end = None
    for item in segments:
        if not isinstance(item, dict):
            raise ValueError("annotation segment must be an object")
        if "start_ms" not in item or "end_ms" not in item:
            raise ValueError("annotation segments must use start_ms/end_ms")
        start_ms = int(round(float(item["start_ms"])))
        end_ms = int(round(float(item["end_ms"])))
        state = str(item.get("state", UNKNOWN_STATE)).upper()
        if state not in ANNOTATION_STATES:
            raise ValueError(f"annotation state must be one of {sorted(ANNOTATION_STATES)}")
        if end_ms <= start_ms:
            raise ValueError("annotation segment end_ms must be greater than start_ms")
        if previous_end is not None and start_ms < previous_end:
            raise ValueError("annotation segments must be sorted and non-overlapping")
        previous_end = end_ms
        normalized.append({
            "start_ms": start_ms,
            "end_ms": end_ms,
            "state": state,
        })
    return _merge_adjacent(normalized)


def _window_from_summary(summary: dict[str, Any], points: list[dict[str, Any]]) -> tuple[int, int]:
    start_s = summary.get("start_seconds")
    end_s = summary.get("end_seconds")
    if start_s is not None and end_s is not None:
        start_ms = int(round(float(start_s) * 1000.0))
        end_ms = int(round(float(end_s) * 1000.0))
        return start_ms, max(start_ms, end_ms)
    if points:
        start_ms = min(int(point["video_time_ms"]) for point in points)
        end_ms = max(int(point["video_time_ms"]) for point in points)
        return start_ms, max(start_ms, end_ms)
    return 0, 0


def _annotation_window(
    annotations: list[dict[str, Any]],
    summary: dict[str, Any],
    points: list[dict[str, Any]],
) -> tuple[int, int]:
    if annotations:
        return (
            min(int(segment["start_ms"]) for segment in annotations),
            max(int(segment["end_ms"]) for segment in annotations),
        )
    return _window_from_summary(summary, points)


def _annotation_gaps(annotations: list[dict[str, Any]], start_ms: int, end_ms: int) -> list[dict[str, int]]:
    gaps: list[dict[str, int]] = []
    cursor = start_ms
    for segment in annotations:
        segment_start = max(start_ms, int(segment["start_ms"]))
        segment_end = min(end_ms, int(segment["end_ms"]))
        if segment_end <= start_ms or segment_start >= end_ms:
            continue
        if segment_start > cursor:
            gaps.append(_interval(cursor, segment_start))
        cursor = max(cursor, segment_end)
    if cursor < end_ms:
        gaps.append(_interval(cursor, end_ms))
    return gaps


def _segments_with_state(segments: list[dict[str, Any]], states: set[str]) -> list[dict[str, Any]]:
    return [dict(segment) for segment in segments if segment["state"] in states and _duration(segment) > 0]


def _intervals_from_segments(segments: list[dict[str, Any]]) -> list[dict[str, int]]:
    return [_interval(int(segment["start_ms"]), int(segment["end_ms"])) for segment in segments if _duration(segment) > 0]


def _clip_segments_to_intervals(
    segments: list[dict[str, Any]],
    intervals: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    clipped: list[dict[str, Any]] = []
    for segment in segments:
        for interval in intervals:
            start_ms = max(int(segment["start_ms"]), int(interval["start_ms"]))
            end_ms = min(int(segment["end_ms"]), int(interval["end_ms"]))
            if end_ms > start_ms:
                clipped.append({
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                    "state": segment["state"],
                })
    return _merge_adjacent(clipped)


def _total_duration(intervals: list[dict[str, Any]]) -> int:
    return sum(max(0, int(interval["end_ms"]) - int(interval["start_ms"])) for interval in intervals)


def _track_key(record: dict[str, Any]) -> str:
    return str(record.get("track_id"))


def _detector_confidence(record: dict[str, Any]) -> float:
    detector = record.get("detector") if isinstance(record.get("detector"), dict) else {}
    try:
        return float(detector.get("conf", 0.0))
    except (TypeError, ValueError):
        return 0.0


def _track_frame_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    track_frames: dict[str, set[tuple[int, int]]] = defaultdict(set)
    for record in records:
        video_time_s = record.get("video_time_s")
        if video_time_s is None:
            continue
        video_time_ms = int(round(float(video_time_s) * 1000.0))
        frame_idx = int(record.get("frame_idx", 0))
        frame_key = (frame_idx, video_time_ms)
        track_id = _track_key(record)
        track_frames[track_id].add(frame_key)
    return {track_id: len(frames) for track_id, frames in track_frames.items()}


def _dominant_track_id(track_ids: set[str], track_frame_counts: dict[str, int]) -> str:
    return sorted(
        track_ids,
        key=lambda track_id: (-track_frame_counts.get(track_id, 0), track_id),
    )[0]


def _dedupe_records_by_frame(
    records: list[dict[str, Any]],
    warnings: list[str] | None,
    track_frame_counts: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    passthrough_records: list[dict[str, Any]] = []
    records_by_frame: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    duplicate_count = 0
    multi_target_frame_count = 0
    for record in records:
        video_time_s = record.get("video_time_s")
        if video_time_s is None:
            passthrough_records.append(record)
            continue
        video_time_ms = int(round(float(video_time_s) * 1000.0))
        frame_idx = int(record.get("frame_idx", 0))
        frame_key = (frame_idx, video_time_ms)
        records_by_frame[frame_key].append(record)

    best_records: list[dict[str, Any]] = []
    counts = track_frame_counts or {}
    for frame_key in sorted(records_by_frame):
        frame_records = records_by_frame[frame_key]
        track_ids = {_track_key(record) for record in frame_records}
        selected_records = frame_records
        if len(track_ids) > 1:
            multi_target_frame_count += 1
            dominant_track_id = _dominant_track_id(track_ids, counts)
            selected_records = [
                record for record in frame_records
                if _track_key(record) == dominant_track_id
            ]
        duplicate_count += max(0, len(frame_records) - 1)
        best_records.append(max(selected_records, key=_detector_confidence))

    if multi_target_frame_count and warnings is not None:
        track_counts = [
            f"{track_id}:{count}"
            for track_id, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        ]
        warnings.append(
            "multiple targets detected; selected per frame by cumulative track frame count "
            f"for {multi_target_frame_count} frame(s), track_counts={track_counts}"
        )
    if duplicate_count and warnings is not None:
        warnings.append(
            "duplicate trace records; "
            f"kept highest detector confidence for {duplicate_count} duplicate record(s)"
        )
    return passthrough_records + best_records


def load_trace_points(track_trace_path: str | Path, warnings: list[str] | None = None) -> list[dict[str, Any]]:
    records = _load_jsonl(track_trace_path)
    track_frame_counts = _track_frame_counts(records)
    records = _dedupe_records_by_frame(records, warnings, track_frame_counts)

    points: list[dict[str, Any]] = []
    seen_frames: set[tuple[int, int]] = set()
    for record in records:
        video_time_s = record.get("video_time_s")
        if video_time_s is None:
            continue
        video_time_ms = int(round(float(video_time_s) * 1000.0))
        frame_idx = int(record.get("frame_idx", 0))
        frame_key = (frame_idx, video_time_ms)
        if frame_key in seen_frames:
            if warnings is not None:
                warnings.append(
                    f"duplicate trace record ignored at frame_idx={frame_idx}, video_time_ms={video_time_ms}"
                )
            continue
        seen_frames.add(frame_key)
        fusion = record.get("fusion") if isinstance(record.get("fusion"), dict) else {}
        state = str(fusion.get("final_state", UNKNOWN_STATE)).upper()
        if state == "MOVING":
            raise ValueError("track trace state MOVING is no longer supported; use ACTIVE")
        if state not in SYSTEM_STATES and state != UNKNOWN_STATE:
            raise ValueError(f"track trace state must be one of {sorted(SYSTEM_STATES | {UNKNOWN_STATE})}")
        points.append({
            "frame_idx": frame_idx,
            "track_id": record.get("track_id"),
            "video_time_ms": video_time_ms,
            "process_time_ms": _round_float(record.get("process_time_ms"), 4) if record.get("process_time_ms") is not None else None,
            "state": state,
            "frame_pipeline_ms": _round_float(record.get("frame_pipeline_ms"), 4),
            "action_ms": _round_float(record.get("action_ms"), 4),
            "target_fps": _round_float(record.get("target_fps"), 4) if record.get("target_fps") is not None else None,
        })
    return sorted(points, key=lambda item: (item["video_time_ms"], item["frame_idx"]))


def _snapshot_has_active_states(snapshot: dict[str, Any]) -> bool:
    active_states = snapshot.get("active_states")
    if isinstance(active_states, dict):
        return bool(active_states)
    if isinstance(active_states, list):
        return bool(active_states)
    return False


def tail_unknown_start_from_results(results_path: str | Path) -> int | None:
    path = Path(results_path)
    if not path.exists():
        return None

    snapshots = [
        snapshot
        for snapshot in _load_jsonl(path)
        if snapshot.get("video_time_s") is not None
    ]
    if not snapshots:
        return None

    snapshots.sort(key=lambda item: float(item["video_time_s"]))
    last_active_index = None
    for index, snapshot in enumerate(snapshots):
        if _snapshot_has_active_states(snapshot):
            last_active_index = index

    if last_active_index is None:
        return None

    for snapshot in snapshots[last_active_index + 1 :]:
        if not _snapshot_has_active_states(snapshot):
            return int(round(float(snapshot["video_time_s"]) * 1000.0))
    return None


def prediction_segments_from_points(
    points: list[dict[str, Any]],
    start_ms: int,
    end_ms: int,
    tail_unknown_start_ms: int | None = None,
) -> list[dict[str, Any]]:
    if end_ms <= start_ms:
        return []
    current_state = UNKNOWN_STATE
    current_track_id = None
    current_start = start_ms
    last_point_time = None

    for point in points:
        if int(point["video_time_ms"]) <= start_ms:
            current_state = point["state"]
            current_track_id = point.get("track_id")
            last_point_time = int(point["video_time_ms"])
        else:
            break

    for point in points:
        point_time = int(point["video_time_ms"])
        if point_time <= start_ms:
            continue
        if point_time >= end_ms:
            break
        next_state = point["state"]
        next_track_id = point.get("track_id")
        track_changed = (
            current_track_id is not None
            and next_track_id is not None
            and next_track_id != current_track_id
        )
        if track_changed:
            old_track_end = last_point_time if last_point_time is not None else current_start
            if current_start < old_track_end:
                yield_segment = {"start_ms": current_start, "end_ms": old_track_end, "state": current_state}
                if _duration(yield_segment) > 0:
                    yield_segment["state"] = yield_segment["state"].upper()
                    yield yield_segment
            unknown_start = max(current_start, old_track_end)
            if unknown_start < point_time:
                yield {"start_ms": unknown_start, "end_ms": point_time, "state": UNKNOWN_STATE}
            current_start = point_time
            current_state = next_state
            current_track_id = next_track_id
            last_point_time = point_time
            continue
        if next_state == current_state:
            current_track_id = next_track_id
            last_point_time = point_time
            continue
        if point_time > current_start:
            yield_segment = {"start_ms": current_start, "end_ms": point_time, "state": current_state}
            current_start = point_time
            current_state = next_state
            current_track_id = next_track_id
            last_point_time = point_time
            if _duration(yield_segment) > 0:
                yield_segment["state"] = yield_segment["state"].upper()
                yield yield_segment
        else:
            current_state = next_state
            current_track_id = next_track_id
            current_start = point_time
            last_point_time = point_time

    tail_unknown_start = None
    if tail_unknown_start_ms is not None and tail_unknown_start_ms < end_ms:
        tail_unknown_start = max(start_ms, int(tail_unknown_start_ms))

    if current_start < end_ms:
        if tail_unknown_start is None or tail_unknown_start <= current_start:
            end_state = UNKNOWN_STATE if tail_unknown_start is not None else current_state
            yield {"start_ms": current_start, "end_ms": end_ms, "state": end_state}
        else:
            yield {"start_ms": current_start, "end_ms": tail_unknown_start, "state": current_state}
            yield {"start_ms": tail_unknown_start, "end_ms": end_ms, "state": UNKNOWN_STATE}


def _states_for_metrics(gt_segments: list[dict[str, Any]], pred_segments: list[dict[str, Any]]) -> list[str]:
    states = {segment["state"] for segment in gt_segments + pred_segments}
    states.discard(UNKNOWN_STATE)
    states.discard(AMBIGUOUS_STATE)
    return sorted(states)


def _segments_for_state(segments: list[dict[str, Any]], state: str) -> list[dict[str, Any]]:
    return [segment for segment in segments if segment["state"] == state and _duration(segment) > 0]


def _total_intersection_ms(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> int:
    total = 0
    for left_segment in left:
        for right_segment in right:
            total += _overlap_ms(left_segment, right_segment)
    return total


def compute_segment_iou(
    gt_segments: list[dict[str, Any]],
    pred_segments: list[dict[str, Any]],
) -> tuple[float | None, dict[str, float]]:
    if not gt_segments:
        return None, {}
    by_class: dict[str, float] = {}
    for state in _states_for_metrics(gt_segments, pred_segments):
        gt_state = _segments_for_state(gt_segments, state)
        pred_state = _segments_for_state(pred_segments, state)
        intersection = _total_intersection_ms(gt_state, pred_state)
        union = sum(_duration(item) for item in gt_state) + sum(_duration(item) for item in pred_state) - intersection
        by_class[state] = intersection / union if union > 0 else 0.0
    macro = sum(by_class.values()) / len(by_class) if by_class else None
    return (_round_float(macro, 4) if macro is not None else None), {state: _round_float(value, 4) for state, value in by_class.items()}


def _gt_events_from_annotations(annotations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    index = 0
    while index < len(annotations):
        segment = annotations[index]
        state = segment["state"]
        if state not in SYSTEM_STATES:
            index += 1
            continue
        event = dict(segment)
        lookahead = index + 1
        while (
            lookahead + 1 < len(annotations)
            and annotations[lookahead]["state"] == AMBIGUOUS_STATE
            and annotations[lookahead + 1]["state"] == state
            and int(annotations[lookahead]["start_ms"]) == int(event["end_ms"])
            and int(annotations[lookahead]["end_ms"]) == int(annotations[lookahead + 1]["start_ms"])
        ):
            event["end_ms"] = annotations[lookahead + 1]["end_ms"]
            lookahead += 2
        events.append(event)
        index = max(index + 1, lookahead)
    return _merge_adjacent(events)


def _event_scores_for_threshold(
    gt_events: list[dict[str, Any]],
    pred_events: list[dict[str, Any]],
    threshold: float,
) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    by_class: dict[str, dict[str, float]] = {}
    per_class_recall: dict[str, float] = {}
    for state in _states_for_metrics(gt_events, pred_events):
        gt_state_events = _segments_for_state(gt_events, state)
        pred_state_events = _segments_for_state(pred_events, state)
        candidates = []
        for gt_index, gt_event in enumerate(gt_state_events):
            for pred_index, pred_event in enumerate(pred_state_events):
                score = _tiou(gt_event, pred_event)
                if score >= threshold:
                    candidates.append((score, gt_index, pred_index))
        candidates.sort(reverse=True)
        matched_gt: set[int] = set()
        matched_pred: set[int] = set()
        for _score, gt_index, pred_index in candidates:
            if gt_index in matched_gt or pred_index in matched_pred:
                continue
            matched_gt.add(gt_index)
            matched_pred.add(pred_index)

        precision = len(matched_pred) / len(pred_state_events) if pred_state_events else 0.0
        recall = len(matched_gt) / len(gt_state_events) if gt_state_events else 0.0
        score_f1 = _f1(precision, recall)
        by_class[state] = {
            "precision": _round_float(precision, 4),
            "recall": _round_float(recall, 4),
            "f1": _round_float(score_f1, 4),
        }
        per_class_recall[state] = _round_float(recall, 4)
    return by_class, per_class_recall


def compute_event_detection(
    gt_events: list[dict[str, Any]],
    pred_events: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, float]]:
    if not gt_events:
        return {"macro": None, "by_class": {}, "by_tiou": {}}, {}
    by_tiou: dict[str, float] = {}
    main_by_class: dict[str, dict[str, float]] = {}
    main_recall: dict[str, float] = {}
    for threshold in EVENT_TIOU_THRESHOLDS:
        by_class, per_class_recall = _event_scores_for_threshold(gt_events, pred_events, threshold)
        macro = sum(item["f1"] for item in by_class.values()) / len(by_class) if by_class else None
        by_tiou[str(threshold)] = _round_float(macro, 4) if macro is not None else None
        if threshold == 0.5:
            main_by_class = by_class
            main_recall = per_class_recall
    return {
        "macro": by_tiou.get("0.5"),
        "by_class": main_by_class,
        "by_tiou": by_tiou,
    }, main_recall


def extract_transitions(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    transitions: list[dict[str, Any]] = []
    for previous, current in zip(segments, segments[1:]):
        from_state = previous["state"]
        to_state = current["state"]
        if from_state == to_state or from_state in {UNKNOWN_STATE, AMBIGUOUS_STATE} or to_state in {UNKNOWN_STATE, AMBIGUOUS_STATE}:
            continue
        transitions.append({
            "time_ms": int(current["start_ms"]),
            "from_state": from_state,
            "to_state": to_state,
        })
    return transitions


def extract_clear_gt_transitions(annotations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    transitions: list[dict[str, Any]] = []
    for previous, current in zip(annotations, annotations[1:]):
        if previous["state"] not in SYSTEM_STATES or current["state"] not in SYSTEM_STATES:
            continue
        if previous["state"] == current["state"]:
            continue
        if int(previous["end_ms"]) != int(current["start_ms"]):
            continue
        transitions.append({
            "time_ms": int(current["start_ms"]),
            "from_state": previous["state"],
            "to_state": current["state"],
        })
    return transitions


def _filter_transitions_outside_intervals(
    transitions: list[dict[str, Any]],
    excluded_intervals: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        transition
        for transition in transitions
        if not _contains_time(excluded_intervals, int(transition["time_ms"]))
    ]


def _match_transitions(
    gt_transitions: list[dict[str, Any]],
    pred_transitions: list[dict[str, Any]],
    tolerance_ms: int,
) -> list[tuple[dict[str, Any], dict[str, Any], int]]:
    candidates = []
    for gt_index, gt_transition in enumerate(gt_transitions):
        for pred_index, pred_transition in enumerate(pred_transitions):
            if gt_transition["from_state"] != pred_transition["from_state"] or gt_transition["to_state"] != pred_transition["to_state"]:
                continue
            offset = abs(int(pred_transition["time_ms"]) - int(gt_transition["time_ms"]))
            if offset <= tolerance_ms:
                candidates.append((offset, gt_index, pred_index))
    candidates.sort()
    matched_gt: set[int] = set()
    matched_pred: set[int] = set()
    matches: list[tuple[dict[str, Any], dict[str, Any], int]] = []
    for offset, gt_index, pred_index in candidates:
        if gt_index in matched_gt or pred_index in matched_pred:
            continue
        matched_gt.add(gt_index)
        matched_pred.add(pred_index)
        matches.append((gt_transitions[gt_index], pred_transitions[pred_index], offset))
    return matches


def compute_transition_metrics(
    gt_transitions: list[dict[str, Any]],
    pred_transitions: list[dict[str, Any]],
    transition_tolerance_ms: int,
    latency_match_window_ms: int,
) -> tuple[dict[str, float | None], float | None, float | None]:
    if not gt_transitions and not pred_transitions:
        return {"f1": None, "precision": None, "recall": None}, None, None
    transition_matches = _match_transitions(gt_transitions, pred_transitions, transition_tolerance_ms)
    precision = len(transition_matches) / len(pred_transitions) if pred_transitions else 0.0
    recall = len(transition_matches) / len(gt_transitions) if gt_transitions else 0.0
    transition_f1 = {
        "f1": _round_float(_f1(precision, recall), 4),
        "precision": _round_float(precision, 4),
        "recall": _round_float(recall, 4),
    }

    boundary_offsets = [float(offset) for _gt, _pred, offset in transition_matches]
    boundary_median = _percentile(boundary_offsets, 50.0)

    latency_matches = _match_transitions(gt_transitions, pred_transitions, latency_match_window_ms)
    latencies = [
        float(pred["time_ms"] - gt["time_ms"])
        for gt, pred, _offset in latency_matches
        if pred["time_ms"] > gt["time_ms"]
    ]
    latency_p90 = _percentile(latencies, 90.0)
    return transition_f1, _round_float(latency_p90, 4) if latency_p90 is not None else None, _round_float(boundary_median, 4) if boundary_median is not None else None


def extract_ambiguous_transitions(
    annotations: list[dict[str, Any]],
    warnings: list[str],
) -> list[dict[str, Any]]:
    ambiguous: list[dict[str, Any]] = []
    for index, segment in enumerate(annotations):
        if segment["state"] != AMBIGUOUS_STATE:
            continue
        previous = annotations[index - 1] if index > 0 else None
        following = annotations[index + 1] if index + 1 < len(annotations) else None
        if not previous or not following:
            continue
        if int(previous["end_ms"]) != int(segment["start_ms"]) or int(segment["end_ms"]) != int(following["start_ms"]):
            continue
        if previous["state"] == following["state"]:
            continue
        if previous["state"] not in MOTION_STATES or following["state"] not in MOTION_STATES:
            warnings.append(
                "ambiguous transition skipped: AMBIGUOUS should only connect ACTIVE/RESTING states"
            )
            continue
        ambiguous.append({
            "ambiguous_start_ms": int(segment["start_ms"]),
            "ambiguous_end_ms": int(segment["end_ms"]),
            "from_state": previous["state"],
            "to_state": following["state"],
        })
    return ambiguous


def compute_ambiguous_transition_metrics(
    ambiguous_transitions: list[dict[str, Any]],
    pred_transitions: list[dict[str, Any]],
    accept_pre_ms: int,
) -> tuple[float | None, float | None]:
    if not ambiguous_transitions:
        return None, None
    hits = 0
    late_commits: list[float] = []
    for transition in ambiguous_transitions:
        matching_predictions = [
            item
            for item in pred_transitions
            if item["from_state"] == transition["from_state"] and item["to_state"] == transition["to_state"]
        ]
        accept_start = int(transition["ambiguous_start_ms"]) - accept_pre_ms
        accept_end = int(transition["ambiguous_end_ms"])
        if any(accept_start <= int(item["time_ms"]) <= accept_end for item in matching_predictions):
            hits += 1
            continue
        late_prediction_times = [
            int(item["time_ms"])
            for item in matching_predictions
            if int(item["time_ms"]) > accept_end
        ]
        if late_prediction_times:
            late_commits.append(float(min(late_prediction_times) - accept_end))
    hit_rate = hits / len(ambiguous_transitions)
    late_p90 = _percentile(late_commits, 90.0)
    return _round_float(hit_rate, 4), _round_float(late_p90, 4) if late_p90 is not None else None


def _compressed_states(segments: list[dict[str, Any]]) -> list[str]:
    states: list[str] = []
    for segment in segments:
        state = segment["state"]
        if state in {UNKNOWN_STATE, AMBIGUOUS_STATE}:
            continue
        if not states or states[-1] != state:
            states.append(state)
    return states


def _levenshtein(left: list[str], right: list[str]) -> int:
    if not left:
        return len(right)
    if not right:
        return len(left)
    previous = list(range(len(right) + 1))
    for left_index, left_value in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_value in enumerate(right, start=1):
            cost = 0 if left_value == right_value else 1
            current.append(min(
                current[-1] + 1,
                previous[right_index] + 1,
                previous[right_index - 1] + cost,
            ))
        previous = current
    return previous[-1]


def compute_segment_edit_score(gt_segments: list[dict[str, Any]], pred_segments: list[dict[str, Any]]) -> float | None:
    if not gt_segments:
        return None
    gt_sequence = _compressed_states(gt_segments)
    pred_sequence = _compressed_states(pred_segments)
    denominator = max(len(gt_sequence), len(pred_sequence))
    if denominator == 0:
        return None
    return _round_float(1.0 - (_levenshtein(pred_sequence, gt_sequence) / denominator), 4)


def _max_transitions_in_window(transitions: list[dict[str, Any]], window_ms: int) -> int:
    times = sorted(int(item["time_ms"]) for item in transitions)
    if not times:
        return 0
    max_count = 0
    right = 0
    for left, start_time in enumerate(times):
        while right < len(times) and times[right] - start_time <= window_ms:
            right += 1
        max_count = max(max_count, right - left)
    return max_count


def compute_oscillation_rate(
    gt_transitions: list[dict[str, Any]],
    pred_transitions: list[dict[str, Any]],
    window_ms: int,
) -> float | None:
    if window_ms <= 0:
        return None
    gt_max = _max_transitions_in_window(gt_transitions, window_ms)
    pred_max = _max_transitions_in_window(pred_transitions, window_ms)
    return _round_float(max(0, pred_max - gt_max) / (window_ms / 1000.0), 4)


def _state_at(segments: list[dict[str, Any]], time_ms: int) -> str:
    for segment in segments:
        if int(segment["start_ms"]) <= time_ms < int(segment["end_ms"]):
            return segment["state"]
    return UNKNOWN_STATE


def compute_confusion_and_false_duration(
    gt_segments: list[dict[str, Any]],
    pred_segments: list[dict[str, Any]],
    limit: int = 5,
) -> tuple[float | None, list[dict[str, Any]]]:
    if not gt_segments:
        return None, []
    boundaries = sorted({
        int(segment["start_ms"])
        for segment in gt_segments + pred_segments
    } | {
        int(segment["end_ms"])
        for segment in gt_segments + pred_segments
    })
    false_runs: list[int] = []
    current_false_start = None
    confusion_durations: dict[tuple[str, str], int] = defaultdict(int)
    for start_ms, end_ms in zip(boundaries, boundaries[1:]):
        if end_ms <= start_ms:
            continue
        gt_state = _state_at(gt_segments, start_ms)
        pred_state = _state_at(pred_segments, start_ms)
        if gt_state == UNKNOWN_STATE:
            if current_false_start is not None:
                false_runs.append(start_ms - current_false_start)
                current_false_start = None
            continue
        mismatch = gt_state != pred_state
        if mismatch:
            confusion_durations[(gt_state, pred_state)] += end_ms - start_ms
            if current_false_start is None:
                current_false_start = start_ms
        elif current_false_start is not None:
            false_runs.append(start_ms - current_false_start)
            current_false_start = None
    if current_false_start is not None and boundaries:
        false_runs.append(boundaries[-1] - current_false_start)

    false_p90 = _percentile([float(item) for item in false_runs], 90.0)
    top_pairs = [
        {"gt": gt, "pred": pred, "duration_ms": duration}
        for (gt, pred), duration in sorted(confusion_durations.items(), key=lambda item: item[1], reverse=True)[:limit]
    ]
    return _round_float(false_p90, 4) if false_p90 is not None else None, top_pairs


def compute_sustained_fps(points: list[dict[str, Any]], warnings: list[str]) -> tuple[float | None, float | None, int]:
    process_times = sorted({
        int(round(float(point["process_time_ms"])))
        for point in points
        if point.get("process_time_ms") is not None
    })
    if not process_times:
        warnings.append("sustained_fps_ratio unavailable: process_time_ms is missing")
        return None, None, 0
    observed_duration_ms = max(process_times)
    sustained_window_ms = min(60000, observed_duration_ms)
    if observed_duration_ms < 5000:
        warnings.append("sustained_fps_ratio unavailable: observed_process_duration_ms < 5000")
        return None, None, sustained_window_ms

    target_fps = next((float(point["target_fps"]) for point in points if point.get("target_fps")), None)
    if not target_fps or target_fps <= 0:
        warnings.append("sustained_fps_ratio unavailable: target_fps is missing")
        return None, None, sustained_window_ms

    window_starts = [0]
    if observed_duration_ms > sustained_window_ms:
        window_starts.append(observed_duration_ms - sustained_window_ms)
        window_starts.extend(time for time in process_times if time <= observed_duration_ms - sustained_window_ms)
    min_count = None
    right = 0
    for start_ms in sorted(set(window_starts)):
        while right < len(process_times) and process_times[right] < start_ms:
            right += 1
        end_ms = start_ms + sustained_window_ms
        scan = right
        while scan < len(process_times) and process_times[scan] <= end_ms:
            scan += 1
        count = scan - right
        min_count = count if min_count is None else min(min_count, count)
    min_count = min_count if min_count is not None else 0
    min_window_fps = min_count / (sustained_window_ms / 1000.0)
    return _round_float(min_window_fps / target_fps, 4), _round_float(min_window_fps, 4), sustained_window_ms


def _compute_params(eval_duration_ms: int, sustained_window_ms: int) -> dict[str, Any]:
    transition_tolerance_ms = int(min(3000, max(1000, 0.2 * eval_duration_ms)))
    return {
        "time_unit": "ms",
        "event_tiou_thresholds": list(EVENT_TIOU_THRESHOLDS),
        "transition_tolerance_ms": transition_tolerance_ms,
        "latency_match_window_ms": int(min(3000, max(2000, 0.3 * eval_duration_ms))),
        "ambiguous_accept_pre_ms": transition_tolerance_ms,
        "sustained_window_ms": int(sustained_window_ms),
        "oscillation_window_ms": int(min(10000, max(5000, 0.25 * eval_duration_ms))),
    }


def _null_quality_metrics() -> tuple[float | None, dict[str, float], dict[str, Any], dict[str, float], dict[str, float | None], float | None, float | None]:
    return (
        None,
        {},
        {"macro": None, "by_class": {}, "by_tiou": {}},
        {},
        {"f1": None, "precision": None, "recall": None},
        None,
        None,
    )


def _timeline_gt_segments(
    annotations: list[dict[str, Any]],
    start_ms: int,
    end_ms: int,
) -> list[dict[str, Any]]:
    if end_ms <= start_ms:
        return []
    segments: list[dict[str, Any]] = []
    cursor = start_ms
    for annotation in annotations:
        segment_start = max(start_ms, int(annotation["start_ms"]))
        segment_end = min(end_ms, int(annotation["end_ms"]))
        if segment_end <= start_ms or segment_start >= end_ms:
            continue
        if segment_start > cursor:
            segments.append({"start_ms": cursor, "end_ms": segment_start, "state": "GAP"})
        if segment_end > segment_start:
            segments.append({
                "start_ms": segment_start,
                "end_ms": segment_end,
                "state": annotation["state"],
            })
        cursor = max(cursor, segment_end)
    if cursor < end_ms:
        segments.append({"start_ms": cursor, "end_ms": end_ms, "state": "GAP"})
    return _merge_adjacent(segments)


def _draw_timeline_band(ax, segments: list[dict[str, Any]], y_start: float):
    for segment in segments:
        state = str(segment["state"]).upper()
        start_s = int(segment["start_ms"]) / 1000.0
        width_s = max(0.001, _duration(segment) / 1000.0)
        kwargs = {
            "facecolors": TIMELINE_STATE_COLORS.get(state, TIMELINE_STATE_COLORS[UNKNOWN_STATE]),
            "edgecolors": "#ffffff",
            "linewidth": 0.6,
        }
        if state == "GAP":
            kwargs["hatch"] = "///"
            kwargs["edgecolors"] = "#bdbdbd"
        ax.broken_barh([(start_s, width_s)], (y_start, 0.72), **kwargs)


def _timeline_legend(states: set[str]) -> list[Patch]:
    handles: list[Patch] = []
    for state in TIMELINE_STATE_ORDER:
        if state not in states:
            continue
        kwargs = {
            "facecolor": TIMELINE_STATE_COLORS.get(state, TIMELINE_STATE_COLORS[UNKNOWN_STATE]),
            "edgecolor": "#bdbdbd" if state == "GAP" else "#ffffff",
            "label": state,
        }
        if state == "GAP":
            kwargs["hatch"] = "///"
        handles.append(Patch(**kwargs))
    return handles


def render_behavior_timeline_plot(
    *,
    trace_path: str | Path,
    summary_path: str | Path,
    annotations_path: str | Path,
    output_path: str | Path,
    results_path: str | Path | None = None,
) -> str:
    summary = _load_json(summary_path)
    annotations = load_annotations(annotations_path)
    warnings: list[str] = []
    points = load_trace_points(trace_path, warnings)
    eval_start_ms, eval_end_ms = _annotation_window(annotations, summary if isinstance(summary, dict) else {}, points)
    gt_segments = _timeline_gt_segments(annotations, eval_start_ms, eval_end_ms)
    tail_unknown_start_ms = tail_unknown_start_from_results(results_path) if results_path is not None else None
    pred_segments = _merge_adjacent(list(prediction_segments_from_points(
        points,
        eval_start_ms,
        eval_end_ms,
        tail_unknown_start_ms=tail_unknown_start_ms,
    )))

    states_present = {segment["state"] for segment in gt_segments + pred_segments}
    duration_s = max(0.001, (eval_end_ms - eval_start_ms) / 1000.0)
    figure_width = min(18.0, max(10.0, duration_s / 5.0))
    figure, ax = plt.subplots(figsize=(figure_width, 3.2), constrained_layout=True)

    _draw_timeline_band(ax, gt_segments, 1.15)
    _draw_timeline_band(ax, pred_segments, 0.15)

    ax.set_ylim(0.0, 2.05)
    ax.set_yticks([1.51, 0.51])
    ax.set_yticklabels(["GT", "System"])
    ax.set_xlim(eval_start_ms / 1000.0, eval_end_ms / 1000.0)
    ax.set_xlabel("Time (s)")
    ax.set_title("GT vs System Behavior Timeline")
    ax.grid(axis="x", color="#d0d0d0", linestyle="--", linewidth=0.7, alpha=0.8)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=12))

    handles = _timeline_legend(states_present)
    if handles:
        ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=min(7, len(handles)), fontsize=8)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)
    return str(output)


def evaluate_behavior_metrics(
    *,
    trace_path: str | Path,
    summary_path: str | Path,
    annotations_path: str | Path,
    results_path: str | Path | None = None,
) -> dict[str, Any]:
    summary = _load_json(summary_path)
    annotations = load_annotations(annotations_path)
    warnings: list[str] = []
    points = load_trace_points(trace_path, warnings)
    eval_start_ms, eval_end_ms = _annotation_window(annotations, summary if isinstance(summary, dict) else {}, points)
    eval_duration_ms = max(0, eval_end_ms - eval_start_ms)

    tail_unknown_start_ms = tail_unknown_start_from_results(results_path) if results_path is not None else None
    raw_pred_segments = _merge_adjacent(list(prediction_segments_from_points(
        points,
        eval_start_ms,
        eval_end_ms,
        tail_unknown_start_ms=tail_unknown_start_ms,
    )))
    clear_gt_segments = _segments_with_state(annotations, SYSTEM_STATES)
    ambiguous_segments = _segments_with_state(annotations, {AMBIGUOUS_STATE})
    coverage_intervals = _intervals_from_segments(annotations)
    clear_intervals = _intervals_from_segments(clear_gt_segments)
    gap_intervals = _annotation_gaps(annotations, eval_start_ms, eval_end_ms)
    excluded_transition_intervals = gap_intervals + _intervals_from_segments(ambiguous_segments)

    sustained_fps_ratio, min_sliding_window_fps, sustained_window_ms = compute_sustained_fps(points, warnings)
    params = _compute_params(eval_duration_ms, sustained_window_ms)

    pred_clear_segments = _clip_segments_to_intervals(raw_pred_segments, clear_intervals)
    pred_event_segments = _clip_segments_to_intervals(raw_pred_segments, coverage_intervals)
    gt_event_segments = _gt_events_from_annotations(annotations)

    raw_pred_transitions = extract_transitions(raw_pred_segments)
    pred_clear_transitions = _filter_transitions_outside_intervals(raw_pred_transitions, excluded_transition_intervals)
    pred_oscillation_transitions = _filter_transitions_outside_intervals(raw_pred_transitions, gap_intervals)
    gt_clear_transitions = extract_clear_gt_transitions(annotations)
    ambiguous_transitions = extract_ambiguous_transitions(annotations, warnings)

    has_clear_gt = bool(clear_gt_segments)
    if not has_clear_gt:
        warnings.append("quality metrics unavailable: no clear annotated intervals")
        (
            segment_iou_macro,
            segment_iou_by_class,
            event_detection_f1,
            per_class_event_recall,
            transition_f1,
            latency_p90_ms,
            boundary_offset_median_ms,
        ) = _null_quality_metrics()
        false_state_duration_p90_ms = None
        top_confusion_pairs: list[dict[str, Any]] = []
        segment_edit_score = None
    else:
        segment_iou_macro, segment_iou_by_class = compute_segment_iou(clear_gt_segments, pred_clear_segments)
        event_detection_f1, per_class_event_recall = compute_event_detection(gt_event_segments, pred_event_segments)
        transition_f1, latency_p90_ms, boundary_offset_median_ms = compute_transition_metrics(
            gt_clear_transitions,
            pred_clear_transitions,
            int(params["transition_tolerance_ms"]),
            int(params["latency_match_window_ms"]),
        )
        false_state_duration_p90_ms, top_confusion_pairs = compute_confusion_and_false_duration(clear_gt_segments, pred_clear_segments)
        segment_edit_score = compute_segment_edit_score(clear_gt_segments, pred_clear_segments)

    ambiguous_hit_rate, late_commit_p90_ms = compute_ambiguous_transition_metrics(
        ambiguous_transitions,
        raw_pred_transitions,
        int(params["ambiguous_accept_pre_ms"]),
    )
    oscillation_rate = (
        compute_oscillation_rate(gt_clear_transitions, pred_oscillation_transitions, int(params["oscillation_window_ms"]))
        if coverage_intervals
        else None
    )

    p95_pipeline = _percentile([float(point["frame_pipeline_ms"]) for point in points], 95.0)
    p95_action = _percentile([float(point["action_ms"]) for point in points], 95.0)

    inputs = {
        "trace_path": str(trace_path),
        "summary_path": str(summary_path),
        "annotations_path": str(annotations_path),
    }
    if results_path is not None:
        inputs["results_path"] = str(results_path)

    return {
        "schema": "behavior_metrics_v3",
        "inputs": inputs,
        "eval_window": {
            "start_ms": eval_start_ms,
            "end_ms": eval_end_ms,
            "duration_ms": eval_duration_ms,
        },
        "params": params,
        "p0_quality": {
            "event_detection_f1": event_detection_f1,
            "segment_iou_macro": segment_iou_macro,
            "segment_iou_by_class": segment_iou_by_class,
            "detection_latency_p90_ms": latency_p90_ms,
            "transition_f1": transition_f1,
            "ambiguous_transition_hit_rate": ambiguous_hit_rate,
        },
        "p0_perf": {
            "p95_pipeline_ms": _round_float(p95_pipeline, 4) if p95_pipeline is not None else None,
            "p95_action_ms": _round_float(p95_action, 4) if p95_action is not None else None,
            "sustained_fps_ratio": sustained_fps_ratio,
            "min_sliding_window_fps": min_sliding_window_fps,
        },
        "diagnostics": {
            "segment_edit_score": segment_edit_score,
            "oscillation_rate": oscillation_rate,
            "boundary_offset_median_ms": boundary_offset_median_ms,
            "per_class_event_recall": per_class_event_recall,
            "false_state_duration_p90_ms": false_state_duration_p90_ms,
            "top_confusion_pairs": top_confusion_pairs,
            "late_commit_p90_ms": late_commit_p90_ms,
        },
        "warnings": warnings,
    }


def write_behavior_metrics(
    *,
    trace_path: str | Path,
    summary_path: str | Path,
    annotations_path: str | Path,
    output_path: str | Path,
    timeline_plot_path: str | Path | None = None,
    results_path: str | Path | None = None,
) -> dict[str, Any]:
    metrics = evaluate_behavior_metrics(
        trace_path=trace_path,
        summary_path=summary_path,
        annotations_path=annotations_path,
        results_path=results_path,
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    plot_path = Path(timeline_plot_path) if timeline_plot_path is not None else output.parent / TIMELINE_PLOT_NAME
    render_behavior_timeline_plot(
        trace_path=trace_path,
        summary_path=summary_path,
        annotations_path=annotations_path,
        output_path=plot_path,
        results_path=results_path,
    )
    metrics["artifacts"] = {
        "timeline_plot": str(plot_path),
    }
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2, sort_keys=True)
    return metrics


def write_behavior_metrics_from_replay_output(
    *,
    system_output_dir: str | Path,
    annotations_path: str | Path,
    metrics_output_dir: str | Path,
    output_name: str | None = None,
    timeline_plot_name: str | None = None,
) -> dict[str, Any]:
    system_output = Path(system_output_dir)
    trace_path = system_output / "track_trace.jsonl"
    summary_path = system_output / "summary.json"
    results_path = system_output / "results.jsonl"
    missing = [str(path) for path in (trace_path, summary_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"system output is missing required file(s): {', '.join(missing)}")

    output_dir = Path(metrics_output_dir)
    resolved_output_name = output_name or default_metrics_output_name(annotations_path)
    resolved_timeline_plot_name = timeline_plot_name or default_timeline_plot_name(annotations_path)
    output_path = output_dir / resolved_output_name
    timeline_plot_path = output_dir / resolved_timeline_plot_name
    return write_behavior_metrics(
        trace_path=trace_path,
        summary_path=summary_path,
        annotations_path=annotations_path,
        output_path=output_path,
        timeline_plot_path=timeline_plot_path,
        results_path=results_path if results_path.exists() else None,
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Compute behavior metrics from offline replay track trace output.")
    parser.add_argument(
        "--system-output-dir",
        required=True,
        help="Offline replay output directory containing track_trace.jsonl and summary.json.",
    )
    parser.add_argument("--annotations", required=True, help="Annotation JSON path.")
    parser.add_argument(
        "--metrics-output-dir",
        required=True,
        help="Directory where annotation-named metrics outputs will be written.",
    )
    parser.add_argument("--output-name", default=None, help="Override the default <annotation>.metrics.json filename.")
    parser.add_argument("--timeline-plot-name", default=None, help="Override the default <annotation>.gt_system_timeline.png filename.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    write_behavior_metrics_from_replay_output(
        system_output_dir=args.system_output_dir,
        annotations_path=args.annotations,
        metrics_output_dir=args.metrics_output_dir,
        output_name=args.output_name,
        timeline_plot_name=args.timeline_plot_name,
    )


if __name__ == "__main__":
    main()
