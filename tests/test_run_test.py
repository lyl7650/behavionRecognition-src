import json
from pathlib import Path

import cv2
import numpy as np
from scripts.pipeline import run_test
from utils.metrics import compute_time_agreement, compute_segment_iou, compute_event_detection, compute_transition_metrics
import torch


def _write_annotation(path: Path, video_name: str):
    path.write_text(json.dumps({"schema": "behavior_eval_annotations_v1", "video_path": video_name,
                                "time_unit": "ms", "segments": [{"start_ms": 0, "end_ms": 1000, "state": "ACTIVE"}]}),
                    encoding="utf-8")


def test_find_rabbit_pairs_matches_video_path_and_ignores_redundant_json(tmp_path):
    videos = tmp_path / "videos"
    annotations = tmp_path / "annotations"
    videos.mkdir()
    annotations.mkdir()
    (videos / "matched.mp4").touch()
    _write_annotation(annotations / "matched.annotations.json", "matched.mp4")
    _write_annotation(annotations / "extra.annotations.json", "not_in_test_set.mp4")

    assert run_test.find_rabbit_pairs(videos, annotations) == [(videos / "matched.mp4", annotations / "matched.annotations.json")]


def test_pet_states_conversion_keeps_all_four_metrics_states(tmp_path):
    states = tmp_path / "sample_states.json"
    states.write_text(json.dumps({"video": str(tmp_path / "sample.mp4"), "frames": [
        {"timestamp_sec": 0.0, "state": "RESTING"}, {"timestamp_sec": 0.5, "state": "EATING"},
        {"timestamp_sec": 1.0, "state": "DRINKING"}, {"timestamp_sec": 1.5, "state": "ACTIVE"}]}), encoding="utf-8")

    annotation = run_test.pet_annotation_from_states(states, tmp_path / "converted.annotations.json")
    payload = json.loads(annotation.read_text(encoding="utf-8"))
    assert [segment["state"] for segment in payload["segments"]] == ["RESTING", "EATING", "DRINKING", "ACTIVE"]


def test_pet_evaluation_uses_four_class_order():
    assert run_test.SPECIES["pet"] == ["eat", "drink", "active", "resting"]


def test_metric_row_uses_event_f1_at_tiou_half():
    metric = {"p0_quality": {"segment_iou_macro": 0.2,
              "time_agreement": {"accuracy": 0.8, "unknown_rate": 0.0, "gt_ms": 1000},
              "segment_iou_by_class": {"EATING": 0.8},
              "transition_f1": {"f1": 0.3, "precision": 0.4, "recall": 0.25, "tp": 1, "fp": 1, "fn": 1},
              "event_detection_f1": {"macro": 0.4, "by_tiou": {"0.3": 0.5, "0.7": 0.2},
                                     "by_class": {"EATING": {"tp": 1, "fp": 2, "fn": 0}}}},
              "artifacts": {"timeline_plot": "x.png"}}
    row = run_test._metric_row(Path("video.mp4"), metric)
    assert row["event_f1_tiou_0.5"] == 0.4
    assert row["transition_f1"] == 0.3
    assert row["time_accuracy"] == 0.8
    assert row["segment_iou_macro"] == 0.2
    assert (row["event_tp"], row["event_fp"], row["event_fn"]) == (1, 2, 0)


def test_time_accuracy_and_event_counts_preserve_fragmentation_penalty():
    gt = [{"start_ms": 0, "end_ms": 1000, "state": "EATING"}]
    pred = [{"start_ms": 0, "end_ms": 200, "state": "ACTIVE"},
            {"start_ms": 200, "end_ms": 700, "state": "EATING"},
            {"start_ms": 700, "end_ms": 800, "state": "UNKNOWN"},
            {"start_ms": 800, "end_ms": 1000, "state": "EATING"}]
    agreement = compute_time_agreement(gt, pred)
    macro, _ = compute_segment_iou(gt, pred)
    events, _ = compute_event_detection(gt, pred)
    assert (agreement["gt_ms"], agreement["correct_ms"], agreement["unknown_ms"]) == (1000, 700, 100)
    assert agreement["accuracy"] == 0.7
    assert macro == 0.35
    assert compute_segment_iou(gt, pred, gt_classes_only=True)[0] == 0.7
    assert events["by_class"]["EATING"]["tp"] == 1
    assert events["by_class"]["EATING"]["fp"] == 1
    assert events["by_class"]["ACTIVE"]["fp"] == 1


def test_pooled_aggregation_uses_durations_and_counts():
    def metric(gt_ms, correct_ms, state, fp):
        return {"p0_quality": {
            "time_agreement": {"gt_ms": gt_ms, "correct_ms": correct_ms, "unknown_ms": 0,
                               "by_class": {state: {"gt_ms": gt_ms, "pred_ms": correct_ms, "intersection_ms": correct_ms}}},
            "event_detection_f1": {"by_class": {state: {"tp": 1, "fp": fp, "fn": 0}}},
            "transition_f1": {"tp": 0, "fp": fp, "fn": 0},
        }}
    agg = run_test._aggregate_quality(
        [{"time_iou_macro": 0.5}, {"time_iou_macro": 0.9}],
        [metric(1000, 500, "EATING", 1), metric(9000, 8100, "EATING", 0)],
    )
    assert agg["time_accuracy"] == 0.86
    assert agg["time_iou_macro"] == 0.86
    assert agg["segment_iou_macro"] == 0.86
    assert agg["time_iou_by_class"] == {"EATING": 0.86}
    assert agg["pooled_time_iou_macro"] == 0.86
    assert agg["transition_counts"] == {"tp": 0, "fp": 1, "fn": 0}
    assert agg["pooled_transition_f1"] == 0.0


def test_pooled_iou_counts_false_predictions_from_videos_without_that_gt_class():
    def metric(state, predicted_counts):
        return {"p0_quality": {
            "time_agreement": {"gt_ms": 1000, "correct_ms": 500, "unknown_ms": 0,
                               "by_class": predicted_counts},
            "event_detection_f1": {"by_class": {}},
            "transition_f1": {"tp": 0, "fp": 0, "fn": 0},
        }}
    eating = metric("EATING", {
        "EATING": {"gt_ms": 1000, "pred_ms": 500, "intersection_ms": 500},
        "ACTIVE": {"gt_ms": 0, "pred_ms": 500, "intersection_ms": 0},
    })
    active = metric("ACTIVE", {
        "ACTIVE": {"gt_ms": 1000, "pred_ms": 1000, "intersection_ms": 1000},
    })
    agg = run_test._aggregate_quality([{}, {}], [eating, active])
    assert agg["time_iou_by_class"] == {"ACTIVE": 0.6667, "EATING": 0.5}
    assert agg["segment_iou_macro"] == 0.5834


def test_no_transition_has_undefined_f1_and_zero_counts():
    result, _, _ = compute_transition_metrics([], [], 3000, 3000)
    assert result == {"f1": None, "precision": None, "recall": None, "tp": 0, "fp": 0, "fn": 0}


def test_mean_handles_transition_f1_and_legacy_dict():
    assert run_test._mean([{"transition_f1": 0.2}, {"transition_f1": 0.4}], "transition_f1") == 0.3
    assert run_test._mean([{"transition_f1": {"f1": 0.5}}], "transition_f1") == 0.5


def test_default_pet_evaluation_root_exists_and_finds_available_annotations(tmp_path):
    assert run_test.PET_VIDEOS.is_dir()
    assert run_test.find_pet_pairs(converted_dir=tmp_path / "annotations")


def test_student_inference_uses_forward_seq_not_single_frame_forward(tmp_path, monkeypatch):
    class FakeStudent:
        def forward(self, _batch):
            raise AssertionError("single-frame forward must not be used for clip inference")

        def forward_seq(self, batch):
            assert batch.ndim == 5
            return torch.tensor([[0.0, 2.0, 0.0, 0.0]])

    monkeypatch.setattr(
        run_test,
        "_video_frames",
        lambda *_args, **_kwargs: iter([(0, 0.0, torch.zeros(16, 3, 2, 2), None)]),
    )
    trace = tmp_path / "track_trace.jsonl"
    summary = tmp_path / "summary.json"
    detect_cfg = {"yolo_model": "fake.pt", "expand_ratio": 1.25, "target_classes": [0], "max_det": 1}
    run_test.infer_video(
        FakeStudent(),
        "student",
        tmp_path / "video.mp4",
        ["eat", "drink", "active", "resting"],
        trace,
        summary,
        0.5,
        16,
        2,
        torch.device("cpu"),
        detector=object(),
        detect_cfg=detect_cfg,
    )
    assert json.loads(trace.read_text(encoding="utf-8"))["fusion"]["final_state"] == "DRINK"


def test_detector_config_points_to_expected_weights():
    rabbit = run_test.DETECTOR_CONFIG["rabbit"]
    pet = run_test.DETECTOR_CONFIG["pet"]
    assert rabbit["yolo_model"].name == "rabbit-yolo11n-0603.pt"
    assert pet["yolo_model"].name == "yolo11n-ssb-0710.pt"
    assert rabbit["target_classes"] == [0]
    assert pet["target_classes"] == [1, 2]
    assert run_test.EXPAND_RATIO == 1.25


def test_video_frames_applies_expanded_crop(monkeypatch, tmp_path):
    """Crop path should call build_smoothed_crop_boxes with expand_ratio and crop the animal."""
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    frame[20:60, 40:100] = 255

    class FakeCapture:
        def isOpened(self):
            return True

        def get(self, prop):
            mapping = {
                cv2.CAP_PROP_FPS: 10.0,
                cv2.CAP_PROP_FRAME_COUNT: 16,
                cv2.CAP_PROP_FRAME_WIDTH: 200,
                cv2.CAP_PROP_FRAME_HEIGHT: 100,
            }
            return mapping.get(prop, 0)

        def set(self, *_args):
            return True

        def read(self):
            return True, frame.copy()

        def release(self):
            return None

    monkeypatch.setattr(run_test.cv2, "VideoCapture", lambda *_a, **_k: FakeCapture())

    class FakeDetector:
        def predict(self, _frame, **_kwargs):
            class BoxItem:
                def __init__(self):
                    self.xyxy = torch.tensor([[40.0, 20.0, 100.0, 60.0]])

            class Boxes:
                def __len__(self):
                    return 1

                def __getitem__(self, index):
                    assert index == 0
                    return BoxItem()

            class Result:
                boxes = Boxes()

            return [Result()]

    seen = {}

    def fake_build(boxes, frame_width, frame_height, fallback_size, expand_ratio, smooth_alpha=0.35):
        seen["expand_ratio"] = expand_ratio
        seen["boxes"] = boxes
        return [(20, 5, 120, 75)] * len(boxes)

    monkeypatch.setattr(run_test, "build_smoothed_crop_boxes", fake_build)

    detect_cfg = {
        "target_classes": [0],
        "max_det": 1,
        "expand_ratio": 1.25,
        "device": "cpu",
        "yolo_model": "fake.pt",
    }
    clips = list(
        run_test._video_frames(
            tmp_path / "v.mp4",
            sample_every_s=1.0,
            temporal_window=4,
            img_size=32,
            detector=FakeDetector(),
            detect_cfg=detect_cfg,
        )
    )
    assert len(clips) >= 1
    assert seen["expand_ratio"] == 1.25
    assert clips[0][2].shape == (4, 3, 32, 32)

def test_dominant_gt_state_picks_longest_total_duration(tmp_path):
    annotation = tmp_path / "a.annotations.json"
    annotation.write_text(json.dumps({"segments": [
        {"start_ms": 0, "end_ms": 1000, "state": "ACTIVE"},
        {"start_ms": 1000, "end_ms": 5000, "state": "RESTING"},
        {"start_ms": 5000, "end_ms": 6000, "state": "ACTIVE"}]}), encoding="utf-8")
    assert run_test._dominant_gt_state(annotation) == "resting"


def test_write_evaluation_report_has_summary_and_one_row_per_video(tmp_path):
    plot = tmp_path / "video_a" / "gt_system_timeline.png"
    plot.parent.mkdir()
    plot.touch()
    summary = {
        "species": "pet", "model_kind": "student", "checkpoint": "ckpt.pth",
        "detector": "yolo.pt", "expand_ratio": 1.25, "annotated_videos": 1,
        "sample_every_seconds": 0.5,
        "aggregate": {"segment_iou_macro": 0.5, "time_iou_macro": 0.5,
                      "time_iou_by_class": {"ACTIVE": 0.25, "RESTING": 0.75}, "event_f1_tiou_0.3": 0.6,
                      "event_f1_tiou_0.5": 0.4, "event_f1_tiou_0.7": 0.2, "transition_f1": None},
        "videos": [{"video": "video_a.mp4", "gt_label": "resting", "eval_seconds": 59.9,
                    "segment_iou_macro": 0.5, "time_iou_by_class": {"RESTING": 0.5},
                    "time_iou_macro": 0.5, "event_f1_tiou_0.3": 0.6, "event_f1_tiou_0.5": 0.4,
                    "event_f1_tiou_0.7": 0.2, "transition_f1": 0.1, "timeline_plot": str(plot)}],
    }
    report = run_test.write_evaluation_report(tmp_path, summary)
    content = report.read_text(encoding="utf-8")
    assert report.name == "evaluation_report.html"
    assert "video_a.mp4" in content
    assert "<img src='video_a/gt_system_timeline.png'" in content
    assert "resting" in content
    assert content.count("Timeline Comparison") == 1
    assert content.count("segment_iou_macro (Time IoU)") == 3
    assert "<th>ACTIVE Time IoU</th><td>0.2500</td>" in content
    assert "<th>RESTING Time IoU</th><td>0.7500</td>" in content
    assert "<td>0.5000</td><td>-</td><td>0.5000</td>" in content
    assert "<tr><td>ACTIVE Time IoU</td><td>-</td></tr>" in content
    assert "Event F1" not in content
    assert "Time accuracy" not in content
    assert "Transition F1" not in content


def test_interactive_args_uses_numbered_choices():
    answers = iter(["2", "3", r"E:\ckpt.pth"])
    argv = run_test._interactive_args(input_fn=lambda _p: next(answers))
    assert argv == ["--species", "rabbit", "--model-kind", "student", "--checkpoint", r"E:\ckpt.pth"]


def test_interactive_rabbit_fusion_uses_dashboard_default():
    answers = iter(["2", "1"])
    argv = run_test._interactive_args(input_fn=lambda _p: next(answers))
    assert argv == ["--species", "rabbit", "--model-kind", "fusion", "--checkpoint", "dashboard_default"]

