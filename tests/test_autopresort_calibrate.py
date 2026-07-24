import importlib


def test_boundary_purity():
    cal = importlib.import_module("auto_presort.calibrate")
    gt = [(0.0, 10.0, "eat"), (10.0, 15.0, "active")]
    assert cal.boundary_purity([(2.0, 8.0, "eat")], gt) == 1.0
    assert abs(cal.boundary_purity([(6.0, 12.0, "eat")], gt) - (4.0 / 6.0)) < 1e-6


def test_recall_unions_overlapping_predictions():
    cal = importlib.import_module("auto_presort.calibrate")
    gt = [(0.0, 10.0, "eat")]
    assert abs(cal.recall([(0.0, 8.0, "eat"), (2.0, 10.0, "eat")], gt) - 1.0) < 1e-6


def test_short_action_loss_requires_min_coverage():
    cal = importlib.import_module("auto_presort.calibrate")
    gt = [(0.0, 10.0, "eat"), (10.0, 12.0, "drink")]
    assert cal.short_action_loss(
        [(2.0, 8.0, "eat"), (10.0, 10.1, "drink")], gt,
        min_gt_dur=3.0, min_cover=0.5) == ["drink"]
    assert cal.short_action_loss(
        [(10.0, 11.5, "drink")], gt, min_gt_dur=3.0, min_cover=0.5) == []


def test_summarize_gate_requires_purity_and_recall():
    cal = importlib.import_module("auto_presort.calibrate")
    rows = [{"purity": 0.98, "recall": 0.30, "short_lost": []}]
    s = cal.summarize(rows, purity_gate=0.95, recall_gate=0.5)
    assert s["gate_pass"] is False
    rows2 = [{"purity": 0.98, "recall": 0.70, "short_lost": []}]
    assert cal.summarize(rows2, purity_gate=0.95, recall_gate=0.5)["gate_pass"] is True


def test_evaluate_docs_computes_per_video_metrics():
    cal = importlib.import_module("auto_presort.calibrate")
    docs = {"v1.mp4": {"segments": [
        {"start": 2.0, "end": 8.0, "label": "eat"}]}}
    gt = {"v1.mp4": [[0.0, 10.0, "eat"], [10.0, 12.0, "drink"]]}
    rows = cal.evaluate_docs(docs, gt, min_gt_dur=3.0, min_cover=0.5)
    assert len(rows) == 1
    assert rows[0]["purity"] == 1.0
    assert abs(rows[0]["recall"] - (6.0 / 12.0)) < 1e-6
    assert rows[0]["short_lost"] == ["drink"]
