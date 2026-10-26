"""Batch scheduling and model reuse without loading actual VLM weights."""
import sys

import pytest
from types import SimpleNamespace

from scripts.vlm_screener import review_infer as infer


def _videos(n):
    return [{"vid": f"v{i:03d}", "rel_path": f"{i:03d}.mp4", "path": "video.mp4",
             "duration": 3.0} for i in range(n)]


@pytest.mark.parametrize("completed,expected_sizes", [
    (0, [50, 50, 5]), (10, [50, 45]),
])
def test_worker_batches_50_and_synthesizes_before_next_batch(
        monkeypatch, completed, expected_sizes):
    monkeypatch.setattr(infer.store, "require_context", lambda target:
                        {"species": "rabbit", "scene": "site_a"})
    videos = _videos(105)
    docs = {v["vid"]: {"combos": {c["combo"]: {} for c in infer.COMBOS}}
            for v in videos[:completed]}
    monkeypatch.setattr(infer.store, "load_videos", lambda target: videos)
    monkeypatch.setattr(infer.store, "list_docs", lambda target: docs)
    monkeypatch.setattr(infer.time, "sleep", lambda seconds: None)
    events = []
    progress = {}

    class Process:
        returncode = 0

        def __init__(self, args, **kwargs):
            job = args[args.index("--job") + 1]
            combo_start = args.index("--combos") + 1
            vids_start = args.index("--vids")
            combos = args[combo_start:vids_start]
            vids = args[vids_start + 1:]
            count = sum(c not in docs.get(v, {}).get("combos", {})
                        for c in combos for v in vids)
            progress.update(job=job, done=count, total=count)
            events.append(("model", job, len(vids), count))

        def poll(self):
            return 0

    monkeypatch.setattr(infer.subprocess, "Popen", Process)
    monkeypatch.setattr(infer, "_read_progress", lambda target: progress)

    def synth(target, combos, vids):
        events.append(("synth", len(vids)))
        return []

    monkeypatch.setattr(infer, "resynthesize_all", synth)
    infer._state.update(running=True, error=None, done=0, total=(105 - completed) * 4)
    infer._worker("target", [c["combo"] for c in infer.COMBOS], False)

    expected = []
    for i, size in enumerate(expected_sizes, 1):
        expected.extend([("model", f"{i}:8b", size, size * 2),
                         ("model", f"{i}:32b", size, size * 2),
                         ("synth", size)])
    assert events == expected
    assert infer._state["error"] is None
    assert infer._state["done"] == infer._state["total"] == (105 - completed) * 4


def test_batch_reuses_model_and_resume_skips_completed_combos(monkeypatch):
    monkeypatch.setattr(infer.store, "require_context", lambda target:
                        {"species": "rabbit", "scene": "site_a"})
    videos = _videos(2)
    docs = {}
    monkeypatch.setattr(infer.store, "load_videos", lambda target: videos)
    monkeypatch.setattr(infer.store, "list_docs", lambda target: docs)
    monkeypatch.setattr(infer.store, "load_doc", lambda target, vid: docs.get(vid))
    monkeypatch.setattr(infer.store, "save_doc", lambda target, doc:
                        docs.__setitem__(doc["vid"], doc))
    monkeypatch.setattr(infer.store, "set_status", lambda *args: None)
    monkeypatch.setattr(infer.store, "load_windows", lambda target: {})
    monkeypatch.setattr(infer.store, "load_gates", lambda target: {})
    monkeypatch.setattr(infer, "_ensure_windows", lambda target, video, all_w:
                        {"duration": 3.0, "windows": [{"w": "w000", "start": 0}]})
    monkeypatch.setattr(infer, "_ensure_frames", lambda *args: None)
    monkeypatch.setattr(infer, "_ensure_gate", lambda *args:
                        {"w000": {"has_pet": True, "det_frames": 8}})
    monkeypatch.setattr(infer, "_load_pil_frames", lambda *args: [object()])
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False)))
    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=lambda _: object()))
    loads = []
    monkeypatch.setattr(infer, "load_model", lambda path, dtype, words:
                        (loads.append((path, dtype)) or (object(), object(), None, False)))
    monkeypatch.setattr(infer, "classify", lambda model, processor, ids, logits,
                        frames, prompt, name, table: (
                            ("eating" if prompt == infer.PROMPTS["v3"] else "active"),
                            0.8, None, False, False))
    progress = []
    monkeypatch.setattr(infer, "_write_progress", lambda target, p: progress.append(p))
    cfgs = infer.COMBOS[:2]
    assert infer.run_model_for_batch("target", cfgs, False,
                                     [v["vid"] for v in videos], "1:8b") == 0
    assert len(loads) == 1
    assert all(set(doc["combos"]) == {"8b-v3", "8b-old"}
               for doc in docs.values())
    assert progress[-1]["done"] == progress[-1]["total"] == 4

    infer.run_model_for_batch("target", cfgs, False,
                              [v["vid"] for v in videos], "1:8b")
    assert len(loads) == 1
    assert progress[-1]["total"] == 0


def test_vote_tie_and_missing_segments():
    def combo(label, confidence):
        return {"windows": [{"start": 0, "gate": "vlm", "label": label,
                             "confidence": confidence}]}

    votes = {"a": combo("eating", 0.8), "b": combo("eating", 0.6),
             "c": combo("active", 0.9), "d": combo("active", 0.9)}
    assert infer.synthesize_segments(votes, 3)[0]["label"] == "active"
    votes["b"] = combo("resting", 0.6)
    assert infer.synthesize_segments(votes, 3) == []  # 2:1:1 is below 3 votes


def test_worker_failure_does_not_advance_or_synthesize(monkeypatch):
    monkeypatch.setattr(infer.store, "require_context", lambda target:
                        {"species": "rabbit", "scene": "site_a"})
    monkeypatch.setattr(infer.store, "load_videos", lambda target: _videos(51))
    monkeypatch.setattr(infer.store, "list_docs", lambda target: {})
    launched = []

    class FailedProcess:
        returncode = 1

        def __init__(self, args, **kwargs):
            launched.append(args)

        def poll(self):
            return 1

    monkeypatch.setattr(infer.subprocess, "Popen", FailedProcess)
    monkeypatch.setattr(infer, "_read_progress", lambda target: {
        "job": "1:8b", "error": "model failed"})
    monkeypatch.setattr(infer, "resynthesize_all", lambda *args:
                        pytest.fail("failed batch must not synthesize"))
    infer._state.update(running=True, error=None, done=0, total=204)
    infer._worker("target", [c["combo"] for c in infer.COMBOS], False)
    assert len(launched) == 1
    assert infer._state["error"] == "model failed"
    assert infer._state["done"] == 0


def test_resynthesize_only_batch_and_preserves_human_edits(monkeypatch):
    def doc(vid, source):
        return {"vid": vid, "rel_path": vid, "duration": 3,
                "combos": {"8b-v3": {"windows": [
                    {"start": 0, "gate": "vlm", "label": "eating", "confidence": 0.9}]}},
                "segments": [{"start": 0, "end": 3, "label": "active",
                              "source": source}], "approved": True}

    docs = {"v1": doc("v1", "model"), "v2": doc("v2", "human"),
            "outside": doc("outside", "model")}
    saved = []
    monkeypatch.setattr(infer.store, "list_docs", lambda target: docs)
    monkeypatch.setattr(infer.store, "save_doc", lambda target, d: saved.append(d["vid"]))
    assert infer.resynthesize_all("target", ["8b-v3"], {"v1", "v2"}) == ["v1"]
    assert saved == ["v1"]
    assert docs["v1"]["segments"][0]["label"] == "eating"
    assert docs["v1"]["approved"] is False
    assert docs["v2"]["approved"] is True


def test_start_counts_only_missing_video_combo_pairs(monkeypatch):
    monkeypatch.setattr(infer.store, "require_context", lambda target:
                        {"species": "rabbit", "scene": "site_a"})
    videos = _videos(2)
    docs = {videos[0]["vid"]: {"combos": {"8b-v3": {}}}}
    monkeypatch.setattr(infer, "precheck", lambda: (True, ""))
    monkeypatch.setattr(infer.store, "load_videos", lambda target: videos)
    monkeypatch.setattr(infer.store, "list_docs", lambda target: docs)
    monkeypatch.setattr(infer.threading, "Thread", lambda **kwargs:
                        SimpleNamespace(start=lambda: None))
    infer._state["running"] = False
    ok, message = infer.start("target", ["8b-v3", "8b-old"], False)
    assert ok and "3 video-combo tasks" in message
    assert infer._state["total"] == 3
    infer._state["running"] = False
