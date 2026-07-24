import importlib


def _params():
    return dict(window_sec=3.0, stride_sec=1.0, threshold=0.90, enter_run=2,
                trim_margin=1.0, min_duration=2.0, imgsz=224, temporal_window=16,
                expand_ratio=1.25, no_animal_overlap=0.5)


def test_label_video_produces_trimmed_segment():
    eng = importlib.import_module("auto_presort.engine")
    track = {"fps": 10.0, "frames": 90, "boxes": [(0, 0, 10, 10)] * 90,
             "flags": [True] * 90, "no_animal_spans": []}
    doc = eng.label_video(
        video_path="fake.mp4", rel_path="fake.mp4", source_id="src1",
        track=track,
        read_window_frames_fn=lambda vp, s, e, fps: ([None], [None]),
        classify_frames_fn=lambda frames, boxes: ("eat", 0.95),
        meta={"class_names": ["eat"]}, params=_params(),
    )
    assert doc["approved"] is False
    assert len(doc["segments"]) == 1
    seg = doc["segments"][0]
    assert seg["label"] == "eat"
    assert seg["start"] == 2.5 and seg["end"] == 6.5


def test_label_video_forces_gap_in_no_animal_region():
    eng = importlib.import_module("auto_presort.engine")
    track = {"fps": 10.0, "frames": 90, "boxes": [None] * 90,
             "flags": [False] * 90, "no_animal_spans": [(0.0, 9.0)]}
    doc = eng.label_video(
        video_path="fake.mp4", rel_path="fake.mp4", source_id="src1",
        track=track,
        read_window_frames_fn=lambda vp, s, e, fps: ([None], [None]),
        classify_frames_fn=lambda frames, boxes: ("eat", 0.99),
        meta={"class_names": ["eat"]}, params=_params(),
    )
    assert doc["segments"] == []
    assert any(g[2] == "no_animal" for g in doc["gaps"])


def test_engine_integration_with_fakes():
    eng = importlib.import_module("auto_presort.engine")
    tl = importlib.import_module("auto_presort.timeline")

    boxes = [(0, 0, 20, 20)] * 60 + [None] * 30
    flags = [True] * 60 + [False] * 30
    dt = importlib.import_module("auto_presort.detect_track")
    track = {"fps": 10.0, "frames": 90, "boxes": dt.propagate_boxes(boxes),
             "flags": flags, "no_animal_spans": dt.no_animal_spans(flags, 10.0, 1.0)}

    seen = {"tensors": 0}

    def reader(vp, s, e, fps):
        return [object(), object(), object()], [(0, 0, 20, 20)] * 3

    def classifier(frames, boxes):
        seen["tensors"] += 1
        return ("eat", 0.95)

    params = dict(window_sec=3.0, stride_sec=1.0, threshold=0.90, enter_run=2,
                  trim_margin=1.0, min_duration=2.0, imgsz=224, temporal_window=16,
                  expand_ratio=1.25, no_animal_overlap=0.5)
    doc = eng.label_video(
        video_path="fake.mp4", rel_path="fake.mp4", source_id="s1", track=track,
        read_window_frames_fn=reader, classify_frames_fn=classifier,
        meta={"class_names": ["eat"], "sha256": "abc"}, params=params,
    )
    assert len(doc["segments"]) == 1 and doc["segments"][0]["label"] == "eat"
    assert any(g[2] == "no_animal" for g in doc["gaps"])
    total_windows = len(tl.iter_windows(9.0, 3.0, 1.0))
    assert seen["tensors"] < total_windows


def test_label_video_short_no_animal_span_still_forces_gap():
    # Regression (hard rule 4): a no-animal span shorter than a window (< 1.5s)
    # must NOT be swallowed into a confident segment.
    eng = importlib.import_module("auto_presort.engine")
    # 20s @ 10fps; animal out at [9.0, 10.4] (1.4s < window 3.0s)
    frames = 200
    flags = [True] * frames
    for i in range(90, 104):  # 9.0s..10.4s
        flags[i] = False
    dt = importlib.import_module("auto_presort.detect_track")
    track = {"fps": 10.0, "frames": frames,
             "boxes": [(0, 0, 10, 10)] * frames,
             "flags": flags,
             "no_animal_spans": dt.no_animal_spans(flags, 10.0, 1.0)}
    assert track["no_animal_spans"]  # sanity: a ~1.4s span was recorded
    doc = eng.label_video(
        video_path="fake.mp4", rel_path="fake.mp4", source_id="s1", track=track,
        read_window_frames_fn=lambda vp, s, e, fps: ([None], [None]),
        classify_frames_fn=lambda frames, boxes: ("eat", 0.99),  # confident throughout
        meta={"class_names": ["eat"]}, params=_params(),
    )
    # No segment may contain the no-animal region [9.0, 10.4]
    for seg in doc["segments"]:
        assert not (seg["start"] < 10.4 and seg["end"] > 9.0), \
            f"segment {seg} spans the no-animal region"
    # And a no_animal gap must be emitted
    assert any(g[2] == "no_animal" for g in doc["gaps"])


def test_label_video_produces_all_conf_levels():
    eng = importlib.import_module("auto_presort.engine")
    tl = importlib.import_module("auto_presort.timeline")
    track = {"fps": 10.0, "frames": 90, "boxes": [(0, 0, 10, 10)] * 90,
             "flags": [True] * 90, "no_animal_spans": []}
    doc = eng.label_video(
        video_path="fake.mp4", rel_path="fake.mp4", source_id="src1",
        track=track,
        read_window_frames_fn=lambda vp, s, e, fps: ([None], [None]),
        classify_frames_fn=lambda frames, boxes: ("eat", 0.95),
        meta={"class_names": ["eat"]}, params=_params(),
    )
    assert set(doc["segment_levels"].keys()) == {"0.5", "0.6", "0.7", "0.8", "0.9"}
    assert doc["segments"] == doc["segment_levels"]["0.9"]


def test_label_video_lower_threshold_merges_across_dip():
    # 中间一个低分窗口(0.60)在 0.9 档打断成两段、在 0.5 档合成一段。
    eng = importlib.import_module("auto_presort.engine")
    # 6 高 + 1 低(0.60) + 6 高 = 13 窗；每侧 6 窗，trim(margin 1.0) 后仍 ≥ min_duration 2.0 得以保留。
    confs = iter([0.95] * 6 + [0.60] + [0.95] * 6)

    def classifier(frames, boxes):
        return ("eat", next(confs))

    track = {"fps": 10.0, "frames": 150, "boxes": [(0, 0, 10, 10)] * 150,
             "flags": [True] * 150, "no_animal_spans": []}
    doc = eng.label_video(
        video_path="fake.mp4", rel_path="fake.mp4", source_id="src1",
        track=track,
        read_window_frames_fn=lambda vp, s, e, fps: ([None], [None]),
        classify_frames_fn=classifier,
        meta={"class_names": ["eat"]}, params=_params(),
    )
    # 0.9 档：0.60 打断 → 两段；0.5 档：0.60 够格 → 一段
    assert len(doc["segment_levels"]["0.9"]) == 2
    assert len(doc["segment_levels"]["0.5"]) == 1
