import importlib


def test_make_segment_has_required_fields():
    tl = importlib.import_module("auto_presort.timeline")
    seg = tl.make_segment(start=1.0, end=4.0, label="eat", confidence=0.93, source="model")
    assert seg == {
        "start": 1.0, "end": 4.0, "label": "eat",
        "confidence": 0.93, "source": "model",
    }


def test_schema_version_is_exposed():
    tl = importlib.import_module("auto_presort.timeline")
    assert isinstance(tl.SCHEMA_VERSION, str) and tl.SCHEMA_VERSION


def test_iter_windows_full_windows_with_stride():
    tl = importlib.import_module("auto_presort.timeline")
    wins = tl.iter_windows(duration=8.0, window_sec=3.0, stride_sec=1.0)
    assert wins[0] == (0.0, 3.0, 1.5)   # (start, end, center)
    assert wins[-1] == (5.0, 8.0, 6.5)
    assert all(round(e - s, 6) == 3.0 for s, e, _ in wins)


def test_iter_windows_drops_incomplete_tail():
    tl = importlib.import_module("auto_presort.timeline")
    wins = tl.iter_windows(duration=3.5, window_sec=3.0, stride_sec=1.0)
    assert wins == [(0.0, 3.0, 1.5)]


def test_iter_windows_too_short_returns_empty():
    tl = importlib.import_module("auto_presort.timeline")
    assert tl.iter_windows(duration=2.0, window_sec=3.0, stride_sec=1.0) == []


def test_merge_windows_two_segments_with_min_conf():
    tl = importlib.import_module("auto_presort.timeline")
    preds = [
        (1.5, "eat", 0.95), (2.5, "eat", 0.96), (3.5, "eat", 0.94),
        (4.5, None, 0.10),
        (5.5, "active", 0.92), (6.5, "active", 0.93),
    ]
    segs = tl.merge_windows(preds, threshold=0.90, enter_run=2)
    assert [s["label"] for s in segs] == ["eat", "active"]
    assert segs[0]["start"] == 1.5 and segs[0]["end"] == 3.5
    assert segs[0]["confidence"] == 0.94
    assert segs[1]["start"] == 5.5 and segs[1]["end"] == 6.5


def test_merge_windows_flicker_splits_into_two_eat_segments():
    tl = importlib.import_module("auto_presort.timeline")
    preds = [
        (1.5, "eat", 0.95), (2.5, "eat", 0.96),
        (3.5, "drink", 0.99),
        (4.5, "eat", 0.95), (5.5, "eat", 0.94),
    ]
    segs = tl.merge_windows(preds, threshold=0.90, enter_run=2)
    assert [s["label"] for s in segs] == ["eat", "eat"]
    assert segs[0]["end"] == 2.5 and segs[1]["start"] == 4.5


def test_merge_windows_below_threshold_is_uncertain():
    tl = importlib.import_module("auto_presort.timeline")
    preds = [(1.5, "eat", 0.80), (2.5, "eat", 0.85)]
    assert tl.merge_windows(preds, threshold=0.90, enter_run=2) == []


def test_trim_returns_trimmed_and_boundary_spans():
    tl = importlib.import_module("auto_presort.timeline")
    seg = tl.make_segment(1.5, 7.5, "eat", 0.95, "model")
    trimmed, boundaries = tl.trim_segments([seg], trim_margin=1.0, min_duration=2.0)
    assert len(trimmed) == 1
    assert trimmed[0]["start"] == 2.5 and trimmed[0]["end"] == 6.5
    assert (1.5, 2.5) in boundaries and (6.5, 7.5) in boundaries


def test_trim_drops_too_short_but_still_reports_boundary():
    tl = importlib.import_module("auto_presort.timeline")
    seg = tl.make_segment(0.0, 3.5, "eat", 0.95, "model")
    trimmed, boundaries = tl.trim_segments([seg], trim_margin=1.0, min_duration=2.0)
    assert trimmed == []
    assert (0.0, 1.0) in boundaries and (2.5, 3.5) in boundaries


def test_compute_gaps_priority_and_full_partition():
    tl = importlib.import_module("auto_presort.timeline")
    segs = [tl.make_segment(2.5, 6.5, "eat", 0.95, "model")]
    gaps = tl.compute_gaps(
        duration=8.0, segments=segs,
        boundary_spans=[(1.5, 2.5), (6.5, 7.5)],
        no_animal_spans=[(7.5, 8.0)],
    )
    assert (0.0, 1.5, "uncertain") in gaps
    assert (1.5, 2.5, "boundary") in gaps
    assert (6.5, 7.5, "boundary") in gaps
    assert (7.5, 8.0, "no_animal") in gaps
    covered = sorted((g[0], g[1]) for g in gaps) + [(2.5, 6.5)]
    covered.sort()
    assert covered[0][0] == 0.0 and covered[-1][1] == 8.0


def test_build_video_doc_shape():
    tl = importlib.import_module("auto_presort.timeline")
    segs = [tl.make_segment(1.0, 3.0, "eat", 0.95, "model")]
    doc = tl.build_video_doc(
        rel_path="video_001.mp4", duration=4.0, segments=segs,
        boundary_spans=[], no_animal_spans=[], source_id="srcA1b2",
        approved=False, meta={"checkpoint_sha256": "abc", "class_names": ["eat"]},
    )
    assert doc["schema_version"] == tl.SCHEMA_VERSION
    assert doc["rel_path"] == "video_001.mp4"
    assert doc["source_id"] == "srcA1b2"
    assert doc["approved"] is False
    assert doc["segments"] == segs
    assert doc["meta"]["checkpoint_sha256"] == "abc"
    assert "gaps" in doc and "duration" in doc


def test_split_out_no_animal_carves_hole_inside_segment():
    # rule 4: a no-animal span inside a confident segment must split it,
    # even when the span is shorter than a window (the window-drop can't catch it).
    tl = importlib.import_module("auto_presort.timeline")
    seg = tl.make_segment(2.0, 12.0, "eat", 0.95, "model")
    out = tl.split_out_no_animal([seg], no_animal_spans=[(6.0, 7.2)])
    assert len(out) == 2
    assert out[0]["start"] == 2.0 and out[0]["end"] == 6.0
    assert out[1]["start"] == 7.2 and out[1]["end"] == 12.0
    # label/confidence preserved on both halves
    assert all(s["label"] == "eat" and s["confidence"] == 0.95 for s in out)


def test_split_out_no_animal_drops_fully_covered_segment():
    tl = importlib.import_module("auto_presort.timeline")
    seg = tl.make_segment(3.0, 5.0, "eat", 0.95, "model")
    out = tl.split_out_no_animal([seg], no_animal_spans=[(2.0, 6.0)])
    assert out == []


def test_split_out_no_animal_noop_when_disjoint():
    tl = importlib.import_module("auto_presort.timeline")
    seg = tl.make_segment(2.0, 5.0, "eat", 0.95, "model")
    out = tl.split_out_no_animal([seg], no_animal_spans=[(6.0, 8.0)])
    assert len(out) == 1
    assert out[0]["start"] == 2.0 and out[0]["end"] == 5.0


def test_split_out_no_animal_trims_edge_overlap():
    tl = importlib.import_module("auto_presort.timeline")
    seg = tl.make_segment(2.0, 8.0, "eat", 0.95, "model")
    out = tl.split_out_no_animal([seg], no_animal_spans=[(7.0, 9.0)])
    assert len(out) == 1
    assert out[0]["start"] == 2.0 and out[0]["end"] == 7.0


def test_conf_levels_constant():
    tl = importlib.import_module("auto_presort.timeline")
    assert tl.CONF_LEVELS == [0.5, 0.6, 0.7, 0.8, 0.9]


def test_schema_version_bumped_to_v2():
    tl = importlib.import_module("auto_presort.timeline")
    assert tl.SCHEMA_VERSION == "autopresort-2"


def test_build_video_doc_includes_segment_levels_when_given():
    tl = importlib.import_module("auto_presort.timeline")
    segs09 = [tl.make_segment(1.0, 3.0, "eat", 0.95, "model")]
    levels = {"0.5": segs09, "0.6": segs09, "0.7": segs09,
              "0.8": segs09, "0.9": segs09}
    doc = tl.build_video_doc(
        rel_path="v.mp4", duration=4.0, segments=segs09,
        boundary_spans=[], no_animal_spans=[], source_id="s1",
        approved=False, meta={"class_names": ["eat"]},
        segment_levels=levels,
    )
    assert set(doc["segment_levels"].keys()) == {"0.5", "0.6", "0.7", "0.8", "0.9"}
    assert doc["segment_levels"]["0.9"] == segs09


def test_build_video_doc_omits_segment_levels_when_none():
    tl = importlib.import_module("auto_presort.timeline")
    segs = [tl.make_segment(1.0, 3.0, "eat", 0.95, "model")]
    doc = tl.build_video_doc(
        rel_path="v.mp4", duration=4.0, segments=segs,
        boundary_spans=[], no_animal_spans=[], source_id="s1",
        approved=False, meta={"class_names": ["eat"]},
    )
    assert "segment_levels" not in doc


def test_build_video_doc_includes_track_boxes_when_given():
    tl = importlib.import_module("auto_presort.timeline")
    segs = [tl.make_segment(1.0, 3.0, "eat", 0.95, "model")]
    doc = tl.build_video_doc(
        rel_path="v.mp4", duration=4.0, segments=segs,
        boundary_spans=[], no_animal_spans=[], source_id="s1",
        approved=False, meta={"class_names": ["eat"]},
        track_boxes=[(10.04, 20.0, 30.0, 40.0), None, (11.0, 21.0, 31.0, 41.0)],
        fps=16.0, frame_w=1920, frame_h=1080,
    )
    # tuples become JSON-friendly lists, coords rounded, missing frame stays null
    assert doc["track_boxes"] == [[10.0, 20.0, 30.0, 40.0], None, [11.0, 21.0, 31.0, 41.0]]
    assert doc["fps"] == 16.0
    assert doc["frame_w"] == 1920 and doc["frame_h"] == 1080


def test_build_video_doc_omits_track_boxes_when_none():
    tl = importlib.import_module("auto_presort.timeline")
    segs = [tl.make_segment(1.0, 3.0, "eat", 0.95, "model")]
    doc = tl.build_video_doc(
        rel_path="v.mp4", duration=4.0, segments=segs,
        boundary_spans=[], no_animal_spans=[], source_id="s1",
        approved=False, meta={"class_names": ["eat"]},
    )
    assert "track_boxes" not in doc
    assert "fps" not in doc
