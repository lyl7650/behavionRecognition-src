import importlib


def _doc(rel, approved, segs, source_id="a1b2c3d4"):
    return {"rel_path": rel, "approved": approved, "source_id": source_id,
            "segments": segs, "duration": 60.0}


def test_plan_cuts_skips_unapproved_and_reports():
    ac = importlib.import_module("auto_presort.apply_clips")
    docs = [
        _doc("v1.mp4", True,  [{"start": 2.5, "end": 6.5, "label": "eat"}]),
        _doc("v2.mp4", False, [{"start": 1.0, "end": 5.0, "label": "active"}]),
    ]
    cuts, skipped = ac.plan_cuts(docs)
    assert skipped == ["v2.mp4"]
    assert len(cuts) == 1
    assert cuts[0]["rel_path"] == "v1.mp4"
    assert cuts[0]["label"] == "eat"


def test_clip_output_name_embeds_source_name_and_span():
    ac = importlib.import_module("auto_presort.apply_clips")
    name = ac.clip_output_name(
        "1057464119_20260716_200550_d62c76eb.mp4", 2.5, 19.5)
    assert name == "1057464119_20260716_200550_d62c76eb-2.5-19.5.mp4"


def test_clip_output_name_strips_dir_and_ext():
    ac = importlib.import_module("auto_presort.apply_clips")
    name = ac.clip_output_name("sub/CAM1_video.mp4", 12.0, 20.0)
    assert name == "CAM1_video-12-20.mp4"


def test_build_ffmpeg_cmd_reencodes_accurately():
    ac = importlib.import_module("auto_presort.apply_clips")
    cmd = ac.build_ffmpeg_cmd("in.mp4", "out/eat/x.mp4", start=2.5, end=6.5)
    s = " ".join(cmd)
    assert cmd[0] == "ffmpeg"
    assert "-c copy" not in s and "copy" not in s
    assert "libx264" in s
    assert "-ss" in cmd and "2.5" in cmd
    assert "-t" in cmd and "4.0" in cmd
    assert cmd[-1] == "out/eat/x.mp4"


def test_duration_ok_within_tolerance():
    ac = importlib.import_module("auto_presort.apply_clips")
    assert ac.duration_ok(expected=4.0, actual=4.05, tolerance=0.25) is True
    assert ac.duration_ok(expected=4.0, actual=5.0, tolerance=0.25) is False


def test_source_id_from_rel_path_is_stable_and_short():
    ac = importlib.import_module("auto_presort.apply_clips")
    a = ac.source_id_for("sub/video_001.mp4")
    b = ac.source_id_for("sub/video_001.mp4")
    c = ac.source_id_for("sub\\video_001.mp4")
    assert a == b == c
    assert len(a) == 8 and "__" not in a
