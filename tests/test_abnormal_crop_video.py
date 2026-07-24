import importlib
from pathlib import Path


def test_get_clip_segments_uses_fixed_max_duration(monkeypatch):
    crop_video = importlib.import_module("utils.abnormal_crop_video")
    monkeypatch.setattr(crop_video, "MAX_CLIP_DURATION", 5)

    segments = crop_video.get_clip_segments(55.727)

    assert len(segments) == 11
    assert segments[0] == (0.0, 5.0)
    assert segments[-1] == (50.0, 55.0)


def test_get_clip_segments_skips_tail_shorter_than_min_duration(monkeypatch):
    crop_video = importlib.import_module("utils.abnormal_crop_video")
    monkeypatch.setattr(crop_video, "MAX_CLIP_DURATION", 4)
    monkeypatch.setattr(crop_video, "MIN_CLIP_DURATION", 2.0)

    segments = crop_video.get_clip_segments(9.5)

    assert segments == [(0.0, 4.0), (4.0, 8.0)]


def test_train_clip_relative_path_matches_training_dataset_layout():
    crop_video = importlib.import_module("utils.abnormal_crop_video")
    video_stem = (
        "CAM1_20260605_195140_8b302346-f0ef-401d-8e71-"
        "cd9129b07870-00.00.00.000-00.00.55.727"
    )

    relative_path = crop_video.get_train_clip_relative_path("eat", video_stem, 0)

    assert relative_path == Path(
        "eat",
        (
            "CAM1_20260605_195140_8b302346-f0ef-401d-8e71-"
            "cd9129b07870-00.00.00.000-00.00.55.727_clip-1.mp4"
        ),
    )


def test_build_segment_frame_ranges_keeps_valid_ranges_only():
    crop_video = importlib.import_module("utils.abnormal_crop_video")

    ranges = crop_video.build_segment_frame_ranges(
        segments=[(0.0, 3.0), (3.0, 3.01)],
        fps=30.0,
        total_frames=100,
    )

    assert ranges == [(0, 0, 90)]


def test_build_smoothed_crop_boxes_keeps_size_stable_and_inside_frame():
    crop_video = importlib.import_module("utils.abnormal_crop_video")
    boxes = [
        (10, 10, 50, 40),
        (100, 60, 200, 120),
        (180, 100, 300, 190),
    ]

    crop_boxes = crop_video.build_smoothed_crop_boxes(
        boxes,
        frame_width=320,
        frame_height=240,
        fallback_size=(80, 60),
        expand_ratio=1.25,
        smooth_alpha=0.5,
    )

    widths = {x2 - x1 for x1, _, x2, _ in crop_boxes}
    heights = {y2 - y1 for _, y1, _, y2 in crop_boxes}
    centers_x = [(x1 + x2) / 2 for x1, _, x2, _ in crop_boxes]

    assert len(widths) == 1
    assert len(heights) == 1
    assert all(0 <= x1 < x2 <= 320 and 0 <= y1 < y2 <= 240 for x1, y1, x2, y2 in crop_boxes)
    assert centers_x[1] - centers_x[0] < 95


def test_source_key_extracts_source_or_falls_back():
    crop_video = importlib.import_module("utils.abnormal_crop_video")
    # auto_presort naming: strip trailing -<start>-<end> span
    assert crop_video.source_key("1057464119_d62c76eb-2.5-19.5.mp4") == "1057464119_d62c76eb"
    assert crop_video.source_key("CAM1_vid-20-25.mp4") == "CAM1_vid"
    # run_crop appends _clip-N to the stem; still recovers the same source
    assert crop_video.source_key("1057464119_d62c76eb-2.5-19.5_clip-1.mp4") == "1057464119_d62c76eb"
    # legacy naming (no -<start>-<end> span): whole filename is the key
    assert crop_video.source_key("CAM1_20260605.mp4") == "CAM1_20260605.mp4"


def test_rebalance_val_keeps_same_source_in_one_split(tmp_path):
    crop_video = importlib.import_module("utils.abnormal_crop_video")
    out = tmp_path / "snap"
    records = {"eat": [
        ("srcA-0-5.mp4",    ["eat/srcA-0-5_clip-1.mp4"]),
        ("srcB-0-5.mp4",    ["eat/srcB-0-5_clip-1.mp4"]),
        ("srcA-10-15.mp4",  ["eat/srcA-10-15_clip-1.mp4"]),
        ("srcB-9-14.mp4",   ["eat/srcB-9-14_clip-1.mp4"]),
    ]}
    for _, clips in records["eat"]:
        for c in clips:
            p = out / "train" / c
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"x")
    cfg = crop_video.default_config()
    cfg["output_base_dir"] = str(out)
    crop_video.rebalance_val(records, cfg)
    for src in ("srcA", "srcB"):
        in_train = list((out / "train").rglob(f"{src}-*"))
        in_val = list((out / "val").rglob(f"{src}-*"))
        assert not (in_train and in_val), f"source {src} leaked across splits"
