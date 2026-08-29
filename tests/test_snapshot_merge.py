from scripts.pipeline import snapshot_merge as sm


def test_find_latest_picks_newest_before_current():
    dirs = ["2026-06-24", "2026-06-26", "2026-07-02", "2026-07-15"]
    latest, ambiguous = sm.find_latest_old_snapshot(
        "dataset/rabbit", "2026-07-15", dir_lister=lambda root: dirs)
    assert latest == "2026-07-02"
    assert ambiguous == []


def test_find_latest_excludes_current_and_future():
    dirs = ["2026-07-15", "2026-07-20"]
    latest, ambiguous = sm.find_latest_old_snapshot(
        "dataset/rabbit", "2026-07-15", dir_lister=lambda root: dirs)
    assert latest is None


def test_find_latest_collects_ambiguous_names():
    dirs = ["2026-07-02", "0703-videos", "0626"]
    latest, ambiguous = sm.find_latest_old_snapshot(
        "dataset/rabbit", "2026-07-15", dir_lister=lambda root: dirs)
    assert latest == "2026-07-02"
    assert set(ambiguous) == {"0703-videos", "0626"}


def _touch(p):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("x", encoding="utf-8")


def test_merge_video_format_copies_and_skips_dupes(tmp_path):
    old = tmp_path / "old"; new = tmp_path / "new"
    _touch(old / "train" / "eat" / "a.mp4")
    _touch(old / "train" / "eat" / "b.mp4")
    _touch(old / "val" / "drink" / "c.mp4")
    _touch(new / "train" / "eat" / "a.mp4")
    stats = sm.merge_snapshot(str(old), str(new), frames=False)
    assert (new / "train" / "eat" / "b.mp4").is_file()
    assert (new / "val" / "drink" / "c.mp4").is_file()
    assert stats == {"copied": 2, "skipped": 1, "filtered": 0}


def test_merge_frame_format_copies_whole_sample_folder(tmp_path):
    old = tmp_path / "old"; new = tmp_path / "new"
    _touch(old / "train" / "0626-eat-x-0" / "frame_0000.jpg")
    _touch(old / "train" / "0626-eat-x-0" / "label.txt")
    stats = sm.merge_snapshot(str(old), str(new), frames=True)
    assert (new / "train" / "0626-eat-x-0" / "frame_0000.jpg").is_file()
    assert (new / "train" / "0626-eat-x-0" / "label.txt").is_file()
    assert stats["copied"] == 1


def test_merge_video_format_filters_class_no_longer_valid(tmp_path):
    old = tmp_path / "old"; new = tmp_path / "new"
    _touch(old / "train" / "eat" / "a.mp4")
    _touch(old / "train" / "drink" / "b.mp4")
    _touch(old / "train" / "other" / "c.mp4")
    stats = sm.merge_snapshot(str(old), str(new), frames=False,
                               class_names=["eat", "drink", "active", "resting"])
    assert (new / "train" / "eat" / "a.mp4").is_file()
    assert (new / "train" / "drink" / "b.mp4").is_file()
    assert not (new / "train" / "other" / "c.mp4").exists()
    assert stats["copied"] == 2
    assert stats["filtered"] == 1


def test_merge_frame_format_filters_class_no_longer_valid(tmp_path):
    old = tmp_path / "old"; new = tmp_path / "new"
    _touch(old / "train" / "site-eat-v1-0" / "frame_0000.jpg")
    _touch(old / "train" / "site-other-v1-0" / "frame_0000.jpg")
    stats = sm.merge_snapshot(str(old), str(new), frames=True,
                               class_names=["eat", "drink", "active", "resting"])
    assert (new / "train" / "site-eat-v1-0" / "frame_0000.jpg").is_file()
    assert not (new / "train" / "site-other-v1-0").exists()
    assert stats["copied"] == 1
    assert stats["filtered"] == 1


def test_merge_class_names_none_copies_everything(tmp_path):
    old = tmp_path / "old"; new = tmp_path / "new"
    _touch(old / "train" / "eat" / "a.mp4")
    _touch(old / "train" / "other" / "c.mp4")
    stats = sm.merge_snapshot(str(old), str(new), frames=False)
    assert (new / "train" / "eat" / "a.mp4").is_file()
    assert (new / "train" / "other" / "c.mp4").is_file()
    assert stats["copied"] == 2
    assert stats["skipped"] == 0


def test_resolve_old_layer_video_prefers_videos_subdir(tmp_path):
    d = tmp_path / "2026-06-26"
    (d / "train").mkdir(parents=True)
    (d / "videos" / "train").mkdir(parents=True)
    layer, other = sm.resolve_old_layer(str(d), frames=False)
    assert layer == str(d / "videos")
    assert other is True


def test_resolve_old_layer_frames_uses_top_level(tmp_path):
    d = tmp_path / "2026-06-26"
    (d / "train").mkdir(parents=True)
    layer, other = sm.resolve_old_layer(str(d), frames=True)
    assert layer == str(d)
    assert other is False


def test_health_report_flags_cross_split_leak_and_unrecognized(tmp_path):
    new = tmp_path / "new"
    _touch(new / "train" / "eat" / "dup.mp4")
    _touch(new / "val" / "eat" / "dup.mp4")
    _touch(new / "train" / "eat" / "mystery.mp4")
    _touch(new / "train" / "nocat" / "weird.mp4")
    rep = sm.health_report(str(new), ["eat", "drink", "active", "resting"], frames=False)
    assert "dup.mp4" in rep["cross_split_leak"]
    assert any("nocat" in u or "weird" in u for u in rep["unrecognized"])


def test_health_report_detects_source_leak(tmp_path):
    for split, fn in [("train", "1057464119_d62c76eb-2.5-19.5_clip-1.mp4"),
                      ("val",   "1057464119_d62c76eb-22-30_clip-1.mp4")]:
        p = tmp_path / split / "eat" / fn
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x")
    report = sm.health_report(str(tmp_path), ["eat"], frames=False)
    assert "1057464119_d62c76eb" in report["source_leak"]
    assert report["cross_split_leak"] == []


def test_health_report_no_source_leak_for_legacy_names(tmp_path):
    for split, fn in [("train", "CAM1_video_clip-1.mp4"),
                      ("val",   "CAM2_video_clip-1.mp4")]:
        p = tmp_path / split / "eat" / fn
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x")
    report = sm.health_report(str(tmp_path), ["eat"], frames=False)
    assert report["source_leak"] == []
