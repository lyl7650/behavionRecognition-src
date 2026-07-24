from scripts.pipeline import scaffold_dirs as sd


def test_build_dirs_creates_records_and_sorted_classes(tmp_path):
    ctx = {"species": "pet", "site": "hkbr", "date": "2026-07-15"}
    created = sd.build_dirs(ctx, root=str(tmp_path))
    base = tmp_path / "dataset" / "raw_videos" / "pet" / "normal" / "hkbr-record"
    assert (base / "records" / "2026-07-15").is_dir()
    for c in ["eat", "drink", "active", "resting"]:
        assert (base / "sorted-2026-07-15" / c).is_dir()
    assert len(created) == 1 + 4


def test_build_dirs_rabbit_has_four_classes(tmp_path):
    ctx = {"species": "rabbit", "site": "hkbr", "date": "2026-07-15"}
    sd.build_dirs(ctx, root=str(tmp_path))
    base = tmp_path / "dataset" / "raw_videos" / "rabbit" / "hkbr-record" / "sorted-2026-07-15"
    for c in ["resting", "active", "eating", "drinking"]:
        assert (base / c).is_dir()
