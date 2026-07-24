import importlib


def test_find_other_dirs_finds_nested_case_insensitive(tmp_path):
    os_mod = importlib.import_module("auto_presort.other_split")
    (tmp_path / "siteA-record" / "sorted-0629" / "other").mkdir(parents=True)
    (tmp_path / "siteB-record" / "sorted-0501" / "Other").mkdir(parents=True)
    (tmp_path / "siteB-record" / "sorted-0501" / "eat").mkdir(parents=True)
    found = os_mod.find_other_dirs(str(tmp_path))
    assert len(found) == 2
    assert all(f.lower().endswith("other") for f in found)


def test_find_other_dirs_skips_hidden_dirs(tmp_path):
    os_mod = importlib.import_module("auto_presort.other_split")
    (tmp_path / ".autopresort" / "other").mkdir(parents=True)
    (tmp_path / "record" / "sorted-1" / "other").mkdir(parents=True)
    found = os_mod.find_other_dirs(str(tmp_path))
    assert len(found) == 1
    assert ".autopresort" not in found[0]


def test_find_other_dirs_empty_when_none(tmp_path):
    os_mod = importlib.import_module("auto_presort.other_split")
    (tmp_path / "record" / "sorted-1" / "eat").mkdir(parents=True)
    assert os_mod.find_other_dirs(str(tmp_path)) == []


def test_split_classes_has_no_ambiguous():
    os_mod = importlib.import_module("auto_presort.other_split")
    assert os_mod.SPLIT_CLASSES == ["resting", "active", "eat", "drink"]


def test_seed_dir_creates_blank_docs_and_skips_existing(tmp_path):
    seed_mod = importlib.import_module("auto_presort_seed_other_split")
    st = importlib.import_module("auto_presort.store")

    other_dir = tmp_path / "other"
    other_dir.mkdir()
    (other_dir / "v1.mp4").write_bytes(b"x")
    (other_dir / "v2.mp4").write_bytes(b"x")

    seeded, skipped = seed_mod.seed_dir(str(other_dir), probe_duration_fn=lambda p: 7.5)
    assert seeded == 2
    assert skipped == 0

    doc1 = st.load_doc(str(other_dir), "v1.mp4")
    assert doc1["segments"] == []
    assert doc1["approved"] is False
    assert doc1["duration"] == 7.5
    assert doc1["meta"]["class_names"] == ["resting", "active", "eat", "drink"]

    # Re-run: both already have docs -> nothing new seeded
    seeded2, skipped2 = seed_mod.seed_dir(str(other_dir), probe_duration_fn=lambda p: 7.5)
    assert seeded2 == 0
    assert skipped2 == 2


def test_seed_dir_ignores_non_video_files(tmp_path):
    seed_mod = importlib.import_module("auto_presort_seed_other_split")
    other_dir = tmp_path / "other"
    other_dir.mkdir()
    (other_dir / "v1.mp4").write_bytes(b"x")
    (other_dir / "readme.txt").write_text("not a video")

    seeded, skipped = seed_mod.seed_dir(str(other_dir), probe_duration_fn=lambda p: 1.0)
    assert seeded == 1
    assert skipped == 0
