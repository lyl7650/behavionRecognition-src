import pytest
from scripts.pipeline import run_crop


def test_validate_sorted_ok(tmp_path):
    (tmp_path / "eat").mkdir()
    (tmp_path / "eat" / "a.mp4").write_text("x", encoding="utf-8")
    run_crop.validate_sorted(str(tmp_path), ["eat", "drink", "active", "resting"])


def test_validate_sorted_raises_when_no_videos(tmp_path):
    (tmp_path / "eat").mkdir()
    with pytest.raises(FileNotFoundError):
        run_crop.validate_sorted(str(tmp_path), ["eat", "drink", "active", "resting"])


def test_assert_snapshot_empty_raises_when_nonempty(tmp_path):
    (tmp_path / "train").mkdir()
    (tmp_path / "train" / "x.mp4").write_text("x", encoding="utf-8")
    with pytest.raises(FileExistsError):
        run_crop.assert_snapshot_empty(str(tmp_path))


def test_assert_snapshot_empty_ok_when_missing(tmp_path):
    run_crop.assert_snapshot_empty(str(tmp_path / "nope"))
