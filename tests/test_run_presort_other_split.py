import importlib
import shutil

import pytest

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")


def _make_video(path, seconds=10, fps=10, w=160, h=120):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    wr = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    for i in range(seconds * fps):
        f = np.full((h, w, 3), i % 255, dtype=np.uint8)
        wr.write(f)
    wr.release()


def test_apply_other_split_skips_unseeded_dirs(tmp_path, capsys):
    apply_mod = importlib.import_module("run_presort_apply_other_split")
    other_dir = tmp_path / "siteA-record" / "sorted-1" / "other"
    other_dir.mkdir(parents=True)
    (other_dir / "clip1.mp4").write_bytes(b"x")

    rc = apply_mod.main(["--root", str(tmp_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "not seeded yet" in out


def test_apply_other_split_reports_no_dirs_found(tmp_path):
    apply_mod = importlib.import_module("run_presort_apply_other_split")
    (tmp_path / "empty").mkdir()
    rc = apply_mod.main(["--root", str(tmp_path)])
    assert rc == 1


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH")
def test_apply_other_split_cuts_into_sorted_date_sibling_dirs(tmp_path):
    seed_mod = importlib.import_module("auto_presort_seed_other_split")
    apply_mod = importlib.import_module("run_presort_apply_other_split")
    st = importlib.import_module("auto_presort.store")

    other_dir = tmp_path / "siteA-record" / "sorted-2026-07-20" / "other"
    other_dir.mkdir(parents=True)
    _make_video(other_dir / "clip1.mp4")

    seed_mod.seed_dir(str(other_dir), probe_duration_fn=lambda p: 10.0)
    doc = st.load_doc(str(other_dir), "clip1.mp4")
    doc["segments"] = [{"start": 1.0, "end": 5.0, "label": "active",
                        "confidence": 1.0, "source": "human"}]
    doc["approved"] = True
    st.save_doc(str(other_dir), doc)

    rc = apply_mod.main(["--root", str(tmp_path)])
    assert rc == 0

    sorted_date_dir = other_dir.parent
    outs = list((sorted_date_dir / "active").glob("*.mp4"))
    assert len(outs) == 1
    # the original other/ clip is left untouched
    assert (other_dir / "clip1.mp4").is_file()
