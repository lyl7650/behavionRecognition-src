import importlib
import json
import shutil
import subprocess

import pytest

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None,
                                reason="ffmpeg not on PATH")


def _make_video(path, seconds=10, fps=10, w=160, h=120):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    wr = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    for i in range(seconds * fps):
        f = np.full((h, w, 3), i % 255, dtype=np.uint8)
        wr.write(f)
    wr.release()


def _probe_duration(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True)
    return float(out.stdout.strip())


def test_apply_cuts_only_approved_and_names_with_source_id(tmp_path):
    apply_mod = importlib.import_module("scripts.presort.auto_presort_apply")
    st = importlib.import_module("auto_presort.store")

    batch = tmp_path / "batch"; batch.mkdir()
    _make_video(batch / "v1.mp4")
    _make_video(batch / "v2.mp4")
    st.save_doc(str(batch), {
        "schema_version": "autopresort-1", "rel_path": "v1.mp4",
        "source_id": "aaaa1111", "duration": 10.0, "approved": True,
        "segments": [{"start": 2.0, "end": 6.0, "label": "eat",
                      "confidence": 0.95, "source": "model"}],
        "gaps": [], "meta": {"class_names": ["eat"]}})
    st.save_doc(str(batch), {
        "schema_version": "autopresort-1", "rel_path": "v2.mp4",
        "source_id": "bbbb2222", "duration": 10.0, "approved": False,
        "segments": [{"start": 1.0, "end": 5.0, "label": "eat",
                      "confidence": 0.95, "source": "model"}],
        "gaps": [], "meta": {"class_names": ["eat"]}})

    out_dir = tmp_path / "sorted-2026-07-17"
    result = apply_mod.run_apply(str(batch), str(out_dir))

    assert result["skipped"] == ["v2.mp4"]
    outs = list(out_dir.rglob("*.mp4"))
    assert len(outs) == 1
    assert outs[0].name == "v1-2-6.mp4"
    assert outs[0].parent.name == "eat"
    assert abs(_probe_duration(outs[0]) - 4.0) <= 0.25


def test_apply_reports_duration_mismatch(tmp_path):
    apply_mod = importlib.import_module("scripts.presort.auto_presort_apply")
    st = importlib.import_module("auto_presort.store")
    batch = tmp_path / "batch"; batch.mkdir()
    _make_video(batch / "v1.mp4", seconds=5)
    st.save_doc(str(batch), {
        "schema_version": "autopresort-1", "rel_path": "v1.mp4",
        "source_id": "cccc3333", "duration": 5.0, "approved": True,
        "segments": [{"start": 3.0, "end": 9.0, "label": "eat",
                      "confidence": 0.95, "source": "model"}],
        "gaps": [], "meta": {"class_names": ["eat"]}})
    result = apply_mod.run_apply(str(batch), str(tmp_path / "out"))
    assert result["warnings"]
