import importlib
import hashlib
import json
import pytest


def _write(tmp_path, content=b"good"):
    p = tmp_path / "teacher.pth"
    p.write_bytes(content)
    return p


def _sidecar(p, class_names, sha):
    meta = {"sha256": sha, "class_names": class_names, "species": "pet"}
    (p.parent / (p.name + ".meta.json")).write_text(json.dumps(meta))


CLASSES = ["eat", "drink", "active", "resting"]


def test_missing_checkpoint_raises(tmp_path):
    g = importlib.import_module("auto_presort.checkpoint_guard")
    with pytest.raises(FileNotFoundError):
        g.resolve_meta(str(tmp_path / "nope.pth"), CLASSES)


def test_empty_class_names_raises(tmp_path):
    g = importlib.import_module("auto_presort.checkpoint_guard")
    p = _write(tmp_path)
    with pytest.raises(ValueError, match="class_names"):
        g.resolve_meta(str(p), [])


def test_no_sidecar_ok_uses_species_class_names(tmp_path):
    g = importlib.import_module("auto_presort.checkpoint_guard")
    p = _write(tmp_path, b"good")
    meta = g.resolve_meta(str(p), CLASSES)
    assert meta["class_names"] == CLASSES
    assert meta["sha256"] == hashlib.sha256(b"good").hexdigest()


def test_sidecar_hash_mismatch_raises(tmp_path):
    g = importlib.import_module("auto_presort.checkpoint_guard")
    p = _write(tmp_path, b"good")
    _sidecar(p, CLASSES, sha="deadbeef")
    with pytest.raises(ValueError, match="hash"):
        g.resolve_meta(str(p), CLASSES)


def test_sidecar_class_names_mismatch_raises(tmp_path):
    g = importlib.import_module("auto_presort.checkpoint_guard")
    p = _write(tmp_path, b"good")
    sha = hashlib.sha256(b"good").hexdigest()
    _sidecar(p, ["resting", "active", "eating", "drinking"], sha=sha)
    with pytest.raises(ValueError, match="class_names"):
        g.resolve_meta(str(p), CLASSES)


def test_sidecar_ok_returns_meta(tmp_path):
    g = importlib.import_module("auto_presort.checkpoint_guard")
    p = _write(tmp_path, b"good")
    sha = hashlib.sha256(b"good").hexdigest()
    _sidecar(p, CLASSES, sha=sha)
    meta = g.resolve_meta(str(p), CLASSES)
    assert meta["class_names"] == CLASSES
    assert meta["sha256"] == sha


def test_head_dim_check_detects_mismatch():
    g = importlib.import_module("auto_presort.checkpoint_guard")
    fake_state = {"head.1.weight": [[0.0]] * 4, "head.1.bias": [0.0] * 4}
    with pytest.raises(ValueError, match="class count"):
        g.check_head_dim(fake_state, expected_num_classes=3)


def test_head_dim_check_passes_when_matching():
    g = importlib.import_module("auto_presort.checkpoint_guard")
    fake_state = {"head.1.weight": [[0.0]] * 3, "head.1.bias": [0.0] * 3}
    g.check_head_dim(fake_state, expected_num_classes=3)
