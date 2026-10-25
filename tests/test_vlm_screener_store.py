"""Regression tests for Windows sharing conflicts on review-store JSON files."""
import json
import os
from types import SimpleNamespace

import pytest

from scripts.vlm_screener import review_store as store


def test_atomic_write_retries_transient_windows_replace(monkeypatch, tmp_path):
    path = tmp_path / "videos.json"
    path.write_text('{"videos": ["old"]}', encoding="utf-8")
    original_replace = os.replace
    attempts = []

    def replace(src, dst):
        attempts.append(src)
        if len(attempts) == 1:
            error = PermissionError("locked by reader")
            error.winerror = 5
            raise error
        original_replace(src, dst)

    monkeypatch.setattr(store, "os", SimpleNamespace(
        name="nt", path=os.path, makedirs=os.makedirs, fdopen=os.fdopen,
        replace=replace, unlink=os.unlink))
    monkeypatch.setattr(store.time, "sleep", lambda _: None)
    store._write_json(str(path), {"videos": ["new"]})
    assert len(attempts) == 2
    assert attempts[0] == attempts[1] != str(path) + ".tmp"
    assert json.loads(path.read_text(encoding="utf-8")) == {"videos": ["new"]}
    assert sorted(p.name for p in tmp_path.iterdir()) == ["videos.json"]


def test_atomic_write_fails_without_overwriting_on_persistent_lock(
        monkeypatch, tmp_path):
    path = tmp_path / "videos.json"
    path.write_text('{"videos": ["old"]}', encoding="utf-8")
    attempts = []

    def replace(src, dst):
        attempts.append(src)
        error = PermissionError("locked")
        error.winerror = 32
        raise error

    monkeypatch.setattr(store, "os", SimpleNamespace(
        name="nt", path=os.path, makedirs=os.makedirs, fdopen=os.fdopen,
        replace=replace, unlink=os.unlink))
    monkeypatch.setattr(store.time, "sleep", lambda _: None)
    with pytest.raises(PermissionError):
        store._write_json(str(path), {"videos": ["new"]})
    assert len(attempts) == 21
    assert json.loads(path.read_text(encoding="utf-8")) == {"videos": ["old"]}
    assert sorted(p.name for p in tmp_path.iterdir()) == ["videos.json"]


def test_set_status_does_not_overwrite_unreadable_or_missing_list(tmp_path):
    target = str(tmp_path)
    data = tmp_path / store.STORE_DIRNAME
    data.mkdir()
    path = data / "videos.json"
    path.write_text("invalid json", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        store.set_status(target, "v001", "inferred")
    assert path.read_text(encoding="utf-8") == "invalid json"
    path.write_text('{"videos": []}', encoding="utf-8")
    with pytest.raises(KeyError, match="v001"):
        store.set_status(target, "v001", "inferred")
    assert json.loads(path.read_text(encoding="utf-8")) == {"videos": []}
