import json
import sys
from types import SimpleNamespace

import pytest

from scripts.vlm_screener import review_infer, review_store as store
from scripts.vlm_screener import server
from scripts.vlm_screener.prompts import COMBOS_BY_SPECIES


def test_scene_is_directory_scoped_and_exported(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "SCENES_PATH", str(tmp_path / "scenes.json"))
    target = str(tmp_path / "videos")
    (tmp_path / "videos").mkdir()
    assert store.ensure_context(target, "pet", "Clinic_A") == {
        "species": "pet", "scene": "Clinic_A"}
    assert store.scenes() == ["Clinic_A"]
    assert store.ensure_context(target, "pet") == store.context(target)
    with pytest.raises(ValueError, match="already assigned"):
        store.ensure_context(target, "rabbit", "Clinic_A")
    with pytest.raises(ValueError, match="already assigned"):
        store.ensure_context(target, "pet", "Clinic_B")
    with pytest.raises(ValueError, match="scene must contain"):
        store.ensure_context(target, "pet", "../bad")

    store.save_doc(target, {"vid": "v1", "rel_path": "a.mp4", "approved": True,
                            "segments": [{"label": "active", "start": 0,
                                          "end": 3, "source": "human"}]})
    record = json.loads(store.export_jsonl(target))
    assert record["species"] == "pet"
    assert record["scene"] == "Clinic_A"
    assert record["source"] == "human"
    assert store.write_finish(target)["scene"] == "Clinic_A"


def test_legacy_rabbit_requires_scene_and_cannot_be_opened_as_pet(tmp_path):
    target = str(tmp_path)
    (tmp_path / store.STORE_DIRNAME / "docs").mkdir(parents=True)
    with pytest.raises(ValueError, match="rabbit"):
        store.ensure_context(target, "pet", "Clinic_A")
    assert store.ensure_context(target, "rabbit") is None
    with pytest.raises(ValueError, match="select a scene"):
        store.export_jsonl(target)


def test_pet_prompt_is_checked_before_gpu_precheck(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "SCENES_PATH", str(tmp_path / "scenes.json"))
    store.ensure_context(str(tmp_path), "pet", "Site_A")
    monkeypatch.setattr(review_infer, "precheck", lambda:
                        pytest.fail("GPU check must not run for an empty prompt"))
    review_infer._state["running"] = False
    ok, message = review_infer.start(str(tmp_path),
                                     [COMBOS_BY_SPECIES["pet"][0]["combo"]], False)
    assert not ok and "prompt v1 is not written yet" in message
    ok, message = review_infer.start(str(tmp_path), ["8b-v3"], False)
    assert not ok and "invalid pet combo" in message


def test_pet_uses_pet_detector_and_its_own_prompt(monkeypatch):
    from scripts.vlm_screener import prompts

    monkeypatch.setitem(prompts.PET_PROMPTS, "v1", "pet prompt for test")
    monkeypatch.setattr(store, "require_context", lambda target:
                        {"species": "pet", "scene": "Site_A"})
    video = {"vid": "v1", "rel_path": "v1.mp4", "path": "v1.mp4"}
    monkeypatch.setattr(store, "load_videos", lambda target: [video])
    monkeypatch.setattr(store, "list_docs", lambda target: {})
    monkeypatch.setattr(store, "load_windows", lambda target: {})
    monkeypatch.setattr(store, "load_gates", lambda target: {})
    monkeypatch.setattr(store, "load_doc", lambda target, vid: None)
    saved = []
    monkeypatch.setattr(store, "save_doc", lambda target, doc: saved.append(doc))
    monkeypatch.setattr(store, "set_status", lambda *args: None)
    monkeypatch.setattr(review_infer, "_write_progress", lambda *args: None)
    monkeypatch.setattr(review_infer, "_ensure_windows", lambda *args:
                        {"duration": 1.0, "windows": []})
    monkeypatch.setattr(review_infer, "_ensure_frames", lambda *args: None)
    monkeypatch.setattr(review_infer, "_ensure_gate", lambda *args: {})
    monkeypatch.setattr(review_infer, "load_model", lambda *args:
                        (object(), object(), [], False))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False)))
    weights = []
    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(
        YOLO=lambda path: weights.append(path)))
    cfg = COMBOS_BY_SPECIES["pet"][0]
    assert review_infer.run_model_for_batch("target", [cfg], False,
                                            ["v1"], "1:8b") == 0
    assert weights == [review_infer.PET_YOLO_WEIGHTS]
    assert saved[0]["combos"]["8b-v1"]["prompt_version"] == "screener-pet-5class-v1"


def test_species_selection_clears_open_directory(monkeypatch, tmp_path):
    monkeypatch.setattr(server.review_infer, "status", lambda: {"running": False})
    monkeypatch.setitem(server._state, "species", "rabbit")
    monkeypatch.setitem(server._state, "target", "old-rabbit-dir")
    monkeypatch.setitem(server._state, "overrides", {"video": "copy"})
    monkeypatch.setitem(server._state, "suggested_dir", "old-rabbit-dir")
    assert server.select_species("pet") == {"species": "pet", "target": None}
    assert server._state["overrides"] == {}
    assert server._state["suggested_dir"] is None
    with pytest.raises(ValueError, match="species changed"):
        server.open_target(str(tmp_path), species="rabbit")
    with pytest.raises(ValueError, match="invalid species"):
        server.select_species("other")


def test_species_cannot_change_during_inference(monkeypatch):
    monkeypatch.setitem(server._state, "species", "rabbit")
    monkeypatch.setattr(server.review_infer, "status", lambda: {"running": True})
    with pytest.raises(RuntimeError, match="inference is running"):
        server.select_species("pet")
    assert server._state["species"] == "rabbit"
