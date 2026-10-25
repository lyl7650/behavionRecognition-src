import pytest

from scripts.pipeline.lineage import store


@pytest.fixture(autouse=True)
def isolate_lineage(tmp_path, monkeypatch):
    root = tmp_path / "lineage"
    root.mkdir()
    monkeypatch.setattr(store, "LINEAGE_ROOT", root)
    monkeypatch.setattr(store, "INGEST_PATH", root / "ingest.jsonl")
    monkeypatch.setattr(store, "DERIVATION_PATH", root / "derivations.jsonl")
    monkeypatch.setattr(store, "SCENES_PATH", root / "snapshots")
