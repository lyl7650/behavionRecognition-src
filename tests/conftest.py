import pytest

from scripts.pipeline.lineage import store


@pytest.fixture(autouse=True)
def isolate_lineage(tmp_path_factory, monkeypatch):
    """Redirect lineage writes away from the repo without touching tmp_path.

    Tests that assert on the exact contents of ``tmp_path`` must stay
    unaffected, so the lineage root lives in a separate throwaway directory.
    """
    root = tmp_path_factory.mktemp("lineage")
    monkeypatch.setattr(store, "LINEAGE_ROOT", root)
    monkeypatch.setattr(store, "INGEST_PATH", root / "ingest.jsonl")
    monkeypatch.setattr(store, "DERIVATION_PATH", root / "derivations.jsonl")
    monkeypatch.setattr(store, "SCENES_PATH", root / "snapshots")
