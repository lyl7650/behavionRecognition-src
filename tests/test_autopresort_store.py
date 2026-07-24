import importlib
import json


def _mkdoc(rel="v1.mp4", approved=False, segs=None):
    return {
        "schema_version": "autopresort-1", "rel_path": rel, "source_id": "abc123",
        "duration": 10.0, "approved": approved,
        "segments": segs if segs is not None else
            [{"start": 1.0, "end": 4.0, "label": "eat", "confidence": 0.95, "source": "model"}],
        "gaps": [], "meta": {"class_names": ["eat"]},
    }


def test_save_and_load_roundtrip(tmp_path):
    st = importlib.import_module("auto_presort.store")
    st.save_doc(str(tmp_path), _mkdoc())
    doc = st.load_doc(str(tmp_path), "v1.mp4")
    assert doc["rel_path"] == "v1.mp4"
    assert (tmp_path / ".autopresort" / "v1.mp4.json").is_file()


def test_nested_rel_path_is_flattened_safely(tmp_path):
    st = importlib.import_module("auto_presort.store")
    st.save_doc(str(tmp_path), _mkdoc(rel="sub/v2.mp4"))
    doc = st.load_doc(str(tmp_path), "sub/v2.mp4")
    assert doc["rel_path"] == "sub/v2.mp4"
    files = [p.name for p in (tmp_path / ".autopresort").iterdir()]
    assert all("/" not in f and "\\" not in f for f in files)


def test_list_docs_returns_all(tmp_path):
    st = importlib.import_module("auto_presort.store")
    st.save_doc(str(tmp_path), _mkdoc(rel="a.mp4"))
    st.save_doc(str(tmp_path), _mkdoc(rel="b.mp4", approved=True))
    docs = st.list_docs(str(tmp_path))
    assert sorted(d["rel_path"] for d in docs) == ["a.mp4", "b.mp4"]


def test_update_doc_edit_forces_approved_false(tmp_path):
    st = importlib.import_module("auto_presort.store")
    st.save_doc(str(tmp_path), _mkdoc(approved=True))
    new_segs = [{"start": 1.5, "end": 4.0, "label": "eat",
                 "confidence": 0.95, "source": "human"}]
    doc = st.update_doc(str(tmp_path), "v1.mp4", segments=new_segs, action="save")
    assert doc["approved"] is False
    assert doc["segments"] == new_segs
    on_disk = st.load_doc(str(tmp_path), "v1.mp4")
    assert on_disk["approved"] is False


def test_update_doc_approve_action_sets_true(tmp_path):
    st = importlib.import_module("auto_presort.store")
    st.save_doc(str(tmp_path), _mkdoc(approved=False))
    doc = st.update_doc(str(tmp_path), "v1.mp4", segments=None, action="approve")
    assert doc["approved"] is True
    assert st.load_doc(str(tmp_path), "v1.mp4")["approved"] is True


def test_update_doc_unknown_action_raises(tmp_path):
    st = importlib.import_module("auto_presort.store")
    st.save_doc(str(tmp_path), _mkdoc())
    try:
        st.update_doc(str(tmp_path), "v1.mp4", segments=None, action="nope")
        assert False, "should raise"
    except ValueError:
        pass
