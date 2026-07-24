import os

import auto_presort.store as store
from scripts.review_tools import pack_review as pk


def _rels(n):
    return [f"v{i}.mp4" for i in range(n)]


def test_select_all_keyword():
    assert pk.select_videos(_rels(3), "all") == ["v0.mp4", "v1.mp4", "v2.mp4"]


def test_select_blank_means_all():
    assert pk.select_videos(_rels(3), "") == ["v0.mp4", "v1.mp4", "v2.mp4"]


def test_select_range_inclusive():
    assert pk.select_videos(_rels(30), "0-19") == _rels(20)


def test_select_explicit_indices():
    assert pk.select_videos(_rels(10), "0,3,5") == ["v0.mp4", "v3.mp4", "v5.mp4"]


def test_select_mixed_and_dedup_and_order():
    # 混合 + 重复索引去重 + 保持原列表顺序(不按输入顺序)
    assert pk.select_videos(_rels(25), "20-22,10,10") == ["v10.mp4", "v20.mp4", "v21.mp4", "v22.mp4"]


def test_select_out_of_range_ignored():
    assert pk.select_videos(_rels(3), "0,99") == ["v0.mp4"]


def _make_fake_project(root):
    # 造出被拷贝的源文件(占位内容即可)
    os.makedirs(os.path.join(root, "auto_presort"))
    os.makedirs(os.path.join(root, "review_launchers"))
    for rel in pk.CODE_FILES:
        p = os.path.join(root, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w") as f:
            f.write("x")
    for src in pk.LAUNCHER_TMPLS:
        with open(os.path.join(root, src), "w") as f:
            f.write("launch")


def _make_fake_records(rec, rels, with_doc):
    os.makedirs(os.path.join(rec, ".autopresort"))
    for rel in rels:
        with open(os.path.join(rec, rel), "wb") as f:
            f.write(b"vid")
        if rel in with_doc:
            store.save_doc(rec, {"schema_version": "autopresort-1", "rel_path": rel,
                                 "source_id": "s", "duration": 1.0, "approved": False,
                                 "segments": [], "gaps": [], "meta": {"class_names": ["eat"]}})


def test_assemble_pack_copies_code_videos_docs_launchers(tmp_path):
    proj = tmp_path / "proj"; rec = tmp_path / "rec"; out = tmp_path / "out"
    _make_fake_project(str(proj))
    _make_fake_records(str(rec), ["a.mp4", "b.mp4"], with_doc={"a.mp4", "b.mp4"})
    info = pk.assemble_pack(str(rec), str(out), ["a.mp4", "b.mp4"], str(proj))

    assert info["n_videos"] == 2 and info["n_docs"] == 2 and info["missing_docs"] == []
    # 代码进 app/
    assert (out / "app" / "auto_presort_review.py").exists()
    assert (out / "app" / "auto_presort" / "review_server.py").exists()
    # 视频 + 标注进 data/
    assert (out / "data" / "a.mp4").exists()
    assert (out / "data" / ".autopresort" / store._doc_filename("a.mp4")).exists()
    # 双击入口留在根目录
    assert (out / "review.bat").exists()
    assert (out / "review.command").exists()
    assert (out / "README.txt").exists()


def test_assemble_pack_records_missing_doc(tmp_path):
    proj = tmp_path / "proj"; rec = tmp_path / "rec"; out = tmp_path / "out"
    _make_fake_project(str(proj))
    _make_fake_records(str(rec), ["a.mp4", "b.mp4"], with_doc={"a.mp4"})  # b 没 doc
    info = pk.assemble_pack(str(rec), str(out), ["a.mp4", "b.mp4"], str(proj))
    assert info["n_videos"] == 2 and info["n_docs"] == 1
    assert info["missing_docs"] == ["b.mp4"]
    assert not (out / "data" / ".autopresort" / store._doc_filename("b.mp4")).exists()
