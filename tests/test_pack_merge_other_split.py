import importlib
import json
import os


def test_build_prefixes_and_resolve_roundtrip(tmp_path):
    osm = importlib.import_module("auto_presort.other_split")
    d1 = tmp_path / "siteA-record" / "sorted-1" / "other"
    d2 = tmp_path / "siteB-record" / "sorted-2" / "other"
    d1.mkdir(parents=True)
    d2.mkdir(parents=True)
    prefixes = osm.build_prefixes(str(tmp_path), [str(d1), str(d2)])

    phys, local = osm.resolve_prefix(prefixes, "siteA-record/sorted-1/other/clip1.mp4")
    assert os.path.realpath(phys) == os.path.realpath(str(d1))
    assert local == "clip1.mp4"

    phys2, local2 = osm.resolve_prefix(prefixes, "siteB-record/sorted-2/other/x.mp4")
    assert os.path.realpath(phys2) == os.path.realpath(str(d2))
    assert local2 == "x.mp4"


def test_resolve_prefix_unknown_returns_none(tmp_path):
    osm = importlib.import_module("auto_presort.other_split")
    d1 = tmp_path / "rec" / "sorted-1" / "other"
    d1.mkdir(parents=True)
    prefixes = osm.build_prefixes(str(tmp_path), [str(d1)])
    assert osm.resolve_prefix(prefixes, "nope/other/z.mp4") == (None, None)


def test_select_clips_range_and_all():
    pack = importlib.import_module("scripts.review_tools.pack_review_other_split")
    clips = [("p", "d", f"v{i}.mp4") for i in range(5)]
    assert pack.select_clips(clips, "all") == clips
    assert pack.select_clips(clips, "") == clips
    assert pack.select_clips(clips, "1-3") == clips[1:4]
    assert pack.select_clips(clips, "0,4") == [clips[0], clips[4]]
    # out-of-range / garbage tokens are ignored
    assert pack.select_clips(clips, "3-99,abc") == [clips[3], clips[4]]


def _seed(other_dir, fn, approved=False, label="active"):
    st = importlib.import_module("auto_presort.store")
    st.save_doc(str(other_dir), {
        "schema_version": "autopresort-2", "rel_path": fn, "source_id": "s1",
        "duration": 10.0, "approved": approved,
        "segments": [{"start": 1.0, "end": 4.0, "label": label,
                      "confidence": 1.0, "source": "human"}] if label else [],
        "gaps": [], "meta": {"class_names": ["resting", "active", "eat", "drink"]}})


def test_pack_then_merge_roundtrips_annotations(tmp_path):
    pack = importlib.import_module("scripts.review_tools.pack_review_other_split")
    merge = importlib.import_module("scripts.review_tools.merge_other_split")
    osm = importlib.import_module("auto_presort.other_split")
    st = importlib.import_module("auto_presort.store")

    root = tmp_path / "raw"
    d1 = root / "siteA-record" / "sorted-1" / "other"
    d2 = root / "siteB-record" / "sorted-2" / "other"
    d1.mkdir(parents=True)
    d2.mkdir(parents=True)
    (d1 / "clip1.mp4").write_bytes(b"aaaa")
    (d2 / "clip2.mp4").write_bytes(b"bbbb")
    _seed(d1, "clip1.mp4", approved=False)
    _seed(d2, "clip2.mp4", approved=False)

    prefixes = osm.build_prefixes(str(root), osm.find_other_dirs(str(root)))
    clips = pack.list_all_clips(prefixes)
    assert [c[2] for c in clips] == ["clip1.mp4", "clip2.mp4"]

    out = tmp_path / "review-pack-x"
    info = pack.assemble_pack(str(out), clips, pack.PROJECT_ROOT)
    assert info["n_videos"] == 2
    assert info["n_docs"] == 2
    # pack preserves namespace path
    assert (out / "data" / "siteA-record" / "sorted-1" / "other" / "clip1.mp4").is_file()
    assert (out / "data" / "siteA-record" / "sorted-1" / "other" / ".autopresort").is_dir()

    # Colleague approves clip1 (edit the doc in the pack), leaves clip2 unapproved.
    doc_path = (out / "data" / "siteA-record" / "sorted-1" / "other"
                / ".autopresort" / "clip1.mp4.json")
    doc = json.loads(doc_path.read_text(encoding="utf-8"))
    doc["approved"] = True
    doc["segments"] = [{"start": 2.0, "end": 6.0, "label": "resting",
                        "confidence": 1.0, "source": "human"}]
    doc_path.write_text(json.dumps(doc), encoding="utf-8")

    result = merge.merge_back(str(out / "data"), prefixes)
    assert result["merged"] == 1
    assert len(result["skipped_unapproved"]) == 1
    # approved clip1 landed back in its original other/ dir
    back = st.load_doc(str(d1), "clip1.mp4")
    assert back["approved"] is True
    assert back["segments"][0]["label"] == "resting"
    # clip2 (unapproved) was not merged; original stays as seeded
    assert st.load_doc(str(d2), "clip2.mp4")["approved"] is False


def test_merge_back_force_merges_unapproved(tmp_path):
    pack = importlib.import_module("scripts.review_tools.pack_review_other_split")
    merge = importlib.import_module("scripts.review_tools.merge_other_split")
    osm = importlib.import_module("auto_presort.other_split")
    st = importlib.import_module("auto_presort.store")

    root = tmp_path / "raw"
    d1 = root / "rec" / "sorted-1" / "other"
    d1.mkdir(parents=True)
    (d1 / "clip1.mp4").write_bytes(b"aaaa")
    _seed(d1, "clip1.mp4", approved=False)

    prefixes = osm.build_prefixes(str(root), osm.find_other_dirs(str(root)))
    clips = pack.list_all_clips(prefixes)
    out = tmp_path / "review-pack-y"
    pack.assemble_pack(str(out), clips, pack.PROJECT_ROOT)

    result = merge.merge_back(str(out / "data"), prefixes, force=True)
    assert result["merged"] == 1
    assert result["skipped_unapproved"] == []
