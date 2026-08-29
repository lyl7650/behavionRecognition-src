import importlib
import json
import threading
import urllib.error
import urllib.parse
import urllib.request


def _start_multi(dirs):
    rs = importlib.import_module("auto_presort.review_server")
    server = rs.make_multi_server([str(d) for d in dirs], port=0)  # port=0 -> random free port
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


def _seed(batch_dir, rel="v1.mp4", label="resting"):
    st = importlib.import_module("auto_presort.store")
    st.save_doc(str(batch_dir), {
        "schema_version": "autopresort-2", "rel_path": rel,
        "source_id": "abc123", "duration": 10.0, "approved": False,
        "segments": [{"start": 1.0, "end": 4.0, "label": label,
                      "confidence": 1.0, "source": "human"}],
        "gaps": [], "meta": {"class_names": ["resting", "active", "eat", "drink"]},
    })


def test_api_videos_merges_across_dirs_with_namespaced_rel_path(tmp_path):
    dir_a = tmp_path / "siteA-record" / "sorted-1" / "pending"
    dir_b = tmp_path / "siteB-record" / "sorted-2" / "pending"
    dir_a.mkdir(parents=True)
    dir_b.mkdir(parents=True)
    _seed(dir_a, rel="clip1.mp4")
    _seed(dir_b, rel="clip2.mp4")

    server, base = _start_multi([dir_a, dir_b])
    try:
        with urllib.request.urlopen(f"{base}/api/videos") as r:
            items = json.loads(r.read())
        rels = sorted(it["rel_path"] for it in items)
        assert rels == [
            "siteA-record/sorted-1/pending/clip1.mp4",
            "siteB-record/sorted-2/pending/clip2.mp4",
        ]
        # No model in this workflow -> conf_tier is faked as "0.9" so the sidebar's
        # default filter (only 0.9 checked) doesn't hide every video.
        assert all(it["conf_tier"] == "0.9" for it in items)
    finally:
        server.shutdown()


def test_api_doc_get_and_post_route_to_correct_physical_dir(tmp_path):
    dir_a = tmp_path / "siteA-record" / "sorted-1" / "pending"
    dir_b = tmp_path / "siteB-record" / "sorted-2" / "pending"
    dir_a.mkdir(parents=True)
    dir_b.mkdir(parents=True)
    _seed(dir_a, rel="clip1.mp4")
    _seed(dir_b, rel="clip2.mp4")

    server, base = _start_multi([dir_a, dir_b])
    try:
        rel = "siteB-record/sorted-2/pending/clip2.mp4"
        with urllib.request.urlopen(f"{base}/api/doc?rel={urllib.parse.quote(rel)}") as r:
            doc = json.loads(r.read())
        assert doc["rel_path"] == "clip2.mp4"

        body = json.dumps({"rel_path": rel, "action": "approve"}).encode()
        req = urllib.request.Request(f"{base}/api/doc", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req) as r:
            assert json.loads(r.read())["approved"] is True

        st = importlib.import_module("auto_presort.store")
        assert st.load_doc(str(dir_b), "clip2.mp4")["approved"] is True
        assert st.load_doc(str(dir_a), "clip1.mp4")["approved"] is False
    finally:
        server.shutdown()


def test_video_range_request_routes_to_correct_dir(tmp_path):
    dir_a = tmp_path / "siteA-record" / "sorted-1" / "pending"
    dir_b = tmp_path / "siteB-record" / "sorted-2" / "pending"
    dir_a.mkdir(parents=True)
    dir_b.mkdir(parents=True)
    _seed(dir_a, rel="clip1.mp4")
    _seed(dir_b, rel="clip2.mp4")
    (dir_a / "clip1.mp4").write_bytes(b"AAAA0000")
    (dir_b / "clip2.mp4").write_bytes(b"BBBB1111")

    server, base = _start_multi([dir_a, dir_b])
    try:
        req = urllib.request.Request(
            f"{base}/video/siteB-record/sorted-2/pending/clip2.mp4",
            headers={"Range": "bytes=0-3"})
        with urllib.request.urlopen(req) as r:
            assert r.status == 206
            assert r.read() == b"BBBB"
    finally:
        server.shutdown()


def test_unknown_rel_path_returns_404(tmp_path):
    dir_a = tmp_path / "siteA-record" / "sorted-1" / "pending"
    dir_a.mkdir(parents=True)
    _seed(dir_a, rel="clip1.mp4")

    server, base = _start_multi([dir_a])
    try:
        try:
            urllib.request.urlopen(f"{base}/api/doc?rel=does/not/exist.mp4")
            status = 200
        except urllib.error.HTTPError as e:
            status = e.code
        assert status == 404
    finally:
        server.shutdown()


def test_api_stats_aggregates_across_dirs(tmp_path):
    dir_a = tmp_path / "siteA-record" / "sorted-1" / "pending"
    dir_b = tmp_path / "siteB-record" / "sorted-2" / "pending"
    dir_a.mkdir(parents=True)
    dir_b.mkdir(parents=True)
    _seed(dir_a, rel="clip1.mp4", label="resting")
    _seed(dir_b, rel="clip2.mp4", label="active")
    st = importlib.import_module("auto_presort.store")
    doc_a = st.load_doc(str(dir_a), "clip1.mp4")
    doc_a["approved"] = True
    st.save_doc(str(dir_a), doc_a)

    server, base = _start_multi([dir_a, dir_b])
    try:
        with urllib.request.urlopen(f"{base}/api/stats") as r:
            s = json.loads(r.read())
        assert s["total_videos"] == 0  # no real files on disk, only .json docs
        assert s["approved_videos"] == 1
        assert s["labels"]["resting"]["count"] == 1
        assert "active" not in s["labels"]  # doc_b never approved
        assert set(s["class_names"]) == {"resting", "active", "eat", "drink"}
    finally:
        server.shutdown()
