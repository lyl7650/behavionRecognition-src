import importlib
import json
import threading
import urllib.error
import urllib.request


def _start(batch_dir):
    rs = importlib.import_module("auto_presort.review_server")
    server = rs.make_server(str(batch_dir), port=0)  # port=0 -> random free port
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


def _seed(batch_dir):
    st = importlib.import_module("auto_presort.store")
    st.save_doc(str(batch_dir), {
        "schema_version": "autopresort-1", "rel_path": "v1.mp4",
        "source_id": "abc123", "duration": 10.0, "approved": False,
        "segments": [{"start": 1.0, "end": 4.0, "label": "eat",
                      "confidence": 0.95, "source": "model"}],
        "gaps": [], "meta": {"class_names": ["eat", "drink"]},
    })


def test_api_videos_lists_docs(tmp_path):
    _seed(tmp_path)
    server, base = _start(tmp_path)
    try:
        with urllib.request.urlopen(f"{base}/api/videos") as r:
            data = json.loads(r.read())
        assert data[0]["rel_path"] == "v1.mp4"
        assert data[0]["approved"] is False
    finally:
        server.shutdown()


def test_api_doc_get_and_post_save_invalidates_approved(tmp_path):
    _seed(tmp_path)
    server, base = _start(tmp_path)
    try:
        body = json.dumps({"rel_path": "v1.mp4", "action": "approve"}).encode()
        req = urllib.request.Request(f"{base}/api/doc", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req) as r:
            assert json.loads(r.read())["approved"] is True

        body = json.dumps({"rel_path": "v1.mp4", "action": "save", "segments": [
            {"start": 1.5, "end": 4.0, "label": "drink",
             "confidence": 0.95, "source": "human"}]}).encode()
        req = urllib.request.Request(f"{base}/api/doc", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req) as r:
            doc = json.loads(r.read())
        assert doc["approved"] is False
        assert doc["segments"][0]["label"] == "drink"
    finally:
        server.shutdown()


def test_video_range_request_returns_206(tmp_path):
    _seed(tmp_path)
    (tmp_path / "v1.mp4").write_bytes(b"0123456789ABCDEF")
    server, base = _start(tmp_path)
    try:
        req = urllib.request.Request(f"{base}/video/v1.mp4",
                                     headers={"Range": "bytes=4-7"})
        with urllib.request.urlopen(req) as r:
            assert r.status == 206
            assert r.read() == b"4567"
            assert r.headers["Content-Range"] == "bytes 4-7/16"
    finally:
        server.shutdown()


def test_video_path_traversal_rejected(tmp_path):
    _seed(tmp_path)
    server, base = _start(tmp_path)
    try:
        try:
            urllib.request.urlopen(f"{base}/video/../secret.txt")
            status = 200
        except urllib.error.HTTPError as e:
            status = e.code
        assert status in (400, 403, 404)
    finally:
        server.shutdown()


def test_video_unsatisfiable_range_returns_416(tmp_path):
    _seed(tmp_path)
    (tmp_path / "v1.mp4").write_bytes(b"0123456789ABCDEF")
    server, base = _start(tmp_path)
    try:
        req = urllib.request.Request(f"{base}/video/v1.mp4",
                                     headers={"Range": "bytes=99-"})
        try:
            urllib.request.urlopen(req)
            status = 200
        except urllib.error.HTTPError as e:
            status = e.code
        assert status == 416
        # 畸形 Range 不崩，按整文件返回
        req2 = urllib.request.Request(f"{base}/video/v1.mp4",
                                      headers={"Range": "bytes=abc-"})
        with urllib.request.urlopen(req2) as r:
            assert r.status == 200
            assert r.read() == b"0123456789ABCDEF"
    finally:
        server.shutdown()


def test_api_stats(tmp_path):
    _seed(tmp_path)  # v1.mp4 doc: 1 x eat 1.0-4.0, not approved
    st = importlib.import_module("auto_presort.store")
    st.save_doc(str(tmp_path), {
        "schema_version": "autopresort-1", "rel_path": "v2.mp4",
        "source_id": "def456", "duration": 8.0, "approved": True,
        "segments": [{"start": 0.0, "end": 2.5, "label": "drink",
                      "confidence": 0.92, "source": "model"},
                     {"start": 3.0, "end": 4.0, "label": "eat",
                      "confidence": 0.91, "source": "model"}],
        "gaps": [], "meta": {"class_names": ["eat", "drink"]},
    })
    st.save_doc(str(tmp_path), {   # 有 doc 但没产出任何段 → 不算"模型已标"
        "schema_version": "autopresort-1", "rel_path": "v3.mp4",
        "source_id": "aaa111", "duration": 5.0, "approved": False,
        "segments": [], "gaps": [], "meta": {"class_names": ["eat", "drink"]},
    })
    for name in ("v1.mp4", "v2.mp4", "v3.mp4", "v4.mp4"):  # v4 未标注，只计入总数
        (tmp_path / name).write_bytes(b"x")
    server, base = _start(tmp_path)
    try:
        with urllib.request.urlopen(f"{base}/api/stats") as r:
            s = json.loads(r.read())
        assert s["total_videos"] == 4
        assert s["labeled_videos"] == 2      # v1、v2 有段；v3 空段不算
        assert s["approved_videos"] == 1
        # 标签统计只认已批准的 v2；未批准 v1 的 eat 不计入
        assert s["labels"]["eat"] == {"count": 1, "duration": 1.0}
        assert s["labels"]["drink"] == {"count": 1, "duration": 2.5}
        # class_names 汇总所有 doc 的动作名单，供前端柱状图为每个动作固定一根柱
        assert set(s["class_names"]) == {"eat", "drink"}
    finally:
        server.shutdown()


def _seed_with_levels(batch_dir):
    st = importlib.import_module("auto_presort.store")
    segs09 = [{"start": 2.5, "end": 6.5, "label": "eat",
               "confidence": 0.95, "source": "model"}]
    segs05 = [{"start": 1.5, "end": 7.5, "label": "eat",
               "confidence": 0.60, "source": "model"}]
    st.save_doc(str(batch_dir), {
        "schema_version": "autopresort-2", "rel_path": "vL.mp4",
        "source_id": "lev123", "duration": 10.0, "approved": False,
        "segments": segs09, "gaps": [],
        "segment_levels": {"0.5": segs05, "0.6": segs05, "0.7": segs09,
                           "0.8": segs09, "0.9": segs09},
        "meta": {"class_names": ["eat", "drink"]},
    })


def test_api_doc_returns_segment_levels(tmp_path):
    _seed_with_levels(tmp_path)
    server, base = _start(tmp_path)
    try:
        with urllib.request.urlopen(f"{base}/api/doc?rel=vL.mp4") as r:
            doc = json.loads(r.read())
        assert set(doc["segment_levels"].keys()) == {"0.5", "0.6", "0.7", "0.8", "0.9"}
        assert doc["segment_levels"]["0.5"][0]["confidence"] == 0.60
    finally:
        server.shutdown()


def test_save_preserves_segment_levels(tmp_path):
    _seed_with_levels(tmp_path)
    server, base = _start(tmp_path)
    try:
        # 前端 save 只发 segments（不含 segment_levels）
        body = json.dumps({"rel_path": "vL.mp4", "action": "save", "segments": [
            {"start": 2.0, "end": 5.0, "label": "drink",
             "confidence": 0.7, "source": "human"}]}).encode()
        req = urllib.request.Request(f"{base}/api/doc", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req) as r:
            doc = json.loads(r.read())
        # segments 被更新，但 segment_levels 仍在（未被覆盖）
        assert doc["segments"][0]["label"] == "drink"
        assert "segment_levels" in doc
        assert set(doc["segment_levels"].keys()) == {"0.5", "0.6", "0.7", "0.8", "0.9"}
    finally:
        server.shutdown()


def test_api_videos_includes_conf_tier(tmp_path):
    """Verify /api/videos includes conf_tier field computed from segment_levels."""
    st = importlib.import_module("auto_presort.store")

    # Video 1: has segments at 0.8 level (not at 0.9) → tier = "0.8"
    st.save_doc(str(tmp_path), {
        "schema_version": "autopresort-2", "rel_path": "v1.mp4",
        "source_id": "s1", "duration": 10.0, "approved": False,
        "segments": [], "gaps": [],
        "segment_levels": {
            "0.5": [{"start":1,"end":3,"label":"eat","confidence":0.6,"source":"model"}],
            "0.6": [{"start":1,"end":3,"label":"eat","confidence":0.65,"source":"model"}],
            "0.7": [{"start":2,"end":4,"label":"eat","confidence":0.75,"source":"model"}],
            "0.8": [{"start":2,"end":4,"label":"eat","confidence":0.85,"source":"model"}],
            "0.9": []  # empty at 0.9 → highest non-empty is 0.8
        },
        "meta": {"class_names": ["eat"]},
    })

    # Video 2: has segments at 0.9 level → tier = "0.9"
    st.save_doc(str(tmp_path), {
        "schema_version": "autopresort-2", "rel_path": "v2.mp4",
        "source_id": "s2", "duration": 12.0, "approved": True,
        "segments": [{"start":1,"end":5,"label":"drink","confidence":0.95,"source":"model"}],
        "gaps": [],
        "segment_levels": {
            "0.5": [{"start":1,"end":5,"label":"drink","confidence":0.95,"source":"model"}],
            "0.9": [{"start":1,"end":5,"label":"drink","confidence":0.95,"source":"model"}],
        },
        "meta": {"class_names": ["drink"]},
    })

    # Video 3: no segment_levels (old format) → tier = null
    st.save_doc(str(tmp_path), {
        "schema_version": "autopresort-1", "rel_path": "v3.mp4",
        "source_id": "s3", "duration": 8.0, "approved": False,
        "segments": [], "gaps": [], "meta": {"class_names": ["eat"]},
    })

    server, base = _start(tmp_path)
    try:
        with urllib.request.urlopen(f"{base}/api/videos") as r:
            items = json.loads(r.read())

        assert len(items) == 3

        v1 = next(x for x in items if x["rel_path"] == "v1.mp4")
        assert v1["conf_tier"] == "0.8", "Should pick highest non-empty tier"

        v2 = next(x for x in items if x["rel_path"] == "v2.mp4")
        assert v2["conf_tier"] == "0.9", "Should pick 0.9 when available"

        v3 = next(x for x in items if x["rel_path"] == "v3.mp4")
        assert v3["conf_tier"] is None, "No segment_levels → null tier"
    finally:
        server.shutdown()
