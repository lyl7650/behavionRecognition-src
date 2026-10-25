"""Review server for the directory-mode VLM screener (port 8768).

Editing-capable successor of the read-only screener server. One target
directory is "open" at a time; its labels live in <target>/.vlm-screener/.

GET  /                  screener.html
GET  /api/state         {target, species, suggested_dir, no_infer, infer: {...}}
POST /api/species       {species} -> select species and clear open directory
POST /api/open          {dir, scene, species} -> recursive scan + open (sync; may take a
                        while on first scan: durations + browser copies)
GET  /api/data          light per-video aggregate for filtering (no windows;
                        includes seg_labels = labels present in the current
                        segments, so manual-only labels are filterable)
GET  /api/doc?vid=      full doc: segments + combos windows + gate
POST /api/doc           {vid, action: save|approve, segments?} — 2b rules:
                        any save clears approved, approve keeps segments
POST /api/resynthesize  {vid} rebuild model segments from combo windows
POST /api/infer/start   {combos: [...], crop: bool} background worker
                        (403 when launched with --no-infer)
GET  /api/infer/status  worker progress
POST /api/finish        write review_state.json snapshot + stats, then start
                        the background clip cut (approved segments ->
                        dataset/raw_videos/<species>/<scene>-record/sorted-<date>/<label>/)
GET  /api/apply/status  clip cut worker progress
GET  /api/export        approved segments as a jsonl download
GET  /api/video?vid=    original mp4, HTTP Range; browser-decodable copy
                        substituted when the codec needs transcoding
GET  /api/frame?vid=&w= window thumbnail jpg
"""
import argparse
import json
import os
import re
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

from scripts.vlm_screener import review_store as store
from scripts.vlm_screener import review_infer
from scripts.vlm_screener import apply_clips
from scripts.vlm_screener.review_infer import synthesize_segments
from scripts.vlm_screener.prompts import COMBOS_BY_SPECIES

HTML_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "screener.html")
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LAST_DIR_FILE = os.path.join(PROJECT_ROOT, "logs", "vlm_screener_last_dir.txt")
VID_RE = re.compile(r"[0-9a-f]{12}")

_state = {"target": None, "suggested_dir": None, "no_infer": False,
          "overrides": {}, "species": "rabbit"}


def select_species(species):
    if species not in COMBOS_BY_SPECIES:
        raise ValueError(f"invalid species: {species}")
    if review_infer.status()["running"]:
        raise RuntimeError("inference is running; wait before changing species")
    if species != _state["species"]:
        _state.update(species=species, target=None, suggested_dir=None,
                      overrides={})
    return {"species": _state["species"], "target": _state["target"]}


def log(msg):
    print(msg, flush=True)


def _source_of(rel_path):
    norm = rel_path.replace("\\", "/")
    return norm.split("/", 1)[0] if "/" in norm else "(root)"


def _has_video_direct(d):
    """Directory directly contains a video file (keeps resolve hits precise)."""
    try:
        for name in os.listdir(d):
            if name.lower().endswith((".mp4", ".mov", ".avi", ".mkv", ".webm")):
                return True
    except OSError:
        return False
    return False


def resolve_folder(name):
    """Map a folder NAME (from the browser picker, which cannot see absolute
    paths) to full path(s). Search roots: ancestors of last-opened dir
    (parent / grandparent / great-grandparent) + project root, walk depth 3,
    prefer directories that directly contain video files."""
    name = (name or "").strip().strip('"')
    if not name or "/" in name or "\\" in name or ":" in name:
        return []
    roots, seen = [], set()
    try:
        with open(LAST_DIR_FILE, encoding="utf-8") as f:
            last = f.read().strip()
    except OSError:
        last = ""
    p = last
    for _ in range(3):
        p = os.path.dirname(p) if p else ""
        if p and os.path.isdir(p) and p not in seen:
            seen.add(p)
            roots.append(p)
        if not p:
            break
    if PROJECT_ROOT not in seen and os.path.isdir(PROJECT_ROOT):
        roots.append(PROJECT_ROOT)
    direct, indirect = [], []
    for r in roots:
        for root, dirs, _files in os.walk(r):
            depth = os.path.relpath(root, r).count(os.sep)
            if depth >= 3:
                dirs[:] = []
                continue
            dirs[:] = [d for d in dirs if not d.startswith(".")]
            for d in dirs:
                if d.lower() == name.lower():
                    full = os.path.join(root, d)
                    if full not in direct and full not in indirect:
                        (direct if _has_video_direct(full)
                         else indirect).append(full)
    return direct + indirect


def open_target(target, scene="", species=None):
    """Scan + open a directory. Sync on purpose: the /api/open caller waits."""
    target = os.path.abspath(target)
    if not os.path.isdir(target):
        raise NotADirectoryError(target)
    if review_infer.status()["running"]:
        raise RuntimeError("inference is running; wait for it to finish "
                           "before switching directory")
    if species is not None and species != _state["species"]:
        raise ValueError("species changed; select the species again before opening")
    provenance = store.ensure_context(target, _state["species"], scene)
    log(f"[review] scanning {target} ...")
    videos = store.scan_dir(target)
    log(f"[review] {len(videos)} videos; preparing browser copies ...")
    overrides = store.browser_video_overrides(target)
    _state["target"] = target
    _state["suggested_dir"] = None
    _state["overrides"] = overrides
    try:
        os.makedirs(os.path.dirname(LAST_DIR_FILE), exist_ok=True)
        with open(LAST_DIR_FILE, "w", encoding="utf-8") as f:
            f.write(target)
    except OSError:
        pass
    docs = store.list_docs(target)
    approved = sum(1 for d in docs.values() if d.get("approved"))
    log(f"[review] open: {target} (species={_state['species']}, "
        f"scene={(provenance or {}).get('scene') or '[unset]'}, {len(videos)} videos, "
        f"{len(docs)} with results, {approved} approved)")
    return {"target": target, "scene": (provenance or {}).get("scene", ""),
            "species": _state["species"], "n_videos": len(videos),
            "n_docs": len(docs), "n_approved": approved}


def aggregate():
    target = _state["target"]
    videos = store.load_videos(target)
    docs = store.list_docs(target)
    items = []
    for v in videos:
        doc = docs.get(v["vid"])
        combos = {}
        n_gate_none = 0
        if doc:
            for combo, cd in (doc.get("combos") or {}).items():
                combos[combo] = {
                    "dominant": cd.get("dominant"),
                    "dominant_share": cd.get("dominant_share"),
                    "dist": cd.get("dist"),
                    "prompt_version": cd.get("prompt_version"),
                }
            gvid = (doc.get("gate") or {}).get("windows") or {}
            n_gate_none = sum(1 for g in gvid.values() if not g.get("has_pet"))
        items.append({
            "vid": v["vid"], "name": v["name"], "rel_path": v["rel_path"],
            "source": _source_of(v["rel_path"]),
            "duration": v["duration"],
            "status": ("approved" if doc and doc.get("approved")
                       else "ready" if doc else "pending"),
            "n_segments": len((doc or {}).get("segments") or []),
            "n_windows": len(((doc or {}).get("gate") or {})
                             .get("windows") or {}),
            "n_gate_none": n_gate_none,
            "seg_labels": sorted({s.get("label") for s in
                                  ((doc or {}).get("segments") or [])
                                  if s.get("label")}),
            "combos": combos,
        })
    return {
        "schema": "vlm-review-data-1",
        "target": target,
        "species": _state["species"],
        "scene": (store.context(target) or {}).get("scene", ""),
        "n_videos": len(items),
        "class_names": store.CLASS_NAMES,
        "infer": review_infer.status(),
        "stats": store.compute_stats(target),
        "videos": items,
    }


def _doc_payload(target, vid):
    doc = store.load_doc(target, vid)
    if doc is None:
        return None
    video = store.video_by_vid(target, vid)
    doc = dict(doc)
    doc["video_path"] = (video or {}).get("path", doc.get("video_path"))
    doc["n_gate_none"] = sum(
        1 for g in ((doc.get("gate") or {}).get("windows") or {}).values()
        if not g.get("has_pet"))
    return doc


class Handler(BaseHTTPRequestHandler):
    server_version = "VlmScreener/2.0"

    def log_message(self, fmt, *args):  # noqa: A003 - quiet default spam
        pass

    def _send(self, code, body, ctype, extra=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (ConnectionAbortedError, BrokenPipeError, ConnectionResetError):
            pass

    def _json(self, obj, code=200, extra=None):
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"),
                   "application/json; charset=utf-8", extra)

    def _body(self):
        length = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(length) or b"{}")

    def do_GET(self):  # noqa: N802
        u = urlparse(self.path)
        q = parse_qs(u.query)
        if u.path in ("/", "/index.html", "/screener.html"):
            try:
                with open(HTML_PATH, "rb") as f:
                    self._send(200, f.read(), "text/html; charset=utf-8")
            except OSError as e:
                self._json({"error": f"html missing: {e}"}, 500)
        elif u.path == "/api/state":
            self._json({"target": _state["target"],
                        "suggested_dir": _state["suggested_dir"],
                        "species": _state["species"],
                        "scene": (store.context(_state["target"]) or {}).get("scene", "") if _state["target"] else "",
                        "scenes": store.scenes(),
                        "no_infer": _state["no_infer"],
                        "infer": review_infer.status()})
        elif u.path == "/api/data":
            if not _state["target"]:
                self._json({"error": "no directory open"}, 400)
                return
            self._json(aggregate())
        elif u.path == "/api/doc":
            vid = (q.get("vid") or [""])[0]
            if not _state["target"]:
                self._json({"error": "no directory open"}, 400)
                return
            doc = _doc_payload(_state["target"], vid)
            if doc is None:
                self._json({"error": "not inferred yet"}, 404)
            else:
                self._json(doc)
        elif u.path == "/api/infer/status":
            self._json(review_infer.status())
        elif u.path == "/api/apply/status":
            self._json(apply_clips.status())
        elif u.path == "/api/export":
            if not _state["target"]:
                self._json({"error": "no directory open"}, 400)
                return
            try:
                body = store.export_jsonl(_state["target"])
            except ValueError as exc:
                self._json({"error": str(exc)}, 409)
                return
            fname = f"review-export-{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
            self._send(200, body, "application/jsonl; charset=utf-8",
                       {"Content-Disposition": f'attachment; filename="{fname}"'})
        elif u.path == "/api/frame":
            vid = (q.get("vid") or [""])[0]
            win = (q.get("w") or ["w000"])[0]
            if not _state["target"] or not VID_RE.fullmatch(vid or ""):
                self._json({"error": "bad vid"}, 400)
                return
            fp = os.path.join(store.data_dir(_state["target"]),
                              "frames", vid, f"{win}_f0.jpg")
            if os.path.isfile(fp):
                with open(fp, "rb") as f:
                    self._send(200, f.read(), "image/jpeg")
            else:
                self._json({"error": "no frame"}, 404)
        elif u.path == "/api/video":
            vid = (q.get("vid") or [""])[0]
            if not _state["target"] or not VID_RE.fullmatch(vid or ""):
                self._json({"error": "bad vid"}, 400)
                return
            video = store.video_by_vid(_state["target"], vid)
            path = video["path"] if video else None
            if path:
                path = _state["overrides"].get(video["rel_path"], path)
            if not path or not os.path.isfile(path):
                self._json({"error": "video not found"}, 404)
                return
            self._serve_file(path)
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):  # noqa: N802
        u = urlparse(self.path)
        try:
            payload = self._body()
        except ValueError:
            self._json({"error": "bad json"}, 400)
            return
        if u.path == "/api/species":
            try:
                self._json(select_species(payload.get("species")))
            except ValueError as e:
                self._json({"error": str(e)}, 400)
            except RuntimeError as e:
                self._json({"error": str(e)}, 409)
            return
        if u.path == "/api/open":
            try:
                self._json(open_target(payload.get("dir", ""),
                                       payload.get("scene", ""),
                                       payload.get("species")))
            except NotADirectoryError:
                self._json({"error": "directory not found"}, 400)
            except (RuntimeError, ValueError) as e:
                self._json({"error": str(e)}, 409)
            return
        if u.path == "/api/resolve-folder":
            hits = resolve_folder(payload.get("name", ""))
            self._json({"hits": hits})
            return
        target = _state["target"]
        if not target:
            self._json({"error": "no directory open"}, 400)
            return
        if u.path == "/api/doc":
            try:
                doc = store.update_doc(target, payload.get("vid", ""),
                                       payload.get("segments"),
                                       payload.get("action", ""))
                self._json(doc)
            except FileNotFoundError:
                self._json({"error": "not inferred yet"}, 404)
            except (KeyError, ValueError) as e:
                self._json({"error": str(e)}, 400)
        elif u.path == "/api/resynthesize":
            doc = store.load_doc(target, payload.get("vid", ""))
            if doc is None:
                self._json({"error": "not inferred yet"}, 404)
                return
            doc["segments"] = synthesize_segments(doc.get("combos") or {},
                                                  doc.get("duration") or 0)
            doc["approved"] = False
            store.save_doc(target, doc)
            self._json(doc)
        elif u.path == "/api/infer/start":
            if _state["no_infer"]:
                self._json({"error": "server runs in view-only mode "
                                     "(--no-infer); use 7_vlm_screener.bat"},
                           403)
                return
            combos = [c for c in (payload.get("combos") or []) if c]
            if not combos:
                self._json({"error": "pick at least one combo"}, 400)
                return
            ok, msg = review_infer.start(target, combos,
                                         bool(payload.get("crop")))
            if ok:
                self._json({"ok": True, "message": msg})
            else:
                self._json({"error": msg}, 409)
        elif u.path == "/api/finish":
            if apply_clips.status()["running"]:
                self._json({"error": "clip cutting already running"}, 409)
                return
            try:
                state = store.write_finish(target)
            except ValueError as exc:
                self._json({"error": str(exc)}, 409)
                return
            ok, msg = apply_clips.start(target)
            self._json({**state,
                        "apply": {"ok": ok, "message": msg,
                                  "status": apply_clips.status()}})
        else:
            self._json({"error": "not found"}, 404)

    def _serve_file(self, path):
        size = os.path.getsize(path)
        rng = self.headers.get("Range")
        start, end = 0, size - 1
        code = 200
        if rng:
            m = re.match(r"bytes=(\d*)-(\d*)", rng)
            if m:
                if m.group(1):
                    start = int(m.group(1))
                if m.group(2):
                    end = int(m.group(2))
                code = 206
        start = max(0, min(start, size - 1))
        end = min(end, size - 1, start + 8 * 1024 * 1024 - 1)
        self.send_response(code)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        try:
            with open(path, "rb") as f:
                f.seek(start)
                remaining = end - start + 1
                while remaining > 0:
                    chunk = f.read(min(1 << 20, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
        except (ConnectionAbortedError, BrokenPipeError, ConnectionResetError):
            pass  # browser seek/pause aborts the stream; harmless (2b rule)


def serve(port=8768, open_browser=False, target=None, no_infer=False,
          species="rabbit"):
    if species not in COMBOS_BY_SPECIES:
        raise ValueError(f"invalid species: {species}")
    _state["species"] = species
    _state["no_infer"] = no_infer
    _state["target"] = None
    _state["suggested_dir"] = target
    _state["overrides"] = {}
    httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{port}"
    log(f"review server on {url} (no_infer={no_infer})")
    if open_browser:
        import threading
        import webbrowser
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    httpd.serve_forever()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8768)
    ap.add_argument("--dir", default=None,
                    help="video directory to prefill on the review page")
    ap.add_argument("--no-infer", action="store_true",
                    help="view-only: disable the inference endpoints")
    ap.add_argument("--view-only", action="store_true",
                    help="alias of --no-infer")
    ap.add_argument("--open-browser", action="store_true")
    ap.add_argument("--species", choices=("rabbit", "pet"), default="rabbit")
    a = ap.parse_args(argv)
    serve(a.port, a.open_browser, a.dir, a.no_infer or a.view_only,
          species=a.species)
    return 0


if __name__ == "__main__":
    sys.exit(main())
