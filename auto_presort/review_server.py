"""审核服务：stdlib http.server。

GET  /               → review_app.html
GET  /api/videos     → [{rel_path, approved, duration, n_segments, class_names}]
GET  /api/stats      → {total_videos, labeled_videos, approved_videos, labels}
GET  /api/doc?rel=X  → 完整 doc
POST /api/doc        → {rel_path, action: save|approve, segments?} → 更新后的 doc
GET  /video/<rel>    → 视频文件（支持 Range/206，长视频拖动必需）
"""
import json
import os
import posixpath
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs, unquote

from auto_presort import store

_HTML_PATH = os.path.join(os.path.dirname(__file__), "review_app.html")

VIDEO_EXTS = (".mp4", ".avi", ".mov", ".webm")


def compute_stats(batch_dir):
    """底部统计栏数据：目录内全部视频数、已标注（有段）/已批准数、
    各标签段数与总时长（秒，只统计已批准的视频）。"""
    total = 0
    for root, dirs, files in os.walk(batch_dir):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        total += sum(1 for f in files if f.lower().endswith(VIDEO_EXTS))
    docs = store.list_docs(batch_dir)
    labels = {}
    class_names = []               # 完整动作类别名单（含 0 段的动作，供柱状图占位）
    for d in docs:
        for c in d.get("meta", {}).get("class_names", []):
            if c not in class_names:
                class_names.append(c)
        if not d["approved"]:          # 标签统计只认已批准的
            continue
        for s in d["segments"]:
            info = labels.setdefault(s["label"], {"count": 0, "duration": 0.0})
            info["count"] += 1
            info["duration"] += max(0.0, s["end"] - s["start"])
    for info in labels.values():
        info["duration"] = round(info["duration"], 1)
    return {
        "total_videos": total,
        "labeled_videos": sum(1 for d in docs if d["segments"]),  # 只算真产出了段的
        "approved_videos": sum(1 for d in docs if d["approved"]),
        "labels": labels,
        "class_names": class_names,
    }


def make_server(batch_dir, port=8765):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # 静默访问日志
            pass

        def _json(self, obj, status=200):
            body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _safe_video_path(self, rel):
            rel = unquote(rel).replace("\\", "/")
            norm = posixpath.normpath(rel)
            if norm.startswith("..") or posixpath.isabs(norm):
                return None
            path = os.path.join(batch_dir, *norm.split("/"))
            real = os.path.realpath(path)
            if not real.startswith(os.path.realpath(batch_dir)):
                return None
            return real if os.path.isfile(real) else None

        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path == "/":
                with open(_HTML_PATH, "rb") as f:
                    body = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif parsed.path == "/api/videos":
                docs = store.list_docs(batch_dir)
                items = []
                for d in docs:
                    # Calculate conf_tier: highest level with non-empty segments
                    conf_tier = None
                    if "segment_levels" in d and d["segment_levels"]:
                        for level in [0.9, 0.8, 0.7, 0.6, 0.5]:
                            key = f"{level:.1f}"
                            if key in d["segment_levels"] and len(d["segment_levels"][key]) > 0:
                                conf_tier = key
                                break
                    items.append({
                        "rel_path": d["rel_path"],
                        "approved": d["approved"],
                        "duration": d["duration"],
                        "n_segments": len(d["segments"]),
                        "class_names": d.get("meta", {}).get("class_names", []),
                        "conf_tier": conf_tier,
                    })
                self._json(items)
            elif parsed.path == "/api/stats":
                self._json(compute_stats(batch_dir))
            elif parsed.path == "/api/doc":
                rel = parse_qs(parsed.query).get("rel", [""])[0]
                try:
                    self._json(store.load_doc(batch_dir, rel))
                except FileNotFoundError:
                    self._json({"error": "not found"}, 404)
            elif parsed.path.startswith("/video/"):
                self._serve_video(parsed.path[len("/video/"):])
            else:
                self._json({"error": "not found"}, 404)

        def do_POST(self):
            parsed = urlparse(self.path)
            if parsed.path != "/api/doc":
                self._json({"error": "not found"}, 404)
                return
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length))
            try:
                doc = store.update_doc(
                    batch_dir, payload["rel_path"],
                    segments=payload.get("segments"),
                    action=payload["action"])
                self._json(doc)
            except FileNotFoundError:
                self._json({"error": "not found"}, 404)
            except (KeyError, ValueError) as e:
                self._json({"error": str(e)}, 400)

        def _serve_video(self, rel):
            path = self._safe_video_path(rel)
            if path is None:
                self._json({"error": "bad path"}, 404)
                return
            size = os.path.getsize(path)
            rng = self.headers.get("Range")
            start, end = 0, size - 1
            status = 200
            if rng and rng.startswith("bytes="):
                try:
                    spec = rng[len("bytes="):].split("-")
                    if spec[0]:
                        start = int(spec[0])
                        if spec[1]:
                            end = min(int(spec[1]), size - 1)
                    elif spec[1]:                       # bytes=-N（末尾 N 字节）
                        start = max(0, size - int(spec[1]))
                    status = 206
                except ValueError:
                    start, end, status = 0, size - 1, 200  # 畸形 Range 当作整文件
                if status == 206 and (start >= size or end < start):
                    self.send_response(416)              # 不可满足的范围
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
            length = end - start + 1
            self.send_response(status)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(length))
            if status == 206:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            with open(path, "rb") as f:
                f.seek(start)
                remaining = length
                while remaining > 0:
                    chunk = f.read(min(1 << 16, remaining))
                    if not chunk:
                        break
                    try:
                        self.wfile.write(chunk)
                    except (ConnectionAbortedError, BrokenPipeError, ConnectionResetError):
                        return                       # 浏览器拖动会中断连接，正常
                    remaining -= len(chunk)

    return _ExclusiveServer(("127.0.0.1", port), Handler)


class _ExclusiveServer(ThreadingHTTPServer):
    # 关掉端口复用：端口被占时第二个进程 bind 直接报错，而不是在 Windows 上静默串到旧服务器。
    allow_reuse_address = False


def compute_stats_multi(dirs):
    """compute_stats 的多目录版本：把各 dir 的统计汇总成一份。"""
    total = 0
    for d in dirs:
        for root, sub, files in os.walk(d):
            sub[:] = [s for s in sub if not s.startswith(".")]
            total += sum(1 for f in files if f.lower().endswith(VIDEO_EXTS))
    labeled_videos, approved_videos = 0, 0
    labels, class_names = {}, []
    for d in dirs:
        for doc in store.list_docs(d):
            for c in doc.get("meta", {}).get("class_names", []):
                if c not in class_names:
                    class_names.append(c)
            if doc["segments"]:
                labeled_videos += 1
            if doc["approved"]:
                approved_videos += 1
            if not doc["approved"]:
                continue
            for s in doc["segments"]:
                info = labels.setdefault(s["label"], {"count": 0, "duration": 0.0})
                info["count"] += 1
                info["duration"] += max(0.0, s["end"] - s["start"])
    for info in labels.values():
        info["duration"] = round(info["duration"], 1)
    return {
        "total_videos": total,
        "labeled_videos": labeled_videos,
        "approved_videos": approved_videos,
        "labels": labels,
        "class_names": class_names,
    }


def make_multi_server(dirs, port=8765):
    """review_app.html 不用改一行：把多个物理目录的视频合并成一份列表喂给它。

    用于"许多散落的 other/ 目录，一次浏览器会话全部审完"这种场景。每个视频的
    API rel_path 前面加一段"该目录相对公共根的路径"作命名空间，例如
    "ft-record/sorted-0629/other/clip1.mp4"；每个物理目录自己的 .autopresort/
    仍由 store.py 原样管理，store.py 本身对这层命名空间一无所知。

    conf_tier 固定回 "0.9"：这条工作流没有模型预标注，review_app.html 的侧栏
    默认只勾选 0.9 档，回真实的 null 会导致视频默认被过滤到看不见；这里只是
    借用现成的 UI 开关，不代表真的有 0.9 置信度。
    """
    norm_dirs = [os.path.realpath(d) for d in dirs]
    common = os.path.commonpath(norm_dirs) if len(norm_dirs) > 1 else os.path.dirname(norm_dirs[0])
    from auto_presort.other_split import build_prefixes, resolve_prefix
    prefixes = build_prefixes(common, norm_dirs)  # [(prefix, physical_dir)], 最长前缀优先

    def resolve(rel):
        return resolve_prefix(prefixes, rel)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _json(self, obj, status=200):
            body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _safe_video_path(self, physical_dir, rel):
            norm = posixpath.normpath(rel)
            if norm.startswith("..") or posixpath.isabs(norm):
                return None
            path = os.path.join(physical_dir, *norm.split("/"))
            real = os.path.realpath(path)
            if not real.startswith(os.path.realpath(physical_dir)):
                return None
            return real if os.path.isfile(real) else None

        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path == "/":
                with open(_HTML_PATH, "rb") as f:
                    body = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif parsed.path == "/api/videos":
                items = []
                for prefix, d in prefixes:
                    for doc in store.list_docs(d):
                        items.append({
                            "rel_path": f"{prefix}/{doc['rel_path']}",
                            "approved": doc["approved"],
                            "duration": doc["duration"],
                            "n_segments": len(doc["segments"]),
                            "class_names": doc.get("meta", {}).get("class_names", []),
                            "conf_tier": "0.9",
                        })
                items.sort(key=lambda it: it["rel_path"])
                self._json(items)
            elif parsed.path == "/api/stats":
                self._json(compute_stats_multi([d for _, d in prefixes]))
            elif parsed.path == "/api/doc":
                rel = parse_qs(parsed.query).get("rel", [""])[0]
                physical_dir, local = resolve(unquote(rel))
                if physical_dir is None:
                    self._json({"error": "not found"}, 404)
                    return
                try:
                    self._json(store.load_doc(physical_dir, local))
                except FileNotFoundError:
                    self._json({"error": "not found"}, 404)
            elif parsed.path.startswith("/video/"):
                physical_dir, local = resolve(unquote(parsed.path[len("/video/"):]))
                if physical_dir is None:
                    self._json({"error": "bad path"}, 404)
                    return
                self._serve_video(physical_dir, local)
            else:
                self._json({"error": "not found"}, 404)

        def do_POST(self):
            parsed = urlparse(self.path)
            if parsed.path != "/api/doc":
                self._json({"error": "not found"}, 404)
                return
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length))
            physical_dir, local = resolve(payload.get("rel_path", ""))
            if physical_dir is None:
                self._json({"error": "not found"}, 404)
                return
            try:
                doc = store.update_doc(
                    physical_dir, local,
                    segments=payload.get("segments"),
                    action=payload["action"])
                self._json(doc)
            except FileNotFoundError:
                self._json({"error": "not found"}, 404)
            except (KeyError, ValueError) as e:
                self._json({"error": str(e)}, 400)

        def _serve_video(self, physical_dir, rel):
            path = self._safe_video_path(physical_dir, rel)
            if path is None:
                self._json({"error": "bad path"}, 404)
                return
            size = os.path.getsize(path)
            rng = self.headers.get("Range")
            start, end = 0, size - 1
            status = 200
            if rng and rng.startswith("bytes="):
                try:
                    spec = rng[len("bytes="):].split("-")
                    if spec[0]:
                        start = int(spec[0])
                        if spec[1]:
                            end = min(int(spec[1]), size - 1)
                    elif spec[1]:                       # bytes=-N（末尾 N 字节）
                        start = max(0, size - int(spec[1]))
                    status = 206
                except ValueError:
                    start, end, status = 0, size - 1, 200  # 畸形 Range 当作整文件
                if status == 206 and (start >= size or end < start):
                    self.send_response(416)              # 不可满足的范围
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
            length = end - start + 1
            self.send_response(status)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(length))
            if status == 206:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            with open(path, "rb") as f:
                f.seek(start)
                remaining = length
                while remaining > 0:
                    chunk = f.read(min(1 << 16, remaining))
                    if not chunk:
                        break
                    try:
                        self.wfile.write(chunk)
                    except (ConnectionAbortedError, BrokenPipeError, ConnectionResetError):
                        return                       # 浏览器拖动会中断连接，正常
                    remaining -= len(chunk)

    return _ExclusiveServer(("127.0.0.1", port), Handler)
