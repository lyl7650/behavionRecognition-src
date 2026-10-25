"""Local web demo: X-CLIP vs 2a dashboard labels, dual-timeline review.

Serves <batch_dir> videos + .autopresort/ + .xclip*/ docs on 127.0.0.1.
Endpoints:
  GET /                        -> review page (single HTML)
  GET /api/data?limit=10&level=0.9&xc=.xclip-p16  -> aggregated JSON
  GET /video/<rel_path>        -> video stream (HTTP Range supported)

Usage:
  python -m scripts.presort.xclip_review --dir <batch_dir> [--port 8766] [--limit 10]
"""
import argparse
import json
import os
import re
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HTML_PATH = os.path.join(PROJECT_ROOT, "scripts", "presort", "xclip_review.html")


def build_data(batch_dir, xc_name, level, limit):
    ap_dir = os.path.join(batch_dir, ".autopresort")
    xc_dir = os.path.join(batch_dir, xc_name)
    ap_docs = {}
    for fn in sorted(os.listdir(ap_dir)):
        if fn.endswith(".json") and fn != "manifest.json":
            with open(os.path.join(ap_dir, fn), encoding="utf-8") as f:
                d = json.load(f)
            ap_docs[d["rel_path"]] = d
    xc_docs = []
    for fn in sorted(os.listdir(xc_dir)):
        if fn.endswith(".json") and fn not in ("manifest.json", "compare_report.json"):
            with open(os.path.join(xc_dir, fn), encoding="utf-8") as f:
                xc_docs.append(json.load(f))
    if limit:
        xc_docs = xc_docs[:limit]

    step = 0.25
    labels = ["resting", "active", "eating", "drinking", "none"]
    videos, total_a, total_x = [], 0, 0
    for doc in xc_docs:
        rel = doc["rel_path"]
        ap_doc = ap_docs.get(rel)
        if ap_doc is None:
            continue
        duration = min(doc.get("duration", 0), ap_doc.get("duration", 0))
        ap_segs = ap_doc.get("segment_levels", {}).get(level) or ap_doc.get("segments", [])
        ap_gaps = ap_doc.get("gaps", [])

        def timeline(segs, gaps):
            n = max(1, int(round(duration / step)))
            labs = [None] * n
            for s in segs:
                i0 = max(0, int(s["start"] / step))
                i1 = min(n, max(i0 + 1, int(round(s["end"] / step))))
                for i in range(i0, i1):
                    labs[i] = s["label"]
            for g in gaps:
                i0 = max(0, int(g[0] / step))
                i1 = min(n, max(i0 + 1, int(round(g[1] / step))))
                for i in range(i0, i1):
                    if labs[i] is None:
                        labs[i] = "none"
            return [l if l in labels else "none" for l in labs]

        ap_labs = timeline(ap_segs, ap_gaps)
        xc_labs = timeline(doc.get("segments", []), doc.get("gaps", []))
        agree = sum(1 for a, b in zip(ap_labs, xc_labs) if a == b)
        total_a += agree
        total_x += len(ap_labs)
        videos.append({
            "rel": rel,
            "duration": round(duration, 3),
            "agree": round(agree / max(1, len(ap_labs)), 4),
            "ap_segments": ap_segs, "ap_gaps": ap_gaps,
            "xc_segments": doc.get("segments", []),
            "xc_gaps": doc.get("gaps", []),
            "xc_windows": doc.get("windows", []),
        })
    return {
        "xc_name": xc_name, "level": level, "step": step,
        "overall_agree": round(total_a / max(1, total_x), 4),
        "n_videos": len(videos), "videos": videos,
    }


def make_handler(args):
    from http.server import BaseHTTPRequestHandler

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, body, ctype):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/" or self.path.startswith("/index"):
                with open(HTML_PATH, "rb") as f:
                    self._send(200, f.read(), "text/html; charset=utf-8")
            elif self.path.startswith("/api/data"):
                m = re.search(r"limit=(\d+)", self.path)
                limit = int(m.group(1)) if m else args.limit
                m = re.search(r"level=([\d.]+)", self.path)
                level = m.group(1) if m else args.level
                m = re.search(r"xc=([.\w-]+)", self.path)
                xc = m.group(1) if m else args.xc_name
                data = build_data(args.dir, xc, level, limit)
                self._send(200, json.dumps(data, ensure_ascii=False).encode("utf-8"),
                           "application/json; charset=utf-8")
            elif self.path.startswith("/video/"):
                rel = self.path[len("/video/"):].split("?")[0]
                vp = os.path.join(args.dir, *rel.split("/"))
                if not os.path.isfile(vp):
                    self._send(404, b"not found", "text/plain")
                    return
                size = os.path.getsize(vp)
                rng = self.headers.get("Range")
                start, end = 0, size - 1
                if rng:
                    m = re.search(r"bytes=(\d+)-(\d*)", rng)
                    if m:
                        start = int(m.group(1))
                        if m.group(2):
                            end = min(int(m.group(2)), size - 1)
                with open(vp, "rb") as f:
                    f.seek(start)
                    chunk = f.read(end - start + 1)
                self.send_response(206 if rng else 200)
                self.send_header("Content-Type", "video/mp4")
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Content-Length", str(len(chunk)))
                if rng:
                    self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
                self.end_headers()
                self.wfile.write(chunk)
            else:
                self._send(404, b"not found", "text/plain")

        def log_message(self, *a):
            pass

    return Handler


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--level", default="0.9")
    ap.add_argument("--xc-name", default=".xclip-p16")
    a = ap.parse_args(argv)

    from http.server import ThreadingHTTPServer
    print(f"serving {a.dir} on http://127.0.0.1:{a.port} (xc={a.xc_name}, limit={a.limit})")
    ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(a)).serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
