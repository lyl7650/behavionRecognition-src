"""Local web demo: Qwen3-VL-8B vs Qwen3-VL-32B labels, dual-timeline review.

Modeled on scripts.presort.xclip_review (the :8766 page). Serves <batch_dir>
videos plus .vlm-<tag>/ docs on 127.0.0.1. Deliberately NO agreement stats
against the dashboard/repvit labels (user: those are not ground truth).

Tabs:
  1. batch comparison (dual timelines from .vlm-<tag>/ docs)
  2. playground: arbitrary local video + editable prompt + ad-hoc
     Qwen3-VL inference (scripts.vlm_label.vlm_playground, results not
     persisted)

Endpoints:
  GET  /                    -> review page (single HTML, tabs)
  GET  /api/data            -> aggregated JSON for both models
                               (?extra=tag1,tag2 -> dynamic x_<tag>_* panels)
  GET  /api/tags            -> every .vlm-<tag> dir usable as a comparison panel
  GET  /video/<rel>         -> batch-dir video stream (HTTP Range)
  GET  /api/video?path=<abs>-> arbitrary local video stream (playground)
  GET  /api/vlm_status      -> playground engine status + default prompt
  POST /api/vlm_infer       -> ad-hoc inference (json body, see playground)
  POST /api/prompt_run      -> launch one custom-prompt batch round
                               (prompt/model/prompt_mode/tag/limit/... body;
                                one at a time; writes .vlm-<tag>/ + archive)
  GET  /api/prompt_run_status -> progress of that round + prompt archives
  GET  /api/prompt_archive?tag= -> raw archived prompt text

Usage:
  python -m scripts.vlm_label.vlm_review --dir <batch_dir> [--port 8767]
"""
import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.parse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HTML_PATH = os.path.join(PROJECT_ROOT, "scripts", "vlm_label", "vlm_review.html")
LABELS = ["resting", "active", "eating", "drinking", "grooming",
          "digging", "bellyup", "none", "unknown"]
# degenerate ~0.1s clips (3-frame junk) carry no reviewable timeline and are
# dropped from the page list entirely; the AP "first 50" set backfills from
# the clips after them so it always covers 50 real videos
DUR_MIN = 0.5
# user-editable prompt archives (one .txt per prompt-experiment round)
PROMPT_ARCHIVE_DIR = ".vlm-prompts"
# tag guard: keeps ?extra= and POST /api/prompt_run from escaping batch_dir
TAG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


def _load_doc_dir(d):
    docs, manifest = {}, None
    if not os.path.isdir(d):
        return docs, manifest
    for fn in sorted(os.listdir(d)):
        if not fn.endswith(".json"):
            continue
        with open(os.path.join(d, fn), encoding="utf-8") as f:
            data = json.load(f)
        if fn == "manifest.json":
            manifest = data
        else:
            docs[data.get("rel_path", fn)] = data
    return docs, manifest


def _load_ap_doc(ap_dir, rel, ap_level):
    """Load one .autopresort doc (autopresort-2 schema) for a page rel.

    Autopresort keeps per-level segment lists in segment_levels; pick the
    requested level (nearest available as fallback). Gaps carry a third
    element (uncertain/boundary) that the page renders as gray bands.
    """
    flat = rel.replace("/", "__") + ".json"
    path = os.path.join(ap_dir, flat)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            doc = json.load(f)
    except (OSError, ValueError):
        return None
    levels = doc.get("segment_levels") or {}
    segs = levels.get(ap_level)
    if segs is None and levels:
        try:
            nearest = min(levels, key=lambda k: abs(float(k) - float(ap_level)))
            segs = levels[nearest]
        except (TypeError, ValueError):
            segs = None
    if segs is None:
        segs = doc.get("segments") or []
    return {"segments": segs, "gaps": doc.get("gaps") or []}


def build_data(batch_dir, a_name, b_name, limit=0, ap_dir="", ap_level="0.5",
               ap_limit=0, c_name="", d_name="",
               old_a_name="", old_b_name="",
               v4_a_name="", v4_b_name="", extra_names=()):
    a_docs, a_man = _load_doc_dir(os.path.join(batch_dir, a_name))
    b_docs, b_man = _load_doc_dir(os.path.join(batch_dir, b_name))
    c_docs, c_man = _load_doc_dir(os.path.join(batch_dir, c_name)) if c_name else ({}, None)
    d_docs, d_man = _load_doc_dir(os.path.join(batch_dir, d_name)) if d_name else ({}, None)
    # legacy (5-class, v2 prompt) dirs -> per-video o_a_/o_b_ fields + statuses
    oa_docs, oa_man = (_load_doc_dir(os.path.join(batch_dir, old_a_name))
                       if old_a_name else ({}, None))
    ob_docs, ob_man = (_load_doc_dir(os.path.join(batch_dir, old_b_name))
                       if old_b_name else ({}, None))
    # compact 9-class prompt (schema v4) dirs -> per-video v4_a_/v4_b_ fields
    v4a_docs, v4a_man = (_load_doc_dir(os.path.join(batch_dir, v4_a_name))
                         if v4_a_name else ({}, None))
    v4b_docs, v4b_man = (_load_doc_dir(os.path.join(batch_dir, v4_b_name))
                         if v4_b_name else ({}, None))
    # dynamic comparison panels (prompt-experiment rounds): each extra tag
    # loads .vlm-<tag>/ and lands in per-video x_<tag>_* fields. The page
    # requests these explicitly (?extra=t1,t2) -- nothing is auto-attached,
    # so the default view stays exactly the fixed panels.
    ex_pairs = []
    for tag in extra_names:
        ed, em = _load_doc_dir(os.path.join(batch_dir, f".vlm-{tag}"))
        ex_pairs.append((tag, ed, em))
    all_rels = (set(a_docs) | set(b_docs) | set(oa_docs) | set(ob_docs)
                | set(v4a_docs) | set(v4b_docs))
    for _tag, ed, _em in ex_pairs:
        all_rels |= set(ed)
    rels = sorted(all_rels)
    if limit:
        rels = rels[:limit]

    def _dur_for(r):
        vals = [(a_docs.get(r) or {}).get("duration") or 0.0,
                (b_docs.get(r) or {}).get("duration") or 0.0,
                (oa_docs.get(r) or {}).get("duration") or 0.0,
                (ob_docs.get(r) or {}).get("duration") or 0.0,
                (v4a_docs.get(r) or {}).get("duration") or 0.0,
                (v4b_docs.get(r) or {}).get("duration") or 0.0]
        for _tag, ed, _em in ex_pairs:
            vals.append((ed.get(r) or {}).get("duration") or 0.0)
        return max(vals)

    rels = [r for r in rels if _dur_for(r) >= DUR_MIN]

    videos = []
    for rel in rels:
        da, db = a_docs.get(rel), b_docs.get(rel)
        dc, dd = c_docs.get(rel), d_docs.get(rel)
        doa, dob = oa_docs.get(rel), ob_docs.get(rel)
        dv4a, dv4b = v4a_docs.get(rel), v4b_docs.get(rel)
        dur = _dur_for(rel)
        v = {
            "rel": rel,
            "source_id": (da or db or doa or dob or dv4a or dv4b or {}).get("source_id"),
            "duration": round(dur, 3),
            "a_segments": (da or {}).get("segments", []),
            "a_gaps": (da or {}).get("gaps", []),
            "a_windows": (da or {}).get("windows", []),
            "b_segments": (db or {}).get("segments", []),
            "b_gaps": (db or {}).get("gaps", []),
            "b_windows": (db or {}).get("windows", []),
            "c_segments": (dc or {}).get("segments", []),
            "c_gaps": (dc or {}).get("gaps", []),
            "c_windows": (dc or {}).get("windows", []),
            "d_segments": (dd or {}).get("segments", []),
            "d_gaps": (dd or {}).get("gaps", []),
            "d_windows": (dd or {}).get("windows", []),
            "o_a_segments": (doa or {}).get("segments", []),
            "o_a_gaps": (doa or {}).get("gaps", []),
            "o_a_windows": (doa or {}).get("windows", []),
            "o_b_segments": (dob or {}).get("segments", []),
            "o_b_gaps": (dob or {}).get("gaps", []),
            "o_b_windows": (dob or {}).get("windows", []),
            "v4_a_segments": (dv4a or {}).get("segments", []),
            "v4_a_gaps": (dv4a or {}).get("gaps", []),
            "v4_a_windows": (dv4a or {}).get("windows", []),
            "v4_b_segments": (dv4b or {}).get("segments", []),
            "v4_b_gaps": (dv4b or {}).get("gaps", []),
            "v4_b_windows": (dv4b or {}).get("windows", []),
            "ap_segments": [],
            "ap_gaps": [],
        }
        for tag, ed, _em in ex_pairs:
            d = ed.get(rel) or {}
            v[f"x_{tag}_segments"] = d.get("segments", [])
            v[f"x_{tag}_gaps"] = d.get("gaps", [])
            v[f"x_{tag}_windows"] = d.get("windows", [])
        videos.append(v)

    # attach .autopresort results (third panel) to the first ap_limit videos
    ap_n = 0
    if ap_dir and os.path.isdir(ap_dir):
        for v in (videos[:ap_limit] if ap_limit else videos):
            doc = _load_ap_doc(ap_dir, v["rel"], ap_level)
            if doc:
                v["ap_segments"] = doc["segments"]
                v["ap_gaps"] = doc["gaps"]
                ap_n += 1

    def status(man, docs, done_flag):
        if man is None and not docs:
            return None
        return {
            "engine": (man or {}).get("inference_engine"),
            "n_videos": len(docs),
            # counted from docs so resume reruns (which skip finished clips
            # and rewrite the manifest with n_windows=0) stay accurate
            "n_windows": sum(len(d.get("windows") or [])
                             for d in docs.values()) or (man or {}).get("n_windows"),
            "label_counts": (man or {}).get("label_counts"),
            "incomplete": bool((man or {}).get("incomplete")),
            "finished_at": (man or {}).get("finished_at"),
            "done": done_flag,
        }

    a_done = a_man is not None
    b_done = b_man is not None
    ap_status = None
    if ap_dir:
        ap_status = {
            "engine": f"repvit+GRU（autopresort-2，segment_level={ap_level}）",
            "n_videos": ap_n,
            "done": True,
        }
    ex_status = {f"x_{tag}": status(em, ed, em is not None)
                 for tag, ed, em in ex_pairs}
    return {
        "labels": LABELS,
        "n_videos": len(videos),
        "extra_tags": [tag for tag, _ed, _em in ex_pairs],
        "status": {
            "a": status(a_man, a_docs, a_done),
            "b": status(b_man, b_docs, b_done),
            "c": status(c_man, c_docs, c_man is not None) if c_name else None,
            "d": status(d_man, d_docs, d_man is not None) if d_name else None,
            "oa": status(oa_man, oa_docs, oa_man is not None),
            "ob": status(ob_man, ob_docs, ob_man is not None),
            "va": status(v4a_man, v4a_docs, v4a_man is not None),
            "vb": status(v4b_man, v4b_docs, v4b_man is not None),
            "ap": ap_status,
            **ex_status,
        },
        "videos": videos,
    }


# ---------------- prompt-experiment batch runner ----------------
# One background task at a time. Launches vlm_label as a subprocess with
# --prompt-file (same code path as the watchdog batch runs), so every round
# is resume-safe and results land in a fresh .vlm-<tag>/ dir.
_PEXP_LOCK = threading.Lock()
_PEXP = {"proc": None, "meta": None}


def _sanitize_tag(tag):
    tag = (tag or "").strip()
    if not tag:
        return ""
    return tag if TAG_RE.match(tag) else None


def list_tags(batch_dir):
    """Every .vlm-<tag> dir under batch_dir that holds at least one doc,
    with manifest status -- the pool the page's comparison selector picks
    from (fixed panels a/b/c/d/oa/ob/va/vb/ap are page-side constants)."""
    tags = []
    if not os.path.isdir(batch_dir):
        return {"tags": tags}
    for fn in sorted(os.listdir(batch_dir)):
        if not fn.startswith(".vlm-"):
            continue
        d = os.path.join(batch_dir, fn)
        if not os.path.isdir(d):
            continue
        tag = fn[len(".vlm-"):]
        if not TAG_RE.match(tag):
            continue
        n_docs = n_win = 0
        man = None
        for j in sorted(os.listdir(d)):
            if not j.endswith(".json"):
                continue
            if j == "manifest.json":
                try:
                    with open(os.path.join(d, j), encoding="utf-8") as f:
                        man = json.load(f)
                except (OSError, ValueError):
                    man = None
                continue
            n_docs += 1
            try:
                with open(os.path.join(d, j), encoding="utf-8") as f:
                    n_win += len((json.load(f) or {}).get("windows") or [])
            except (OSError, ValueError):
                pass
        if not n_docs:
            continue  # manifest-only dir (or empty) is not comparable
        tags.append({
            "tag": tag,
            "n_docs": n_docs, "n_windows": n_win,
            "engine": (man or {}).get("inference_engine"),
            "prompt_version": (man or {}).get("prompt_version"),
            "model_id": (man or {}).get("model_id"),
            "finished_at": (man or {}).get("finished_at"),
            "incomplete": bool((man or {}).get("incomplete")),
            "has_custom_prompt": "custom_prompt" in (man or {}),
        })
    return {"tags": tags}


def start_prompt_run(batch_dir, body):
    """Validate + launch one custom-prompt batch round. Returns
    (payload, http_status). Existing result dirs are never overwritten."""
    from scripts.vlm_label import vlm_playground as VP

    with _PEXP_LOCK:
        proc = _PEXP.get("proc")
        if proc is not None and proc.poll() is None:
            meta = _PEXP.get("meta") or {}
            return ({"ok": False,
                     "error": f"已有批量任务在跑（tag={meta.get('tag', '?')}），"
                              f"等它结束或重启 server 后再启动"}, 409)
        prompt = str(body.get("prompt") or "").strip()
        if not prompt:
            return ({"ok": False, "error": "prompt 不能为空"}, 400)
        model_key = str(body.get("model") or "8b")
        m = VP.MODELS.get(model_key)
        if not m:
            return ({"ok": False, "error": f"未知模型: {model_key}"}, 400)
        if not os.path.isfile(os.path.join(m["path"], ".download_done")):
            return ({"ok": False, "error": f"{m['desc']} 未下载完成"}, 400)
        prompt_mode = str(body.get("prompt_mode") or "full")
        if prompt_mode not in ("full", "compact"):
            prompt_mode = "full"
        try:
            limit = max(1, min(int(body.get("limit") or 50), 500))
        except (TypeError, ValueError):
            limit = 50
        tag = _sanitize_tag(body.get("tag"))
        if tag is None:
            return ({"ok": False,
                     "error": "tag 只能包含字母/数字/-/_，长度 ≤64"}, 400)
        if not tag:
            tag = "pexp-" + time.strftime("%m%d-%H%M%S")
        out_dir = os.path.join(batch_dir, f".vlm-{tag}")
        if os.path.isdir(out_dir) and any(
                fn.endswith(".json") for fn in os.listdir(out_dir)):
            return ({"ok": False,
                     "error": f"tag「{tag}」已有结果，换个 tag（已有结果永不覆盖）"}, 409)
        # VRAM pre-check (same policy as the playground: refuse readably
        # instead of OOM-ing whatever owns the GPU right now)
        free = VP._vram_free_gib()
        if free >= 0 and free < m["need_gib"]:
            return ({"ok": False,
                     "error": f"显存不足：空闲 {free:.1f} GiB，{m['desc']} 约需 "
                              f"{m['need_gib']:.0f} GiB；GPU 很可能被批量标注"
                              f"或其他任务占用，请稍后再试"}, 409)

        arch_dir = os.path.join(batch_dir, PROMPT_ARCHIVE_DIR)
        os.makedirs(arch_dir, exist_ok=True)
        arch_path = os.path.join(arch_dir, f"{tag}.txt")
        with open(arch_path, "w", encoding="utf-8", newline="\n") as f:
            f.write(prompt)

        venv_py = os.path.join(PROJECT_ROOT, ".venv-vlm", "Scripts", "python.exe")
        env = dict(os.environ)
        env.update({"HF_HUB_OFFLINE": "1", "PYTHONIOENCODING": "utf-8",
                    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
        os.makedirs(os.path.join(PROJECT_ROOT, "logs"), exist_ok=True)
        log_path = os.path.join(PROJECT_ROOT, "logs", f"vlm_pexp_{tag}.log")
        old = _PEXP.get("meta") or {}
        if old.get("_logf"):
            try:
                old["_logf"].close()
            except OSError:
                pass
        logf = open(log_path, "ab")
        cmd = [venv_py, "-m", "scripts.vlm_label.vlm_label",
               "--dir", batch_dir, "--tag", tag, "--model-id", m["path"],
               "--dtype", m["dtype"], "--limit", str(limit),
               "--prompt", prompt_mode, "--prompt-file", arch_path]
        if m.get("gpu_mem_gib"):
            cmd += ["--gpu-mem-gib", str(m["gpu_mem_gib"])]
        for k, flag in (("window_sec", "--window-sec"),
                        ("stride_sec", "--stride-sec"),
                        ("max_side", "--max-side")):
            if body.get(k):
                cmd += [flag, str(body[k])]
        if body.get("frames"):
            cmd += ["--frames", str(int(body["frames"]))]
        proc = subprocess.Popen(cmd, cwd=PROJECT_ROOT, env=env,
                                stdout=logf, stderr=subprocess.STDOUT)
        _PEXP["proc"] = proc
        _PEXP["meta"] = {"tag": tag, "model": model_key,
                         "model_desc": m["desc"],
                         "prompt_mode": prompt_mode, "limit": limit,
                         "archive": arch_path, "log": log_path,
                         "_logf": logf,
                         "pid": proc.pid,
                         "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                         "started_epoch": time.time()}
        return ({"ok": True, "tag": tag, "limit": limit,
                 "model_desc": m["desc"], "prompt_mode": prompt_mode,
                 "log": log_path}, 200)


def prompt_run_status(batch_dir):
    """Progress of the current/last prompt-experiment round + the prompt
    archive list (for reloading an old prompt into the editor)."""
    meta = dict(_PEXP.get("meta") or {})
    logf = meta.pop("_logf", None)
    proc = _PEXP.get("proc")
    running = bool(proc is not None and proc.poll() is None)
    rc = proc.poll() if proc is not None else None
    if proc is not None and not running and logf:
        try:
            logf.close()
        except OSError:
            pass
        _PEXP["meta"] = meta
    tag = meta.get("tag", "")
    progress = None
    if tag:
        out_dir = os.path.join(batch_dir, f".vlm-{tag}")
        n_docs = n_err = n_win = 0
        if os.path.isdir(out_dir):
            for fn in os.listdir(out_dir):
                if not fn.endswith(".json") or fn == "manifest.json":
                    continue
                n_docs += 1
                try:
                    with open(os.path.join(out_dir, fn), encoding="utf-8") as f:
                        d = json.load(f)
                    n_err += 1 if d.get("error") else 0
                    n_win += len(d.get("windows") or [])
                except (OSError, ValueError):
                    pass
        progress = {"tag": tag, "n_docs": n_docs, "n_errors": n_err,
                    "n_windows": n_win, "running": running}
    prompts = []
    arch_dir = os.path.join(batch_dir, PROMPT_ARCHIVE_DIR)
    if os.path.isdir(arch_dir):
        for fn in sorted(os.listdir(arch_dir)):
            if fn.endswith(".txt"):
                p = os.path.join(arch_dir, fn)
                try:
                    st = os.stat(p)
                    prompts.append({"tag": fn[:-4], "mtime": int(st.st_mtime),
                                    "size": st.st_size})
                except OSError:
                    pass
    return {"running": running, "returncode": rc, "meta": meta,
            "progress": progress, "prompts": prompts}


def make_handler(args):
    from http.server import BaseHTTPRequestHandler

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, body, ctype):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _video_response(self, vp):
            if not os.path.isfile(vp):
                self._send(404, b"video not found", "text/plain")
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

        def do_GET(self):
            path = urllib.parse.unquote(self.path)
            if path == "/" or path.startswith("/index"):
                with open(HTML_PATH, "rb") as f:
                    self._send(200, f.read(), "text/html; charset=utf-8")
            elif path.startswith("/api/vlm_status"):
                from scripts.vlm_label import vlm_playground
                st = vlm_playground.status()
                st["batch_dir"] = args.dir
                self._send(200, json.dumps(st, ensure_ascii=False).encode("utf-8"),
                           "application/json; charset=utf-8")
            elif path.startswith("/api/prompt_run_status"):
                self._send(200, json.dumps(prompt_run_status(args.dir),
                                           ensure_ascii=False).encode("utf-8"),
                           "application/json; charset=utf-8")
            elif path.startswith("/api/prompt_archive"):
                q = urllib.parse.parse_qs(path.split("?", 1)[1]) \
                    if "?" in path else {}
                tag = _sanitize_tag((q.get("tag") or [""])[0])
                if not tag:
                    self._send(400, b"bad tag", "text/plain")
                    return
                p = os.path.join(args.dir, PROMPT_ARCHIVE_DIR, tag + ".txt")
                if not os.path.isfile(p):
                    self._send(404, b"not found", "text/plain")
                    return
                with open(p, "rb") as f:
                    self._send(200, f.read(), "text/plain; charset=utf-8")
            elif path.startswith("/api/tags"):
                self._send(200, json.dumps(list_tags(args.dir),
                                           ensure_ascii=False).encode("utf-8"),
                           "application/json; charset=utf-8")
            elif path.startswith("/api/data"):
                m = re.search(r"limit=(\d+)", path)
                limit = int(m.group(1)) if m else args.limit
                ex = re.search(r"[?&]extra=([A-Za-z0-9_,%-]+)", path)
                extras = [t for t in (ex.group(1).split(",")
                                      if ex else [])
                          if TAG_RE.match(t)]
                data = build_data(args.dir, args.a_name, args.b_name, limit,
                                  ap_dir=args.ap_dir, ap_level=args.ap_level,
                                  ap_limit=args.ap_limit,
                                  c_name=args.c_name, d_name=args.d_name,
                                  old_a_name=args.old_a_name,
                                  old_b_name=args.old_b_name,
                                  v4_a_name=args.v4_a_name,
                                  v4_b_name=args.v4_b_name,
                                  extra_names=extras)
                self._send(200, json.dumps(data, ensure_ascii=False).encode("utf-8"),
                           "application/json; charset=utf-8")
            elif path.startswith("/api/video?"):
                q = urllib.parse.parse_qs(path.split("?", 1)[1])
                self._video_response(q.get("path", [""])[0])
            elif path.startswith("/video/"):
                rel = path[len("/video/"):].split("?")[0]
                self._video_response(os.path.join(args.dir, *rel.split("/")))
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self):
            if self.path.startswith("/api/prompt_run"):
                n = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(n) or b"{}")
                try:
                    res, code = start_prompt_run(args.dir, body)
                except Exception as e:  # noqa: BLE001 - report to the page
                    res, code = ({"ok": False,
                                  "error": f"{type(e).__name__}: {e}"}, 400)
                self._send(code, json.dumps(res, ensure_ascii=False)
                           .encode("utf-8"),
                           "application/json; charset=utf-8")
            elif self.path.startswith("/api/vlm_infer"):
                n = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(n) or b"{}")
                try:
                    from scripts.vlm_label import vlm_playground
                    res = vlm_playground.run_adhoc(
                        video_path=body.get("video_path", ""),
                        prompt=body.get("prompt", ""),
                        model_key=body.get("model", "8b"),
                        window_sec=float(body.get("window_sec", 6.0)),
                        stride_sec=float(body.get("stride_sec", 3.0)),
                        frames=int(body.get("frames", 8)),
                        max_side=int(body.get("max_side", 448)),
                        free_form=bool(body.get("free_form", False)),
                        max_new_tokens=int(body.get("max_new_tokens") or 0),
                        max_windows=int(body.get("max_windows") or 0),
                    )
                    self._send(200, json.dumps(res, ensure_ascii=False).encode("utf-8"),
                               "application/json; charset=utf-8")
                except Exception as e:  # noqa: BLE001 - report to the page
                    self._send(400, json.dumps(
                        {"ok": False, "error": f"{type(e).__name__}: {e}"},
                        ensure_ascii=False).encode("utf-8"),
                        "application/json; charset=utf-8")
            else:
                self._send(404, b"not found", "text/plain")

        def log_message(self, *a):
            pass

    return Handler


def serve(dir_path, port=8767, a_name=".vlm-qwen3vl-8b", b_name=".vlm-qwen3vl-32b",
          limit=0, in_thread=False, ap_dir="", ap_level="0.5", ap_limit=50,
          c_name="", d_name="",
          old_a_name=".vlm-qwen3vl-8b", old_b_name=".vlm-qwen3vl-32b",
          v4_a_name=".vlm-qwen3vl-8b-v4", v4_b_name=".vlm-qwen3vl-32b-v4"):
    from http.server import ThreadingHTTPServer

    args = argparse.Namespace(dir=dir_path, a_name=a_name, b_name=b_name,
                              limit=limit, ap_dir=ap_dir, ap_level=ap_level,
                              ap_limit=ap_limit, c_name=c_name, d_name=d_name,
                              old_a_name=old_a_name, old_b_name=old_b_name,
                              v4_a_name=v4_a_name, v4_b_name=v4_b_name)
    srv = ThreadingHTTPServer(("127.0.0.1", port), make_handler(args))
    srv.daemon_threads = True
    if in_thread:
        t = threading.Thread(target=srv.serve_forever, daemon=True)
        t.start()
        return srv
    srv.serve_forever()
    return srv


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--port", type=int, default=8767)
    ap.add_argument("--a-name", default=".vlm-qwen3vl-8b")
    ap.add_argument("--b-name", default=".vlm-qwen3vl-32b")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ap-dir", default="",
                    help=".autopresort dir for the third comparison panel")
    ap.add_argument("--ap-level", default="0.5",
                    help="autopresort segment_level to display")
    ap.add_argument("--ap-limit", type=int, default=50,
                    help="attach autopresort docs to the first N videos")
    ap.add_argument("--c-name", default="",
                    help="YOLO-crop round doc dir for model c (e.g. .vlm-qwen3vl-8b-crop)")
    ap.add_argument("--d-name", default="",
                    help="YOLO-crop round doc dir for model d (e.g. .vlm-qwen3vl-32b-crop)")
    ap.add_argument("--old-a-name", default=".vlm-qwen3vl-8b",
                    help="legacy 5-class (v2 prompt) doc dir for old-model-a panel")
    ap.add_argument("--old-b-name", default=".vlm-qwen3vl-32b",
                    help="legacy 5-class (v2 prompt) doc dir for old-model-b panel")
    ap.add_argument("--v4-a-name", default=".vlm-qwen3vl-8b-v4",
                    help="compact 9-class (schema v4) doc dir for model a panel")
    ap.add_argument("--v4-b-name", default=".vlm-qwen3vl-32b-v4",
                    help="compact 9-class (schema v4) doc dir for model b panel")
    a = ap.parse_args(argv)

    print(f"serving {a.dir} on http://127.0.0.1:{a.port} "
          f"(a={a.a_name}, b={a.b_name}, c={a.c_name or 'off'}, "
          f"d={a.d_name or 'off'}, old={a.old_a_name}/{a.old_b_name}, "
          f"v4={a.v4_a_name}/{a.v4_b_name}, ap={a.ap_dir or 'off'})")
    serve(a.dir, a.port, a.a_name, a.b_name, a.limit, in_thread=False,
          ap_dir=a.ap_dir, ap_level=a.ap_level, ap_limit=a.ap_limit,
          c_name=a.c_name, d_name=a.d_name,
          old_a_name=a.old_a_name, old_b_name=a.old_b_name,
          v4_a_name=a.v4_a_name, v4_b_name=a.v4_b_name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
