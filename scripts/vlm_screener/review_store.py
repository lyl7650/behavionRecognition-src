"""Per-directory label store for the review workflow (2b-compatible docs).

Layout under <target_dir>/.vlm-screener/:
  videos.json        scanned list [{vid, rel_path, name, path, duration, status}]
  windows.json       {vid: {duration, windows: [{w, start, end, length, ts}]}}
  gates.json         {vid: {wNNN: {det_frames, has_pet, boxes}}}
  docs/<vid>.json    full review doc (2b-style segments/approved + combos)
  review_state.json  Finish snapshot (stats + finished_at)
  review-cache/      H.264 browser copies for non-native codecs

Conventions (mirroring auto_presort.store):
  update_doc(action="save") forces approved=False (any edit clears approval);
  update_doc(action="approve") only sets approved=True, never touches segments.
  vid = sha1(rel_path)[:12] so the whole target dir can move around freely.
All writes are atomic (tmp + os.replace).
"""
import hashlib
import json
import os
import re
import tempfile
import time

from auto_presort.review_server import (
    VIDEO_EXTS, _probe_video_codec, prepare_browser_video_overrides)

STORE_DIRNAME = ".vlm-screener"
# VLM inference only ever produces VLM_CLASS_NAMES; the screener never asks
# the model about the MANUAL_CLASS_NAMES. The manual labels exist solely for
# human annotators in the page editor, and flow through save / stats / export
# / apply like any other segment label. "defecating" is both: rabbit prompts
# cannot emit it (manual only), pet prompts can (see prompts.PET_LABELS).
VLM_CLASS_NAMES = ["eating", "drinking", "active", "resting", "unknown"]
VLM_CLASS_NAMES_PET = VLM_CLASS_NAMES + ["defecating"]
MANUAL_CLASS_NAMES = ["grooming", "others", "defecating", "stand"]


def vlm_class_names(species):
    return VLM_CLASS_NAMES_PET if species == "pet" else VLM_CLASS_NAMES


def class_names(species):
    """Label list for one species: model labels, then manual-only ones."""
    vlm = vlm_class_names(species)
    return vlm + [c for c in MANUAL_CLASS_NAMES if c not in vlm]


SCHEMA_VERSION = "vlm-review-1"
BROWSER_PROBE_MAX = 2000
SCENE_RE = re.compile(r"^[\u3400-\u9fffA-Za-z0-9_]+$")
SCENES_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "config", "vlm_screener_scenes.json")


def data_dir(target):
    return os.path.join(target, STORE_DIRNAME)


def context(target):
    path = os.path.join(data_dir(target), "context.json")
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def scenes():
    if not os.path.isfile(SCENES_PATH):
        return []
    with open(SCENES_PATH, encoding="utf-8") as f:
        return json.load(f)["scenes"]


def ensure_context(target, species, scene=""):
    if species not in ("rabbit", "pet"):
        raise ValueError(f"invalid species: {species}")
    if not isinstance(scene, str):
        raise ValueError("scene must be text")
    scene = scene.strip()
    if scene and (len(scene) > 80 or not SCENE_RE.fullmatch(scene)):
        raise ValueError("scene must contain only Chinese, letters, digits or underscore")
    saved = context(target)
    if saved:
        if saved["species"] != species or (scene and saved["scene"] != scene):
            raise ValueError(f"directory is already assigned to {saved['species']} / {saved['scene']}")
        return saved
    # Legacy directory caches were all produced by the rabbit-only tool.
    if species != "rabbit" and any(os.path.exists(os.path.join(data_dir(target), name))
                                   for name in ("docs", "gates.json")):
        raise ValueError("existing unmarked review data belongs to rabbit")
    if not scene:
        return None
    saved = {"species": species, "scene": scene}
    _write_json(os.path.join(data_dir(target), "context.json"), saved)
    known = scenes()
    if scene not in known:
        _write_json(SCENES_PATH, {"scenes": sorted([*known, scene])})
    return saved


def require_context(target):
    saved = context(target)
    if not saved or not saved.get("scene"):
        raise ValueError("select a scene for this directory before inference or export")
    return saved


def _docs_dir(target):
    return os.path.join(data_dir(target), "docs")


def vid_for(rel_path):
    return hashlib.sha1(rel_path.replace("\\", "/").encode("utf-8")).hexdigest()[:12]


def _read_json(path, default):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def _write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=os.path.basename(path) + ".",
                               suffix=".tmp", dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        for attempt in range(21):
            try:
                os.replace(tmp, path)
                if attempt:
                    print(f"[review-store] replace succeeded after {attempt} "
                          f"retries: {path}", flush=True)
                break
            except PermissionError as exc:
                if os.name != "nt" or getattr(exc, "winerror", None) not in (5, 32) \
                        or attempt == 20:
                    raise
                if attempt == 0:
                    print(f"[review-store] replace blocked ({exc}); "
                          "retrying up to 10 seconds", flush=True)
                time.sleep(0.5)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def probe_duration(path):
    import cv2
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return 0.0
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return (n / fps) if (n > 0 and fps > 0) else 0.0


def scan_dir(target):
    """(Re)scan target recursively; reuse known durations, probe new files."""
    ddir = data_dir(target)
    old = {}
    vpath = os.path.join(ddir, "videos.json")
    if os.path.isfile(vpath):
        for v in load_videos(target):
            old[v["rel_path"]] = v
    videos = []
    for root, dirs, files in os.walk(target):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for name in sorted(files):
            if not name.lower().endswith(VIDEO_EXTS):
                continue
            full = os.path.join(root, name)
            rel = os.path.relpath(full, target).replace("\\", "/")
            prev = old.get(rel)
            duration = prev["duration"] if prev else probe_duration(full)
            if prev and prev.get("path") != full:
                prev = None  # path changed outside: refresh absolute path
            videos.append({
                "vid": vid_for(rel),
                "rel_path": rel,
                "name": name,
                "path": full,
                "duration": round(float(duration), 3),
                "status": prev["status"] if prev else "pending",
            })
    videos.sort(key=lambda v: v["rel_path"])
    _write_json(vpath, {"schema": "vlm-review-videos-1",
                        "root": os.path.abspath(target),
                        "scanned_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "videos": videos})
    return videos


def load_videos(target):
    path = os.path.join(data_dir(target), "videos.json")
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)["videos"]
    except FileNotFoundError:
        return []


def video_by_vid(target, vid):
    for v in load_videos(target):
        if v["vid"] == vid:
            return v
    return None


def set_status(target, vid, status):
    vids = load_videos(target)
    for v in vids:
        if v["vid"] == vid:
            v["status"] = status
            break
    else:
        raise KeyError(f"video missing from videos.json: {vid}")
    _write_json(os.path.join(data_dir(target), "videos.json"),
                {"schema": "vlm-review-videos-1",
                 "root": os.path.abspath(target),
                 "scanned_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                 "videos": vids})


def load_windows(target):
    return _read_json(os.path.join(data_dir(target), "windows.json"), {})


def save_windows(target, win):
    _write_json(os.path.join(data_dir(target), "windows.json"), win)


def load_gates(target):
    return _read_json(os.path.join(data_dir(target), "gates.json"), {})


def save_gates(target, gates):
    _write_json(os.path.join(data_dir(target), "gates.json"), gates)


def load_doc(target, vid):
    path = os.path.join(_docs_dir(target), f"{vid}.json")
    if not os.path.isfile(path):
        return None
    return _read_json(path, None)


def save_doc(target, doc):
    doc["schema_version"] = SCHEMA_VERSION
    ctx = context(target) or {}
    doc.setdefault("meta", {})["class_names"] = class_names(
        ctx.get("species", "rabbit"))
    _write_json(os.path.join(_docs_dir(target), f"{doc['vid']}.json"), doc)


def list_docs(target):
    out = {}
    ddir = _docs_dir(target)
    if os.path.isdir(ddir):
        for name in sorted(os.listdir(ddir)):
            if name.endswith(".json"):
                doc = _read_json(os.path.join(ddir, name), None)
                if doc and doc.get("vid"):
                    out[doc["vid"]] = doc
    return out


def update_doc(target, vid, segments, action):
    """2b semantics: any save clears approval; approve never touches segments."""
    doc = load_doc(target, vid)
    if doc is None:
        raise FileNotFoundError(vid)
    if action == "save":
        if segments is not None:
            doc["segments"] = segments
        doc["approved"] = False
    elif action == "approve":
        doc["approved"] = True
    else:
        raise ValueError(f"unknown action: {action}")
    save_doc(target, doc)
    return doc


def compute_stats(target):
    docs = list_docs(target)
    total = len(load_videos(target))
    labels = {}
    for doc in docs.values():
        if not doc.get("approved"):
            continue
        for s in doc.get("segments", []):
            info = labels.setdefault(s["label"], {"count": 0, "duration": 0.0})
            info["count"] += 1
            info["duration"] += max(0.0, s["end"] - s["start"])
    for info in labels.values():
        info["duration"] = round(info["duration"], 1)
    ctx = context(target) or {}
    return {
        "total_videos": total,
        "labeled_videos": sum(1 for d in docs.values() if d.get("segments")),
        "approved_videos": sum(1 for d in docs.values() if d.get("approved")),
        "labels": labels,
        "class_names": class_names(ctx.get("species", "rabbit")),
    }


def write_finish(target):
    provenance = require_context(target)
    state = {"status": "finished",
             **provenance,
             "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
             "stats": compute_stats(target)}
    _write_json(os.path.join(data_dir(target), "review_state.json"), state)
    return state


def export_jsonl(target):
    """One line per approved segment across all approved videos."""
    provenance = require_context(target)
    lines = []
    for doc in sorted(list_docs(target).values(), key=lambda d: d["rel_path"]):
        if not doc.get("approved"):
            continue
        for s in doc.get("segments", []):
            lines.append(json.dumps({
                **provenance,
                "vid": doc["vid"], "rel_path": doc["rel_path"],
                "duration": doc.get("duration"), "label": s["label"],
                "start": s["start"], "end": s["end"],
                "confidence": s.get("confidence"),
                "source": s.get("source", "human"),
            }, ensure_ascii=False))
    return ("\n".join(lines) + ("\n" if lines else "")).encode("utf-8")


def browser_video_overrides(target):
    """H.264 review copies for codecs browsers cannot decode (2b logic).

    Large dirs (>BROWSER_PROBE_MAX videos) skip the per-file codec probe
    entirely: probing 18k files on a cold USB/network drive costs hours and
    blocks /api/open. Big camera dumps are mp4/H.264 anyway; /api/video
    serves the originals directly there.
    """
    n = len(load_videos(target))
    if n > BROWSER_PROBE_MAX:
        print(f"[review] {n} videos > {BROWSER_PROBE_MAX}: "
              f"skipping browser-copy probe (originals served as-is)",
              flush=True)
        return {}
    try:
        return prepare_browser_video_overrides(target)
    except RuntimeError as exc:
        print(f"[review] browser copy prep failed: {exc}", flush=True)
        return {}
