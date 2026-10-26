"""Background worker: cut approved screener segments into label-sorted clips.

Started by POST /api/finish right after the review_state.json snapshot.
Nothing here is interactive and no GPU is needed; ffmpeg + ffprobe must be
on PATH.

Output root is derived from the directory context (species + scene, see
review_store.ensure_context) plus the recording date detected in the video
paths, i.e. exactly pipeline_config.sorted_dir(species, scene, date):

  dataset/raw_videos/<species>/<scene>-record/sorted-<date>/<label>/

which is the directory 3_run_crop reads. Clip length = the segment's own
end - start (variable, never a fixed window). Cutting reuses
auto_presort.apply_clips (precise re-encode: -ss before -i + libx264,
never -c copy) and verifies each clip's duration with ffprobe.

Clip naming: <source>-<start>-<end>.mp4. Every source in an aws_backup
batch is named video.mp4, so the bare basename collides; the source token
is the record_uuid prefix (greppable in the G:\\aws_backup path) with a
sha1-of-rel_path fallback. abnormal_crop_video.source_key strips the
trailing -<start>-<end>, so train/val rebalance still groups clips by
their source video.

Date rule: the single distinct date= found in approved videos' rel_paths;
on mixed dates the most frequent one wins (with a warning); with no
date= token at all, today is used (with a warning).
"""
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from collections import Counter
from datetime import datetime

from auto_presort.apply_clips import build_ffmpeg_cmd, duration_ok, source_id_for
from scripts.pipeline import pipeline_config as pc
from scripts.pipeline.lineage.store import record_derivation, record_input_if_root
from scripts.vlm_screener import review_store as store

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATE_RE = re.compile(r"date=(\d{4}-\d{2}-\d{2})")
MAX_WARNINGS = 50

# Screener labels are verb forms; the training dirs use the class names from
# pipeline_config.SPECIES (pet: eat/drink). Labels with no class dir in that
# species (unknown, defecating) keep their own name so the clips stay visible
# under the sorted dir instead of being dropped.
LABEL_DIR_ALIASES = {"pet": {"eating": "eat", "drinking": "drink"}}


def label_dir_name(species, label):
    return LABEL_DIR_ALIASES.get(species, {}).get(label, label)


_state = {"running": False, "target": None, "out_dir": None, "date": None,
          "done": 0, "total": 0, "current": None, "written": 0,
          "by_label": {}, "warnings": [], "error": None,
          "started_at": None, "finished_at": None}


def log(msg):
    print(f"[apply-clips] {msg}", flush=True)


def status():
    return dict(_state)


def source_token(rel_path):
    """record_uuid prefix (greppable in G:\\aws_backup paths) or sha1 fallback."""
    m = re.search(r"record_uuid=([0-9a-fA-F-]{36})", rel_path.replace("\\", "/"))
    if m:
        return m.group(1)[:8]
    return source_id_for(rel_path)


def clip_name(rel_path, start, end):
    return f"{source_token(rel_path)}-{round(start, 3):g}-{round(end, 3):g}.mp4"


def detect_date(target):
    """(date, note) from the approved videos' rel_paths; see module docstring."""
    dates = Counter()
    for doc in store.list_docs(target).values():
        if not doc.get("approved"):
            continue
        m = DATE_RE.search(doc.get("rel_path", ""))
        if m:
            dates[m.group(1)] += 1
    if len(dates) == 1:
        return next(iter(dates)), None
    if len(dates) > 1:
        (date, _n), = dates.most_common(1)
        return date, ("mixed recording dates: "
                      + ", ".join(f"{d}x{n}" for d, n in sorted(dates.items()))
                      + f"; using {date}")
    return datetime.now().strftime("%Y-%m-%d"), \
        "no date= token in video paths; using today"


def output_dir(target):
    """Abs path of sorted_dir(species, scene, date) for this target."""
    ctx = store.require_context(target)
    date, note = detect_date(target)
    out = pc.sorted_dir(ctx["species"], ctx["scene"], date)
    if not os.path.isabs(out):
        out = os.path.join(PROJECT_ROOT, out)
    return os.path.abspath(out), date, note


def plan(target):
    """(cuts, skipped_rel_paths). Only approved docs (Review-all gate)."""
    cuts, skipped = [], []
    for doc in sorted(store.list_docs(target).values(),
                      key=lambda d: d["rel_path"]):
        if not doc.get("approved"):
            skipped.append(doc["rel_path"])
            continue
        for seg in doc.get("segments", []):
            cuts.append({
                "rel_path": doc["rel_path"],
                "vid": doc.get("vid"),
                "label": seg["label"],
                "start": float(seg["start"]),
                "end": float(seg["end"]),
            })
    return cuts, skipped


def _source_file(target, cut):
    """Absolute path of the source video, or None if missing."""
    video = store.video_by_vid(target, cut["vid"]) if cut.get("vid") else None
    if video and os.path.isfile(video.get("path", "")):
        return video["path"]
    cand = os.path.join(target, *cut["rel_path"].replace("\\", "/").split("/"))
    return cand if os.path.isfile(cand) else None


def _probe_duration(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", path],
        capture_output=True, text=True)
    try:
        return float(out.stdout.strip())
    except ValueError:
        return -1.0


def _warn(state, msg):
    if len(state["warnings"]) < MAX_WARNINGS:
        state["warnings"].append(msg)
    elif len(state["warnings"]) == MAX_WARNINGS:
        state["warnings"].append(f"... further warnings suppressed")


def _worker(target, out_dir, cuts, tolerance, species):
    try:
        for i, cut in enumerate(cuts):
            _state["current"] = cut["rel_path"]
            _state["done"] = i
            if cut["end"] - cut["start"] <= 0:
                _warn(_state, f"non-positive duration: {cut['rel_path']} "
                              f"{cut['start']}-{cut['end']}")
                continue
            src = _source_file(target, cut)
            if src is None:
                _warn(_state, f"missing source video: {cut['rel_path']}")
                continue
            name = clip_name(cut["rel_path"], cut["start"], cut["end"])
            dst = os.path.join(out_dir, label_dir_name(species, cut["label"]),
                               name)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            cmd = build_ffmpeg_cmd(src, dst, cut["start"], cut["end"])
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0 or not os.path.isfile(dst):
                _warn(_state, f"ffmpeg failed: {name}: {proc.stderr[-300:]}")
                continue
            actual = _probe_duration(dst)
            expected = cut["end"] - cut["start"]
            if not duration_ok(expected, actual, tolerance):
                _warn(_state, f"duration mismatch: {name} "
                              f"expected={expected:.2f}s actual={actual:.2f}s")
                continue
            record_input_if_root(src)
            record_derivation(
                dst, src, operation="vlm_review_cut", label=cut["label"],
                parameters={"start": cut["start"], "end": cut["end"],
                            "review_target": os.path.abspath(target)},
            )
            _state["written"] += 1
            _state["by_label"][cut["label"]] = \
                _state["by_label"].get(cut["label"], 0) + 1
        _state["done"] = len(cuts)
        _state["current"] = None
        log(f"done: {_state['written']}/{len(cuts)} clips -> {out_dir} "
            f"({len(_state['warnings'])} warning(s))")
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        log(f"[ERROR] clip cutting: {e}")
        _state["error"] = str(e)
    finally:
        _state["running"] = False
        _state["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")


def start(target, tolerance=0.25):
    """Kick off the cutting thread; returns (ok, message)."""
    if _state["running"]:
        return False, "clip cutting already running"
    if not (shutil.which("ffmpeg") and shutil.which("ffprobe")):
        return False, "ffmpeg/ffprobe not found on PATH"
    try:
        out_dir, date, note = output_dir(target)
        species = store.require_context(target)["species"]
    except ValueError as exc:
        return False, str(exc)
    cuts, _skipped = plan(target)
    _state.update({"running": True, "target": os.path.abspath(target),
                   "out_dir": out_dir, "date": date, "done": 0,
                   "total": len(cuts), "current": None, "written": 0,
                   "by_label": {}, "warnings": ([note] if note else []),
                   "error": None,
                   "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                   "finished_at": None})
    if not cuts:
        _state.update(running=False,
                       finished_at=time.strftime("%Y-%m-%d %H:%M:%S"))
        return True, "no approved segments; nothing to cut"
    log(f"start: {len(cuts)} clip(s) -> {out_dir} (date={date})")
    t = threading.Thread(target=_worker,
                         args=(os.path.abspath(target), out_dir, cuts,
                               tolerance, species),
                         daemon=True)
    t.start()
    return True, f"{len(cuts)} clip(s) queued -> {out_dir}"


def main(argv=None):
    """Foreground re-cut without the page (repair / testing).

    Usage: python -m scripts.vlm_screener.apply_clips --target <dir>
    """
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--target", required=True,
                    help="screener target video dir (contains .vlm-screener/)")
    a = ap.parse_args(argv)
    ok, msg = start(a.target)
    print(msg)
    if not ok:
        return 2
    while status()["running"]:
        time.sleep(2)
    s = status()
    print(f"written={s['written']} by_label={s['by_label']} "
          f"warnings={len(s['warnings'])} error={s['error']}")
    return 1 if s["error"] else 0


if __name__ == "__main__":
    sys.exit(main())
