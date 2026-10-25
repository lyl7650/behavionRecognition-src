"""Stage 1: sample videos -> plan windows -> dump frames -> YOLO gate.

Inputs (hardcoded defaults per user spec):
  E-pool: dataset/raw_videos/rabbit/hkbr-record/records/2026-06-25 and
          .../2026-06-24  (recursive *.mp4) -> random 30
  G-pool: G:/aws_backup/2026-10-07/raw/rabbit/<NNNNNNNN>/video.mp4
          -> random 20

Outputs under <project>/.vlm-screener/:
  manifest.json   sampled 50 videos (path/source/vid), reusable forever
  windows.json    per-video duration + 6s/3s window plan (8 frames each)
  gates.json      per-window YOLO rabbit vote {det_frames, has_pet}
  frames/<vid>/wNNN_fK.jpg   resized frames (long side <= 448)

Resume-safe: every artifact skips if it already exists. Use --resample to
force a fresh sample draw (regenerates everything downstream).
"""
import argparse
import glob
import hashlib
import json
import os
import random
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, ".vlm-screener")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")

E_DIRS = [
    os.path.join(PROJECT_ROOT, "dataset", "raw_videos", "rabbit",
                 "hkbr-record", "records", "2026-06-25"),
    os.path.join(PROJECT_ROOT, "dataset", "raw_videos", "rabbit",
                 "hkbr-record", "records", "2026-06-24"),
]
G_DIR = r"G:\aws_backup\2026-10-07\raw\rabbit"
YOLO_WEIGHTS = (r"E:\ActionRecognition\detection-src\rabbit-detect\runs"
                r"\rabbit-yolo11n-0603\weights\rabbit-yolo11n-0603.pt")

WINDOW_SEC = 6.0
STRIDE_SEC = 3.0
N_FRAMES = 8
MAX_SIDE = 448
JPG_QUALITY = 88

VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv", ".webm")


def log(msg):
    line = f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        with open(os.path.join(LOG_DIR, "vlm_screener.log"), "a",
                  encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass


def vid_for(path):
    return hashlib.sha1(path.encode("utf-8")).hexdigest()[:12]


def list_e_videos(dirs):
    out = []
    for d in dirs:
        for root, _dirs2, files in os.walk(d):
            for fn in sorted(files):
                if fn.lower().endswith(VIDEO_EXTS):
                    out.append(os.path.join(root, fn))
    return sorted(out)


def list_g_videos(gdir):
    out = []
    for sub in sorted(os.listdir(gdir)):
        meta = os.path.join(gdir, sub, "metadata.json")
        for ext in VIDEO_EXTS:
            p = os.path.join(gdir, sub, f"video{ext}")
            if os.path.isfile(p):
                entry = {"path": p}
                if os.path.isfile(meta):
                    try:
                        with open(meta, encoding="utf-8") as f:
                            m = json.load(f)
                        entry["camera_meta"] = {
                            k: m.get(k) for k in
                            ("trigger_action", "max_confidence", "start_time",
                             "end_time", "device_id") if m.get(k) is not None}
                    except (OSError, ValueError):
                        pass
                out.append(entry)
                break
    return out


def probe_duration(path):
    import cv2
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return 0.0
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return (n / fps) if (n > 0 and fps > 0) else 0.0


def window_ranges(duration):
    if duration < WINDOW_SEC:
        return [(0.0, max(duration, 0.5))]
    out, t = [], 0.0
    while t + WINDOW_SEC <= duration + 1e-6:
        out.append((round(t, 3), WINDOW_SEC))
        t += STRIDE_SEC
    if not out:
        out.append((0.0, duration))
    return out


def sample_timestamps(start, length, n=N_FRAMES):
    return [start + (i + 0.5) * length / n for i in range(n)]


def stage_sample(n_e, n_g, seed, resample):
    mpath = os.path.join(DATA_DIR, "manifest.json")
    if os.path.isfile(mpath) and not resample:
        with open(mpath, encoding="utf-8") as f:
            m = json.load(f)
        log(f"sample: reuse manifest ({len(m['videos'])} videos, "
            f"seed={m.get('seed')})")
        return m
    e_pool = list_e_videos(E_DIRS)
    g_pool = list_g_videos(G_DIR)
    log(f"sample: pools e={len(e_pool)} g={len(g_pool)}")
    if len(e_pool) < n_e or len(g_pool) < n_g:
        log(f"[ERROR] pool too small: need e>={n_e} g>={n_g}")
        raise SystemExit(1)
    rng = random.Random(seed)
    e_pick = sorted(rng.sample(e_pool, n_e))
    g_pick = rng.sample(g_pool, n_g)
    videos = []
    for p in e_pick:
        videos.append({"vid": vid_for(p), "source": "e_hkbr", "path": p,
                       "name": os.path.basename(p)})
    for entry in g_pick:
        p = entry["path"]
        videos.append({"vid": vid_for(p), "source": "g_backup", "path": p,
                       "name": f"{os.path.basename(os.path.dirname(p))}.mp4",
                       "camera_meta": entry.get("camera_meta")})
    manifest = {"created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "seed": seed, "n_e": n_e, "n_g": n_g,
                "e_dirs": E_DIRS, "g_dir": G_DIR, "videos": videos}
    tmp = mpath + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    os.replace(tmp, mpath)
    log(f"sample: drew {len(videos)} videos (e={n_e} g={n_g}) "
        f"seed={seed} -> {mpath}")
    return manifest


def stage_windows(manifest):
    wpath = os.path.join(DATA_DIR, "windows.json")
    if os.path.isfile(wpath):
        with open(wpath, encoding="utf-8") as f:
            w = json.load(f)
        log(f"windows: reuse ({len(w['videos'])} videos, "
            f"{sum(len(v['windows']) for v in w['videos'])} windows)")
        return w
    videos = []
    durs = []
    for v in manifest["videos"]:
        dur = probe_duration(v["path"])
        if dur <= 0:
            log(f"  [WARN] unreadable duration, force 1s window: {v['path']}")
            dur = 1.0
        ranges = window_ranges(dur)
        durs.append(dur)
        wins = []
        for i, (start, length) in enumerate(ranges):
            wins.append({"w": f"w{i:03d}", "start": start, "end": None,
                         "length": length,
                         "ts": [round(t, 3) for t in
                                sample_timestamps(start, length)]})
        videos.append({"vid": v["vid"], "duration": round(dur, 3),
                       "windows": wins})
    durs.sort()
    n = len(durs)
    med = durs[n // 2] if n else 0
    log(f"windows: duration min={durs[0]:.1f}s med={med:.1f}s "
        f"max={durs[-1]:.1f}s total_windows="
        f"{sum(len(v['windows']) for v in videos)}")
    out = {"schema": "vlm-screener-windows-1", "videos": videos}
    tmp = wpath + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    os.replace(tmp, wpath)
    return out


def stage_frames(manifest, windows):
    import cv2
    import numpy as np
    path_by_vid = {v["vid"]: v["path"] for v in manifest["videos"]}
    t0 = time.time()
    n_new = 0
    for wv in windows["videos"]:
        vdir = os.path.join(DATA_DIR, "frames", wv["vid"])
        need = []
        for win in wv["windows"]:
            for k in range(N_FRAMES):
                fp = os.path.join(vdir, f"{win['w']}_f{k}.jpg")
                if not os.path.isfile(fp):
                    need.append((win, k, fp))
        if not need:
            continue
        os.makedirs(vdir, exist_ok=True)
        cap = cv2.VideoCapture(path_by_vid[wv["vid"]])
        if not cap.isOpened():
            log(f"  [ERROR] cannot open {path_by_vid[wv['vid']]}")
            continue
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        want = {}
        for win, k, fp in need:
            t = win["ts"][k]
            want.setdefault(max(0, min(int(round(t * fps)), n_total - 1)),
                            []).append((win, k, fp))
        got = {}
        i = 0
        last_needed = max(want) if want else -1
        while i <= last_needed:
            ok, frame = cap.read()
            if not ok:
                break
            if i in want:
                got[i] = frame
            i += 1
        cap.release()
        for win, k, fp in need:
            frame = got.get(max(0, min(int(round(win["ts"][k] * fps)),
                                       max(0, n_total - 1))))
            if frame is None:
                frame = next(iter(got.values()), None)
            if frame is None:
                continue
            h, w = frame.shape[:2]
            m = max(h, w)
            if m > MAX_SIDE:
                s = MAX_SIDE / m
                frame = cv2.resize(frame, (int(round(w * s)),
                                           int(round(h * s))),
                                   interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode(".jpg", frame,
                                   [cv2.IMWRITE_JPEG_QUALITY, JPG_QUALITY])
            if ok:
                tmp = fp + ".tmp"
                with open(tmp, "wb") as f:
                    f.write(buf.tobytes())
                os.replace(tmp, fp)
                n_new += 1
    log(f"frames: wrote {n_new} new jpgs in {time.time() - t0:.0f}s "
        f"(total windows "
        f"{sum(len(v['windows']) for v in windows['videos'])})")


def stage_gate(manifest, windows):
    gpath = os.path.join(DATA_DIR, "gates.json")
    if os.path.isfile(gpath):
        with open(gpath, encoding="utf-8") as f:
            g = json.load(f)
        log(f"gate: reuse ({sum(len(v) for v in g['windows'].values())} "
            f"windows)")
        return g
    from ultralytics import YOLO
    from scripts.vlm_screener.prompts import GATE_MIN_DET, YOLO_CONF
    yolo = YOLO(YOLO_WEIGHTS)
    t0 = time.time()
    win_gate = {}
    n_pet = 0
    n_all = 0
    for wv in windows["videos"]:
        vid = wv["vid"]
        vdir = os.path.join(DATA_DIR, "frames", vid)
        gates = {}
        for win in wv["windows"]:
            det = 0
            frames = []
            for k in range(N_FRAMES):
                fp = os.path.join(vdir, f"{win['w']}_f{k}.jpg")
                if os.path.isfile(fp):
                    frames.append(fp)
            if frames:
                try:
                    res = yolo.predict(frames, conf=YOLO_CONF, device="cpu",
                                       verbose=False)
                    for r in res:
                        if r.boxes is not None and len(r.boxes) > 0:
                            det += 1
                except Exception as e:  # noqa: BLE001
                    log(f"  [WARN] yolo fail {vid}/{win['w']}: {e}")
                    det = GATE_MIN_DET  # fail-open: let the VLM decide
            has_pet = det >= GATE_MIN_DET
            gates[win["w"]] = {"det_frames": det, "has_pet": bool(has_pet)}
            n_all += 1
            n_pet += int(has_pet)
        win_gate[vid] = gates
    out = {"schema": "vlm-screener-gates-1",
           "rule": f"det_frames >= {GATE_MIN_DET} of {N_FRAMES} "
                   f"(conf >= {YOLO_CONF}, cpu)",
           "windows": win_gate}
    tmp = gpath + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    os.replace(tmp, gpath)
    log(f"gate: {n_all} windows, {n_pet} has_pet ({100.0 * n_pet / max(1, n_all):.0f}%) "
        f"in {time.time() - t0:.0f}s -> {gpath}")
    return out


def main(argv=None):
    global DATA_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-e", type=int, default=30)
    ap.add_argument("--n-g", type=int, default=20)
    ap.add_argument("--seed", type=int, default=20261013)
    ap.add_argument("--data-dir", default=DATA_DIR,
                    help="override output dir (smoke tests use a scratch "
                         "dir so the real .vlm-screener stays clean)")
    ap.add_argument("--resample", action="store_true",
                    help="force a fresh random draw (regenerates manifest "
                         "only; delete windows/gates/frames to redo those)")
    ap.add_argument("--skip-gate", action="store_true")
    a = ap.parse_args(argv)
    DATA_DIR = a.data_dir
    os.makedirs(DATA_DIR, exist_ok=True)
    manifest = stage_sample(a.n_e, a.n_g, a.seed, a.resample)
    windows = stage_windows(manifest)
    stage_frames(manifest, windows)
    if not a.skip_gate:
        stage_gate(manifest, windows)
    log("prepare: DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
