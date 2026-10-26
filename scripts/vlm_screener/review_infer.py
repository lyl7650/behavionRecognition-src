"""Background inference worker for the per-directory review workflow.

Reuses the window plan / frame dump / YOLO gate semantics of prepare.py and
the model loading / classify / merge logic of run_combos.py, but writes into
the target directory's .vlm-screener/ store and produces 2b-style docs.

Pipeline in batches of 50 videos: within each batch, run the selected
combos grouped by model (one load for both prompts), then synthesize.
Missing videos -> windows plan -> full-frame 448 jpgs -> YOLO gate
(stores per-frame best bbox) -> per-window VLM labels -> doc update.

Crop mode: when crop=True, VLM frames are the gate bbox expanded 1.25x
(cropped in memory from the stored 448 frames, video-pixel coords never
leave the pipeline); frames without a detection fall back to full frame.

Segment synthesis: untied winners need a strict majority of run combos;
ties use the highest mean confidence and bypass that threshold. YOLO none
windows stay gaps; segment confidence is the agreeing votes' mean.
"""
import json
import os
import subprocess
import sys
import threading
import time

from scripts.pipeline import pipeline_config as pc
from scripts.vlm_screener import review_store as store
from scripts.vlm_screener.prepare import (
    WINDOW_SEC, STRIDE_SEC, N_FRAMES, MAX_SIDE, JPG_QUALITY,
    window_ranges, sample_timestamps, YOLO_WEIGHTS)
from scripts.vlm_screener.prompts import (
    COMBOS, PROMPTS, MODEL_KEYS, GATE_MIN_DET,
    YOLO_CONF, SCHEMA_VERSION, COMBOS_BY_SPECIES, PROMPTS_BY_SPECIES,
    VERSIONS_BY_SPECIES, TABLES)
from scripts.vlm_screener.run_combos import (
    load_model, classify, merge_segments)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
LOCK_PATH = os.path.join(LOG_DIR, "vlm_screener.lock")
OVERNIGHT_LOCK = os.path.join(LOG_DIR, "vlm_overnight.lock")
MIN_FREE_VRAM_MB = 7000
CROP_EXPAND = 1.25
BATCH_SIZE = 50
PET_YOLO_WEIGHTS = (r"E:\ActionRecognition\detection-src\cat_dog_detect\runs"
                    r"\yolo11n_scale0.6_fliplr0.42\weights\yolo11n-ssb-0710.pt")

_state = {"running": False, "target": None, "combos": [], "crop": False,
          "done": 0, "total": 0, "current": None, "error": None,
          "started_at": None}


def status():
    return dict(_state)


def _pid_alive(pid):
    try:
        import subprocess
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}"],
                             capture_output=True, text=True).stdout
        return str(pid) in out
    except Exception:  # noqa: BLE001
        return False


def precheck():
    """(ok, message) before starting inference: overnight lock + VRAM."""
    try:
        with open(OVERNIGHT_LOCK, encoding="utf-8") as f:
            pid = int(f.read().strip() or 0)
        if pid and pid != os.getpid() and _pid_alive(pid):
            return False, ("8767 overnight orchestrator (vlm_overnight) is "
                           f"still running (pid {pid}); wait for it to exit")
    except (OSError, ValueError):
        pass
    try:
        import subprocess
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15).stdout.strip()
        free_mb = int(out.splitlines()[0])
        if free_mb < MIN_FREE_VRAM_MB:
            return False, (f"GPU free VRAM {free_mb}MB < {MIN_FREE_VRAM_MB}MB; "
                           "close other GPU jobs first")
    except Exception as e:  # noqa: BLE001
        return False, f"cannot query GPU: {e}"
    return True, ""


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


# ---------------------------------------------------------------- per-video

def _ensure_windows(target, video, win_all):
    vid = video["vid"]
    if vid in win_all and win_all[vid].get("windows"):
        return win_all[vid]
    dur = video["duration"] or 1.0
    wins = []
    for i, (start, length) in enumerate(window_ranges(dur)):
        wins.append({"w": f"w{i:03d}", "start": start, "end": None,
                     "length": length,
                     "ts": [round(t, 3) for t in
                            sample_timestamps(start, length)]})
    entry = {"duration": round(dur, 3), "windows": wins}
    win_all[vid] = entry
    store.save_windows(target, win_all)
    return entry


def _ensure_frames(target, video, wv):
    """Dump missing full-frame 448 jpgs (same naming as prepare.py)."""
    import cv2
    vdir = os.path.join(store.data_dir(target), "frames", video["vid"])
    need = []
    for win in wv["windows"]:
        for k in range(N_FRAMES):
            fp = os.path.join(vdir, f"{win['w']}_f{k}.jpg")
            if not os.path.isfile(fp):
                need.append((win, k, fp))
    if not need:
        return
    os.makedirs(vdir, exist_ok=True)
    cap = cv2.VideoCapture(video["path"])
    if not cap.isOpened():
        log(f"  [ERROR] cannot open {video['path']}")
        return
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


def _ensure_gate(target, video, wv, gates_all, yolo, classes):
    """YOLO gate restricted to the species' detector classes.

    `classes` is pipeline_config's target_classes (pet 1=cat/2=dog, rabbit 0).
    Without it the co-trained non-pet class (index 0) opens the gate and its
    larger bbox becomes the crop. Gates cached before the filter existed carry
    no "classes" key and are recomputed.
    """
    vid = video["vid"]
    gvid = gates_all.get(vid)
    if gvid and all("boxes" in g and g.get("classes") == classes
                    for g in gvid.values()) \
            and len(gvid) == len(wv["windows"]):
        return gvid
    gvid = gvid or {}
    vdir = os.path.join(store.data_dir(target), "frames", vid)
    for win in wv["windows"]:
        if win["w"] in gvid and "boxes" in gvid[win["w"]] \
                and gvid[win["w"]].get("classes") == classes:
            continue
        frames = []
        for k in range(N_FRAMES):
            fp = os.path.join(vdir, f"{win['w']}_f{k}.jpg")
            if os.path.isfile(fp):
                frames.append(fp)
        det, boxes = 0, []
        if frames:
            try:
                res = yolo.predict(frames, conf=YOLO_CONF, device="cpu",
                                   classes=classes, verbose=False)
                for r in res:
                    if r.boxes is not None and len(r.boxes) > 0:
                        det += 1
                        xy = r.boxes.xyxy.cpu().tolist()
                        bx = max(xy, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
                        boxes.append([round(v, 1) for v in bx])
                    else:
                        boxes.append(None)
            except Exception as e:  # noqa: BLE001
                log(f"  [WARN] yolo fail {vid}/{win['w']}: {e}")
                det = GATE_MIN_DET      # fail-open, same as prepare.py
                boxes = [None] * len(frames)
        gvid[win["w"]] = {"det_frames": det,
                          "has_pet": bool(det >= GATE_MIN_DET),
                          "boxes": boxes,
                          "classes": list(classes)}
    gates_all[vid] = gvid
    store.save_gates(target, gates_all)
    return gvid


def _load_pil_frames(target, video, win):
    from PIL import Image
    vdir = os.path.join(store.data_dir(target), "frames", video["vid"])
    out = []
    for k in range(N_FRAMES):
        fp = os.path.join(vdir, f"{win['w']}_f{k}.jpg")
        if os.path.isfile(fp):
            out.append(Image.open(fp).convert("RGB"))
    return out


def _crop_pil(pil_frames, boxes, expand=CROP_EXPAND):
    """Expand the per-frame bbox by 1.25 and crop; miss -> full frame."""
    out = []
    for im, bx in zip(pil_frames, boxes or []):
        if not bx:
            out.append(im)
            continue
        W, H = im.size
        x1, y1, x2, y2 = bx
        w, h = x2 - x1, y2 - y1
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        nx1 = max(0, int(cx - w * expand / 2))
        ny1 = max(0, int(cy - h * expand / 2))
        nx2 = min(W, int(cx + w * expand / 2))
        ny2 = min(H, int(cy + h * expand / 2))
        if nx2 - nx1 < 8 or ny2 - ny1 < 8:
            out.append(im)
        else:
            out.append(im.crop((nx1, ny1, nx2, ny2)))
    return out


# ------------------------------------------------------------- synthesis

def synthesize_segments(combos_data, duration):
    """Window-level vote -> 2b segments. Rule confirmed 10-13:
      - YOLO none windows never vote (stay gap);
      - window label = most votes; ties are broken by mean confidence and
        the tie-break winner is EXEMPT from the majority gate;
      - untied winners need >= strict majority (n//2+1) votes, else gap;
      - each window contributes [start, next_start) so segments never
        overlap even though windows do (6s window, 3s stride)."""
    base = None
    for combo in combos_data:
        if combos_data[combo].get("windows"):
            base = combos_data[combo]["windows"]
            break
    if not base:
        return []
    n = len([c for c in combos_data if combos_data[c].get("windows")])
    min_n = n // 2 + 1
    raw = []
    for i, ref in enumerate(base):
        votes = {}
        for combo in combos_data:
            wins = combos_data[combo].get("windows") or []
            if i >= len(wins):
                continue
            rec = wins[i]
            if rec.get("gate") != "vlm" or rec.get("label") in (None, "none"):
                continue
            votes.setdefault(rec["label"], []).append(rec.get("confidence") or 0.0)
        if not votes:
            continue
        top = max(len(v) for v in votes.values())
        leaders = {k: v for k, v in votes.items() if len(v) == top}
        if len(leaders) == 1:
            label, confs = next(iter(leaders.items()))
            if top < min_n:
                continue                  # weak but untied majority -> gap
        else:
            label, confs = max(
                leaders.items(),
                key=lambda kv: (sum(kv[1]) / len(kv[1]), len(kv[1])))
            # tie broken by mean confidence: exempt from min_n by design
        start = ref["start"]
        end = base[i + 1]["start"] if i + 1 < len(base) else duration
        raw.append((start, end, label, sum(confs) / len(confs)))
    segs = merge_segments(raw)
    for s in segs:
        s["source"] = "model"
    return segs


def resynthesize_all(target, combos, vids=None):
    """Rebuild segments for completed docs (optionally only batch vids).
    Human-edited segments are never touched; approval is cleared only when
    the rebuilt segments actually differ. Returns changed rel_paths."""
    want = set(combos)
    changed = []
    for vid, doc in store.list_docs(target).items():
        if vids is not None and vid not in vids:
            continue
        if not want <= set((doc.get("combos") or {}).keys()):
            continue
        segs = doc.get("segments") or []
        if any(s.get("source") == "human" for s in segs):
            continue                      # human work wins, never overwrite
        new = synthesize_segments(doc.get("combos") or {},
                                  doc.get("duration") or 0)
        if json.dumps(segs, sort_keys=True, ensure_ascii=False) != \
                json.dumps(new, sort_keys=True, ensure_ascii=False):
            doc["segments"] = new
            if doc.get("approved"):
                doc["approved"] = False   # content changed -> re-approve
            store.save_doc(target, doc)
            changed.append(doc["rel_path"])
    return changed


# ---------------------------------------------------------------- worker

def _progress_path(target):
    return os.path.join(store.data_dir(target), "infer_progress.json")


def _write_progress(target, data):
    try:
        path = _progress_path(target)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        os.replace(tmp, path)
    except OSError:
        pass


def _read_progress(target):
    try:
        with open(_progress_path(target), encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def run_model_for_batch(target, cfgs, crop, batch_vids, job):
    """Run one model's selected prompts over a batch in one child process."""
    import torch
    from ultralytics import YOLO
    species = store.require_context(target)["species"]
    allowed = COMBOS_BY_SPECIES[species]
    if any(c not in allowed for c in cfgs):
        raise ValueError(f"combos do not belong to {species}")
    prompts = PROMPTS_BY_SPECIES[species]
    versions = VERSIONS_BY_SPECIES[species]
    table = TABLES[species]
    for cfg in cfgs:
        if not prompts[cfg["prompt_key"]]:
            raise ValueError(f"{species} prompt {cfg['prompt_key']} is not written yet")

    model_keys = {(c["model_key"], c["dtype"]) for c in cfgs}
    if len(model_keys) != 1:
        raise ValueError("a model batch must use one model and dtype")
    model_key, dtype = model_keys.pop()
    selected = set(batch_vids)
    videos = [v for v in store.load_videos(target) if v["vid"] in selected]
    docs = store.list_docs(target)
    todo_by_combo = {
        cfg["combo"]: [v for v in videos
                       if cfg["combo"] not in (docs.get(v["vid"], {}).get("combos") or {})]
        for cfg in cfgs
    }
    total = sum(len(todo) for todo in todo_by_combo.values())
    _write_progress(target, {"job": job, "done": 0, "total": total,
                             "current": None, "error": None})
    if not total:
        log(f"=== batch {job}: nothing to do ===")
        return 0

    win_all = store.load_windows(target)
    gates_all = store.load_gates(target)
    weights = YOLO_WEIGHTS if species == "rabbit" else PET_YOLO_WEIGHTS
    gate_classes = list(pc.SPECIES[species]["target_classes"])
    log(f"batch {job}: species={species}, yolo={weights}, "
        f"gate_classes={gate_classes}")
    yolo = YOLO(weights)
    model_dir = os.path.join(PROJECT_ROOT, "models-vlm", MODEL_KEYS[model_key])
    log(f"=== batch {job}: {total} video-combo tasks (crop={crop}) ===")
    model, processor, cand_ids, use_logits_conf = load_model(
        model_dir, dtype, table["words"])
    done = 0
    try:
        for cfg in cfgs:
            combo = cfg["combo"]
            prompt = prompts[cfg["prompt_key"]]
            todo = todo_by_combo[combo]
            for vi, video in enumerate(todo, 1):
                _write_progress(target, {"job": job, "done": done,
                                         "total": total,
                                         "current": video["rel_path"],
                                         "error": None})
                vid = video["vid"]
                wv = _ensure_windows(target, video, win_all)
                _ensure_frames(target, video, wv)
                gvid = _ensure_gate(target, video, wv, gates_all, yolo,
                                    gate_classes)
                records = []
                dist = {l: 0 for l in table["labels"]}
                dist["none"] = 0
                wins = wv["windows"]
                for idx, win in enumerate(wins):
                    g = gvid.get(win["w"], {})
                    det = int(g.get("det_frames", 0))
                    end = (wv["duration"] if idx == len(wins) - 1
                           else wins[idx + 1]["start"])
                    ran_vlm = False
                    if not g.get("has_pet", False):
                        label, conf, probs, perr = "none", 1.0, None, False
                    else:
                        frames = _load_pil_frames(target, video, win)
                        if not frames:
                            label, conf, probs, perr = "none", 1.0, None, False
                        else:
                            if crop:
                                frames = _crop_pil(frames, g.get("boxes"))
                            label, conf, probs, retried, perr = classify(
                                model, processor, cand_ids,
                                use_logits_conf, frames, prompt,
                                f"{vid}/{win['w']}", table)
                            ran_vlm = True
                    records.append({
                        "w": win["w"], "start": win["start"], "end": end,
                        "gate": "vlm" if ran_vlm else "yolo",
                        "det_frames": det, "label": label,
                        "confidence": conf, "probs": probs,
                        "parse_error": bool(perr)})
                    dist[label] = dist.get(label, 0) + 1
                total_w = max(1, sum(dist.values()))
                dom = max(dist.items(), key=lambda kv: kv[1])[0]
                combo_data = {
                    "dominant": dom if dist[dom] else "none",
                    "dominant_share": round(dist[dom] / total_w, 4),
                    "dist": dist,
                    "prompt_version": versions[cfg["prompt_key"]],
                    "windows": records,
                }
                doc = store.load_doc(target, vid)
                if doc is None:
                    doc = {
                        "vid": vid, "rel_path": video["rel_path"],
                        "video_path": video["path"],
                        "duration": wv["duration"], "approved": False,
                        "segments": [], "combos": {},
                        "gate": {"windows": gvid},
                        "inference": {"crop": crop, "combos": []},
                    }
                    fresh = True
                else:
                    fresh = not doc.get("segments")
                    doc["video_path"] = video["path"]
                    doc["duration"] = wv["duration"]
                    doc["gate"] = {"windows": gvid}
                doc.setdefault("combos", {})[combo] = combo_data
                doc.setdefault("inference", {"crop": crop, "combos": []})
                doc["inference"]["crop"] = crop
                if combo not in doc["inference"]["combos"]:
                    doc["inference"]["combos"].append(combo)
                if fresh:
                    doc["segments"] = synthesize_segments(
                        doc["combos"], doc["duration"])
                store.save_doc(target, doc)
                store.set_status(target, vid, "inferred")
                done += 1
                doms = " ".join(f"{r['label'][:3]}:{r['confidence']:.2f}"
                                for r in records[:6])
                log(f"  ({vi}/{len(todo)}) {video['rel_path']} "
                    f"combo={combo} wins={len(records)} "
                    f"dom={combo_data['dominant']}"
                    f"({combo_data['dominant_share']:.0%}) [{doms}]")
                if vi % 10 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
    finally:
        del model, processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc
        gc.collect()
        log(f"batch {job}: model released (child exits -> VRAM freed)")
    _write_progress(target, {"job": job, "done": done, "total": total,
                             "current": None, "error": None})
    log(f"batch {job}: {done} video-combo tasks updated")
    return 0


def _worker(target, combos, crop):
    """Run 50-video batches; one child per model in each batch."""
    try:
        env = dict(os.environ)
        env.setdefault("PYTHONIOENCODING", "utf-8")
        env.setdefault("HF_HUB_OFFLINE", "1")
        species = store.require_context(target)["species"]
        cfgs = [c for c in COMBOS_BY_SPECIES[species] if c["combo"] in combos]
        videos = store.load_videos(target)
        docs = store.list_docs(target)
        pending = [v for v in videos if any(
            c["combo"] not in (docs.get(v["vid"], {}).get("combos") or {})
            for c in cfgs)]
        groups = []
        for cfg in cfgs:
            key = (cfg["model_key"], cfg["dtype"])
            if groups and groups[-1][0] == key:
                groups[-1][1].append(cfg)
            else:
                groups.append((key, [cfg]))
        done_before = 0
        for start_idx in range(0, len(pending), BATCH_SIZE):
            batch = pending[start_idx:start_idx + BATCH_SIZE]
            batch_vids = [v["vid"] for v in batch]
            log(f"=== batch {start_idx // BATCH_SIZE + 1}: "
                f"{len(batch)} videos ===")
            for (model_key, _dtype), group_cfgs in groups:
                if not any(c["combo"] not in
                           (docs.get(v["vid"], {}).get("combos") or {})
                           for c in group_cfgs for v in batch):
                    continue
                job = f"{start_idx // BATCH_SIZE + 1}:{model_key}"
                proc = subprocess.Popen(
                    [sys.executable, "-m",
                     "scripts.vlm_screener.review_infer_child",
                     "--target", target, "--job", job,
                     "--combos", *[c["combo"] for c in group_cfgs],
                     "--vids", *batch_vids]
                    + (["--crop"] if crop else []),
                    cwd=PROJECT_ROOT, env=env)
                while proc.poll() is None:
                    p = _read_progress(target)
                    if p.get("job") == job:
                        _state["done"] = done_before + int(p.get("done") or 0)
                        _state["current"] = p.get("current")
                    time.sleep(1)
                p = _read_progress(target)
                if proc.returncode != 0:
                    err = (p.get("error") if p.get("job") == job else None) or \
                          f"batch {job} exited with code {proc.returncode}"
                    log(f"[ERROR] {err}")
                    _state["error"] = err
                    break
                if p.get("job") != job or p.get("done") != p.get("total"):
                    raise RuntimeError(f"batch {job} finished without complete progress")
                done_before += int(p["done"])
                _state["done"] = done_before
            if _state["error"]:
                break
            changed = resynthesize_all(target, [c["combo"] for c in cfgs],
                                       set(batch_vids))
            log(f"=== batch {start_idx // BATCH_SIZE + 1} synthesized: "
                f"{len(changed)} docs rebuilt ===")
        if _state["error"]:
            log(f"=== review inference FAILED (done={_state['done']}, "
                f"error={_state['error']}) ===")
        else:
            log(f"=== review inference DONE (done={_state['done']}) ===")
        _state["running"] = False
        _state["current"] = None
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        log(f"[ERROR] review inference: {e}")
        _state["error"] = str(e)
        _state["running"] = False


def start(target, combos, crop):
    """Kick off the worker thread; returns (ok, message)."""
    if _state["running"]:
        return False, "inference already running"
    try:
        species = store.require_context(target)["species"]
    except ValueError as exc:
        return False, str(exc)
    allowed = COMBOS_BY_SPECIES[species]
    if not combos or len(combos) != len(set(combos)) or any(
            c not in {cfg["combo"] for cfg in allowed} for c in combos):
        return False, f"invalid {species} combo selection"
    for cfg in allowed:
        if cfg["combo"] in combos and not PROMPTS_BY_SPECIES[species][cfg["prompt_key"]]:
            return False, f"{species} prompt {cfg['prompt_key']} is not written yet"
    ok, msg = precheck()
    if not ok:
        return False, msg
    _state.update({"running": True, "target": os.path.abspath(target),
                   "combos": list(combos), "crop": bool(crop),
                   "done": 0, "total": 0, "current": None, "error": None,
                   "started_at": time.strftime("%Y-%m-%d %H:%M:%S")})
    videos = store.load_videos(target)
    doc_by_vid = store.list_docs(target)
    total = 0
    for c in combos:
        total += sum(1 for v in videos
                     if c not in (doc_by_vid.get(v["vid"], {}).get("combos")
                                  or {}))
    _state["total"] = total
    t = threading.Thread(target=_worker, args=(target, combos, crop),
                         daemon=True)
    t.start()
    return True, f"{total} video-combo tasks queued"
