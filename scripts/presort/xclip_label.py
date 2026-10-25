"""X-CLIP zero-shot labeling for comparison with 2a auto_presort labels.

Sliding 3s windows / 1s stride, 8 frames each (PIL frames -- numpy input
breaks the VideoMAE image processor in transformers 4.53). Softmax over
per-class prompt groups (mean over prompts), argmax -> segments; windows
whose argmax is "other" become gaps, mirroring the .autopresort doc.

Output: <dir>/.xclip/<flat rel_path>.json + manifest.json
(schema_version xclip-1; segments/gaps share field names with autopresort-2).

Usage:
  python -m scripts.presort.xclip_label --dir <batch_dir> [--limit 10]

Logs to logs/xclip_label.log (English only, cp1252-safe).
"""
import argparse
import json
import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOG_PATH = os.path.join(PROJECT_ROOT, "logs", "xclip_label.log")
MODEL_DIR = os.path.join(PROJECT_ROOT, "checkpoints", "xclip-base-patch32")

VIDEO_EXTS = (".mp4", ".avi", ".mov", ".webm")

# Rabbit 4-class alignment:进食/喝水/睡觉/跑动/其他 -> eating/drinking/resting/active + other->gap
PROMPT_TEXTS = {
    "eating": ["a rabbit eating food", "a rabbit chewing hay"],
    "drinking": ["a rabbit drinking water"],
    "resting": ["a rabbit resting", "a rabbit sitting still"],
    "active": ["a rabbit running", "a rabbit hopping around"],
    "other": ["an empty scene without any animal"],
}
SCHEMA_VERSION = "xclip-1"


def log(msg):
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def collect_videos(batch_dir, limit):
    """Recursive video walk (skip dot-dirs), sorted, first `limit` entries."""
    rels = []
    for root, dirs, files in os.walk(batch_dir):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for fn in sorted(files):
            if fn.lower().endswith(VIDEO_EXTS):
                rel = os.path.relpath(os.path.join(root, fn), batch_dir)
                rels.append(rel.replace("\\", "/"))
    rels.sort()
    return rels[:limit] if limit else rels


def window_ranges(duration, window_sec, stride_sec):
    """[(start, length), ...]. Videos shorter than one window -> single full clip."""
    if duration < window_sec:
        return [(0.0, duration)]
    out = []
    t = 0.0
    while t + window_sec <= duration + 1e-6:
        out.append((round(t, 3), window_sec))
        t += stride_sec
    if not out:
        out.append((0.0, duration))
    return out


def sample_timestamps(start, length, n_frames):
    return [start + (i + 0.5) * length / n_frames for i in range(n_frames)]


def read_frames_at(path, timestamps):
    """Sequential cv2 read; returns (fps, duration, [BGR uint8 HWC frames]).

    Frame ids are NOT deduplicated and missing tails are backfilled with the
    last readable frame: the caller needs exactly len(timestamps) frames
    (X-CLIP reshapes internally by num_frames=8 -- fewer frames makes the
    batch dim compute to 0 and the forward crashes). Returns BGR (YOLO input
    convention); convert to RGB only when feeding the X-CLIP processor.
    """
    import cv2

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = (n_frames / fps) if (n_frames > 0 and fps > 0) else 0.0
    ids = [max(0, min(int(round(t * fps)), n_frames - 1)) for t in timestamps]
    want = set(ids)
    frames, last, i = {}, None, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i in want:
            last = frame
            frames[i] = last
        i += 1
    cap.release()
    out = [frames.get(fid, last) for fid in ids]
    if any(f is None for f in out):
        raise RuntimeError(f"no readable frames in {path}")
    return fps, duration, out


def merge_windows(windows, gap_reason="other"):
    """[(start, end, label, conf)] -> (segments, gaps) with adjacent-merge."""
    segments, gaps = [], []
    for start, end, label, conf in windows:
        target = gaps if label == "other" else segments
        if label == "other":
            if target and abs(target[-1][1] - start) < 1e-6:
                target[-1][1] = end
                continue
            target.append([start, end])
            continue
        if target and target[-1]["label"] == label and abs(target[-1]["end"] - start) < 1e-6:
            prev = target[-1]
            prev["confidence"] = (prev["confidence"] * prev["_w"] + conf * (end - start)) / (prev["_w"] + (end - start))
            prev["_w"] += end - start
            prev["end"] = end
        else:
            target.append({"start": start, "end": end, "label": label,
                           "confidence": conf, "source": "xclip", "_w": end - start})
    for seg in segments:
        seg.pop("_w", None)
    return segments, gaps


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="batch dir containing videos")
    ap.add_argument("--limit", type=int, default=0, help="only first N videos (0=all)")
    ap.add_argument("--model-dir", default=MODEL_DIR)
    ap.add_argument("--window-sec", type=float, default=3.0)
    ap.add_argument("--stride-sec", type=float, default=1.0)
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--batch-windows", type=int, default=32)
    ap.add_argument("--no-crop", action="store_true",
                    help="disable YOLO cropping (feed full frames)")
    ap.add_argument("--yolo-model", default=None,
                    help="default: SPECIES['rabbit']['yolo_model']")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--expand-ratio", type=float, default=1.25)
    ap.add_argument("--out-name", default=".xclip",
                    help="output dir name under batch dir (e.g. .xclip-p16)")
    a = ap.parse_args(argv)

    import cv2
    import numpy as np
    import torch
    from PIL import Image
    from transformers import XCLIPModel, XCLIPProcessor

    from auto_presort.apply_clips import source_id_for
    from scripts.pipeline import pipeline_config as pc
    from utils.abnormal_crop_video import build_smoothed_crop_boxes

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"xclip_label start dir={a.dir} limit={a.limit} device={device} crop={not a.no_crop}")

    sp = pc.SPECIES["rabbit"]
    yolo = None
    if not a.no_crop:
        from ultralytics import YOLO
        yolo = YOLO(a.yolo_model or sp["yolo_model"]).to(device)
        log(f"yolo loaded: {a.yolo_model or sp['yolo_model']}")

    processor = XCLIPProcessor.from_pretrained(a.model_dir, local_files_only=True)
    model = XCLIPModel.from_pretrained(a.model_dir, local_files_only=True).to(device).eval()
    texts = [t for ts in PROMPT_TEXTS.values() for t in ts]
    class_slices, idx0 = [], 0
    for name, ts in PROMPT_TEXTS.items():
        class_slices.append((name, list(range(idx0, idx0 + len(ts)))))
        idx0 += len(ts)

    rels = collect_videos(a.dir, a.limit)
    if not rels:
        log("[ERROR] no videos found")
        return 1
    out_dir = os.path.join(a.dir, a.out_name)
    os.makedirs(out_dir, exist_ok=True)

    t_all = time.time()
    for i, rel in enumerate(rels, 1):
        t0 = time.time()
        vp = os.path.join(a.dir, *rel.split("/"))
        fps, duration, _ = read_frames_at(vp, [0.0])
        ranges = window_ranges(duration, a.window_sec, a.stride_sec)
        ts_by_win = [sample_timestamps(s, L, a.frames) for s, L in ranges]
        all_ts = [t for ts in ts_by_win for t in ts]
        _, dur2, flat_frames = read_frames_at(vp, all_ts)
        duration = dur2 or duration

        win_records, raw_windows = [], []
        for w_idx, (start, length) in enumerate(ranges):
            frames_bgr = flat_frames[w_idx * a.frames:(w_idx + 1) * a.frames]
            if yolo is not None:
                results = yolo.predict(frames_bgr, classes=sp["target_classes"],
                                       conf=a.conf, verbose=False)
                boxes = []
                for r in results:
                    if r.boxes is not None and len(r.boxes):
                        ci = int(r.boxes.conf.argmax())
                        boxes.append(tuple(float(v) for v in r.boxes.xyxy[ci].tolist()))
                    else:
                        boxes.append(None)
                fh, fw = frames_bgr[0].shape[:2]
                crop_boxes = build_smoothed_crop_boxes(
                    boxes, frame_width=fw, frame_height=fh,
                    fallback_size=(fw, fh), expand_ratio=a.expand_ratio)
                frames_in = []
                for f, cb in zip(frames_bgr, crop_boxes):
                    x1, y1, x2, y2 = cb
                    x1, y1 = max(0, int(round(x1))), max(0, int(round(y1)))
                    x2, y2 = min(fw, int(round(x2))), min(fh, int(round(y2)))
                    frames_in.append(f[y1:y2, x1:x2])
            else:
                frames_in = frames_bgr
            pil = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
                   for f in frames_in]
            with torch.no_grad():
                inputs = processor(videos=[pil], text=texts,
                                   return_tensors="pt", padding=True).to(device)
                out = model(**inputs)
            probs = out.logits_per_video.softmax(dim=-1)[0].cpu().numpy()
            class_prob = {name: float(np.mean([probs[j] for j in idxs]))
                          for name, idxs in class_slices}
            label = max(class_prob, key=class_prob.get)
            # Responsibility interval: this window's vote covers [start, start+stride);
            # the last window extends to the video end (adjacent intervals then
            # touch, which is what merge_windows requires for sliding windows).
            if w_idx == len(ranges) - 1:
                end = duration
            else:
                end = min(start + a.stride_sec, duration)
            raw_windows.append((start, end, label, class_prob[label]))
            win_records.append({"start": start, "end": end,
                                "coverage": min(start + length, duration),
                                "probs": {k: round(v, 4) for k, v in class_prob.items()}})

        segments, gaps = merge_windows(raw_windows)
        doc = {
            "schema_version": SCHEMA_VERSION,
            "rel_path": rel,
            "source_id": source_id_for(rel),
            "duration": round(duration, 3),
            "approved": False,
            "segments": segments,
            "gaps": [[round(g[0], 3), round(g[1], 3), "other"] for g in gaps],
            "windows": win_records,
            "meta": {
                "model_dir": a.model_dir,
                "engine": "xclip-base-patch32 zero-shot",
                "prompts": PROMPT_TEXTS,
                "window_sec": a.window_sec, "stride_sec": a.stride_sec,
                "frames_per_window": a.frames,
                "yolo_crop": not a.no_crop,
                "yolo_model": None if a.no_crop else (a.yolo_model or sp["yolo_model"]),
            },
        }
        flat = rel.replace("/", "__") + ".json"
        with open(os.path.join(out_dir, flat), "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False, indent=2)
        log(f"  ({i}/{len(rels)}) {rel} dur={duration:.1f}s wins={len(ranges)} "
            f"segs={len(segments)} gaps={len(gaps)} {time.time() - t0:.1f}s")

    manifest = {
        "species": "rabbit", "inference_engine": SCHEMA_VERSION,
        "model_dir": a.model_dir, "prompts": PROMPT_TEXTS,
        "params": {"window_sec": a.window_sec, "stride_sec": a.stride_sec,
                   "frames_per_window": a.frames},
        "n_videos": len(rels),
    }
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    log(f"[DONE] {len(rels)} videos in {time.time() - t_all:.1f}s -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
