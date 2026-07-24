"""Step 1 of auto_presort: model pre-labels a batch of raw videos into .autopresort/ JSONs.

Heavy entry: needs GPU + torch + YOLO + teacher checkpoint.
Class names come from pipeline_config.SPECIES[--species]; an optional
<checkpoint>.meta.json sidecar is cross-checked if present.
English console prints (cp1252-safe).
Usage:
  python auto_presort_label.py --species rabbit --dir <raw_batch_dir> \
      --checkpoint <teacher.pth> [--threshold 0.90] [--trim-margin 1.0]
"""
import argparse
import os
import sys

VIDEO_EXTS = (".mp4", ".avi", ".mov", ".webm")


def collect_videos(batch_dir):
    """Recursively find raw videos; returns rel paths with forward slashes."""
    rels = []
    for root, dirs, files in os.walk(batch_dir):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for fn in sorted(files):
            if fn.lower().endswith(VIDEO_EXTS):
                rel = os.path.relpath(os.path.join(root, fn), batch_dir)
                rels.append(rel.replace("\\", "/"))
    return rels


def default_params(threshold, trim_margin):
    return dict(window_sec=3.0, stride_sec=1.0, threshold=threshold, enter_run=2,
                trim_margin=trim_margin, min_duration=2.0, imgsz=224,
                temporal_window=16, expand_ratio=1.25, no_animal_overlap=0.5)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--species", required=True, choices=["pet", "rabbit"])
    ap.add_argument("--dir", required=True, help="raw batch dir with long videos")
    ap.add_argument("--checkpoint", required=True, help="teacher .pth (.meta.json sidecar optional)")
    ap.add_argument("--threshold", type=float, default=0.90)
    ap.add_argument("--trim-margin", type=float, default=1.0)
    a = ap.parse_args(argv)

    from scripts.pipeline import pipeline_config as pc
    from ultralytics import YOLO
    from auto_presort import detect_track, engine, store
    from auto_presort.apply_clips import source_id_for
    from auto_presort.checkpoint_guard import verify_and_load_teacher

    sp = pc.SPECIES[a.species]
    rels = collect_videos(a.dir)
    if not rels:
        print(f"[ERROR] No videos found under: {a.dir}")
        return 1

    print(f"[1/3] Verifying teacher checkpoint (hard-fail on any mismatch)...")
    model, meta = verify_and_load_teacher(a.checkpoint, sp["class_names"])

    print(f"[2/3] Loading YOLO: {sp['yolo_model']}")
    yolo = YOLO(sp["yolo_model"])

    params = default_params(a.threshold, a.trim_margin)
    reader = engine.build_real_window_reader()
    classifier = engine.build_real_classifier(model, meta["class_names"], params)

    print(f"[3/3] Labeling {len(rels)} video(s)...")
    for i, rel in enumerate(rels, 1):
        vp = os.path.join(a.dir, *rel.split("/"))
        print(f"  ({i}/{len(rels)}) {rel}")
        track = detect_track.build_track(vp, yolo, sp["target_classes"])
        doc = engine.label_video(
            video_path=vp, rel_path=rel, source_id=source_id_for(rel), track=track,
            read_window_frames_fn=lambda v, s, e, fps, _b=track["boxes"]:
                reader(v, s, e, fps, _b),
            classify_frames_fn=classifier, meta=meta, params=params)
        store.save_doc(a.dir, doc)
        print(f"      segments={len(doc['segments'])} gaps={len(doc['gaps'])}")

    store.save_manifest(a.dir, {
        "species": a.species, "checkpoint_sha256": meta["sha256"],
        "class_names": meta["class_names"], "params": params,
        "n_videos": len(rels)})
    print(f"[DONE] JSONs written to {os.path.join(a.dir, '.autopresort')}")
    print("Next: python auto_presort_review.py --dir <this dir>")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
