"""Step 1 of auto_presort: model pre-labels a batch of raw videos into .autopresort/ JSONs.

Runs the configured dashboard offline inference engine (model + FSM + rules).
Class names come from pipeline_config.SPECIES[--species].
English console prints (cp1252-safe).
Usage:
  python auto_presort_label.py --species rabbit --dir <raw_batch_dir> \
      [--threshold 0.90] [--trim-margin 1.0]
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
    ap.add_argument("--threshold", type=float, default=0.90)
    ap.add_argument("--trim-margin", type=float, default=1.0)
    ap.add_argument("--beh-model", default=None,
                    help="dashboard behavior model (.pth/.enc) for the fusion engine; "
                         "default: $<SPECIES>_BEH_MODEL env var, else the dashboard's "
                         "built-in encrypted model")
    a = ap.parse_args(argv)

    from scripts.pipeline import pipeline_config as pc
    from auto_presort import engine, store
    from auto_presort.fused_inference import infer_video
    from auto_presort.apply_clips import source_id_for

    beh_model = a.beh_model or os.environ.get(f"{a.species.upper()}_BEH_MODEL")

    sp = pc.SPECIES[a.species]
    rels = collect_videos(a.dir)
    if not rels:
        print(f"[ERROR] No videos found under: {a.dir}")
        return 1

    meta = {"class_names": list(sp["class_names"])}
    print("[1/3] Dashboard inference: model probabilities + FSM + rules")

    params = default_params(a.threshold, a.trim_margin)
    print(f"[2/3] Labeling {len(rels)} video(s)...")
    for i, rel in enumerate(rels, 1):
        vp = os.path.join(a.dir, *rel.split("/"))
        print(f"  ({i}/{len(rels)}) {rel}")
        track, classifier, model_info = infer_video(
            a.species, vp, meta["class_names"], beh_model=beh_model)
        print(f"      dashboard_models={model_info}")
        doc = engine.label_video(
            video_path=vp, rel_path=rel, source_id=source_id_for(rel), track=track,
            read_window_frames_fn=None, classify_frames_fn=None, meta=meta, params=params,
            classify_window_fn=classifier)
        store.save_doc(a.dir, doc)
        print(f"      segments={len(doc['segments'])} gaps={len(doc['gaps'])}")

    store.save_manifest(a.dir, {
        "species": a.species, "inference_engine": "dashboard model+FSM+rules",
        "class_names": meta["class_names"], "params": params,
        "n_videos": len(rels)})
    print(f"[DONE] JSONs written to {os.path.join(a.dir, '.autopresort')}")
    print("Next: python auto_presort_review.py --dir <this dir>")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
