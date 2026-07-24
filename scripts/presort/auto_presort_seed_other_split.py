"""Step 2a2 of the other-split workflow: seed blank annotation docs for other/ clips.

Light entry: stdlib + ffprobe on PATH. No model involved (unlike auto_presort_label.py) —
every video just gets an empty-segments doc so it shows up in the review page. Skips
videos that already have a doc, so re-running after adding new other/ clips is safe.
Usage: python auto_presort_seed_other_split.py [--root <dir>]
"""
import argparse
import os
import subprocess
import sys

from auto_presort import store, timeline as tl
from auto_presort.apply_clips import source_id_for
from auto_presort.other_split import find_other_dirs, SPLIT_CLASSES

VIDEO_EXTS = (".mp4", ".avi", ".mov", ".webm")


def _probe_duration(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", path],
        capture_output=True, text=True)
    try:
        return float(out.stdout.strip())
    except ValueError:
        return 0.0


def seed_dir(other_dir, probe_duration_fn=_probe_duration):
    """Create a blank doc for every video in other_dir that doesn't have one yet.

    Returns (seeded_count, already_done_count).
    """
    existing = {d["rel_path"] for d in store.list_docs(other_dir)}
    seeded, skipped = 0, 0
    for fn in sorted(os.listdir(other_dir)):
        if not fn.lower().endswith(VIDEO_EXTS):
            continue
        if fn in existing:
            skipped += 1
            continue
        duration = probe_duration_fn(os.path.join(other_dir, fn))
        doc = tl.build_video_doc(
            rel_path=fn, duration=duration, segments=[], boundary_spans=[],
            no_animal_spans=[], source_id=source_id_for(fn), approved=False,
            meta={"class_names": SPLIT_CLASSES})
        store.save_doc(other_dir, doc)
        seeded += 1
    return seeded, skipped


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None,
                    help="raw root to scan (default: pet raw_root from pipeline_config)")
    a = ap.parse_args(argv)

    from scripts.pipeline import pipeline_config as pc
    root = a.root or pc.SPECIES["pet"]["raw_root"]

    other_dirs = find_other_dirs(root)
    if not other_dirs:
        print(f"[ERROR] No 'other' directories found under: {root}")
        return 1

    print(f"[INFO] Found {len(other_dirs)} 'other' dir(s) under {root}")
    total_seeded, total_skipped = 0, 0
    for d in other_dirs:
        seeded, skipped = seed_dir(d)
        total_seeded += seeded
        total_skipped += skipped
        print(f"  {d}: seeded={seeded} already_done={skipped}")

    print(f"[DONE] seeded={total_seeded} already_done={total_skipped} dirs={len(other_dirs)}")
    print("Next: double-click 2b2_presort_review_other_split.bat")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]) or 0)
