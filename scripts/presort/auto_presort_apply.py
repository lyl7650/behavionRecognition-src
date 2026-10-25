"""Step 3 of auto_presort: cut approved segments into sorted-<date>/<label>/.

Light entry: stdlib + ffmpeg on PATH. Skips unapproved videos (Review-all gate).
Accurate re-encode cutting (never -c copy). English console prints (cp1252-safe).
Usage: python auto_presort_apply.py --dir <raw_batch_dir> --out <sorted_dir>
"""
import argparse
import os
import subprocess
import sys

from auto_presort import store
from auto_presort.apply_clips import plan_cuts, clip_output_name, build_ffmpeg_cmd, duration_ok
from scripts.pipeline.lineage.store import record_derivation, record_input_if_root


def _probe_duration(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", path],
        capture_output=True, text=True)
    try:
        return float(out.stdout.strip())
    except ValueError:
        return -1.0


def run_apply(batch_dir, out_dir, tolerance=0.25):
    docs = store.list_docs(batch_dir)
    cuts, skipped = plan_cuts(docs)
    if skipped:
        print(f"[GATE] Skipped {len(skipped)} unapproved video(s): {skipped}")
    written, warnings = [], []
    for cut in cuts:
        src = os.path.join(batch_dir, *cut["rel_path"].replace("\\", "/").split("/"))
        if not os.path.isfile(src):
            warnings.append(f"missing source video: {cut['rel_path']}")
            continue
        name = clip_output_name(cut["rel_path"], cut["start"], cut["end"])
        dst = os.path.join(out_dir, cut["label"], name)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        cmd = build_ffmpeg_cmd(src, dst, cut["start"], cut["end"])
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0 or not os.path.isfile(dst):
            warnings.append(f"ffmpeg failed: {name}: {proc.stderr[-300:]}")
            continue
        actual = _probe_duration(dst)
        expected = cut["end"] - cut["start"]
        if not duration_ok(expected, actual, tolerance):
            warnings.append(
                f"duration mismatch: {name} expected={expected:.2f}s actual={actual:.2f}s")
        record_input_if_root(src)
        record_derivation(
            dst, src, operation="presort_cut", label=cut["label"],
            parameters={"start": cut["start"], "end": cut["end"],
                        "batch_dir": os.path.abspath(batch_dir)},
        )
        written.append(dst)
    print(f"[OK] Wrote {len(written)} clip(s) to {out_dir}")
    for w in warnings:
        print(f"[WARN] {w}")
    return {"written": written, "skipped": skipped, "warnings": warnings}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="raw batch dir (contains .autopresort/)")
    ap.add_argument("--out", required=True,
                    help="output sorted dir, e.g. dataset/raw_videos/rabbit/hkbr-record/sorted-2026-07-17")
    a = ap.parse_args(argv)
    run_apply(a.dir, a.out)


if __name__ == "__main__":
    main(sys.argv[1:])
