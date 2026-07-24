"""Step 2c2 of the other-split workflow: cut approved active/resting/eat/drink clips.

Walks every scattered other/ dir under the pet raw_root and, for each one that has been
seeded/reviewed (has a .autopresort/ store), applies it via the EXISTING
auto_presort_apply.run_apply (unchanged) — output lands in
<that other/ dir's parent sorted-<date> dir>/<label>/, i.e. alongside the existing
eat/drink/other folders.
Usage: python run_presort_apply_other_split.py [--root <dir>]
"""
import argparse
import os
import sys

from auto_presort.other_split import find_other_dirs


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None,
                    help="raw root to scan (default: pet raw_root from pipeline_config)")
    a = ap.parse_args(argv)

    from scripts.pipeline import pipeline_config as pc
    from . import auto_presort_apply
    root = a.root or pc.SPECIES["pet"]["raw_root"]

    other_dirs = find_other_dirs(root)
    if not other_dirs:
        print(f"[ERROR] No 'other' directories found under: {root}")
        return 1

    total_written, total_skipped, total_warnings = 0, 0, 0
    for other_dir in other_dirs:
        if not os.path.isdir(os.path.join(other_dir, ".autopresort")):
            print(f"[SKIP] {other_dir}: not seeded yet (run 2a2/2b2 first)")
            continue
        out_dir = os.path.dirname(other_dir)  # the sorted-<date> parent
        print(f"[APPLY] {other_dir} -> {out_dir}")
        result = auto_presort_apply.run_apply(other_dir, out_dir)
        total_written += len(result["written"])
        total_skipped += len(result["skipped"])
        total_warnings += len(result["warnings"])

    print(f"[DONE] total written={total_written} skipped(unapproved)={total_skipped} "
          f"warnings={total_warnings}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]) or 0)
