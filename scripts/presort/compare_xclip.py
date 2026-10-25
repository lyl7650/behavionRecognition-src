"""Compare X-CLIP zero-shot labels vs 2a dashboard (model+FSM+rules) labels.

Reads <dir>/.autopresort/ (2a baseline, schema autopresort-2, uses the chosen
segment_levels threshold) and <dir>/.xclip/ (xclip-1), builds per-side 0.25s
grid label timelines (4 behaviors + "none" for gaps), and reports per-video
and overall agreement plus a confusion matrix. Writes
<dir>/.xclip/compare_report.json.

Usage:
  python -m scripts.presort.compare_xclip --dir <batch_dir> [--level 0.9] [--limit 10]
"""
import argparse
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOG_PATH = os.path.join(PROJECT_ROOT, "logs", "compare_xclip.log")
LABELS = ["resting", "active", "eating", "drinking", "none"]


def log(msg):
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def flat_name(rel_path):
    return rel_path.replace("/", "__") + ".json"


def build_timeline(segments, gaps, duration, step):
    """Grid of labels; segments override gaps; uncovered -> None."""
    n = max(1, int(round(duration / step)))
    labs = [None] * n
    for s in segments:
        i0 = max(0, int(s["start"] / step))
        i1 = min(n, max(i0 + 1, int(round(s["end"] / step))))
        for i in range(i0, i1):
            labs[i] = s["label"]
    for g in gaps:
        i0 = max(0, int(g[0] / step))
        i1 = min(n, max(i0 + 1, int(round(g[1] / step))))
        for i in range(i0, i1):
            if labs[i] is None:
                labs[i] = "none"
    return [lab if lab in LABELS else "none" for lab in labs]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--level", default="0.9", help="2a segment_levels threshold key")
    ap.add_argument("--step", type=float, default=0.25)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--xc-dir-name", default=".xclip",
                    help="xclip output dir name (e.g. .xclip-p16)")
    a = ap.parse_args(argv)

    log(f"compare start dir={a.dir} level={a.level}")
    ap_dir = os.path.join(a.dir, ".autopresort")
    xc_dir = os.path.join(a.dir, a.xc_dir_name)
    ap_docs = {}
    for fn in sorted(os.listdir(ap_dir)):
        if fn.endswith(".json") and fn != "manifest.json":
            with open(os.path.join(ap_dir, fn), encoding="utf-8") as f:
                d = json.load(f)
            ap_docs[d["rel_path"]] = d

    xc_rels = []
    for fn in sorted(os.listdir(xc_dir)):
        if fn.endswith(".json") and fn != "manifest.json" and fn != "compare_report.json":
            with open(os.path.join(xc_dir, fn), encoding="utf-8") as f:
                xc_rels.append(json.load(f))
    if a.limit:
        xc_rels = xc_rels[:a.limit]

    confusion = {al: {xl: 0 for xl in LABELS} for al in LABELS}
    per_video = []
    for doc in xc_rels:
        rel = doc["rel_path"]
        ap_doc = ap_docs.get(rel)
        if ap_doc is None:
            log(f"  skip {rel}: no 2a doc")
            continue
        duration = min(doc.get("duration", 0), ap_doc.get("duration", 0))
        ap_segs = ap_doc.get("segment_levels", {}).get(a.level) or ap_doc.get("segments", [])
        ap_labs = build_timeline(ap_segs, ap_doc.get("gaps", []), duration, a.step)
        xc_labs = build_timeline(doc.get("segments", []), doc.get("gaps", []), duration, a.step)
        agree = sum(1 for x, y in zip(ap_labs, xc_labs) if x == y)
        total = max(1, min(len(ap_labs), len(xc_labs)))
        for al, xl in zip(ap_labs, xc_labs):
            confusion[al][xl] += 1
        per_video.append({"rel_path": rel, "agree": round(agree / total, 4),
                          "n": total,
                          "ap_labels": sorted(set(ap_labs)), "xc_labels": sorted(set(xc_labs))})
        log(f"  {rel}: agree={agree}/{total}={agree/total:.2f} "
            f"2a={sorted(set(ap_labs))} xclip={sorted(set(xc_labs))}")

    grand = {xl: sum(confusion[al][xl] for al in LABELS) for xl in LABELS}
    total_pts = sum(grand.values())
    diag = sum(confusion[al][al] for al in LABELS)
    report = {
        "level_2a": a.level, "step": a.step, "n_videos": len(per_video),
        "overall_agree": round(diag / max(1, total_pts), 4),
        "per_video": per_video,
        "confusion_2a_row_xclip_col": confusion,
        "col_totals_xclip": grand,
    }
    out = os.path.join(xc_dir, "compare_report.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    log(f"[DONE] overall agree={report['overall_agree']} over {total_pts} grid pts "
        f"({len(per_video)} videos) -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
