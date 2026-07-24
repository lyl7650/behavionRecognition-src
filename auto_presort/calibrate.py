"""离线校准：用已知 GT 边界量化引擎，产出 go/no-go。含可运行 CLI（见 main）。"""
import json


def _overlap(a0, a1, b0, b1):
    return max(0.0, min(a1, b1) - max(a0, b0))


def _union_len(spans):
    if not spans:
        return 0.0
    spans = sorted(spans)
    total, cs, ce = 0.0, spans[0][0], spans[0][1]
    for s, e in spans[1:]:
        if s > ce:
            total += ce - cs; cs, ce = s, e
        else:
            ce = max(ce, e)
    return total + (ce - cs)


def boundary_purity(pred, gt):
    total = sum(pe - ps for ps, pe, _ in pred)
    if total <= 0:
        return 0.0
    pure = 0.0
    for ps, pe, pl in pred:
        for gs, ge, gl in gt:
            if gl == pl:
                pure += _overlap(ps, pe, gs, ge)
    return pure / total


def recall(pred, gt):
    total = sum(ge - gs for gs, ge, _ in gt)
    if total <= 0:
        return 0.0
    covered = 0.0
    for gs, ge, gl in gt:
        clipped = [(max(gs, ps), min(ge, pe)) for ps, pe, pl in pred
                   if pl == gl and _overlap(gs, ge, ps, pe) > 0]
        covered += _union_len(clipped)
    return covered / total


def short_action_loss(pred, gt, min_gt_dur, min_cover):
    missed = []
    for gs, ge, gl in gt:
        dur = ge - gs
        if dur >= min_gt_dur:
            continue
        clipped = [(max(gs, ps), min(ge, pe)) for ps, pe, pl in pred
                   if pl == gl and _overlap(gs, ge, ps, pe) > 0]
        if _union_len(clipped) < min_cover * dur:
            missed.append(gl)
    return missed


def summarize(per_video_metrics, purity_gate=0.95, recall_gate=0.5):
    n = len(per_video_metrics) or 1
    mean_purity = sum(r["purity"] for r in per_video_metrics) / n
    mean_recall = sum(r["recall"] for r in per_video_metrics) / n
    short_lost_total = {}
    for r in per_video_metrics:
        for lb in r.get("short_lost", []):
            short_lost_total[lb] = short_lost_total.get(lb, 0) + 1
    return {
        "mean_purity": mean_purity,
        "mean_recall": mean_recall,
        "short_lost_total": short_lost_total,
        "gate_pass": mean_purity >= purity_gate and mean_recall >= recall_gate,
    }


def _doc_pred(doc):
    return [(s["start"], s["end"], s["label"]) for s in doc["segments"]]


def evaluate_docs(docs, gt, min_gt_dur=3.0, min_cover=0.5):
    """docs: {rel_path: doc}; gt: {rel_path: [[s,e,label],...]}。返回每视频度量行。"""
    rows = []
    for rel, doc in docs.items():
        g = [tuple(x) for x in gt.get(rel, [])]
        pred = _doc_pred(doc)
        rows.append({
            "rel_path": rel,
            "purity": boundary_purity(pred, g),
            "recall": recall(pred, g),
            "short_lost": short_action_loss(pred, g, min_gt_dur, min_cover),
        })
    return rows


def write_reports(rows, summary, csv_path, json_path):
    import csv
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rel_path", "purity", "recall", "short_lost"])
        for r in rows:
            w.writerow([r["rel_path"], f"{r['purity']:.4f}",
                        f"{r['recall']:.4f}", ";".join(r["short_lost"])])
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "rows": rows}, f, indent=2)


def main(argv=None):
    """真实运行：需 GPU/teacher/YOLO。人工执行。

    用法: python -m auto_presort.calibrate --species pet --videos <dir> \\
          --gt gt.json --checkpoint <teacher.pth> --out-dir <dir>
    """
    import argparse
    import os
    import pipeline_config as pc
    from auto_presort import detect_track, engine
    from auto_presort.checkpoint_guard import verify_and_load_teacher

    ap = argparse.ArgumentParser()
    ap.add_argument("--species", required=True)
    ap.add_argument("--videos", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--threshold", type=float, default=0.90)
    ap.add_argument("--trim-margin", type=float, default=1.0)
    a = ap.parse_args(argv)

    sp = pc.SPECIES[a.species]
    from ultralytics import YOLO
    yolo = YOLO(sp["yolo_model"])
    model, meta = verify_and_load_teacher(a.checkpoint, sp["class_names"])
    reader = engine.build_real_window_reader()
    classifier = engine.build_real_classifier(model, meta["class_names"],
        params=dict(imgsz=224, temporal_window=16, expand_ratio=1.25))
    params = dict(window_sec=3.0, stride_sec=1.0, threshold=a.threshold, enter_run=2,
                  trim_margin=a.trim_margin, min_duration=2.0, imgsz=224,
                  temporal_window=16, expand_ratio=1.25, no_animal_overlap=0.5)

    gt = json.loads(open(a.gt, encoding="utf-8").read())
    docs = {}
    for rel in gt:
        vp = os.path.join(a.videos, rel)
        track = detect_track.build_track(vp, yolo, sp["target_classes"])
        docs[rel] = engine.label_video(
            video_path=vp, rel_path=rel, source_id=rel, track=track,
            read_window_frames_fn=lambda v, s, e, fps, _b=track["boxes"]:
                reader(v, s, e, fps, _b),
            classify_frames_fn=classifier, meta=meta, params=params)

    rows = evaluate_docs(docs, gt)
    summary = summarize(rows)
    os.makedirs(a.out_dir, exist_ok=True)
    write_reports(rows, summary,
                  os.path.join(a.out_dir, "calib.csv"),
                  os.path.join(a.out_dir, "calib.json"))
    print(f"gate_pass={summary['gate_pass']} "
          f"purity={summary['mean_purity']:.3f} recall={summary['mean_recall']:.3f}")


if __name__ == "__main__":
    main()
