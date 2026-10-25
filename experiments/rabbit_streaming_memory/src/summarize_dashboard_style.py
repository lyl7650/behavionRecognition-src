"""Aggregate per-video dashboard-style replay metrics and per-class events."""
from __future__ import annotations
import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
LAB = HERE.parent
RUNS = ("oct20", "jun24", "t0_seed44", "t1_seed42", "t2_seed43")
METRICS = ("time_iou_macro", "event_f1_tiou_05", "transition_f1", "p95_action_ms_mean_per_video")


def main():
    root = LAB / "outputs/dashboard_style_4hz"
    summaries = {}
    per_class_rows = []
    for run in RUNS:
        run_dir = root / run
        summary_path = run_dir / "metrics_summary.json"
        if not summary_path.is_file():
            raise FileNotFoundError(summary_path)
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        if payload["video_count"] != 30:
            raise RuntimeError(f"{run}: expected 30 scored videos, found {payload['video_count']}")
        summaries[run] = payload
        counts = {}
        for video_dir in sorted(p for p in run_dir.iterdir() if p.is_dir()):
            metrics_path = video_dir / "metrics.json"
            if not metrics_path.is_file():
                raise FileNotFoundError(metrics_path)
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            event_classes = metrics["p0_quality"]["event_detection_f1"]["by_class"]
            for label, item in event_classes.items():
                row = counts.setdefault(label, {"tp": 0, "fp": 0, "fn": 0})
                for key in ("tp", "fp", "fn"):
                    row[key] += int(item[key])
        for label, row in sorted(counts.items()):
            tp, fp, fn = row["tp"], row["fp"], row["fn"]
            precision = tp / (tp + fp) if tp + fp else None
            recall = tp / (tp + fn) if tp + fn else None
            f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None
            per_class_rows.append({"run": run, "class": label, **row, "precision": precision, "recall": recall, "f1": f1})

    summary_rows = []
    for run, payload in summaries.items():
        row = {"run": run, "videos": payload["video_count"], **payload["aggregate"]}
        summary_rows.append(row)
    summary_csv = root / "dashboard_style_comparison.csv"
    with summary_csv.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    class_csv = root / "dashboard_style_by_class.csv"
    with class_csv.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(per_class_rows[0]))
        writer.writeheader()
        writer.writerows(per_class_rows)
    report = {
        "scope": "4 Hz dashboard-style local replay; all five runs share videos, sample times, detector boxes, preprocessing, and isolated FusionAgent snapshot.",
        "comparability_warning": "Not a deployed dashboard evaluation. Uses training-repository YOLO and one assumed track per video. Do not compare causally with 2 Hz temporal-window runs because preprocessing and cadence differ.",
        "runs": summary_rows,
        "by_class": per_class_rows,
        "outputs": [str(summary_csv), str(class_csv)],
    }
    (root / "dashboard_style_comparison.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary_rows, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
