from __future__ import annotations
import csv, json
from pathlib import Path

LAB = Path(__file__).resolve().parents[1]
OUT = LAB / "outputs"
RUNS = [
    "baseline_oct20_i0", "baseline_oct20_i1", "baseline_oct20_i2_fsm",
    "baseline_oct20_i1_reset2", "baseline_oct20_i3_forced_reset8",
    "baseline_jun24_i0", "baseline_jun24_i1",
]

def main():
    overall=[]
    per_video=[]
    artifact_counts={}
    for run in RUNS:
        folder=OUT/run
        summary=json.loads((folder/"metrics_summary.json").read_text(encoding="utf-8"))
        run_json=folder/"run_summary.json"
        metadata=json.loads(run_json.read_text(encoding="utf-8")) if run_json.exists() else {}
        agg=summary["aggregate"]
        overall.append({
            "run":run,"checkpoint":Path(metadata["checkpoint"]).name if metadata.get("checkpoint") else "",
            "mode":metadata.get("mode", "fsm_postprocess" if "fsm" in run else ""),
            "reset_after_s":metadata.get("reset_after", ""),
            "force_reset_every":metadata.get("force_reset_every", ""),
            "videos":summary["video_count"],
            "time_iou_macro":agg["time_iou_macro"],
            "event_f1_tiou_05":agg["event_f1_tiou_05"],
            "transition_f1":agg["transition_f1"],
            "reported_mean_video_p95_action_ms_unverified":agg.get("p95_action_ms_mean_per_video", ""),
        })
        for row in summary["videos"]:
            per_video.append({"run":run,**row})
        video_dirs=[p for p in folder.iterdir() if p.is_dir()]
        artifact_counts[run]={"video_dirs":len(video_dirs),"metric_json":sum((p/"metrics.json").is_file() for p in video_dirs),"timeline_png":sum((p/"gt_system_timeline.png").is_file() for p in video_dirs),"trace_jsonl":sum((p/"track_trace.jsonl").is_file() for p in video_dirs)}
    for filename, rows in [("comparison_overall.csv",overall),("comparison_per_video.csv",per_video)]:
        path=OUT/filename
        with path.open("w",newline="",encoding="utf-8-sig") as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]))
            writer.writeheader();writer.writerows(rows)
    (OUT/"comparison_artifact_audit.json").write_text(json.dumps(artifact_counts,indent=2),encoding="utf-8")
    print(json.dumps({"overall_rows":len(overall),"per_video_rows":len(per_video),"artifacts":artifact_counts},indent=2))
if __name__=="__main__": main()
