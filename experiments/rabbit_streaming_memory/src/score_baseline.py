"""Score an isolated baseline directory with the project's read-only metrics module."""
from __future__ import annotations
import argparse
import importlib.util
import json
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
LAB=HERE.parent
ROOT=LAB.parents[1]

def load_metrics():
    path=ROOT/'utils/metrics.py'
    sys.path.insert(0, str(ROOT))
    spec=importlib.util.spec_from_file_location('experiment_metrics', path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run-dir', type=Path, required=True)
    args=p.parse_args()
    metrics=load_metrics()
    rows=[]
    for video_dir in sorted(p for p in args.run_dir.iterdir() if p.is_dir()):
        summary=video_dir/'summary.json'
        trace=video_dir/'track_trace.jsonl'
        if not summary.exists() or not trace.exists():
            continue
        payload=json.loads(summary.read_text(encoding='utf-8'))
        annotation=Path(payload['annotation'])
        out=video_dir/'metrics.json'
        result=metrics.write_behavior_metrics(trace_path=trace, summary_path=summary, annotations_path=annotation, output_path=out, timeline_plot_path=video_dir/'gt_system_timeline.png', segment_iou_gt_classes_only=True)
        quality=result['p0_quality']; perf=result.get('p0_perf', {})
        rows.append({'video':video_dir.name, 'time_iou_macro':quality.get('segment_iou_macro'), 'event_f1_tiou_05':quality.get('event_detection_f1',{}).get('by_tiou',{}).get('0.5'), 'transition_f1':quality.get('transition_f1',{}).get('f1'), 'p95_action_ms':perf.get('p95_action_ms')})
    def mean(key):
        vals=[float(r[key]) for r in rows if r.get(key) is not None]
        return sum(vals)/len(vals) if vals else None
    report={'run_dir':str(args.run_dir.resolve()), 'video_count':len(rows), 'aggregate':{'time_iou_macro':mean('time_iou_macro'), 'event_f1_tiou_05':mean('event_f1_tiou_05'), 'transition_f1':mean('transition_f1'), 'p95_action_ms_mean_per_video':mean('p95_action_ms')}, 'videos':rows}
    (args.run_dir/'metrics_summary.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(report['aggregate'],indent=2,ensure_ascii=False))
if __name__=='__main__': main()
