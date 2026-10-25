"""Read-only baseline runner on the existing 30 annotated long videos.

It reuses the production repository only as an import source and writes every
result under the isolated experiment directory. It runs both independent
16-frame windows (I0) and continuous single-frame hidden recurrence (I1).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
LAB = HERE.parent
ROOT = LAB.parents[1]
sys.path.insert(0, str(HERE))
from replay_eval import load_model, replay_stream, replay_window  # noqa: E402


def import_run_test():
    path = ROOT / "scripts/pipeline/run_test.py"
    sys.path.insert(0, str(ROOT))
    spec = importlib.util.spec_from_file_location("production_run_test_snapshot", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description="Run isolated I0/I1 baselines on annotated rabbit videos")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--sample-every", type=float, default=0.5)
    parser.add_argument("--mode", choices=("window", "stream"), required=True)
    parser.add_argument("--reset-after", type=float, default=None, help="hidden reset gap in seconds; omit to retain hidden within each video")
    parser.add_argument("--force-reset-every", type=int, default=None, help="controlled intervention: clear hidden state every N sampled records")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    run_test = import_run_test()
    pairs = run_test.find_rabbit_pairs()
    if len(pairs) != 30:
        raise RuntimeError(f"expected 30 annotated videos, got {len(pairs)}")
    device = torch.device(args.device)
    model = load_model(args.checkpoint, device)
    detector, detect_cfg = run_test.load_detector("rabbit", device)
    cfg = run_test._recognizer_config()
    output_rows = []
    all_action_ms = []
    for index, (video, annotation) in enumerate(pairs, 1):
        print(f"[{index}/{len(pairs)}] {video.name}", flush=True)
        frames = []
        times = []
        boxes = []
        for _, t, clip, bbox in run_test._video_frames(video, args.sample_every, 16, int(cfg.get("imgsz", 224)), detector, detect_cfg):
            frames.append(clip[-1])
            times.append(t)
            boxes.append(bbox)
        frame_tensor = torch.stack(frames)
        started = time.perf_counter()
        force_reset_at = set()
        if args.force_reset_every is not None:
            if args.force_reset_every <= 0:
                raise ValueError("--force-reset-every must be positive")
            force_reset_at = set(range(args.force_reset_every, len(frame_tensor), args.force_reset_every))
        records = replay_window(model, frame_tensor, device, 16, 1.0 / args.sample_every) if args.mode == "window" else replay_stream(model, frame_tensor, device, args.reset_after, 1.0 / args.sample_every, force_reset_at)
        wall = time.perf_counter() - started
        video_dir = args.output_dir / video.stem
        video_dir.mkdir(parents=True, exist_ok=True)
        trace_path = video_dir / "track_trace.jsonl"
        with trace_path.open("w", encoding="utf-8") as handle:
            for record, t, bbox in zip(records, times, boxes):
                record["video_time_s"] = t
                record["track_id"] = 0
                record["fusion"] = {"final_state": record["raw_argmax"].upper()}
                record["bbox"] = None if bbox is None else {"x1": float(bbox[0]), "y1": float(bbox[1]), "x2": float(bbox[2]), "y2": float(bbox[3])}
                record["frame_pipeline_ms"] = 0.0
                record["target_fps"] = 1.0 / args.sample_every
                handle.write(json.dumps(record) + "\n")
        reset_count = sum(record["reset_reason"] is not None for record in records)
        (video_dir / "summary.json").write_text(json.dumps({"video": str(video), "annotation": str(annotation), "mode": args.mode, "reset_after": args.reset_after, "force_reset_every": args.force_reset_every, "reset_count": reset_count, "sample_every_s": args.sample_every, "wall_time_seconds": wall, "records": len(records)}, indent=2), encoding="utf-8")
        all_action_ms.extend(r["action_ms"] for r in records)
        output_rows.append({"video": video.name, "annotation": str(annotation), "records": len(records), "reset_count": reset_count, "wall_time_seconds": wall, "p95_action_ms": float(np.percentile([r["action_ms"] for r in records], 95))})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "run_summary.json").write_text(json.dumps({"checkpoint": str(args.checkpoint.resolve()), "mode": args.mode, "reset_after": args.reset_after, "force_reset_every": args.force_reset_every, "videos": output_rows, "aggregate": {"video_count": len(output_rows), "record_count": len(all_action_ms), "reset_count": sum(row["reset_count"] for row in output_rows), "p95_action_ms": float(np.percentile(all_action_ms, 95))}}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()


