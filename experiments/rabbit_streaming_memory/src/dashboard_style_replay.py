"""4 Hz dashboard-style replay using isolated model and fusion code."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
LAB = HERE.parent
ROOT = LAB.parents[1]
sys.path.insert(0, str(HERE))
from dashboard_fusion_snapshot import CLASS_NAMES, DashboardFusionSnapshot
from replay_eval import load_model

CHECKPOINTS = {
    "oct20": ROOT / "outputs/rabbit/2026-10-20/repvit_best_distilled_2026-10-20_ema.pth",
    "jun24": ROOT / "outputs/rabbit/2026-06-24/0624_repvit_best_distilled_ema.pth",
    "t0_seed44": LAB / "outputs/checkpoints/t0_seed44/model_ema.pth",
    "t1_seed42": LAB / "outputs/checkpoints/t1_seed42/model_ema.pth",
    "t2_seed43": LAB / "outputs/checkpoints/t2_seed43/model_ema.pth",
}
FUSION_SOURCE = Path(r"E:\ActionRecognition\handover\ppc_dashboard\backend\ml\decision\fusion_agent.py")
RECOGNIZER_SOURCE = Path(r"E:\ActionRecognition\handover\ppc_dashboard\backend\ml\behavior\recognizer.py")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def import_run_test():
    path = ROOT / "scripts/pipeline/run_test.py"
    sys.path.insert(0, str(ROOT))
    spec = importlib.util.spec_from_file_location("dashboard_style_run_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def crop_tensor(frame: np.ndarray, box, device: torch.device) -> torch.Tensor:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = map(int, box)
    pad_w = int((x2 - x1) * 0.25 / 2)
    pad_h = int((y2 - y1) * 0.25 / 2)
    x1, y1 = max(0, x1 - pad_w), max(0, y1 - pad_h)
    x2, y2 = min(width, x2 + pad_w), min(height, y2 + pad_h)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        raise RuntimeError(f"empty crop from box={box}, frame={frame.shape}")
    rgb = cv2.cvtColor(cv2.resize(crop, (224, 224)), cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb.copy()).permute(2, 0, 1).to(device=device, dtype=torch.float32) / 255.0
    mean = tensor.new_tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = tensor.new_tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (tensor - mean) / std


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=LAB / "outputs/dashboard_style_4hz")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--fps", type=float, default=4.0)
    parser.add_argument("--limit-videos", type=int, default=None, help="smoke-test only; omit to run all annotated videos")
    args = parser.parse_args()
    if args.fps <= 0:
        parser.error("--fps must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    for name, path in CHECKPOINTS.items():
        if not path.is_file():
            raise FileNotFoundError(f"{name} checkpoint not found: {path}")
    if not FUSION_SOURCE.is_file() or not RECOGNIZER_SOURCE.is_file():
        raise FileNotFoundError("dashboard source snapshot inputs are unavailable")

    run_test = import_run_test()
    pairs = run_test.find_rabbit_pairs()
    if len(pairs) != 30:
        raise RuntimeError(f"expected 30 annotated videos, got {len(pairs)}")
    device = torch.device(args.device)
    models = {name: load_model(checkpoint, device) for name, checkpoint in CHECKPOINTS.items()}
    detector, detect_cfg = run_test.load_detector("rabbit", device)
    output_rows = {name: [] for name in models}

    if args.limit_videos is not None:
        if args.limit_videos < 1:
            parser.error("--limit-videos must be positive")
        pairs = pairs[:args.limit_videos]
    for video_index, (video, annotation) in enumerate(pairs, 1):
        agents = {name: DashboardFusionSnapshot(track_id=0) for name in models}
        print(f"[{video_index}/{len(pairs)}] {video.name}", flush=True)
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            raise RuntimeError(f"cannot open video: {video}")
        native_fps = float(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if native_fps <= 0 or frame_count <= 0:
            cap.release()
            raise RuntimeError(f"invalid video metadata: {video} fps={native_fps} frames={frame_count}")
        sample_times = np.arange(0.0, frame_count / native_fps, 1.0 / args.fps)
        last_box = None
        rows = {name: [] for name in models}

        for timestamp in sample_times:
            pipeline_start = time.perf_counter()
            frame_index = min(frame_count - 1, int(round(timestamp * native_fps)))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = cap.read()
            if not ok or frame is None:
                cap.release()
                raise RuntimeError(f"failed to decode {video} at t={timestamp:.3f}s frame={frame_index}")
            box = run_test._detect_box(detector, frame, detect_cfg, last_box)
            if box is None:
                continue
            last_box = box
            tensor = crop_tensor(frame, box, device)
            pipeline_ms = (time.perf_counter() - pipeline_start) * 1000.0

            for name, model in models.items():
                action_times = []
                prob_map = None

                def predict(previous_hidden):
                    nonlocal prob_map
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    started = time.perf_counter()
                    with torch.inference_mode():
                        logits, next_hidden = model(tensor.unsqueeze(0), previous_hidden)
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    action_times.append((time.perf_counter() - started) * 1000.0)
                    probs = torch.softmax(logits, dim=-1)[0].detach().cpu().numpy()
                    if not np.isfinite(probs).all():
                        raise RuntimeError(f"non-finite probabilities: {name} {video} {timestamp}")
                    prob_map = {label: float(value) for label, value in zip(CLASS_NAMES, probs)}
                    return probs, next_hidden

                state = agents[name].update_with_predictor(box, predict, float(timestamp))
                rows[name].append({
                    "frame_idx": frame_index,
                    "video_time_s": float(timestamp),
                    "raw_argmax": None if prob_map is None else max(prob_map, key=prob_map.get),
                    "probabilities": prob_map or {},
                    "reset_reason": "gap>2s" if state["reset_after_gap"] else None,
                    "action_ms": action_times[0] if action_times else 0.0,
                    "frame_pipeline_ms": pipeline_ms,
                    "bbox": {"x1": float(box[0]), "y1": float(box[1]), "x2": float(box[2]), "y2": float(box[3])},
                    "target_fps": args.fps,
                    "track_id": 0,
                    "fusion": {"final_state": state["final_state"]},
                    "dashboard_fusion_snapshot": state,
                })
        cap.release()

        for name in models:
            out_dir = args.output_root / name / video.stem
            out_dir.mkdir(parents=True, exist_ok=True)
            trace_path = out_dir / "track_trace.jsonl"
            with trace_path.open("w", encoding="utf-8") as stream:
                for row in rows[name]:
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            (out_dir / "summary.json").write_text(json.dumps({
                "video": str(video.resolve()), "annotation": str(annotation.resolve()),
                "mode": "dashboard_style_4hz_single_track", "target_fps": args.fps,
                "native_fps": native_fps, "record_count": len(rows[name]),
                "detector_weights": detect_cfg["yolo_model"],
                "detector_expand_ratio": detect_cfg["expand_ratio"],
                "recognizer_context_expansion": 0.25,
                "scope_warning": "training-repo YOLO and single-track replay; not dashboard detector/tracker end-to-end",
            }, indent=2, ensure_ascii=False), encoding="utf-8")
            actions = [row["action_ms"] for row in rows[name] if row["action_ms"] > 0]
            output_rows[name].append({
                "video": video.name, "records": len(rows[name]),
                "model_calls": len(actions),
                "p95_model_call_ms": float(np.percentile(actions, 95)) if actions else None,
            })
        print(f"  samples={len(sample_times)} tracked={len(rows['oct20'])}", flush=True)

    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "replay_scope": "4Hz local replay using isolated FusionAgent candidate/throttle/FSM and 2s hidden reset; actual RepViT-GRU models; dashboard-like 25% context crop and ImageNet normalization.",
        "limitations": [
            "Uses training-repository YOLO rather than dashboard runtime detector.",
            "Assumes one persistent rabbit track per annotated video; does not reproduce production tracker lifecycle.",
            "No live camera, dashboard scheduler, or deployed runtime is included.",
            "action_ms is synchronized model-only latency; frame_pipeline_ms includes sampled-frame decode, detection, crop, and normalization.",
        ],
        "target_fps": args.fps,
        "video_count": len(pairs),
        "smoke_test_limited": args.limit_videos is not None,
        "device": str(device),
        "dashboard_sources": {
            "fusion_agent": {"path": str(FUSION_SOURCE), "sha256": sha256(FUSION_SOURCE)},
            "recognizer": {"path": str(RECOGNIZER_SOURCE), "sha256": sha256(RECOGNIZER_SOURCE)},
        },
        "models": {
            name: {"checkpoint": str(path.resolve()), "sha256": sha256(path), "videos": output_rows[name]}
            for name, path in CHECKPOINTS.items()
        },
    }
    (args.output_root / "dashboard_style_replay_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({name: {"videos": len(rows), "records": sum(row["records"] for row in rows), "model_calls": sum(row["model_calls"] for row in rows)} for name, rows in output_rows.items()}, indent=2))


if __name__ == "__main__":
    main()
