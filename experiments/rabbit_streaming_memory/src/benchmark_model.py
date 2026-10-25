from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil
import torch

HERE = Path(__file__).resolve().parent
LAB = HERE.parent
ROOT = LAB.parents[1]
sys.path.insert(0, str(ROOT))
from timm_gru_student import TimmGRUStudent  # noqa: E402

CHECKPOINTS = {
    "oct20": ROOT / "outputs/rabbit/2026-10-20/repvit_best_distilled_2026-10-20_ema.pth",
    "jun24": ROOT / "outputs/rabbit/2026-06-24/0624_repvit_best_distilled_ema.pth",
}


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure(call, warmup: int, iterations: int, device: torch.device) -> dict:
    with torch.inference_mode():
        for _ in range(warmup):
            call()
        sync(device)
        durations = []
        for _ in range(iterations):
            start = time.perf_counter_ns()
            call()
            sync(device)
            durations.append((time.perf_counter_ns() - start) / 1e6)
    ordered = sorted(durations)
    p95 = ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))]
    return {
        "iterations": iterations, "warmup": warmup,
        "p50_ms": statistics.median(durations), "p95_ms": p95,
        "mean_ms": statistics.fmean(durations),
        "steps_per_second": 1000.0 / statistics.mean(durations),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Local batch-1 RepViT-GRU latency benchmark; no camera required")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--output", type=Path, default=LAB / "outputs/local_latency.json")
    parser.add_argument("--checkpoint", action="append", default=[], help="additional NAME=PATH checkpoint to benchmark")
    args = parser.parse_args()
    if args.warmup < 1 or args.iterations < 100:
        parser.error("use at least 1 warmup and 100 measured iterations")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    device = torch.device("cuda" if args.device == "cuda" or
                          (args.device == "auto" and torch.cuda.is_available()) else "cpu")

    image = torch.zeros(1, 3, 224, 224, device=device)
    clip = torch.zeros(1, 16, 3, 224, 224, device=device)
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "local model-only synthetic-input microbenchmark; not camera or end-to-end deployment performance",
        "environment": {
            "os": platform.platform(), "processor": platform.processor(),
            "logical_cpu_count": psutil.cpu_count(), "ram_bytes": psutil.virtual_memory().total,
            "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
            "device": str(device), "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
            "input_size": [1, 3, 224, 224], "clip_size": [1, 16, 3, 224, 224],
            "batch_size": 1,
        },
        "settings": {"warmup": args.warmup, "iterations": args.iterations},
        "models": {},
    }
    checkpoints = dict(CHECKPOINTS)
    for item in args.checkpoint:
        if "=" not in item:
            parser.error("--checkpoint must be NAME=PATH")
        name, raw_path = item.split("=", 1)
        if not name or not Path(raw_path).is_file():
            parser.error(f"invalid checkpoint: {item}")
        checkpoints[name] = Path(raw_path)
    for name, checkpoint in checkpoints.items():
        model = TimmGRUStudent("repvit_m1", 4, hidden_dim=256, pretrained=False).to(device).eval()
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        hidden = None

        def stream_step():
            nonlocal hidden
            logits, hidden = model(image, hidden)
            if not torch.isfinite(logits).all():
                raise RuntimeError(f"non-finite stream output for {name}")

        def window_step():
            logits = model.forward_seq(clip)
            if not torch.isfinite(logits).all():
                raise RuntimeError(f"non-finite window output for {name}")

        hidden = None
        stream = measure(stream_step, args.warmup, args.iterations, device)
        report["models"][name] = {
            "checkpoint": str(checkpoint), "stream_single_frame": stream,
            "independent_16_frame_window": measure(window_step, args.warmup, args.iterations, device),
        }
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
