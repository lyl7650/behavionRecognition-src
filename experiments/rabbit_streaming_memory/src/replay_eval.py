"""Model-only replay for isolated I0/I1/I3 comparisons.

This deliberately evaluates already-cropped/decoded frame sequences supplied by
an npz file, so it can validate temporal state handling without copying or
modifying production dashboard code. Video+YOLO replay is implemented later
through the same trace schema.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
LAB = HERE.parent
sys.path.insert(0, str(HERE))
from timm_gru_student import TimmGRUStudent  # noqa: E402

CLASS_NAMES = ["resting", "active", "eating", "drinking"]


def load_model(checkpoint: Path, device: torch.device) -> TimmGRUStudent:
    model = TimmGRUStudent("repvit_m1", 4, 256, pretrained=False).to(device).eval()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
    return model


def replay_stream(model, frames: torch.Tensor, device: torch.device, reset_after: float | None, fps: float, force_reset_at: set[int] | None = None):
    hidden = None
    records = []
    prev_time = None
    with torch.inference_mode():
        for index, frame in enumerate(frames):
            current_time = index / fps
            reset_reason = None
            if force_reset_at and index in force_reset_at:
                hidden = None
                reset_reason = "forced_reset"
            elif prev_time is not None and reset_after is not None and current_time - prev_time > reset_after:
                hidden = None
                reset_reason = f"gap>{reset_after}s"
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            logits, hidden = model(frame.unsqueeze(0).to(device), hidden)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
            pred = int(np.argmax(probs))
            records.append({
                "frame_idx": index,
                "video_time_s": current_time,
                "raw_argmax": CLASS_NAMES[pred],
                "probabilities": {name: float(value) for name, value in zip(CLASS_NAMES, probs)},
                "reset_reason": reset_reason,
                "action_ms": elapsed_ms,
            })
            prev_time = current_time
    return records



def replay_window(model, frames: torch.Tensor, device: torch.device, window_size: int, fps: float):
    records = []
    with torch.inference_mode():
        for index in range(len(frames)):
            start = max(0, index - window_size + 1)
            clip = frames[start:index + 1]
            if len(clip) < window_size:
                clip = torch.cat([clip[:1].repeat(window_size - len(clip), 1, 1, 1), clip], dim=0)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            logits = model.forward_seq(clip.unsqueeze(0).to(device))
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
            pred = int(np.argmax(probs))
            records.append({"frame_idx": index, "video_time_s": index / fps, "raw_argmax": CLASS_NAMES[pred], "probabilities": {name: float(value) for name, value in zip(CLASS_NAMES, probs)}, "reset_reason": None, "action_ms": elapsed_ms})
    return records

def main():
    parser = argparse.ArgumentParser(description="Replay a synthetic/cropped frame sequence with GRU memory")
    parser.add_argument("--frames-npz", type=Path, required=True, help="NPZ containing frames [T,3,H,W] float/uint8")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--fps", type=float, default=4.0)
    parser.add_argument("--reset-after", type=float, default=2.0)
    parser.add_argument("--force-reset-at", type=int, action="append", default=[], help="frame index at which to clear hidden state")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    payload = np.load(args.frames_npz)
    if "frames" not in payload:
        raise ValueError("NPZ must contain frames")
    frames = torch.from_numpy(payload["frames"])
    if frames.ndim != 4 or frames.shape[1] != 3:
        raise ValueError(f"frames must have shape [T,3,H,W], got {tuple(frames.shape)}")
    frames = frames.float()
    if float(frames.max()) > 1.5:
        frames = frames / 255.0
    model = load_model(args.checkpoint, torch.device(args.device))
    records = replay_stream(model, frames, torch.device(args.device), args.reset_after, args.fps, set(args.force_reset_at))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"checkpoint": str(args.checkpoint.resolve()), "fps": args.fps, "reset_after": args.reset_after, "force_reset_at": args.force_reset_at, "records": records}, indent=2), encoding="utf-8")
    print(json.dumps({"records": len(records), "output": str(args.output), "p95_action_ms": float(np.percentile([r["action_ms"] for r in records], 95))}, indent=2))


if __name__ == "__main__":
    main()


