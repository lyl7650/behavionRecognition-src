from __future__ import annotations
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
LAB = HERE.parent
ROOT = LAB.parents[1]
sys.path.insert(0, str(HERE))
from data_selection import load_split_exclusions  # noqa: E402
from stream_dataset import StreamingVideoDataset, scan_videos  # noqa: E402
from timm_gru_student import TimmGRUStudent  # noqa: E402

CLASS_NAMES = ["resting", "active", "eating", "drinking"]


def checkpoint_model(path: Path, device: torch.device):
    model = TimmGRUStudent("repvit_m1", 4, hidden_dim=256, pretrained=False).to(device).eval()
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
    return model


def main():
    parser = argparse.ArgumentParser(description="Validate stream data/model/loss without training")
    parser.add_argument("--data-root", type=Path, default=ROOT / "dataset/rabbit/2026-10-22")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "outputs/rabbit/2026-10-20/repvit_best_distilled_2026-10-20_ema.pth")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--audit-manifest", type=Path, default=LAB / "manifests/audit_final_2026-10-24.json")
    parser.add_argument("--selection-manifest", type=Path, default=LAB / "manifests/training_selection_approved.json")
    parser.add_argument("--sample-root", type=Path, help="optional small readable root with class folders for interface validation")
    args = parser.parse_args()
    device = torch.device(args.device)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    scan_root = args.sample_root or args.data_root
    if args.sample_root:
        train_excluded=set(); val_excluded=set()
    else:
        train_excluded, val_excluded = load_split_exclusions(
            args.audit_manifest, args.selection_manifest, args.data_root
        )
    train_root = scan_root / "train" if (scan_root / "train").is_dir() else scan_root
    val_root = scan_root / "val" if (scan_root / "val").is_dir() else scan_root
    train = scan_videos(train_root, CLASS_NAMES, train_excluded)
    val = scan_videos(val_root, CLASS_NAMES, val_excluded)
    dataset = StreamingVideoDataset(train, block_seconds=12, burn_in_seconds=5, imgsz=224, fps=8, is_train=False)
    batch = dataset[args.sample_index % len(dataset)]
    burn_in = int(batch["burn_in_frames"])
    x = batch["pixel_values"].unsqueeze(0).to(device)
    y = batch["labels"].unsqueeze(0).to(device)
    model = checkpoint_model(args.checkpoint, device)
    with torch.inference_mode():
        stream_logits, stream_hidden = model.forward_stream(x)
        hidden = None
        step_logits = []
        for frame_index in range(min(8, x.shape[1])):
            logits, hidden = model(x[:, frame_index], hidden)
            step_logits.append(logits)
        step_logits = torch.stack(step_logits, dim=1)
    torch.testing.assert_close(step_logits, stream_logits[:, :len(step_logits[0])], rtol=2e-3, atol=2e-3)
    if stream_logits.shape[:2] != y.shape:
        raise AssertionError((stream_logits.shape, y.shape))
    uniform = torch.nn.functional.cross_entropy(stream_logits[:, burn_in:].reshape(-1, 4), y[:, burn_in:].reshape(-1))
    step = torch.nn.functional.cross_entropy(stream_logits[:, -1], y[:, -1])
    result = {"train_videos": len(train), "val_videos": len(val), "block_shape": list(x.shape), "hidden_shape": list(stream_hidden.shape), "burn_in_frames": burn_in, "uniform_loss": float(uniform), "step_loss": float(step), "equivalence": True, "sample": batch["video_path"]}
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

