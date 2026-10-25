"""Train T0/T1/T2 in the isolated experiment package.

T0 uses the copied model with an independent 16-frame window and endpoint loss.
T1 uses contiguous blocks with uniform loss after burn-in.
T2 uses the same blocks but endpoint-only (MiniROAD step-function) loss.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
LAB = HERE.parent
ROOT = LAB.parents[1]
sys.path.insert(0, str(HERE))
from stream_dataset import StreamingVideoDataset, scan_videos  # noqa: E402
from data_selection import load_split_exclusions  # noqa: E402
from timm_gru_student import TimmGRUStudent  # noqa: E402


CLASS_NAMES = ["resting", "active", "eating", "drinking"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_model(checkpoint: Path, device: torch.device) -> TimmGRUStudent:
    model = TimmGRUStudent("repvit_m1", len(CLASS_NAMES), hidden_dim=256, pretrained=False).to(device)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    return model


def loss_for_mode(logits: torch.Tensor, labels: torch.Tensor, burn_in: int, mode: str) -> torch.Tensor:
    # logits [B,T,K], labels [B,T]. Burn-in gives the hidden state time to warm up.
    if burn_in >= logits.shape[1]:
        raise ValueError(f"burn_in={burn_in} leaves no supervised steps")
    supervised_logits = logits[:, burn_in:, :]
    supervised_labels = labels[:, burn_in:]
    if mode == "uniform":
        return F.cross_entropy(supervised_logits.reshape(-1, logits.shape[-1]), supervised_labels.reshape(-1))
    if mode == "step":
        return F.cross_entropy(supervised_logits[:, -1, :], supervised_labels[:, -1])
    raise ValueError(f"unsupported loss mode: {mode}")


def loss_for_window(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 2 or labels.ndim != 2 or logits.shape[0] != labels.shape[0]:
        raise ValueError(f"window logits/labels must be [B,K] and [B,T], got {tuple(logits.shape)} and {tuple(labels.shape)}")
    return F.cross_entropy(logits, labels[:, -1])


def evaluate(model: TimmGRUStudent, loader: DataLoader, device: torch.device, mode: str) -> dict:
    model.eval()
    total = correct = 0
    loss_sum = 0.0
    with torch.inference_mode():
        for batch in loader:
            x = batch["pixel_values"].to(device)
            y = batch["labels"].to(device)
            if mode == "window":
                logits = model.forward_seq(x)
                loss = loss_for_window(logits, y)
            else:
                logits, _ = model.forward_stream(x)
                loss = loss_for_mode(logits, y, int(batch["burn_in_frames"][0]), mode)
                logits = logits[:, -1]
            pred = logits.argmax(dim=-1)
            correct += int((pred == y[:, -1]).sum())
            total += y.shape[0]
            loss_sum += float(loss)
    return {"loss": loss_sum / max(1, len(loader)), "accuracy": correct / max(1, total), "samples": total}


def train_one(args: argparse.Namespace, mode: str, seed: int) -> Path:
    set_seed(seed)
    device = torch.device(args.device)
    data_root = Path(args.data_root)
    audit_path = Path(args.audit_manifest).resolve()
    selection_path = Path(args.selection_manifest).resolve()
    train_excluded, val_excluded = load_split_exclusions(
        audit_path, selection_path, data_root
    )
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    train_specs = scan_videos(data_root / "train", CLASS_NAMES, train_excluded)
    val_specs = scan_videos(data_root / "val", CLASS_NAMES, val_excluded)
    expected_counts = selection.get("expected_counts_after_selection")
    if expected_counts:
        actual_counts = {
            "train": {name: sum(spec.label == index for spec in train_specs) for index, name in enumerate(CLASS_NAMES)},
            "val": {name: sum(spec.label == index for spec in val_specs) for index, name in enumerate(CLASS_NAMES)},
        }
        if actual_counts != expected_counts:
            raise RuntimeError(f"selected dataset counts differ from approved manifest: expected={expected_counts}, actual={actual_counts}")
    if mode == "window":
        block_seconds = 16.0 / args.fps
        burn_in_seconds = 0.0
    else:
        block_seconds = args.block_seconds
        burn_in_seconds = args.burn_in_seconds
    train_ds = StreamingVideoDataset(train_specs, block_seconds, burn_in_seconds, args.imgsz, args.fps, True, args.fliplr, seed, pad_short=(mode == "window"))
    per_class = max(1, args.val_max_videos // len(CLASS_NAMES))
    selected_val=[]
    for class_index in range(len(CLASS_NAMES)):
        class_specs=[spec for spec in val_specs if spec.label==class_index]
        selected_val.extend(class_specs[:per_class])
    val_ds = StreamingVideoDataset(selected_val, block_seconds, burn_in_seconds, args.imgsz, args.fps, False, 0.0, seed, pad_short=(mode == "window"))
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    model = load_model(Path(args.init_checkpoint), device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best = float("inf")
    best_epoch = -1
    out_dir = Path(args.output_dir) / f"{args.mode}_seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    history = []
    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for batch in train_loader:
            x = batch["pixel_values"].to(device)
            y = batch["labels"].to(device)
            if mode == "window":
                logits = model.forward_seq(x)
                loss = loss_for_window(logits, y)
            else:
                logits, _ = model.forward_stream(x)
                loss = loss_for_mode(logits, y, int(batch["burn_in_frames"][0]), mode)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += float(loss.detach())
        metrics = evaluate(model, val_loader, device, mode)
        row = {"epoch": epoch + 1, "train_loss": running / max(1, len(train_loader)), **metrics}
        history.append(row)
        print(f"{mode} seed={seed} epoch={epoch + 1}/{args.epochs} train={row['train_loss']:.4f} val={row['loss']:.4f} acc={row['accuracy']:.4f}", flush=True)
        if metrics["loss"] < best:
            best = metrics["loss"]
            best_epoch = epoch + 1
            torch.save(model.state_dict(), out_dir / "model_ema.pth")
    sidecar = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_mode": args.mode,
        "loss_mode": mode,
        "seed": seed,
        "class_names": CLASS_NAMES,
        "class_order_source": "isolated trial config; matched pipeline_config.py rabbit order",
        "init_checkpoint": str(Path(args.init_checkpoint).resolve()),
        "init_checkpoint_sha256": sha256(Path(args.init_checkpoint)),
        "trainer_sha256": sha256(Path(__file__).resolve()),
        "model_copy_sha256": sha256(HERE / "timm_gru_student.py"),
        "trial_config_sha256": sha256(LAB / "configs/trial.json"),
        "audit_manifest": str(audit_path.resolve()),
        "audit_manifest_sha256": sha256(audit_path),
        "selection_manifest": str(selection_path.resolve()),
        "selection_manifest_sha256": sha256(selection_path),
        "data_root": str(data_root.resolve()),
        "block_seconds": block_seconds,
        "burn_in_seconds": burn_in_seconds,
        "fps": args.fps,
        "epochs": args.epochs,
        "best_epoch": best_epoch,
        "history": history,
    }
    (out_dir / "sidecar.json").write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    return out_dir / "model_ema.pth"


def main() -> None:
    parser = argparse.ArgumentParser(description="Isolated streaming-memory training")
    parser.add_argument("--mode", choices=("t0", "t1", "t2"), required=True)
    parser.add_argument("--data-root", type=Path, default=ROOT / "dataset/rabbit/2026-10-22")
    parser.add_argument("--init-checkpoint", type=Path, default=ROOT / "outputs/rabbit/2026-10-20/repvit_best_distilled_2026-10-20_ema.pth")
    parser.add_argument("--output-dir", type=Path, default=LAB / "outputs/checkpoints")
    parser.add_argument("--selection-manifest", type=Path, default=LAB / "manifests/training_selection_approved.json")
    parser.add_argument("--audit-manifest", type=Path, default=LAB / "manifests/audit_final_2026-10-24.json")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--block-seconds", type=float, default=12.0)
    parser.add_argument("--burn-in-seconds", type=float, default=5.0)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--fliplr", type=float, default=0.5)
    parser.add_argument("--val-max-videos", type=int, default=48)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true", help="audit and load one block without optimizer updates")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    mode_map = {"t0": "window", "t1": "uniform", "t2": "step"}
    if args.dry_run:
        args.epochs = 0
        train_one(args, mode_map[args.mode], args.seed)
    else:
        train_one(args, mode_map[args.mode], args.seed)


if __name__ == "__main__":
    main()
