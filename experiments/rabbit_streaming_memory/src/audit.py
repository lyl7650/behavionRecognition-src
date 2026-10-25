from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import cv2
import torch

HERE = Path(__file__).resolve().parent
LAB = HERE.parent
ROOT = LAB.parents[1]
sys.path.insert(0, str(ROOT))
from timm_gru_student import TimmGRUStudent  # noqa: E402

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".webm"}
CHECKPOINTS = {
    "oct20": ROOT / "outputs/rabbit/2026-10-20/repvit_best_distilled_2026-10-20_ema.pth",
    "jun24": ROOT / "outputs/rabbit/2026-06-24/0624_repvit_best_distilled_ema.pth",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_snapshot() -> dict:
    source = ROOT / "models/timm_gru_student.py"
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
        capture_output=True, text=True,
    )
    status = subprocess.run(
        ["git", "status", "--short", "--", "models/timm_gru_student.py"],
        cwd=ROOT, check=True, capture_output=True, text=True,
    )
    return {
        "source_path": str(source),
        "source_sha256": sha256(source),
        "copied_path": str(HERE / "timm_gru_student.py"),
        "copied_sha256": sha256(HERE / "timm_gru_student.py"),
        "git_head": result.stdout.strip(),
        "source_has_uncommitted_changes": bool(status.stdout.strip()),
    }


def inspect_video(path: Path) -> dict:
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"cannot open video: {path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0 or frames <= 0:
            raise RuntimeError(f"invalid video metadata: {path} (fps={fps}, frames={frames})")
    finally:
        capture.release()
    stat = path.stat()
    return {
        "path": str(path.resolve()), "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns, "fps": fps,
        "frame_count": frames, "duration_seconds": frames / fps,
    }


def scan_dataset(root: Path, full_hash: bool) -> tuple[list[dict], dict, list[dict]]:
    rows = []
    errors = []
    totals: dict[str, Counter] = {"train": Counter(), "val": Counter()}
    batches = [root] if any((root / split).is_dir() for split in ("train", "val")) else sorted(root.iterdir())
    for batch in batches:
        if not batch.is_dir():
            continue
        for split in ("train", "val"):
            split_dir = batch / split
            if not split_dir.is_dir():
                continue
            for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
                for path in sorted(class_dir.rglob("*")):
                    if not path.is_file() or path.suffix.lower() not in VIDEO_EXTENSIONS:
                        continue
                    try:
                        row = inspect_video(path)
                    except Exception as error:
                        errors.append({"path": str(path.resolve()), "error": str(error)})
                        continue
                    row.update({"batch": "root" if batch == root else batch.name,
                                "split": split, "label": class_dir.name})
                    if full_hash:
                        row["sha256"] = sha256(path)
                    totals[split][class_dir.name] += 1
                    rows.append(row)
    if not rows and not errors:
        raise RuntimeError(f"no videos found below {root}")
    return rows, {split: dict(counts) for split, counts in totals.items()}, errors


def audit_checkpoints() -> dict:
    reports = {}
    for name, path in CHECKPOINTS.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        model = TimmGRUStudent("repvit_m1", 4, hidden_dim=256, pretrained=False)
        state = torch.load(path, map_location="cpu", weights_only=True)
        incompatible = model.load_state_dict(state, strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(f"strict load failed for {path}: {incompatible}")
        reports[name] = {
            "path": str(path), "sha256": sha256(path),
            "size_bytes": path.stat().st_size,
            "strict_load": True, "class_count": int(model.fc.out_features),
            "class_order": None,
        }
    return reports


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only audit for rabbit streaming-memory trials")
    parser.add_argument("--data-root", type=Path, default=ROOT / "dataset/rabbit/2026-10-22")
    parser.add_argument("--full-hash", action="store_true", help="SHA-256 every dataset video")
    parser.add_argument("--output", type=Path, default=LAB / "manifests/audit.json")
    args = parser.parse_args()

    rows, counts, video_errors = scan_dataset(args.data_root, args.full_hash)
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(args.data_root.resolve()),
        "full_video_hashes_computed": args.full_hash,
        "counts": counts,
        "video_count": len(rows),
        "invalid_videos": video_errors,
        "duplicate_sha256_across_splits": [],
        "videos": rows,
        "checkpoints": audit_checkpoints(),
        "model_source_snapshot": source_snapshot(),
        "class_order_confirmed": False,
        "class_order_note": "Raw state_dict files do not contain class names; provide and verify a separate mapping before scoring.",
    }
    if args.full_hash:
        owners: dict[str, list[dict]] = {}
        for row in rows:
            owners.setdefault(row["sha256"], []).append(row)
        report["duplicate_sha256_across_splits"] = [
            group for group in owners.values()
            if len({row["split"] for row in group}) > 1
        ]
    report["audit_passed"] = (
        args.full_hash and not video_errors and not report["duplicate_sha256_across_splits"]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"counts": counts, "video_count": len(rows),
                      "checkpoint_strict_load": {k: v["strict_load"] for k, v in report["checkpoints"].items()},
                      "invalid_video_count": len(video_errors),
                      "audit_passed": report["audit_passed"],
                      "manifest": str(args.output)}, indent=2))
    if not report["audit_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
