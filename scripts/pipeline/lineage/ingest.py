"""Register a raw batch and its files in the append-only lineage ledger."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .store import aws_identity_from_path, record_ingest

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".webm", ".mkv"}


def register_tree(source_root: str, raw_root: str, *, species: str | None = None,
                  batch_id: str | None = None, source_base: str | None = None) -> int:
    source = Path(source_root).resolve()
    raw = Path(raw_root).resolve()
    count = 0
    for path in sorted(raw.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in VIDEO_EXTS:
            continue
        if source_base is None:
            source_path = source / path.relative_to(raw)
        else:
            source_path = Path(source_base) / path.name
        record_ingest(str(source_path), str(path), species=species, batch_id=batch_id,
                      metadata=aws_identity_from_path(source_path))
        count += 1
    return count


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_root", help="G:\\aws_backup batch or source tree")
    parser.add_argument("raw_root", help="matching dataset/raw_videos tree")
    parser.add_argument("--species")
    parser.add_argument("--batch-id")
    args = parser.parse_args(argv)
    print(f"registered {register_tree(args.source_root, args.raw_root, species=args.species, batch_id=args.batch_id)} video(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
