"""Inspect and compare newly recorded dataset lineage."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import store


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    trace = sub.add_parser("trace", help="show output -> raw -> aws source lineage")
    trace.add_argument("path")
    diff = sub.add_parser("diff", help="compare two snapshot manifests")
    diff.add_argument("left")
    diff.add_argument("right")
    manifest = sub.add_parser("manifest", help="write a snapshot manifest")
    manifest.add_argument("snapshot_root")
    manifest.add_argument("--species", required=True)
    manifest.add_argument("--date", required=True)
    manifest.add_argument("--frames", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "trace":
        chain = store.trace(args.path)
        if not chain:
            print(f"No lineage record for: {Path(args.path).resolve()}", file=sys.stderr)
            return 2
        print(json.dumps(chain, ensure_ascii=False, indent=2))
    elif args.command == "diff":
        print(json.dumps(store.diff_snapshots(args.left, args.right), ensure_ascii=False, indent=2))
    else:
        print(store.write_snapshot(args.snapshot_root, species=args.species,
                                   date=args.date, frames=args.frames))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
