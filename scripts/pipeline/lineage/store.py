"""Append-only provenance records for newly produced dataset artifacts."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
LINEAGE_ROOT = PROJECT_ROOT / "dataset" / "lineage"
INGEST_PATH = LINEAGE_ROOT / "ingest.jsonl"
DERIVATION_PATH = LINEAGE_ROOT / "derivations.jsonl"
SCENES_PATH = LINEAGE_ROOT / "snapshots"
CHUNK_SIZE = 8 * 1024 * 1024


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def normalize_path(path: str | os.PathLike[str]) -> str:
    return os.path.abspath(os.path.normpath(os.fspath(path)))


def content_id(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while True:
            chunk = stream.read(CHUNK_SIZE)
            if not chunk:
                break
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records = []
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid lineage JSON at {path}:{line_no}: {exc}") from exc
    return records


def _append_unique(path: Path, record: dict[str, Any], key_fields: tuple[str, ...]) -> dict[str, Any]:
    _ensure_parent(path)
    key = tuple(record.get(field) for field in key_fields)
    for old in _read_jsonl(path):
        if tuple(old.get(field) for field in key_fields) == key:
            return old
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    return record


def find_records(path: Path, **filters: Any) -> list[dict[str, Any]]:
    return [record for record in _read_jsonl(path)
            if all(record.get(k) == v for k, v in filters.items())]


def record_ingest(source_path: str, raw_path: str, *, species: str | None = None,
                  batch_id: str | None = None, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    source = normalize_path(source_path)
    raw = normalize_path(raw_path)
    record = {
        "schema": "lineage-ingest-v1",
        "content_id": content_id(raw),
        "source_path": source,
        "raw_path": raw,
        "species": species,
        "batch_id": batch_id,
        "metadata": metadata or {},
        "recorded_at": now_utc(),
    }
    return _append_unique(INGEST_PATH, record, ("content_id", "raw_path"))


def aws_identity_from_path(path: str) -> dict[str, str]:
    """Extract deployment/device/date/record_uuid from an AWS backup-style path."""
    values = {}
    for part in Path(normalize_path(path)).parts:
        if "=" in part:
            key, value = part.split("=", 1)
            if key in {"deployment_id", "device_id", "date", "record_uuid"}:
                values[key] = value
    return values


def record_input_if_root(path: str, *, species: str | None = None) -> dict[str, Any] | None:
    """Record a true input only when it has no known producer in this ledger."""
    normalized = normalize_path(path)
    if find_records(DERIVATION_PATH, output_path=normalized):
        return None
    existing = find_records(INGEST_PATH, raw_path=normalized)
    if existing:
        return existing[-1]
    return record_ingest(normalized, normalized, species=species,
                         metadata={"source_kind": "pipeline_input",
                                   **aws_identity_from_path(normalized)})


def record_derivation(output_path: str, parent_path: str, *, operation: str,
                      label: str | None = None, parent_content_id: str | None = None,
                      parameters: dict[str, Any] | None = None,
                      source_path: str | None = None,
                      extra: dict[str, Any] | None = None) -> dict[str, Any]:
    output = normalize_path(output_path)
    parent = normalize_path(parent_path)
    if not os.path.isfile(output):
        raise FileNotFoundError(f"cannot record missing output: {output}")
    if not os.path.isfile(parent):
        raise FileNotFoundError(f"cannot record missing parent: {parent}")
    record = {
        "schema": "lineage-derivation-v1",
        "content_id": content_id(output),
        "parent_content_id": parent_content_id or content_id(parent),
        "output_path": output,
        "parent_path": parent,
        "source_path": normalize_path(source_path) if source_path else None,
        "operation": operation,
        "label": label,
        "parameters": parameters or {},
        "recorded_at": now_utc(),
    }
    if extra:
        record.update(extra)
    return _append_unique(DERIVATION_PATH, record, ("content_id", "output_path", "parent_content_id"))


def trace(path: str) -> list[dict[str, Any]]:
    target = normalize_path(path)
    ingests = _read_jsonl(INGEST_PATH)
    derivations = _read_jsonl(DERIVATION_PATH)
    by_output = {normalize_path(r["output_path"]): r for r in derivations}
    chain = []
    current = target
    seen = set()
    while current not in seen:
        seen.add(current)
        derivation = by_output.get(current)
        if derivation:
            chain.append(derivation)
            current = normalize_path(derivation["parent_path"])
            continue
        matching = [r for r in ingests if normalize_path(r["raw_path"]) == current]
        if matching:
            chain.extend(matching)
        break
    return chain


def write_snapshot(snapshot_root: str, *, species: str, date: str, frames: bool = False) -> Path:
    root = Path(normalize_path(snapshot_root))
    derivations = _read_jsonl(DERIVATION_PATH)
    files = []
    for split in ("train", "val"):
        split_root = root / split
        if not split_root.is_dir():
            continue
        for path in sorted(p for p in split_root.rglob("*") if p.is_file() and not p.name.endswith(".json")):
            item = {"path": str(path), "split": split, "content_id": content_id(path)}
            matches = [r for r in derivations if normalize_path(r["output_path"]) == normalize_path(path)]
            if matches:
                item["lineage"] = matches[-1]
            files.append(item)
    manifest = {
        "schema": "lineage-snapshot-v1",
        "species": species,
        "date": date,
        "frames": frames,
        "snapshot_root": str(root),
        "created_at": now_utc(),
        "files": files,
    }
    out = SCENES_PATH / f"{species}-{date}-{'frames' if frames else 'videos'}.manifest.json"
    _ensure_parent(out)
    fd, temp_name = tempfile.mkstemp(prefix=out.name + ".", suffix=".tmp", dir=str(out.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(manifest, stream, ensure_ascii=False, indent=2)
        os.replace(temp_name, out)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)
    return out


def diff_snapshots(left: str, right: str) -> dict[str, list[dict[str, Any]]]:
    with open(left, encoding="utf-8") as stream:
        a = json.load(stream)
    with open(right, encoding="utf-8") as stream:
        b = json.load(stream)
    amap = {item["content_id"]: item for item in a.get("files", [])}
    bmap = {item["content_id"]: item for item in b.get("files", [])}
    return {
        "added": [bmap[k] for k in sorted(set(bmap) - set(amap))],
        "removed": [amap[k] for k in sorted(set(amap) - set(bmap))],
        "common": [bmap[k] for k in sorted(set(amap) & set(bmap))
                   if (amap[k].get("path"), amap[k].get("split")) !=
                      (bmap[k].get("path"), bmap[k].get("split"))],
    }
