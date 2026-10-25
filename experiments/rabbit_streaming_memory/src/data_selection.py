from __future__ import annotations
import hashlib
import json
from pathlib import Path


def _normalized(path: str | Path) -> str:
    return str(Path(path).resolve()).lower()


def load_split_exclusions(audit_path: Path, selection_path: Path, data_root: Path) -> tuple[set[str], set[str]]:
    """Validate approved data-selection policy and return per-split exclusions."""
    audit_path=audit_path.resolve(); selection_path=selection_path.resolve(); data_root=data_root.resolve()
    audit=json.loads(audit_path.read_text(encoding="utf-8"))
    selection=json.loads(selection_path.read_text(encoding="utf-8"))
    if not audit.get("full_video_hashes_computed"):
        raise RuntimeError("full-hash audit is required before training")
    if Path(audit["dataset_root"]).resolve()!=data_root:
        raise RuntimeError("audit dataset_root does not match --data-root")
    if Path(selection["source_audit"]).resolve()!=audit_path:
        raise RuntimeError("selection manifest does not reference the supplied full-hash audit")
    expected_audit_hash=selection.get("selection_sha256_basis",{}).get("audit_sha256")
    if not expected_audit_hash or hashlib.sha256(audit_path.read_bytes()).hexdigest()!=expected_audit_hash:
        raise RuntimeError("full-hash audit contents changed after training selection review")
    if not selection.get("approved"):
        raise RuntimeError("training selection manifest is not approved")

    invalid={_normalized(row["path"]) for row in audit.get("invalid_videos",[])}
    selected_invalid={_normalized(path) for path in selection.get("excluded_invalid_paths",[])}
    if selected_invalid!=invalid:
        raise RuntimeError("selection manifest damaged paths do not exactly match the full-hash audit")

    train_duplicates=set(); val_duplicates=set()
    for group in audit.get("duplicate_sha256_across_splits",[]):
        train_group={_normalized(row["path"]) for row in group if row["split"]=="train"}
        val_group={_normalized(row["path"]) for row in group if row["split"]=="val"}
        if len(train_group)!=1 or len(val_group)!=1:
            raise RuntimeError("audit duplicate group does not contain exactly one train and one val file")
        train_duplicates.update(train_group); val_duplicates.update(val_group)
    selected_train={_normalized(path) for path in selection.get("excluded_train_duplicate_paths",[])}
    selected_val={_normalized(path) for path in selection.get("retained_validation_duplicate_paths",[])}
    if selected_train!=train_duplicates or selected_val!=val_duplicates:
        raise RuntimeError("selection manifest does not exactly implement keep-val/exclude-train for every audited duplicate")

    train_root=(data_root/"train").resolve(); val_root=(data_root/"val").resolve()
    invalid_train={path for path in invalid if Path(path).is_relative_to(train_root)}
    invalid_val={path for path in invalid if Path(path).is_relative_to(val_root)}
    if invalid_train|invalid_val != invalid:
        raise RuntimeError("invalid-video audit contains a path outside train/val")
    train_excluded=invalid_train|train_duplicates
    val_excluded=invalid_val
    if any(not Path(path).is_relative_to(train_root) for path in train_excluded):
        raise RuntimeError("train exclusion contains a path outside the train split")
    return train_excluded,val_excluded
