"""Merge a colleague's reviewed annotations back into the original other/ dirs.

Point --pack at the review-pack folder they returned (or just its data/ subdir). The pack
lays videos out as data/<prefix>/<video> and annotations as
data/<prefix>/.autopresort/<video>.json, where <prefix> ends in ".../other" and is the
other/ dir's path relative to the pet raw_root — so each returned doc maps straight back
to the original other/ dir's .autopresort/.

Only .json annotation files are copied back (never the videos). By default an incoming doc
overwrites the local one only if it is approved; pass --force to overwrite regardless.
Usage: python merge_other_split.py --pack review-pack-colleague1 [--force]
"""
import argparse
import json
import os
import sys

import auto_presort.store as store
from auto_presort.other_split import find_other_dirs, build_prefixes, resolve_prefix
from scripts.pipeline import pipeline_config as pc


def _iter_incoming_docs(data_dir):
    """产出 (namespaced_rel, doc_path)：namespaced_rel = "<prefix>/<video filename>"。

    .autopresort/<flat>.json 的文件名是把 rel_path 的 "/" 换成 "__"，这里只用它的
    rel_path 字段（本工作流里就是纯文件名），前缀取该 .autopresort 的父目录相对 data_dir 的路径。
    """
    for dirpath, dirnames, files in os.walk(data_dir):
        if os.path.basename(dirpath) != ".autopresort":
            continue
        parent = os.path.dirname(dirpath)
        prefix = os.path.relpath(parent, data_dir).replace("\\", "/")
        for fn in files:
            if fn.endswith(".json") and fn != "manifest.json":
                with open(os.path.join(dirpath, fn), encoding="utf-8") as f:
                    doc = json.load(f)
                yield f"{prefix}/{doc['rel_path']}", doc


def merge_back(data_dir, prefixes, force=False):
    """把 data_dir 里的标注 doc 按命名空间放回各自原来的 other/ 目录。返回统计。"""
    merged, skipped_unapproved, unresolved = 0, [], []
    for namespaced_rel, doc in _iter_incoming_docs(data_dir):
        physical_dir, local = resolve_prefix(prefixes, namespaced_rel)
        if physical_dir is None:
            unresolved.append(namespaced_rel)
            continue
        if not force and not doc.get("approved"):
            skipped_unapproved.append(namespaced_rel)
            continue
        store.save_doc(physical_dir, doc)
        merged += 1
    return {"merged": merged, "skipped_unapproved": skipped_unapproved,
            "unresolved": unresolved}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True,
                    help="the returned review-pack folder (or its data/ subdir)")
    ap.add_argument("--force", action="store_true",
                    help="overwrite local docs even if the incoming one is not approved")
    a = ap.parse_args(argv)

    data_dir = a.pack
    if os.path.isdir(os.path.join(a.pack, "data")):
        data_dir = os.path.join(a.pack, "data")
    if not os.path.isdir(data_dir):
        print(f"[ERROR] Not a folder: {data_dir}")
        return 1

    root = pc.SPECIES["pet"]["raw_root"]
    prefixes = build_prefixes(root, find_other_dirs(root))

    info = merge_back(data_dir, prefixes, force=a.force)
    print(f"[DONE] merged={info['merged']} "
          f"skipped_unapproved={len(info['skipped_unapproved'])} "
          f"unresolved={len(info['unresolved'])}")
    for rel in info["skipped_unapproved"]:
        print(f"  [SKIP unapproved] {rel} (use --force to merge anyway)")
    for rel in info["unresolved"]:
        print(f"  [WARN unresolved] {rel} (no matching other/ dir under {root})")
    if info["merged"]:
        print("Next: double-click 2c2_apply_other_split.bat to cut the approved segments.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]) or 0)
