"""Pack a subset of scattered other/ clips into a standalone review bundle for a colleague.

Runs on your machine: lists every video across all other/ dirs (under pet raw_root),
you pick a range (e.g. 0-30), and it copies the review front-end + chosen videos +
their .autopresort/*.json + double-click launchers into a self-contained folder your
colleague can run on Windows/Mac with no conda/GPU.

Layout inside the pack mirrors each video's namespace path so merge_other_split.py can
put returned annotations back into the right original other/ dir:
    data/<prefix>/<video>.mp4
    data/<prefix>/.autopresort/<video>.mp4.json
where <prefix> is the other/ dir's path relative to raw_root (ends in ".../other").
"""
import os
import shutil
import sys

import auto_presort.store as store
from auto_presort.other_split import find_other_dirs, build_prefixes
from scripts.pipeline import pipeline_config as pc

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

VIDEO_EXTS = (".mp4", ".avi", ".mov", ".webm")

CODE_FILES = [
    "scripts/presort/auto_presort_review_multidir.py",
    "auto_presort/__init__.py",
    "auto_presort/store.py",
    "auto_presort/other_split.py",
    "auto_presort/review_server.py",
    "auto_presort/review_app.html",
]
LAUNCHER_TMPLS = {   # src (相对项目根) -> 包内目标名
    "review_launchers_other_split/review.bat.tmpl": "review.bat",
    "review_launchers_other_split/review.command.tmpl": "review.command",
    "review_launchers_other_split/README.txt.tmpl": "README.txt",
}


def list_all_clips(prefixes):
    """把所有 other 目录里的视频摊平成 [(prefix, physical_dir, filename)]，按命名空间排序。"""
    clips = []
    for prefix, d in prefixes:
        for fn in sorted(os.listdir(d)):
            if os.path.splitext(fn)[1].lower() in VIDEO_EXTS:
                clips.append((prefix, d, fn))
    clips.sort(key=lambda c: (c[0], c[2]))
    return clips


def select_clips(all_clips, spec):
    """按 spec 从 all_clips 中挑选，保持列表顺序、去重。

    spec 为空或 "all" -> 全部。否则解析逗号分隔的 "N" 或闭区间 "A-B"；越界/非法跳过。
    """
    s = spec.strip()
    if s == "" or s == "all":
        return list(all_clips)
    n = len(all_clips)
    idxs = set()
    for token in s.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            if "-" in token:
                a, b = token.split("-", 1)
                for i in range(int(a), int(b) + 1):
                    if 0 <= i < n:
                        idxs.add(i)
            else:
                i = int(token)
                if 0 <= i < n:
                    idxs.add(i)
        except ValueError:
            continue
    return [all_clips[i] for i in sorted(idxs)]


def assemble_pack(out_dir, clips, project_root):
    """把 review 代码 + 选中视频 + 对应标注 doc + 启动器拷入 out_dir，返回统计。

    根目录只放双击入口；代码放 app/，视频与标注按 data/<prefix>/ 平铺。
    """
    for rel in CODE_FILES:
        dst_rel = os.path.basename(rel) if rel.startswith("scripts/presort/") else rel
        dst = os.path.join(out_dir, "app", dst_rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(os.path.join(project_root, rel), dst)
    for src, dst_name in LAUNCHER_TMPLS.items():
        dst = os.path.join(out_dir, dst_name)
        shutil.copy2(os.path.join(project_root, src), dst)

    n_docs, missing_docs = 0, []
    for prefix, physical_dir, fn in clips:
        vid_dst = os.path.join(out_dir, "data", *prefix.split("/"), fn)
        os.makedirs(os.path.dirname(vid_dst), exist_ok=True)
        shutil.copy2(os.path.join(physical_dir, fn), vid_dst)
        doc_name = store._doc_filename(fn)
        doc_src = os.path.join(physical_dir, ".autopresort", doc_name)
        if os.path.exists(doc_src):
            doc_dst = os.path.join(out_dir, "data", *prefix.split("/"),
                                   ".autopresort", doc_name)
            os.makedirs(os.path.dirname(doc_dst), exist_ok=True)
            shutil.copy2(doc_src, doc_dst)
            n_docs += 1
        else:
            missing_docs.append(f"{prefix}/{fn}")
    return {"n_videos": len(clips), "n_docs": n_docs, "missing_docs": missing_docs}


def main():
    root = pc.SPECIES["pet"]["raw_root"]
    other_dirs = find_other_dirs(root)
    if not other_dirs:
        print(f"No 'other' directories found under: {root}")
        return 1
    prefixes = build_prefixes(root, other_dirs)
    all_clips = list_all_clips(prefixes)
    if not all_clips:
        print(f"No videos found in any other/ dir under: {root}")
        return 1

    for i, (prefix, _, fn) in enumerate(all_clips):
        print(f"[{i}] {prefix}/{fn}")

    spec = input("Which videos to send? (e.g. 0-30 / all): ")
    clips = select_clips(all_clips, spec)
    if not clips:
        print("No videos selected. Nothing to do.")
        return 1

    name = input("Pack name suffix (e.g. colleague1): ").strip() or "other-split"
    out = os.path.join(PROJECT_ROOT, f"review-pack-{name}")
    if os.path.exists(out):
        print(f"Note: output folder already exists, it will be overwritten/merged: {out}")

    info = assemble_pack(out, clips, PROJECT_ROOT)

    print("")
    print("Pack complete.")
    print(f"  videos: {info['n_videos']}")
    print(f"  annotation docs: {info['n_docs']}")
    if info["missing_docs"]:
        print(f"  missing docs ({len(info['missing_docs'])}): run 2a2_seed first for these")
        for m in info["missing_docs"]:
            print(f"    - {m}")
    print(f"  output folder: {out}")
    print("  layout: review.bat / review.command / README.txt at top; code in app/, videos+annotations in data/")
    print("Next step: Send the whole folder to your colleague.")
    print("When they return it, run merge_other_split.py to put annotations back.")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
