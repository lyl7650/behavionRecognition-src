"""review-pack 打包脚本。

在本机运行：询问物种/站点/日期，列出该批次视频，让用户挑选要发送的视频，
把 review 前端 + 所选视频 + 对应 .autopresort/*.json 标注 + 启动器模板
拷贝到一个自包含的输出文件夹，方便同事在自己的机器（Windows/Mac）上
无需 conda/GPU 即可完成审阅。
"""
import os
import shutil
import sys

import auto_presort.store as store
from scripts.pipeline import pipeline_config as pc

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

VIDEO_EXTS = (".mp4", ".avi", ".mov", ".webm")

CODE_FILES = [
    "scripts/presort/auto_presort_review.py",
    "auto_presort/__init__.py",
    "auto_presort/store.py",
    "auto_presort/review_server.py",
    "auto_presort/review_app.html",
]
LAUNCHER_TMPLS = {   # src (相对项目根) -> 包内目标名
    "review_launchers/review.bat.tmpl": "review.bat",
    "review_launchers/review.command.tmpl": "review.command",
    "review_launchers/README.txt.tmpl": "README.txt",
}


def select_videos(all_rels, spec):
    """按 spec 从 all_rels 中挑选视频，保持原列表顺序、去重。

    spec 为空或 "all" -> 返回全部。否则解析逗号分隔的 token，每个 token
    为单个索引 "N" 或闭区间 "A-B"。越界索引忽略，非法/负数 token 跳过。
    """
    s = spec.strip()
    if s == "" or s == "all":
        return list(all_rels)
    n = len(all_rels)
    idxs = set()
    for token in s.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            if "-" in token:
                a, b = token.split("-", 1)
                start = int(a)
                end = int(b)
                for i in range(start, end + 1):
                    if 0 <= i < n:
                        idxs.add(i)
            else:
                i = int(token)
                if 0 <= i < n:
                    idxs.add(i)
        except ValueError:
            continue
    return [all_rels[i] for i in sorted(idxs)]


def assemble_pack(records_dir, out_dir, rels, project_root):
    """把 review 代码 + 视频 + 标注 doc + 启动器拷入 out_dir，返回统计信息。

    目录管理：根目录只放双击入口（review.bat/review.command/README.txt）；
    审核程序放 app/ 子目录，视频与 .autopresort 标注放 data/ 子目录。
    """
    # 1. review 代码文件 -> app/
    for rel in CODE_FILES:
        dst_rel = os.path.basename(rel) if rel.startswith("scripts/presort/") else rel
        dst = os.path.join(out_dir, "app", dst_rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(os.path.join(project_root, rel), dst)
    # 2. 启动器模板 (去掉 .tmpl 后缀) -> 根目录
    for src, dst_name in LAUNCHER_TMPLS.items():
        dst = os.path.join(out_dir, dst_name)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(os.path.join(project_root, src), dst)
    # 3. 视频 + 对应标注 doc -> data/
    n_docs = 0
    missing_docs = []
    for rel in rels:
        vid_dst = os.path.join(out_dir, "data", rel)
        os.makedirs(os.path.dirname(vid_dst), exist_ok=True)
        shutil.copy2(os.path.join(records_dir, rel), vid_dst)
        doc = store._doc_filename(rel)
        doc_src = os.path.join(records_dir, ".autopresort", doc)
        if os.path.exists(doc_src):
            doc_dst = os.path.join(out_dir, "data", ".autopresort", doc)
            os.makedirs(os.path.dirname(doc_dst), exist_ok=True)
            shutil.copy2(doc_src, doc_dst)
            n_docs += 1
        else:
            missing_docs.append(rel)
    return {"n_videos": len(rels), "n_docs": n_docs, "missing_docs": missing_docs}


def main():
    ctx = pc.ask_context()
    records = pc.records_dir(ctx["species"], ctx["site"], ctx["date"])
    if not os.path.isdir(records):
        print(f"Records directory not found: {records}")
        return 1

    all_rels = []
    for root, dirs, files in os.walk(records):
        # skip the .autopresort store dir and any dotdir
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for fn in files:
            if os.path.splitext(fn)[1].lower() in VIDEO_EXTS:
                rel = os.path.relpath(os.path.join(root, fn), records).replace("\\", "/")
                all_rels.append(rel)
    all_rels.sort()

    if not all_rels:
        print(f"No videos found in: {records}")
        return 1

    for i, rel in enumerate(all_rels):
        print(f"[{i}] {rel}")

    spec = input("Which videos to send? (e.g. 0-19 / all): ")
    rels = select_videos(all_rels, spec)
    if not rels:
        print("No videos selected. Nothing to do.")
        return 1

    out = os.path.join(PROJECT_ROOT, f"review-pack-{ctx['site']}-{ctx['date']}")
    if os.path.exists(out):
        print(f"Note: output folder already exists, it will be overwritten/merged: {out}")

    info = assemble_pack(records, out, rels, PROJECT_ROOT)

    print("")
    print("Pack complete.")
    print(f"  videos: {info['n_videos']}")
    print(f"  annotation docs: {info['n_docs']}")
    if info["missing_docs"]:
        print(f"  missing docs ({len(info['missing_docs'])}): {', '.join(info['missing_docs'])}")
    print(f"  output folder: {out}")
    print("  layout: review.bat / review.command / README.txt at top; code in app/, videos+annotations in data/")
    print("Next step: Send the whole folder to your colleague.")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
