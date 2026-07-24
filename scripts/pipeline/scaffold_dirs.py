"""脚手架入口：给定物种/机构/日期，建好空目录供人放视频/剪辑。"""
import os
from . import pipeline_config as pc


def build_dirs(ctx, root="."):
    """按 ctx（{"species","site","date"}）在 root 下建 records/<date> 目录 + sorted-<date>/<各类别> 目录。

    返回创建的目录列表（records 一个 + 每个类别一个）。
    """
    species, site, date = ctx["species"], ctx["site"], ctx["date"]
    created = []

    # 创建 records 目录
    records = os.path.join(root, pc.records_dir(species, site, date))
    os.makedirs(records, exist_ok=True)
    created.append(records)

    # 创建 sorted 下的各类别目录
    for class_dir in pc.sorted_class_dirs(species, site, date):
        full = os.path.join(root, class_dir)
        os.makedirs(full, exist_ok=True)
        created.append(full)

    return created


def main():
    """交互式构建脚手架目录。"""
    ctx = pc.ask_context()
    created = build_dirs(ctx)
    print("Created the following directories:")
    for d in created:
        print("  ", d)
    print("\nNext: put the downloaded long videos into the records directory; "
          "put the trimmed clips into the matching class subfolders under sorted.")


if __name__ == "__main__":
    main()
