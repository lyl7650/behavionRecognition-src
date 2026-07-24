from pathlib import Path
import shutil
import sys

SPLIT_PREFIX = "3.20重制数据集-"
SOURCE_DIR_NAME = "3.20重制数据集"
MERGED_DIR_NAME = "重制数据集-merged"
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}


def remove_empty_dirs(root: Path) -> None:
    for path in sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        if path.is_dir():
            try:
                path.rmdir()
            except OSError:
                pass

    try:
        root.rmdir()
    except OSError:
        pass


def merge_split_dirs(root: Path) -> None:
    merged_root = root / MERGED_DIR_NAME
    merged_root.mkdir(exist_ok=True)

    moved_count = 0
    deleted_count = 0

    for split_dir in sorted(root.iterdir()):
        if not split_dir.is_dir() or not split_dir.name.startswith(SPLIT_PREFIX):
            continue

        source_root = split_dir / SOURCE_DIR_NAME
        if not source_root.is_dir():
            continue

        for src_file in sorted(source_root.rglob("*")):
            if not src_file.is_file() or src_file.suffix.lower() not in VIDEO_SUFFIXES:
                continue

            relative_path = src_file.relative_to(source_root)
            dst_file = merged_root / relative_path
            dst_file.parent.mkdir(parents=True, exist_ok=True)

            if dst_file.exists():
                src_file.unlink()
                deleted_count += 1
                continue

            shutil.move(str(src_file), str(dst_file))
            moved_count += 1

        remove_empty_dirs(split_dir)

    print(f"moved: {moved_count}")
    print(f"deleted duplicates: {deleted_count}")
    print(f"merged directory: {merged_root}")


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage: python utils/merge_vomit_convulsion_raw_video.py <raw_video_root>"
        )

    root = Path(sys.argv[1]).expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"Directory not found: {root}")

    merge_split_dirs(root)


if __name__ == "__main__":
    main()
