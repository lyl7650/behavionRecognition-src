import os
import re
import shutil
from datetime import date

_ISO = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")

# auto_presort 切片命名 <原视频名>-<start>-<end>.mp4；run_crop 会再追加 _clip-N。
# 剥掉尾部时间段（及可选的 _clip-N 与扩展名）还原来源，供同源跨 split 泄漏检测。
_SPAN = re.compile(r"^(.*?)-\d+(?:\.\d+)?-\d+(?:\.\d+)?(?:_clip-\d+)?$")


def _source_key(name):
    stem = re.sub(r"\.(mp4|avi|mov|webm)$", "", name, flags=re.IGNORECASE)
    m = _SPAN.match(stem)
    return m.group(1) if m else name


def _parse_iso(name):
    m = _ISO.match(name)
    if not m:
        return None
    try:
        return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    except ValueError:
        return None


def _os_dir_lister(root):
    return [n for n in os.listdir(root) if os.path.isdir(os.path.join(root, n))]


def find_latest_old_snapshot(dataset_root, current_date, dir_lister=_os_dir_lister):
    current = _parse_iso(current_date)
    latest_name, latest_date, ambiguous = None, None, []
    for name in dir_lister(dataset_root):
        d = _parse_iso(name)
        if d is None:
            if re.search(r"\d", name):
                ambiguous.append(name)
            continue
        if current is not None and d >= current:
            continue
        if latest_date is None or d > latest_date:
            latest_name, latest_date = name, d
    return latest_name, ambiguous


def _iter_samples(split_dir, frames):
    if not os.path.isdir(split_dir):
        return
    if frames:
        for name in sorted(os.listdir(split_dir)):
            p = os.path.join(split_dir, name)
            if os.path.isdir(p):
                yield name, p
    else:
        for cls in sorted(os.listdir(split_dir)):
            cls_dir = os.path.join(split_dir, cls)
            if not os.path.isdir(cls_dir):
                continue
            for fn in sorted(os.listdir(cls_dir)):
                if fn.lower().endswith((".mp4", ".mov", ".avi", ".webm")):
                    yield os.path.join(cls, fn), os.path.join(cls_dir, fn)


def merge_snapshot(old_root, new_root, frames, class_names=None):
    copied, skipped, filtered = 0, 0, 0
    for split in ("train", "val"):
        old_split = os.path.join(old_root, split)
        new_split = os.path.join(new_root, split)
        for rel, src in _iter_samples(old_split, frames):
            if class_names is not None and _class_of(rel, class_names) is None:
                filtered += 1
                continue
            dst = os.path.join(new_split, rel)
            if os.path.exists(dst):
                skipped += 1
                continue
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if frames:
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)
            copied += 1
    return {"copied": copied, "skipped": skipped, "filtered": filtered}


def resolve_old_layer(old_date_dir, frames):
    videos_layer = os.path.join(old_date_dir, "videos")
    has_videos = os.path.isdir(os.path.join(videos_layer, "train")) or \
                 os.path.isdir(os.path.join(videos_layer, "val"))
    has_top = os.path.isdir(os.path.join(old_date_dir, "train")) or \
              os.path.isdir(os.path.join(old_date_dir, "val"))
    if frames:
        return old_date_dir, has_videos
    if has_videos:
        return videos_layer, has_top
    return old_date_dir, False


def _class_of(name, class_names):
    low = name.lower()
    for c in class_names:
        if c.lower() in low:
            return c
    return None


def health_report(new_root, class_names, frames):
    per_class = {c: 0 for c in class_names}
    unrecognized, seen_split, cross_split_leak, label_mismatch = [], {}, [], []
    seen_source_split, source_leak = {}, []
    for split in ("train", "val"):
        for rel, path in _iter_samples(os.path.join(new_root, split), frames):
            base = os.path.basename(rel)
            if base in seen_split and seen_split[base] != split:
                cross_split_leak.append(base)
            seen_split[base] = split
            # 同源跨 split 检测（auto_presort 命名 <原视频名>-<start>-<end>[...]）
            src = _source_key(base)
            if src != base:
                if src in seen_source_split and seen_source_split[src] != split:
                    if src not in source_leak:
                        source_leak.append(src)
                seen_source_split[src] = split
            cls = _class_of(rel, class_names)
            if cls is None:
                unrecognized.append(os.path.join(split, rel))
            else:
                per_class[cls] += 1
            if frames:
                label_file = os.path.join(path, "label.txt")
                if os.path.isfile(label_file):
                    try:
                        idx = int(open(label_file).read().strip())
                        if cls is not None and 0 <= idx < len(class_names) and class_names[idx] != cls:
                            label_mismatch.append(os.path.join(split, rel))
                    except (ValueError, IOError):
                        pass
    return {"per_class": per_class, "cross_split_leak": cross_split_leak,
            "unrecognized": unrecognized, "label_mismatch": label_mismatch,
            "source_leak": source_leak}
