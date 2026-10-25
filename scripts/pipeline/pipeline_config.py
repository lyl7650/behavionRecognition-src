"""流水线共享配置：物种查表 + 路径推导 + 交互问答。

唯一集中写死"物种规则"的地方。
"""
import os
from datetime import datetime


# ===== 第一部分：物种查表 =====
SPECIES = {
    "pet": {
        "class_names": ["eat", "drink", "active", "resting"],
        "label_map": {"eat": 0, "drink": 1, "active": 2, "resting": 3},
        "yolo_model": r"E:\ActionRecognition\detection-src\cat_dog_detect\runs\yolo11n_scale0.6_fliplr0.42\weights\yolo11n-ssb-0710.pt",
        "target_classes": [1, 2],
        "raw_root": r"dataset\raw_videos\pet\normal",
        "dataset_root": r"dataset\pet\normal",
        "path_token": "pet",
        "teacher_checkpoint": r"E:\ActionRecognition\behavionRecognition-src\outputs\pet\2026-07-24\pet_teacher_2026-07-24_ema-videos.pth",
    },
    "rabbit": {
        "class_names": ["resting", "active", "eating", "drinking"],
        "label_map": {"resting": 0, "active": 1, "eating": 2, "drinking": 3},
        "yolo_model": r"E:\ActionRecognition\detection-src\rabbit-detect\runs\yolo11s_scale0.6_fliplr0.4\weights\rabbit-yolo11s-0923.pt",
        "target_classes": [0],
        "raw_root": r"dataset\raw_videos\rabbit",
        "dataset_root": r"dataset\rabbit",
        "path_token": "rabbit",
        "teacher_checkpoint": r"E:\ActionRecognition\behavionRecognition-src\outputs\rabbit\2026-06-24\0624_repvit_best_distilled_ema.pth",
    },
}


# ===== 第二部分：路径推导 =====
def records_dir(species: str, site: str, date: str) -> str:
    """推导 records 目录路径。

    返回 os.path.join(raw_root, f"{site}-record", "records", date)
    """
    raw_root = SPECIES[species]["raw_root"]
    return os.path.join(raw_root, f"{site}-record", "records", date)


def sorted_dir(species: str, site: str, date: str) -> str:
    """推导 sorted 目录路径。

    返回 os.path.join(raw_root, f"{site}-record", f"sorted-{date}")
    """
    raw_root = SPECIES[species]["raw_root"]
    return os.path.join(raw_root, f"{site}-record", f"sorted-{date}")


def sorted_class_dirs(species: str, site: str, date: str) -> list:
    """推导 sorted 下各类别目录列表。

    返回 [os.path.join(sorted_dir(...), c) for c in class_names]
    """
    base = sorted_dir(species, site, date)
    class_names = SPECIES[species]["class_names"]
    return [os.path.join(base, c) for c in class_names]


def snapshot_dir(species: str, date: str, frames: bool = False) -> str:
    """推导 snapshot 目录路径。

    frames=False: 返回 os.path.join(dataset_root, date, "videos")
    frames=True:  返回 os.path.join(dataset_root, date)
    """
    dataset_root = SPECIES[species]["dataset_root"]
    if frames:
        return os.path.join(dataset_root, date)
    else:
        return os.path.join(dataset_root, date, "videos")


# ===== 第三部分：ask_context 交互问答 =====
def _today_str() -> str:
    """返回今天的日期字符串 YYYY-MM-DD."""
    return datetime.now().strftime("%Y-%m-%d")


def _ask_choice(label: str, options: list[str], input_fn) -> str:
    """Binary/list choice via number (1/2/...) or legacy option name.

    Prints a numbered menu and retries until the user picks a valid entry.
    """
    if not options:
        raise ValueError("options must not be empty")
    by_number = {str(i + 1): name for i, name in enumerate(options)}
    by_name = {name.lower(): name for name in options}
    n = len(options)
    while True:
        print(f"{label}:")
        for i, name in enumerate(options, start=1):
            print(f"  {i}) {name}")
        raw = input_fn(f"Choose [1-{n}]: ").strip()
        key = raw.lower()
        if key in by_number:
            return by_number[key]
        if key in by_name:
            return by_name[key]


def _ask_species(input_fn):
    """Prompt for species until a valid one is entered."""
    # Keep order stable and shared with other entrypoints: 1=pet, 2=rabbit.
    return _ask_choice("Species", list(SPECIES.keys()), input_fn)


def _ask_date(input_fn, today_fn):
    """Prompt for date, defaulting to today (YYYY-MM-DD)."""
    today = today_fn()
    date_input = input_fn(f"Date [default {today}]: ").strip()
    return date_input if date_input else today


def ask_context(input_fn=None, today_fn=None) -> dict:
    """Interactively ask species/site/date. Returns {"species", "site", "date"}.

    Used by the scaffold and crop entry points, which need the site to locate
    raw_videos/<species>/<site>-record/... directories.
    """
    if input_fn is None:
        input_fn = input
    if today_fn is None:
        today_fn = _today_str

    species = _ask_species(input_fn)
    site = input_fn("Site (e.g. hkbr): ").strip()
    date = _ask_date(input_fn, today_fn)

    return {"species": species, "site": site, "date": date}


def ask_context_no_site(input_fn=None, today_fn=None) -> dict:
    """Interactively ask species/date only. Returns {"species", "date"}.

    Used by the train/distill entry point, which operates on the already-crop'd
    snapshot dataset/<species>/<date> — that path carries no site, so asking for
    a site would be misleading.
    """
    if input_fn is None:
        input_fn = input
    if today_fn is None:
        today_fn = _today_str

    species = _ask_species(input_fn)
    date = _ask_date(input_fn, today_fn)

    return {"species": species, "date": date}
