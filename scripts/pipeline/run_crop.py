import argparse
import os
import sys

from . import pipeline_config as pc


def validate_sorted(sorted_dir, class_names):
    if not os.path.isdir(sorted_dir):
        raise FileNotFoundError(f"sorted directory not found: {sorted_dir}")
    exts = (".mp4", ".mov", ".avi", ".webm")
    for c in class_names:
        cdir = os.path.join(sorted_dir, c)
        if os.path.isdir(cdir):
            for _, _, files in os.walk(cdir):
                if any(f.lower().endswith(exts) for f in files):
                    return
    raise FileNotFoundError(f"no videos under sorted directory (not trimmed yet?): {sorted_dir}")


def assert_snapshot_empty(snapshot_dir):
    if os.path.isdir(snapshot_dir) and any(os.scandir(snapshot_dir)):
        raise FileExistsError(
            f"target snapshot directory is not empty; to avoid mixing old artifacts into this batch, "
            f"clear it or use a different date: {snapshot_dir}")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", action="store_true",
                        help="use frame format (default: video clip)")
    args = parser.parse_args(argv)
    ctx = pc.ask_context()
    species, site, date = ctx["species"], ctx["site"], ctx["date"]
    sp = pc.SPECIES[species]
    sorted_dir = pc.sorted_dir(species, site, date)
    validate_sorted(sorted_dir, sp["class_names"])
    snapshot = pc.snapshot_dir(species, date, frames=args.frames)
    assert_snapshot_empty(snapshot)
    if args.frames:
        from utils import abnormal_crop as crop
    else:
        from utils import abnormal_crop_video as crop
    cfg = crop.default_config()
    cfg.update({
        "input_base_dir": sorted_dir,
        "lineage_species": species,
        "output_base_dir": snapshot,
        "class_names": sp["class_names"],
        "yolo_model_path": sp["yolo_model"],
        "target_classes": sp["target_classes"],
    })
    if args.frames:
        cfg["label_map"] = sp["label_map"]
    crop.run(cfg)
    print(f"\ncrop done, output at: {snapshot}")


if __name__ == "__main__":
    main(sys.argv[1:])
