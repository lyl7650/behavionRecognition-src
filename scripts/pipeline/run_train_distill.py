"""脚本B：合并旧快照 → 体检确认 → 改 config → teacher 训练 → 回填 → student 蒸馏。

训练相关依赖延迟到函数内 import，避免测试仅 import 本模块就拉起 torch 等重依赖。

merge_and_report / train_and_distill 是可复用单元：编排器
（scripts/scene_retrain/pipeline.py，无人值守模式）与脚本 B（交互模式）
共用同一套合并与训练语义，避免两处实现漂移。
"""
import os
import sys

from . import config_editor as ce
from . import pipeline_config as pc
from . import snapshot_merge as sm


def teacher_output_path(train_cfg):
    """teacher 训练产物路径 = output_dir + output_model_name。

    output_model_name 本身已是 EMA 名，不要再套任何 EMA 命名函数。
    """
    return os.path.join(train_cfg["output_dir"], train_cfg["output_model_name"])


def print_health_report(report, class_names):
    print("\n===== Health Report =====")
    print("Samples per class:")
    for cls in class_names:
        print(f"  {cls}: {report['per_class'].get(cls, 0)}")
    if report["cross_split_leak"]:
        print(f"[LEAK] Same-named samples appear in both train/val: {report['cross_split_leak']}")
    if report.get("source_leak"):
        print(f"[SOURCE LEAK] Same source video appears in both train/val: {report['source_leak']}")
    if report["unrecognized"]:
        print(f"[SILENTLY SKIPPED] Unclassifiable samples (not counted in any class): {report['unrecognized']}")
    if report["label_mismatch"]:
        print(f"[LABEL MISMATCH] Folder name disagrees with label.txt: {report['label_mismatch']}")
    print("Note: classification relies on class keywords in file/dir names; every sample name must contain one configured class name.")
    print("=========================")


def merge_and_report(dataset_root, current_date, new_snapshot, frames,
                     class_names, assume_yes=False, input_fn=None):
    """找最近旧快照 → 回补合并 → 打印体检报告。返回 health_report dict。

    无旧快照时返回 None（调用方自行对"只有本批数据"的快照做体检）。
    assume_yes=False（脚本 B 交互模式）：ambiguous 旧快照要求人工确认。
    assume_yes=True（编排器无人值守模式）：只打 [WARNING] 自动选最新，不暂停。
    """
    if input_fn is None:
        input_fn = input

    latest, ambiguous = sm.find_latest_old_snapshot(dataset_root, current_date)

    if ambiguous:
        print(f"\n[WARNING] Snapshot dirs with unparseable dates: {ambiguous}")
        print(f"Auto-selected latest old snapshot: {latest}")
        print("Merge is irreversible; please confirm the auto-selection is correct.")
        if not assume_yes:
            ans = input("Type 'yes' to continue merging, anything else to abort: ").strip().lower()
            if ans != "yes":
                print("Aborted; nothing was merged.")
                raise SystemExit(0)
        else:
            print("[WARNING] Unattended mode: proceeding with the auto-selection.")

    if not latest:
        print("\nNo old snapshot found; using this batch only.")
        return None

    old_date_dir = os.path.join(dataset_root, latest)
    old_layer, other_exists = sm.resolve_old_layer(old_date_dir, frames)
    if other_exists:
        print(f"[NOTE] Old snapshot {latest} also has data in the other layer; merging only: {old_layer}")
    result = sm.merge_snapshot(old_layer, new_snapshot, frames, class_names=class_names)
    print(f"\nMerged old snapshot {latest}: copied {result['copied']}, skipped {result['skipped']} (already present), filtered {result['filtered']} (class no longer valid).")

    report = sm.health_report(new_snapshot, class_names, frames)
    print_health_report(report, class_names)
    return report


def train_and_distill(species, dataset_path, frames=False):
    """改写 config.yaml → teacher 训练 → 回填 teacher 路径 → student 蒸馏。

    返回 teacher 产物路径。config.yaml 是 cwd 相对路径——调用方必须从项目根运行。
    """
    # Critical ordering: rewrite config.yaml first, then build (builder reads from disk).
    ce.apply_species_config("config.yaml", species, dataset_path)

    from training.config import build_train_config
    from train.train_finetune_normal_3 import train as teacher_train
    from distill.distill_normal_3 import train_distill

    train_cfg = build_train_config("normal")

    print("\n===== Starting teacher training =====")
    teacher_train(config_override=train_cfg)

    teacher_path = teacher_output_path(train_cfg)
    if not os.path.isfile(teacher_path):
        raise FileNotFoundError(
            f"Teacher training artifact not found, aborting distillation: {teacher_path}")

    # Write the teacher path back into config for distillation to use.
    ce.set_teacher_checkpoint("config.yaml", teacher_path)

    print("\n===== Starting student distillation =====")
    train_distill()

    return teacher_path


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    frames = "--frames" in argv

    ctx = pc.ask_context_no_site()
    species, date = ctx["species"], ctx["date"]
    sp = pc.SPECIES[species]
    dataset_root = sp["dataset_root"]
    class_names = sp["class_names"]

    new_snapshot = pc.snapshot_dir(species, date, frames=frames)
    if not os.path.isdir(new_snapshot):
        raise FileNotFoundError(
            f"Batch snapshot not found (run run_crop first): {new_snapshot}")

    report = merge_and_report(dataset_root, date, new_snapshot, frames, class_names)
    if report is None:
        report = sm.health_report(new_snapshot, class_names, frames)
        print_health_report(report, class_names)

    from scripts.pipeline.lineage.store import write_snapshot
    manifest_path = write_snapshot(new_snapshot, species=species, date=date, frames=frames)
    print(f"Dataset manifest: {manifest_path}")

    input("\nHealth check OK? Press Enter to continue, Ctrl+C to abort")

    teacher_path = train_and_distill(species, new_snapshot, frames)

    print(f"\nAll done. teacher: {teacher_path}")
    print("For deployment use the student's *_ema.pth (under outputs/<species>/<date>/).")


if __name__ == "__main__":
    main(sys.argv[1:])
