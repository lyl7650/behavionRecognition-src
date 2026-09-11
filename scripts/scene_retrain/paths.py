"""扁平 retrain 的路径推导（无 site 维度）。

与 detection-src/scene_retrain/paths.py 的扁平化对齐：一批新数据进来直接与
旧快照合并训练，不再按场地隔离。物种规则唯一来源仍是 pipeline_config.SPECIES：

    工作区      <raw_root>/_retrain/<date>/          （job.json、sorted-<date>/、
                                                      _review_state.json）
    切条输出    <workspace>/sorted-<date>/<class>/   （apply 的产物，crop 的输入）
    本次快照    <dataset_root>/<date>/videos         （clip 模式，frames 模式无 videos 层；
                                                      复用 pipeline_config.snapshot_dir，
                                                      本来就不带 site）
    合并基      <dataset_root>                        （find_latest_old_snapshot 扫描旧快照）
"""
import os

from scripts.pipeline import pipeline_config as pc


def build_retrain_context(species: str, date: str, frames: bool = False) -> dict:
    """推导扁平 retrain 全部路径（不创建目录，纯逻辑）。"""
    sp = pc.SPECIES[species]
    raw_root = sp["raw_root"]
    workspace = os.path.join(raw_root, "_retrain", date)
    return {
        "species": species,
        "date": date,
        "frames": frames,
        "raw_root": raw_root,
        "dataset_root": sp["dataset_root"],
        "workspace": workspace,
        "sorted_dir": os.path.join(workspace, f"sorted-{date}"),
        "state_file": os.path.join(workspace, "_review_state.json"),
        "snapshot_dir": pc.snapshot_dir(species, date, frames=frames),
        "classes": list(sp["class_names"]),
        "teacher_checkpoint": sp["teacher_checkpoint"],
    }


def ask_retrain_context(input_fn=None, today_fn=None) -> dict:
    """交互式问 物种/日期（无 site），返回 build_retrain_context 的结果。"""
    ctx = pc.ask_context_no_site(input_fn=input_fn, today_fn=today_fn)
    return build_retrain_context(ctx["species"], ctx["date"])
