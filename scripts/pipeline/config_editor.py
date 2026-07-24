"""按物种自动改写 config.yaml。

写 YAML 时把反斜杠路径转成正斜杠，避免 YAML 把 "\n" 当换行；
改动前做时间戳备份，写临时文件后原子替换。
"""
import os
import shutil
from datetime import datetime

import yaml

from . import pipeline_config


def _now_str():
    return datetime.now().strftime("%Y%m%d-%H%M")


def clean_path(value):
    """剥掉 r"..." / 引号包装并 os.path.normpath。None 透传。"""
    if value is None:
        return value
    text = str(value).strip()
    if len(text) >= 3 and text[0] in ("r", "R") and text[1] in ("'", '"'):
        text = text[1:].strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in ("'", '"'):
        text = text[1:-1]
    return os.path.normpath(text)


def yaml_safe_path(path):
    """清洗后把反斜杠转正斜杠，供 YAML 安全写入。"""
    return clean_path(path).replace("\\", "/")


def _swap_token(text, token):
    """把 "pet"/"rabbit" 中不是 token 的那个替换成 token。"""
    other = "rabbit" if token == "pet" else "pet"
    return text.replace(other, token)


def _dump(data, config_path):
    """写临时文件后 os.replace 原子替换。"""
    tmp_path = f"{config_path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, allow_unicode=True, sort_keys=False,
                       default_flow_style=False)
    os.replace(tmp_path, config_path)


def apply_species_config(config_path, species, dataset_path, now_fn=_now_str):
    """按物种改写 config：数据集路径、类别名、路径 token。"""
    with open(config_path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    normal = data["task_configs"]["normal"]
    normal["dataset_path"] = yaml_safe_path(dataset_path)
    normal["class_names"] = list(pipeline_config.SPECIES[species]["class_names"])

    token = pipeline_config.SPECIES[species]["path_token"]
    finetune = normal["finetune"]
    distill = normal["distillation"]
    finetune["output_dir"] = _swap_token(finetune["output_dir"], token)
    finetune["output_model_name"] = _swap_token(finetune["output_model_name"], token)
    distill["output_dir"] = _swap_token(distill["output_dir"], token)

    for field in (finetune["output_dir"], finetune["output_model_name"],
                  distill["output_dir"]):
        if token not in field:
            raise ValueError(f"token {token!r} not found in field: {field!r}")

    backup_dir = os.path.join(os.path.dirname(os.path.abspath(config_path)), "config", "backups")
    os.makedirs(backup_dir, exist_ok=True)
    backup_path = os.path.join(backup_dir, f"{os.path.basename(config_path)}.{now_fn()}.bak")
    shutil.copy2(config_path, backup_path)
    try:
        _dump(data, config_path)
    except Exception:
        print(f"Write failed; restore from backup: copy {backup_path} {config_path}")
        raise


def set_teacher_checkpoint(config_path, teacher_path):
    """更新 distillation.teacher_checkpoint 为 YAML 安全路径。"""
    with open(config_path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    data["distillation"]["teacher_checkpoint"] = yaml_safe_path(teacher_path)
    _dump(data, config_path)
