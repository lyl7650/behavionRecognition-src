import copy
import os
from datetime import datetime

import yaml


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TRAIN_DEFAULTS = {
    "batch_size": 4,
    "accum_steps": 8,
    "epochs": 35,
    "lr": 4e-5,
    "num_workers": 4,
    "imgsz": 224,
    "temporal_window": 16,
}

TRAIN_TASK_DEFAULTS = {
    "normal": {
        "output_dir": os.path.normpath("outputs/rabbit/2026-05-07/teacher_finetuned_4cls_0507"),
        "output_model_name": "teacher_best_0223_ema.pth",
    },
    "abnormal": {
        "output_dir": os.path.normpath("outputs/teacher_finetuned_5cls"),
        "output_model_name": "teacher_best_5cls_ema.pth",
    },
}

DISTILL_DEFAULTS = {
    "temperature": 4.0,
    "loss_weights": {
        "gt": 1.0,
        "kd": 8.0,
        "feature": 1.0,
    },
    "learning_rate": 3e-4,
    "batch_size": 16,
    "epochs": 50,
    "num_workers": 4,
}

DISTILL_TASK_DEFAULTS = {
    "normal": {
        "output_dir": os.path.normpath("outputs/distilled_final/rabbit/0508"),
        "output_model_name": "repvit_best_distilled.pth",
    },
    "abnormal": {
        "output_dir": os.path.normpath("outputs/distilled_5cls_final"),
        "output_model_name": "repvit_best_distilled_5cls.pth",
    },
}


def resolve_ema_model_name(model_name):
    root, ext = os.path.splitext(model_name)
    if not ext:
        return f"{model_name}_ema"
    return f"{root}_ema{ext}"


def sanitize_path(value):
    if value is None:
        return value
    text = str(value).strip()
    if len(text) >= 3 and text[0] in ("r", "R") and text[1] in ("'", '"'):
        text = text[1:].strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in ("'", '"'):
        text = text[1:-1]
    return os.path.normpath(text)


def load_root_config(config_path=None, config_dict=None):
    if config_dict is not None:
        return copy.deepcopy(config_dict)

    resolved_path = config_path or os.path.join(ROOT_DIR, "config.yaml")
    with open(resolved_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_mode(root_cfg, mode=None, purpose="train"):
    if mode:
        return mode
    if purpose == "distill":
        return root_cfg.get("distillation", {}).get("mode_to_use", root_cfg.get("mode", "normal"))
    return root_cfg.get("mode", "normal")


def build_train_config(mode=None, config_path=None, config_dict=None):
    root_cfg = load_root_config(config_path=config_path, config_dict=config_dict)
    resolved_mode = resolve_mode(root_cfg, mode=mode, purpose="train")
    task_cfg = copy.deepcopy(root_cfg["task_configs"][resolved_mode])
    finetune_cfg = copy.deepcopy(task_cfg.get("finetune", {}))
    recognizer_cfg = copy.deepcopy(task_cfg.get("recognizer", {}))
    class_names = list(task_cfg.get("class_names") or recognizer_cfg.get("class_names") or [])
    if not class_names:
        raise ValueError(f"task_configs.{resolved_mode}.class_names is required")

    imgsz = int(recognizer_cfg.get("imgsz", TRAIN_DEFAULTS["imgsz"]))
    temporal_window = int(recognizer_cfg.get("temporal_window", TRAIN_DEFAULTS["temporal_window"]))

    recognizer_cfg["class_names"] = class_names
    recognizer_cfg["num_classes"] = len(class_names)
    recognizer_cfg["imgsz"] = imgsz
    recognizer_cfg["temporal_window"] = temporal_window

    return {
        "mode": resolved_mode,
        "dataset_path": sanitize_path(task_cfg["dataset_path"]),
        "class_names": class_names,
        "num_classes": len(class_names),
        "checkpoint_path": sanitize_path(finetune_cfg.get("checkpoint_path")),
        "output_dir": sanitize_path(
            (finetune_cfg.get("output_dir") or TRAIN_TASK_DEFAULTS[resolved_mode]["output_dir"])
            .replace("{date}", datetime.now().strftime("%Y-%m-%d"))
        ),
        "output_model_name": (
            (finetune_cfg.get("output_model_name") or TRAIN_TASK_DEFAULTS[resolved_mode]["output_model_name"])
            .replace("{date}", datetime.now().strftime("%Y-%m-%d"))
        ),
        "batch_size": int(finetune_cfg.get("batch_size", TRAIN_DEFAULTS["batch_size"])),
        "accum_steps": int(finetune_cfg.get("accum_steps", TRAIN_DEFAULTS["accum_steps"])),
        "epochs": int(finetune_cfg.get("epochs", TRAIN_DEFAULTS["epochs"])),
        "lr": float(finetune_cfg.get("lr", TRAIN_DEFAULTS["lr"])),
        "num_workers": int(finetune_cfg.get("num_workers", TRAIN_DEFAULTS["num_workers"])),
        "imgsz": imgsz,
        "temporal_window": temporal_window,
        "loss_type": finetune_cfg.get("loss_type", "ce"),
        "recognizer": recognizer_cfg,
    }


def build_distill_config(mode=None, config_path=None, config_dict=None):
    root_cfg = load_root_config(config_path=config_path, config_dict=config_dict)
    resolved_mode = resolve_mode(root_cfg, mode=mode, purpose="distill")
    task_cfg = copy.deepcopy(root_cfg["task_configs"][resolved_mode])
    recognizer_cfg = copy.deepcopy(task_cfg.get("recognizer", {}))
    global_distill_cfg = copy.deepcopy(root_cfg.get("distillation", {}))
    task_distill_cfg = copy.deepcopy(task_cfg.get("distillation", {}))
    class_names = list(task_cfg.get("class_names") or recognizer_cfg.get("class_names") or [])
    if not class_names:
        raise ValueError(f"task_configs.{resolved_mode}.class_names is required")

    recognizer_cfg["class_names"] = class_names
    recognizer_cfg["num_classes"] = len(class_names)

    weights = copy.deepcopy(DISTILL_DEFAULTS["loss_weights"])
    if "gamma" in global_distill_cfg:
        weights["gt"] = float(global_distill_cfg["gamma"])
    if "beta" in global_distill_cfg:
        weights["kd"] = float(global_distill_cfg["beta"])
    if "alpha" in global_distill_cfg:
        weights["feature"] = float(global_distill_cfg["alpha"])
    weights.update(global_distill_cfg.get("loss_weights", {}))
    weights.update(task_distill_cfg.get("loss_weights", {}))

    resolved_output_model_name = (
        task_distill_cfg.get("output_model_name")
        or global_distill_cfg.get("output_model_name")
        or DISTILL_TASK_DEFAULTS[resolved_mode]["output_model_name"]
    ).replace("{date}", datetime.now().strftime("%Y-%m-%d"))

    return {
        "mode": resolved_mode,
        "dataset_path": sanitize_path(task_cfg["dataset_path"]),
        "class_names": class_names,
        "num_classes": len(class_names),
        "recognizer": recognizer_cfg,
        "teacher_checkpoint": sanitize_path(task_distill_cfg.get("teacher_checkpoint") or global_distill_cfg.get("teacher_checkpoint")),
        "temperature": float(task_distill_cfg.get("temperature", global_distill_cfg.get("temperature", DISTILL_DEFAULTS["temperature"]))),
        "loss_weights": {
            "gt": float(weights["gt"]),
            "kd": float(weights["kd"]),
            "feature": float(weights["feature"]),
        },
        "learning_rate": float(task_distill_cfg.get("learning_rate", global_distill_cfg.get("learning_rate", DISTILL_DEFAULTS["learning_rate"]))),
        "batch_size": int(task_distill_cfg.get("batch_size", global_distill_cfg.get("batch_size", DISTILL_DEFAULTS["batch_size"]))),
        "epochs": int(task_distill_cfg.get("epochs", global_distill_cfg.get("epochs", DISTILL_DEFAULTS["epochs"]))),
        "num_workers": int(task_distill_cfg.get("num_workers", global_distill_cfg.get("num_workers", DISTILL_DEFAULTS["num_workers"]))),
        "output_dir": sanitize_path(
            (
                task_distill_cfg.get("output_dir")
                or global_distill_cfg.get("output_dir")
                or DISTILL_TASK_DEFAULTS[resolved_mode]["output_dir"]
            )
            .replace("{date}", datetime.now().strftime("%Y-%m-%d"))
        ),
        "output_model_name": resolved_output_model_name,
        "output_ema_model_name": resolve_ema_model_name(resolved_output_model_name),
    }
