import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from training.config import build_distill_config
from training.distillation import run_distillation_training
from training.distillation_losses import build_priority_composite_distill_loss


DEFAULT_MODE = "normal"


def _build_runtime_config(config_override=None, config_path=None):
    resolved = config_override
    if resolved is None:
        resolved = build_distill_config(DEFAULT_MODE, config_path=config_path)
    return {
        "dataset_path": resolved["dataset_path"],
        "class_names": list(resolved["class_names"]),
        "recognizer": dict(resolved["recognizer"]),
        "teacher_checkpoint": resolved["teacher_checkpoint"],
        "temperature": float(resolved["temperature"]),
        "loss_weights": {
            "gt": float(resolved["loss_weights"]["gt"]),
            "kd": float(resolved["loss_weights"]["kd"]),
            "feature": float(resolved["loss_weights"]["feature"]),
        },
        "learning_rate": float(resolved["learning_rate"]),
        "batch_size": int(resolved["batch_size"]),
        "epochs": int(resolved["epochs"]),
        "num_workers": int(resolved["num_workers"]),
        "output_dir": resolved["output_dir"],
        "output_model_name": resolved["output_model_name"],
        "output_ema_model_name": resolved["output_ema_model_name"],
    }


def _resolve_anchor_idx(class_names):
    try:
        return class_names.index("resting")
    except ValueError as exc:
        raise ValueError("normal distillation requires a 'resting' class") from exc


def _build_criterion(runtime_cfg):
    return build_priority_composite_distill_loss(
        anchor_idx=_resolve_anchor_idx(runtime_cfg["class_names"]),
        alpha=runtime_cfg["loss_weights"]["feature"],
        beta=runtime_cfg["loss_weights"]["kd"],
        gamma=runtime_cfg["loss_weights"]["gt"],
        temperature=runtime_cfg["temperature"],
    )


def train_distill(config_override=None, config_path=None):
    runtime_cfg = _build_runtime_config(config_override=config_override, config_path=config_path)
    criterion = _build_criterion(runtime_cfg)
    return run_distillation_training(
        runtime_cfg=runtime_cfg,
        criterion=criterion,
        best_model_name=runtime_cfg["output_model_name"],
        best_ema_model_name=runtime_cfg["output_ema_model_name"],
    )


if __name__ == "__main__":
    train_distill()
