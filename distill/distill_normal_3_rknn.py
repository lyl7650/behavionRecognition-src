"""Normal four-class RKNN-friendly student distillation entry point."""
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from training.config_rknn import build_distill_config_rknn
from training.distillation_losses import build_priority_composite_distill_loss
from training.distillation_rknn import run_distillation_training_rknn


def _build_runtime_config(config_override=None, config_path=None):
    resolved = config_override or build_distill_config_rknn("normal", config_path=config_path)
    required = ("dataset_path", "class_names", "recognizer", "teacher_checkpoint", "temperature", "loss_weights", "learning_rate", "batch_size", "epochs", "num_workers", "output_dir", "output_model_name", "output_ema_model_name")
    cfg = {key: resolved[key] for key in required}
    cfg.update({key: resolved[key] for key in resolved if key.startswith("rknn_")})
    defaults = {"rknn_finetune_epochs": 4, "rknn_finetune_lr_ratio": 0.1, "rknn_bn_gamma_weight_decay": 1e-4, "rknn_reparam_tolerance": 1e-3}
    for key, value in defaults.items():
        cfg.setdefault(key, value)
    return cfg


def train_distill(config_override=None, config_path=None):
    cfg = _build_runtime_config(config_override, config_path)
    try:
        anchor = cfg["class_names"].index("resting")
    except ValueError as exc:
        raise ValueError("normal RKNN distillation requires a 'resting' class") from exc
    loss = build_priority_composite_distill_loss(anchor, alpha=cfg["loss_weights"]["feature"], beta=cfg["loss_weights"]["kd"], gamma=cfg["loss_weights"]["gt"], temperature=cfg["temperature"])
    return run_distillation_training_rknn(cfg, loss, cfg["output_model_name"], cfg["output_ema_model_name"])


if __name__ == "__main__":
    train_distill()
