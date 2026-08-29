"""Opt-in RKNN configuration adapter; normal training config remains untouched."""

from training.config import build_distill_config, build_train_config, load_root_config, resolve_mode


RKNN_DEFAULTS = {
    "finetune_epochs": 4,
    "finetune_lr_ratio": 0.1,
    "bn_gamma_weight_decay": 1e-4,
    "reparam_tolerance": 1e-3,
}


def _rknn_options(root_cfg, mode, section):
    result = dict(RKNN_DEFAULTS)
    result.update(root_cfg.get("rknn", {}))
    result.update(root_cfg.get("task_configs", {}).get(mode, {}).get(section, {}).get("rknn", {}))
    epochs = int(result["finetune_epochs"])
    if not 3 <= epochs <= 5:
        raise ValueError("rknn.finetune_epochs must be between 3 and 5")
    return result


def build_distill_config_rknn(mode="normal", config_path=None, config_dict=None):
    cfg = build_distill_config(mode, config_path=config_path, config_dict=config_dict)
    root = load_root_config(config_path=config_path, config_dict=config_dict)
    options = _rknn_options(root, cfg["mode"], "distillation")
    cfg.update({f"rknn_{key}": value for key, value in options.items()})
    return cfg


def build_train_config_rknn(mode="normal", config_path=None, config_dict=None):
    cfg = build_train_config(mode, config_path=config_path, config_dict=config_dict)
    root = load_root_config(config_path=config_path, config_dict=config_dict)
    options = _rknn_options(root, cfg["mode"], "finetune")
    cfg["rknn_bn_gamma_weight_decay"] = options["bn_gamma_weight_decay"]
    return cfg
