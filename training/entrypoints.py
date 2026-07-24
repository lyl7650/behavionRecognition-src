import importlib

from training.config import build_distill_config, build_train_config


TRAIN_MODULES = {
    "normal": "train.train_finetune_normal_3",
    "abnormal": "train.train_finetune_ab_5",
}

DISTILL_MODULES = {
    "normal": "distill.distill_normal_3",
    "abnormal": "distill.distill_abnormal_5",
}


def _patch_train_module(module, resolved):
    if resolved["mode"] == "abnormal" and hasattr(module, "TEACHER_SETTINGS"):
        module.TEACHER_SETTINGS.update(
            {
                "dataset_path": resolved["dataset_path"],
                "checkpoint_path": resolved["checkpoint_path"],
                "output_dir": resolved["output_dir"],
                "num_classes": resolved["num_classes"],
                "batch_size": resolved["batch_size"],
                "accum_steps": resolved["accum_steps"],
                "epochs": resolved["epochs"],
                "lr": resolved["lr"],
                "num_workers": resolved["num_workers"],
            }
        )


def _patch_distill_module(module, resolved):
    module._UNIFIED_CONFIG_OVERRIDE = resolved


def run_train(mode=None, config_path=None):
    resolved = build_train_config(mode=mode, config_path=config_path)
    module = importlib.import_module(TRAIN_MODULES[resolved["mode"]])
    _patch_train_module(module, resolved)
    return module.train(config_override=resolved)


def run_distill(mode=None, config_path=None):
    resolved = build_distill_config(mode=mode, config_path=config_path)
    module = importlib.import_module(DISTILL_MODULES[resolved["mode"]])
    _patch_distill_module(module, resolved)
    return module.train_distill(config_override=resolved)
