from training.config import build_distill_config, build_train_config, load_root_config, resolve_mode
from training.entrypoints import run_distill, run_train

__all__ = [
    "build_distill_config",
    "build_train_config",
    "build_priority_composite_distill_loss",
    "CompositeDistillLoss",
    "load_root_config",
    "resolve_mode",
    "run_distillation_training",
    "run_distill",
    "run_train",
]


def __getattr__(name):
    if name == "run_distillation_training":
        from training.distillation import run_distillation_training

        return run_distillation_training
    if name in {"CompositeDistillLoss", "build_priority_composite_distill_loss"}:
        from training.distillation_losses import CompositeDistillLoss, build_priority_composite_distill_loss

        return {
            "CompositeDistillLoss": CompositeDistillLoss,
            "build_priority_composite_distill_loss": build_priority_composite_distill_loss,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
