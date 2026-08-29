"""Normal-teacher finetune entry point with RKNN BN-gamma regularization."""

import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from train.train_finetune_normal_3 import HierarchicalPriorityLoss, SimpleCELoss
from training.config_rknn import build_train_config_rknn
from training.teacher_finetune_rknn import run_teacher_finetune_rknn


def train(config_override=None, config_path=None):
    cfg = config_override or build_train_config_rknn("normal", config_path=config_path)
    cfg.setdefault("rknn_bn_gamma_weight_decay", 1e-4)
    if cfg.get("loss_type", "ce") == "hierarchical":
        try:
            criterion = HierarchicalPriorityLoss(cfg["class_names"].index("resting"), binary_weight=1.1, fine_weight=1.0)
        except ValueError as exc:
            raise ValueError("hierarchical normal RKNN training requires a 'resting' class") from exc
    else:
        criterion = SimpleCELoss()
    return run_teacher_finetune_rknn(cfg, criterion, cfg["output_model_name"], print_config=True, pass_num_classes_to_teacher=False)


if __name__ == "__main__":
    train()
