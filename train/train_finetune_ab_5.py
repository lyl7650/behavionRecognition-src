import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from training.config import build_train_config
from training.teacher_finetune import run_teacher_finetune


BEST_MODEL_NAME = "teacher_best_5cls_ema.pth"


class HierarchicalPriorityLoss(nn.Module):
    def __init__(self, normal_idx=0, binary_weight=1.1, fine_weight=1.0):
        super().__init__()
        self.normal_idx = normal_idx
        self.w_binary = binary_weight
        self.w_fine = fine_weight
        self.ce = nn.CrossEntropyLoss(label_smoothing=0.1)

    def forward(self, logits, targets):
        binary_targets = (targets != self.normal_idx).float()
        probs = F.softmax(logits, dim=1)
        prob_normal = probs[:, self.normal_idx]
        prob_abnormal = 1.0 - prob_normal + 1e-7

        loss_binary = -(
            binary_targets * torch.log(prob_abnormal)
            + (1 - binary_targets) * torch.log(prob_normal)
        ).mean()

        mask_abnormal = targets != self.normal_idx
        loss_fine = torch.tensor(0.0, device=logits.device)
        if mask_abnormal.sum() > 0:
            abnormal_logits = logits[mask_abnormal][:, 1:]
            abnormal_targets = targets[mask_abnormal] - 1
            loss_fine = self.ce(abnormal_logits, abnormal_targets)

        total_loss = self.w_binary * loss_binary + self.w_fine * loss_fine
        return total_loss, loss_binary.item(), loss_fine.item()


def train(config_override=None, config_path=None):
    cfg = config_override or build_train_config("abnormal", config_path=config_path)
    criterion = HierarchicalPriorityLoss(normal_idx=0, binary_weight=1.1, fine_weight=1.0)
    return run_teacher_finetune(
        cfg,
        criterion,
        BEST_MODEL_NAME,
        print_config=False,
        pass_num_classes_to_teacher=True,
    )


if __name__ == "__main__":
    train()
