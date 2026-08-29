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


class HierarchicalPriorityLoss(nn.Module):
    def __init__(self, anchor_idx, binary_weight=1.1, fine_weight=1.0):
        super().__init__()
        self.anchor_idx = anchor_idx
        self.w_binary = binary_weight
        self.w_fine = fine_weight
        self.ce = nn.CrossEntropyLoss(label_smoothing=0.1)

    def forward(self, logits, targets):
        num_classes = int(logits.shape[1])
        if self.anchor_idx < 0 or self.anchor_idx >= num_classes:
            raise ValueError(f"anchor_idx={self.anchor_idx} out of range for num_classes={num_classes}")

        binary_targets = (targets != self.anchor_idx).float()
        probs = F.softmax(logits, dim=1)
        prob_anchor = probs[:, self.anchor_idx]
        prob_action = 1.0 - prob_anchor + 1e-7

        loss_binary = -(
            binary_targets * torch.log(prob_action) + (1 - binary_targets) * torch.log(prob_anchor)
        ).mean()

        mask_action = targets != self.anchor_idx
        loss_fine = torch.tensor(0.0, device=logits.device)
        if mask_action.sum() > 0:
            keep_indices = [idx for idx in range(num_classes) if idx != self.anchor_idx]
            action_logits = logits[mask_action][:, keep_indices]
            action_targets = targets[mask_action].clone()
            action_targets[action_targets > self.anchor_idx] -= 1
            loss_fine = self.ce(action_logits, action_targets)

        total_loss = self.w_binary * loss_binary + self.w_fine * loss_fine
        return total_loss, loss_binary.item(), loss_fine.item()


class SimpleCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(label_smoothing=0.1)

    def forward(self, logits, targets):
        loss = self.ce(logits, targets)
        return loss, 0.0, loss.item()


def load_train_config(config_path=None):
    if config_path is not None:
        return build_train_config("normal", config_path=config_path)
    return build_train_config("normal")


def train(config_override=None, config_path=None):
    cfg = config_override or load_train_config(config_path=config_path)
    if cfg.get("loss_type", "ce") == "hierarchical":
        try:
            anchor_idx = cfg["class_names"].index("resting")
        except ValueError as exc:
            raise ValueError("hierarchical normal training requires a 'resting' class") from exc
        criterion = HierarchicalPriorityLoss(anchor_idx=anchor_idx, binary_weight=1.1, fine_weight=1.0)
    else:
        criterion = SimpleCELoss()
    return run_teacher_finetune(
        cfg,
        criterion,
        cfg["output_model_name"],
        print_config=True,
        pass_num_classes_to_teacher=False,
    )


if __name__ == "__main__":
    train()
