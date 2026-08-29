"""Independent distillation loop for the RKNN-friendly student.

It deliberately does not change ``training.distillation``: use it only from
``distill/distill_normal_3_rknn.py``.
"""

import os

import torch
import torch.nn as nn
from timm.utils import ModelEmaV2
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import get_dataset
from models.student_factory_rknn import build_student_rknn
from models.teacher import SwinTeacher
from training.rknn_optim import adamw_groups_with_bn_gamma_decay
from training.rknn_reparameterization import verify_reparameterization


class FeatureHook:
    def __init__(self, module):
        self.features = None
        self.hook = module.register_forward_hook(self._capture)

    def _capture(self, module, inputs, output):
        self.features = output[0] if isinstance(output, (list, tuple)) else output

    def remove(self):
        self.hook.remove()


class FeatureProjector(nn.Module):
    def __init__(self, student_channels, teacher_channels):
        super().__init__()
        self.proj = nn.ModuleList([
            nn.Sequential(nn.Conv2d(s, t, 1, bias=False), nn.BatchNorm2d(t), nn.ReLU())
            for s, t in zip(student_channels, teacher_channels)
        ])

    def forward(self, feats):
        return [layer(feat) for layer, feat in zip(self.proj, feats)]


def _hook_features(hooks):
    return [hook.features[0] if isinstance(hook.features, (list, tuple)) else hook.features for hook in hooks]


def _make_optimizer(student, projector, learning_rate, bn_gamma_weight_decay):
    groups = adamw_groups_with_bn_gamma_decay(
        student, base_weight_decay=0.05, bn_gamma_weight_decay=bn_gamma_weight_decay
    )
    if projector is not None:
        groups.extend(adamw_groups_with_bn_gamma_decay(
            projector, base_weight_decay=0.05, bn_gamma_weight_decay=bn_gamma_weight_decay
        ))
    return torch.optim.AdamW(groups, lr=learning_rate)


def _validate(model, val_loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for batch in val_loader:
            logits = model.forward_seq(batch["pixel_values"].to(device))
            labels = batch["labels"].to(device)
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.numel()
    return correct / total if total else 0.0


def _train_phase(student, teacher, projector, hooks, train_loader, val_loader, criterion, optimizer, epochs, device, model_ema):
    scaler = torch.amp.GradScaler(device.type)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    for epoch in range(epochs):
        student.train()
        if projector is not None:
            projector.train()
        train_correct = train_total = 0
        for batch in tqdm(train_loader, desc=f"RKNN Ep {epoch + 1}/{epochs}"):
            videos = batch["pixel_values"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            with torch.amp.autocast(device.type):
                with torch.no_grad():
                    teacher_logits, teacher_all = teacher(videos)
                    teacher_features = teacher_all[-len(hooks):]
                student_logits = student.forward_seq(videos)
                projected = projector(_hook_features(hooks)) if projector is not None else []
                loss, *_ = criterion(
                    student_logits, teacher_logits, projected, teacher_features if projector is not None else [],
                    labels, videos.shape[0], videos.shape[1], epoch / max(epochs, 1),
                )
            train_correct += (student_logits.argmax(1) == labels).sum().item()
            train_total += labels.numel()
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            model_ema.update(student)
        scheduler.step()
        train_acc = train_correct / train_total if train_total else 0.0
        val_acc = _validate(model_ema.module, val_loader, device)
        print(
            f"RKNN Ep {epoch + 1}/{epochs} | "
            f"Train Acc: {train_acc:.2%} | Val Acc (EMA): {val_acc:.2%}"
        )


def run_distillation_training_rknn(runtime_cfg, criterion, best_model_name, best_ema_model_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = runtime_cfg["class_names"]
    recognizer_cfg = dict(runtime_cfg["recognizer"])
    recognizer_cfg.update(class_names=class_names, num_classes=len(class_names))
    train_ds = get_dataset(runtime_cfg["dataset_path"], "train", recognizer_cfg, is_train=True)
    val_ds = get_dataset(runtime_cfg["dataset_path"], "val", recognizer_cfg, is_train=False)
    if not len(train_ds) or not len(val_ds):
        raise ValueError("RKNN distillation requires non-empty train and val splits")
    train_loader = DataLoader(train_ds, batch_size=runtime_cfg["batch_size"], shuffle=True, num_workers=runtime_cfg["num_workers"], pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=runtime_cfg["batch_size"], shuffle=False, num_workers=runtime_cfg["num_workers"])
    teacher = SwinTeacher(checkpoint_path=runtime_cfg["teacher_checkpoint"], num_classes=len(class_names)).to(device).eval()
    for param in teacher.parameters():
        param.requires_grad = False
    student = build_student_rknn(recognizer_cfg, len(class_names)).to(device)
    hooks = [FeatureHook(student.backbone)]
    with torch.no_grad():
        sample = next(iter(val_loader))["pixel_values"].to(device)
        _, teacher_all = teacher(sample)
        student.forward_seq(sample)
    projector = FeatureProjector([_hook_features(hooks)[0].shape[1]], [teacher_all[-1].shape[1]]).to(device)
    gamma_decay = runtime_cfg["rknn_bn_gamma_weight_decay"]
    model_ema = ModelEmaV2(student, decay=0.995)
    optimizer = _make_optimizer(student, projector, runtime_cfg["learning_rate"], gamma_decay)
    try:
        _train_phase(student, teacher, projector, hooks, train_loader, val_loader, criterion, optimizer, runtime_cfg["epochs"], device, model_ema)
        pre_acc = _validate(model_ema.module, val_loader, device)
        print(f"Pre-reparameterization EMA accuracy: {pre_acc:.2%}")
        os.makedirs(runtime_cfg["output_dir"], exist_ok=True)
        torch.save(student.state_dict(), os.path.join(runtime_cfg["output_dir"], f"prefold_{best_model_name}"))
        torch.save(model_ema.module.state_dict(), os.path.join(runtime_cfg["output_dir"], f"prefold_{best_ema_model_name}"))
        verify_reparameterization(student, sample, runtime_cfg["rknn_reparam_tolerance"])
        for hook in hooks:
            hook.remove()
        hooks = [FeatureHook(student.backbone)]
        model_ema = ModelEmaV2(student, decay=0.995)  # keys/shapes change after fusion
        tune_lr = runtime_cfg["learning_rate"] * runtime_cfg["rknn_finetune_lr_ratio"]
        optimizer = _make_optimizer(student, projector, tune_lr, gamma_decay)
        _train_phase(student, teacher, projector, hooks, train_loader, val_loader, criterion, optimizer, runtime_cfg["rknn_finetune_epochs"], device, model_ema)
        acc = _validate(model_ema.module, val_loader, device)
        print(f"Post-reparameterization EMA accuracy: {acc:.2%}")
        os.makedirs(runtime_cfg["output_dir"], exist_ok=True)
        torch.save(student.state_dict(), os.path.join(runtime_cfg["output_dir"], best_model_name))
        torch.save(model_ema.module.state_dict(), os.path.join(runtime_cfg["output_dir"], best_ema_model_name))
        return acc
    finally:
        for hook in hooks:
            hook.remove()
