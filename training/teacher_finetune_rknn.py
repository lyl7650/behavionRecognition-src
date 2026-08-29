"""Opt-in teacher finetune loop with mild BatchNorm gamma regularization."""

import os

import torch
import torch.optim as optim
from timm.utils import ModelEmaV2
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import get_dataset
from training.rknn_optim import adamw_groups_with_bn_gamma_decay
from training.teacher_finetune import TrainableSwin


def run_teacher_finetune_rknn(cfg, criterion, save_name, *, print_config=False, pass_num_classes_to_teacher=False):
    """Copy-only normal-teacher trainer. BN bias stays undecayed; gamma gets 1e-5..1e-4."""
    rec_cfg = {"imgsz": cfg["imgsz"], "temporal_window": cfg["temporal_window"], "class_names": cfg["class_names"]}
    if "augment" in cfg.get("recognizer", {}):
        rec_cfg["augment"] = cfg["recognizer"]["augment"]
    train_ds = get_dataset(cfg["dataset_path"], "train", rec_cfg, is_train=True)
    val_ds = get_dataset(cfg["dataset_path"], "val", rec_cfg, is_train=False)
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg["num_workers"], pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"])
    model = TrainableSwin(cfg["num_classes"], cfg["checkpoint_path"], pass_num_classes_to_teacher).cuda()
    ema = ModelEmaV2(model, decay=0.995)
    optimizer = optim.AdamW(
        adamw_groups_with_bn_gamma_decay(model, base_weight_decay=0.1, bn_gamma_weight_decay=cfg["rknn_bn_gamma_weight_decay"]),
        lr=cfg["lr"],
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"], eta_min=1e-6)
    scaler = GradScaler()
    best_acc = -1.0
    os.makedirs(cfg["output_dir"], exist_ok=True)
    for epoch in range(cfg["epochs"]):
        model.train()
        for i, batch in enumerate(tqdm(train_loader, desc=f"RKNN teacher {epoch + 1}/{cfg['epochs']}")):
            x, y = batch["pixel_values"].cuda(non_blocking=True), batch["labels"].cuda(non_blocking=True)
            with autocast("cuda"):
                logits = model(x)
                loss, _, _ = criterion(logits, y)
                loss = loss / cfg["accum_steps"]
            scaler.scale(loss).backward()
            if (i + 1) % cfg["accum_steps"] == 0 or i + 1 == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                ema.update(model)
        scheduler.step()
        ema.module.eval()
        correct = total = 0
        with torch.no_grad():
            for batch in val_loader:
                logits = ema.module(batch["pixel_values"].cuda())
                labels = batch["labels"].cuda()
                correct += (logits.argmax(1) == labels).sum().item()
                total += labels.numel()
        acc = correct / total if total else 0.0
        print(f"RKNN teacher EMA val accuracy: {acc:.2%}")
        if acc > best_acc:
            best_acc = acc
            torch.save(ema.module.state_dict(), os.path.join(cfg["output_dir"], save_name))
    return best_acc
