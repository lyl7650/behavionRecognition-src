import os

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from sklearn.metrics import confusion_matrix
from timm.utils import ModelEmaV2
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import get_dataset
from models.teacher import SwinTeacher


def apply_stochastic_depth(model, drop_prob=0.2):
    count = 0
    for module in model.modules():
        if isinstance(module, torchvision.ops.StochasticDepth):
            module.p = drop_prob
            count += 1
    print(f"[Regularization] Updated {count} StochasticDepth layers to p={drop_prob}")


class TrainableSwin(nn.Module):
    def __init__(self, num_classes, checkpoint_path=None, pass_num_classes_to_teacher=False):
        super().__init__()
        print("Initializing Swin-B teacher...")

        if checkpoint_path is None or not os.path.exists(checkpoint_path):
            print(f"Checkpoint not found: {checkpoint_path}; fallback to pretrained/random init")
            checkpoint_path = None

        teacher_num_classes = num_classes if pass_num_classes_to_teacher else 400
        self.base = SwinTeacher(checkpoint_path=checkpoint_path, num_classes=teacher_num_classes)
        apply_stochastic_depth(self.base.model, drop_prob=0.2)

        for _, param in self.base.model.named_parameters():
            param.requires_grad = True

        old_head = self.base.model.head
        in_features = (
            old_head.in_features
            if not isinstance(old_head, nn.Sequential)
            else old_head[-1].in_features
        )
        self.base.model.head = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(in_features, num_classes),
        )

        for param in self.base.model.head.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.base(x)[0]


def run_teacher_finetune(
    cfg,
    criterion,
    save_name,
    *,
    print_config=False,
    pass_num_classes_to_teacher=False,
):
    print("\n[Teacher] Swin-B finetuning")

    rec_cfg = {
        "imgsz": cfg["imgsz"],
        "temporal_window": cfg["temporal_window"],
        "class_names": cfg["class_names"],
    }
    if "augment" in cfg.get("recognizer", {}):
        rec_cfg["augment"] = cfg["recognizer"]["augment"]

    if print_config:
        print(f"[Config] dataset_path = {cfg['dataset_path']}")
        print(f"[Config] class_names = {cfg['class_names']}")
        print(f"[Config] num_classes = {cfg['num_classes']}")
        print(f"[Config] checkpoint_path = {cfg['checkpoint_path']}")
        print(f"[Config] output_dir = {cfg['output_dir']}")

    train_ds = get_dataset(cfg["dataset_path"], "train", rec_cfg, is_train=True)
    val_ds = get_dataset(cfg["dataset_path"], "val", rec_cfg, is_train=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
    )

    model = TrainableSwin(
        num_classes=cfg["num_classes"],
        checkpoint_path=cfg["checkpoint_path"],
        pass_num_classes_to_teacher=pass_num_classes_to_teacher,
    ).cuda()
    model_ema = ModelEmaV2(model, decay=0.995)

    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "head" in name:
            head_params.append(param)
        else:
            backbone_params.append(param)

    optimizer = optim.AdamW(
        [
            {"params": backbone_params, "lr": cfg["lr"]},
            {"params": head_params, "lr": cfg["lr"]},
        ],
        weight_decay=0.1,
    )
    scaler = GradScaler()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg["epochs"],
        eta_min=1e-6,
    )

    best_acc = 0.0
    best_val_loss = float("inf")
    patience = 7
    no_improve_epochs = 0
    os.makedirs(cfg["output_dir"], exist_ok=True)

    for epoch in range(cfg["epochs"]):
        model.train()
        avg_bin = 0.0
        avg_fine = 0.0
        train_correct = 0
        train_total = 0
        train_loss_total = 0.0
        pbar = tqdm(train_loader, desc=f"Ep {epoch + 1}/{cfg['epochs']}")

        for i, batch in enumerate(pbar):
            x = batch["pixel_values"].cuda(non_blocking=True)
            y = batch["labels"].cuda(non_blocking=True)

            with autocast("cuda"):
                logits = model(x)
                loss, l_bin, l_fine = criterion(logits, y)
                loss = loss / cfg["accum_steps"]

            scaler.scale(loss).backward()

            if (i + 1) % cfg["accum_steps"] == 0 or (i + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                model_ema.update(model)

            train_correct += (logits.detach().argmax(1) == y).sum().item()
            train_total += y.size(0)
            train_loss_total += loss.item() * cfg["accum_steps"]

            avg_bin = 0.9 * avg_bin + 0.1 * l_bin if i > 0 else l_bin
            avg_fine = 0.9 * avg_fine + 0.1 * l_fine if i > 0 else l_fine
            pbar.set_postfix(
                L=f"{loss.item() * cfg['accum_steps']:.3f}",
                Bin=f"{avg_bin:.3f}",
                Fine=f"{avg_fine:.3f}",
            )

        train_acc = train_correct / train_total if train_total > 0 else 0.0
        train_loss_avg = train_loss_total / len(train_loader)
        print(f"\nTrain Acc: {train_acc:.2%} | Train Loss: {train_loss_avg:.4f}")

        scheduler.step()

        model_ema.module.eval()
        all_preds = []
        all_targets = []
        val_loss_total = 0.0

        with torch.no_grad():
            for batch in val_loader:
                x = batch["pixel_values"].cuda()
                y = batch["labels"].cuda()
                with autocast("cuda"):
                    logits = model_ema.module(x)
                    v_loss, _, _ = criterion(logits, y)
                    val_loss_total += v_loss.item()

                all_preds.extend(logits.argmax(1).cpu().numpy())
                all_targets.extend(y.cpu().numpy())

        val_loss_avg = val_loss_total / len(val_loader)
        val_acc = (
            sum(pred == target for pred, target in zip(all_preds, all_targets)) / len(all_preds)
            if all_preds
            else 0.0
        )
        print(f"\nEMA Val Acc: {val_acc:.2%} | Val Loss: {val_loss_avg:.4f}")

        try:
            labels = list(range(cfg["num_classes"]))
            cm = confusion_matrix(all_targets, all_preds, labels=labels)
            print("Confusion Matrix:")
            class_names = cfg["class_names"]
            for idx, row in enumerate(cm):
                print(f"True {class_names[idx][:8]:<8}: {row}")
        except Exception:
            pass

        is_best = False
        if val_acc > best_acc:
            best_acc = val_acc
            is_best = True
        elif val_acc == best_acc and val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            is_best = True

        if is_best:
            best_val_loss = min(best_val_loss, val_loss_avg)
            torch.save(
                model_ema.module.state_dict(),
                os.path.join(cfg["output_dir"], save_name),
            )
            print("Saved new best EMA teacher.")
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            print(f"No improvement for {no_improve_epochs} epoch(s).")

        if no_improve_epochs >= patience:
            print(f"Early stopping triggered after {patience} epochs without improvement.")
            break

        print("-" * 50)
