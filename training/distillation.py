import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.utils import ModelEmaV2
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from sklearn.metrics import confusion_matrix
except ImportError:
    confusion_matrix = None

from dataset import get_dataset
from models.student_factory import build_student
from models.teacher import SwinTeacher


torch.backends.cudnn.benchmark = False  # MobileNetV4 等模型触发 cuDNN autotuner 卡死，关闭后用默认算法
torch.backends.cudnn.deterministic = False


class FeatureHook:
    def __init__(self, module):
        self.features = None
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, inputs, output):
        self.features = output[0] if isinstance(output, (list, tuple)) else output

    def remove(self):
        self.hook.remove()


def _build_student_hooks(backbone) -> list:
    """
    在 backbone 上注册单个 hook，适配所有 timm backbone。
    timm FeatureListNet 的内部子模块名在不同 backbone 间不一致（如 stages.0 vs stages_0），
    用 feature_info + get_submodule 无法跨 backbone 可靠工作，
    因此统一在 backbone 整体上注册一个 hook，确保所有 backbone 行为一致。
    """
    return [FeatureHook(backbone)]


class FeatureProjector(nn.Module):
    def __init__(self, student_channels, teacher_channels):
        super().__init__()
        self.proj = nn.ModuleList(
            [
                nn.Sequential(nn.Conv2d(s, t, 1, bias=False), nn.BatchNorm2d(t), nn.ReLU())
                for s, t in zip(student_channels, teacher_channels)
            ]
        )

    def forward(self, feats):
        return [proj(feat) for proj, feat in zip(self.proj, feats)]


def run_distillation_training(
    runtime_cfg,
    criterion,
    best_model_name,
    best_ema_model_name,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = runtime_cfg["class_names"]
    num_classes = len(class_names)
    recognizer_cfg = dict(runtime_cfg["recognizer"])
    recognizer_cfg["class_names"] = class_names
    recognizer_cfg["num_classes"] = num_classes

    print(f"\nStart Distillation | Classes: {class_names}")

    dataset_root = runtime_cfg["dataset_path"]
    train_ds = get_dataset(dataset_root, "train", recognizer_cfg, is_train=True)
    val_ds = get_dataset(dataset_root, "val", recognizer_cfg, is_train=False)
    if len(train_ds) <= 0:
        raise ValueError(f"Empty train split: dataset_path={dataset_root!r}")
    if len(val_ds) <= 0:
        raise ValueError(f"Empty val split: dataset_path={dataset_root!r}")

    train_loader = DataLoader(
        train_ds,
        batch_size=runtime_cfg["batch_size"],
        shuffle=True,
        num_workers=runtime_cfg["num_workers"],
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=runtime_cfg["batch_size"],
        shuffle=False,
        num_workers=runtime_cfg["num_workers"],
    )

    teacher_ckpt = runtime_cfg["teacher_checkpoint"]
    print(f"Loading Teacher from: {teacher_ckpt}")
    teacher = SwinTeacher(checkpoint_path=teacher_ckpt, num_classes=num_classes).to(device).eval()
    for param in teacher.parameters():
        param.requires_grad = False

    backbone_name = recognizer_cfg.get("backbone") or recognizer_cfg.get("timm_model", "repvit_m1")
    print(f"Building Student ({backbone_name})...")
    student = build_student(recognizer_cfg, num_classes).to(device)
    model_ema = ModelEmaV2(student, decay=0.995)

    # 3D 模型（MoViNet 等）跳过 feature loss：无 backbone 属性，不注册 hook
    use_feature_loss = not getattr(student, "is_3d", False)

    if use_feature_loss:
        s_hooks = _build_student_hooks(student.backbone)
        dummy = torch.randn(2, 16, 3, 224, 224).to(device)
        with torch.no_grad():
            _, teacher_all = teacher(dummy)
            teacher_feats = teacher_all[-len(s_hooks):]
            student.forward_seq(dummy)
        teacher_channels = [feat.shape[1] for feat in teacher_feats]
        student_channels = [
            hook.features[0].shape[1] if isinstance(hook.features, (list, tuple)) else hook.features.shape[1]
            for hook in s_hooks
        ]
        projector = FeatureProjector(student_channels, teacher_channels).to(device)
    else:
        print("3D student detected: feature loss disabled.")
        s_hooks = []
        projector = None

    optimizer = torch.optim.AdamW(
        list(student.parameters()) + (list(projector.parameters()) if projector else []),
        lr=runtime_cfg["learning_rate"],
        weight_decay=0.05,
    )
    epochs = runtime_cfg["epochs"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler(device.type)

    best_acc = 0.0
    out_dir = runtime_cfg["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    try:
        for epoch in range(epochs):
            student.train()
            if projector is not None:
                projector.train()
            epoch_ratio = epoch / epochs
            avg_bin = 0.0
            avg_fine = 0.0
            avg_kd = 0.0
            avg_ft = 0.0
            train_correct = 0
            train_total = 0
            train_loss_total = 0.0
            pbar = tqdm(train_loader, desc=f"Ep {epoch + 1}/{epochs}")

            for i, batch in enumerate(pbar):
                videos = batch["pixel_values"].to(device, non_blocking=True)
                labels = batch["labels"].to(device, non_blocking=True)
                batch_size = videos.shape[0]
                num_frames = videos.shape[1]

                with torch.amp.autocast(device.type):
                    with torch.no_grad():
                        teacher_logits, teacher_all = teacher(videos)
                        teacher_target_feats = teacher_all[-len(s_hooks):] if s_hooks else []

                    student_logits = student.forward_seq(videos)

                    if use_feature_loss:
                        student_raw_feats = [
                            hook.features[0] if isinstance(hook.features, (list, tuple)) else hook.features
                            for hook in s_hooks
                        ]
                        student_proj_feats = projector(student_raw_feats)
                    else:
                        student_proj_feats = []
                        teacher_target_feats = []

                    loss, l_bin, l_fine, l_kd, l_feat = criterion(
                        student_logits,
                        teacher_logits,
                        student_proj_feats,
                        teacher_target_feats,
                        labels,
                        batch_size,
                        num_frames,
                        epoch_ratio,
                    )

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                model_ema.update(student)

                train_correct += (student_logits.detach().argmax(1) == labels).sum().item()
                train_total += labels.size(0)
                train_loss_total += loss.item()

                avg_bin = (avg_bin * i + l_bin) / (i + 1)
                avg_fine = (avg_fine * i + l_fine) / (i + 1)
                avg_kd = (avg_kd * i + l_kd) / (i + 1)
                avg_ft = (avg_ft * i + l_feat) / (i + 1)
                pbar.set_postfix(Bin=f"{avg_bin:.3f}", Fine=f"{avg_fine:.3f}", KD=f"{avg_kd:.2f}", Ft=f"{avg_ft:.2f}")

            train_acc = train_correct / train_total if train_total > 0 else 0.0
            train_loss_avg = train_loss_total / len(train_loader)
            print(f"\nTrain Acc: {train_acc:.2%} | Train Loss: {train_loss_avg:.4f}")

            scheduler.step()

            student.eval()
            all_preds = []
            all_targets = []
            with torch.no_grad():
                for batch in val_loader:
                    videos = batch["pixel_values"].to(device)
                    labels = batch["labels"].to(device)
                    logits = model_ema.module.forward_seq(videos)
                    all_preds.extend(logits.argmax(1).cpu().numpy())
                    all_targets.extend(labels.cpu().numpy())

            acc = (
                sum(pred == target for pred, target in zip(all_preds, all_targets)) / len(all_preds)
                if all_preds
                else 0.0
            )
            print(f"\nVal Acc (EMA): {acc:.2%}")

            try:
                if confusion_matrix is None:
                    raise RuntimeError("sklearn is unavailable")
                cm = confusion_matrix(all_targets, all_preds, labels=list(range(num_classes)))
                print("Confusion Matrix:")
                for idx, row in enumerate(cm):
                    print(f"True {class_names[idx][:8]:<8}: {row}")
            except Exception:
                pass

            if acc > best_acc:
                best_acc = acc
                torch.save(student.state_dict(), os.path.join(out_dir, best_model_name))
                torch.save(model_ema.module.state_dict(), os.path.join(out_dir, best_ema_model_name))
                print("Saved best student model.")
            print("-" * 50)
    finally:
        for hook in s_hooks:
            hook.remove()



def run_distillation(runtime_cfg, criterion, best_model_name, best_ema_model_name):
    return run_distillation_training(
        runtime_cfg=runtime_cfg,
        criterion=criterion,
        best_model_name=best_model_name,
        best_ema_model_name=best_ema_model_name,
    )
