# Multi-Backbone Student Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为蒸馏训练流程新增 5 种 2D backbone 支持（MobileNetV4-Conv-M、EfficientNet-B2/B4、ConvNeXt-Pico/Nano），backbone 通过 config.yaml 的 `backbone` 字段切换，现有 RepViT_GRU 接口完全向后兼容。

**Architecture:** 新增 `TimmGRUStudent`（泛化 backbone 参数的 GRU 学生）和 `build_student` 工厂函数；`models/student.py` 改为薄 shim；`training/distillation.py` 用工厂替换硬编码实例化，并修复 feature hook 注册逻辑使其适配任意 timm backbone。

**Tech Stack:** Python 3.14, PyTorch 2.12, timm, torchvision 0.27, pyyaml

---

## 文件改动清单

| 操作 | 文件 | 说明 |
|---|---|---|
| 新建 | `models/timm_gru_student.py` | 泛化版 backbone+GRU，核心实现 |
| 修改 | `models/student.py` | 改为 shim，保持 RepViT_GRU 接口 |
| 新建 | `models/student_factory.py` | build_student 工厂 + hidden size 查表 |
| 修改 | `training/distillation.py` | 5 处修改（见各 Task） |
| 修改 | `config.yaml` | 每个 recognizer 块新增 backbone/gru_hidden |
| 新建 | `smoke_test_backbones.py` | 快速验证所有 backbone 可加载并前向传播 |
| 新建 | `run_all_backbones.py` | 顺序训练所有 5 个 backbone |
| 新建 | `logs/` 目录 | 存放训练日志 |

---

## Task 1: 创建 models/timm_gru_student.py

**Files:**
- Create: `models/timm_gru_student.py`

- [ ] **Step 1: 创建文件**

```python
# models/timm_gru_student.py
import timm
import torch
import torch.nn as nn


class TimmGRUStudent(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        num_classes: int,
        hidden_dim: int = 256,
        pretrained: bool = True,
        dropout: float = 0.2,
    ):
        super().__init__()
        print(f"Initializing {backbone_name} Backbone (Pretrained={pretrained})...")
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=[-1],
        )
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            feat = self.backbone(dummy)[0]
            in_channels = feat.shape[1]
            print(f"Backbone Output Channels: {in_channels}")

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.gru = nn.GRU(input_size=in_channels, hidden_size=hidden_dim, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x, hidden_state=None):
        """单帧推理: x [B, C, H, W] → (logits [B, C], next_hidden)"""
        features = self.backbone(x)[0]
        x = self.gap(features)
        x = self.flatten(x)
        x = x.unsqueeze(1)
        out, next_hidden = self.gru(x, hidden_state)
        logits = self.fc(out[:, -1, :])
        return logits, next_hidden

    def forward_seq(self, x):
        """Clip 训练: x [B, T, C, H, W] → logits [B, num_classes]"""
        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w)
        features = self.backbone(x)[0]
        x = self.gap(features)
        x = self.flatten(x)
        x = x.view(b, t, -1)
        out, _ = self.gru(x)
        logits = self.fc(out[:, -1, :])
        return logits
```

---

## Task 2: 更新 models/student.py（改为薄 shim）

**Files:**
- Modify: `models/student.py`

- [ ] **Step 1: 替换全部内容**

将现有 `models/student.py` 内容全部替换为：

```python
# models/student.py
# 向后兼容 shim：保持 RepViT_GRU 类名可导入
from models.timm_gru_student import TimmGRUStudent


class RepViT_GRU(TimmGRUStudent):
    def __init__(self, num_classes=5, hidden_dim=256, pretrained=True):
        super().__init__("repvit_m1", num_classes, hidden_dim, pretrained=pretrained)
```

---

## Task 3: 创建 models/student_factory.py

**Files:**
- Create: `models/student_factory.py`

- [ ] **Step 1: 创建文件**

```python
# models/student_factory.py
from models.timm_gru_student import TimmGRUStudent

# GRU hidden size 查表（gru_hidden="auto" 时生效）
# 维度来自本机 dummy forward 实测（features_only=True, out_indices=[-1]）
BACKBONE_GRU_HIDDEN = {
    "repvit_m1": 256,              # 实测特征维度 384
    "mobilenetv4_conv_medium": 512, # 实测特征维度 960
    "efficientnet_b2": 256,         # 实测特征维度 352
    "efficientnet_b4": 384,         # 估算特征维度 ~448，实现时以 dummy forward 为准
    "convnext_pico": 256,           # 实测特征维度 512
    "convnext_nano": 384,           # 估算特征维度 ~640，实现时以 dummy forward 为准
}


def build_student(recognizer_cfg: dict, num_classes: int) -> TimmGRUStudent:
    """
    根据 recognizer_cfg 构建学生模型。
    backbone 字段优先于旧 timm_model 字段（向后兼容）。
    gru_hidden 为 "auto" 时查表，否则直接用整数值。
    """
    backbone = recognizer_cfg.get("backbone") or recognizer_cfg.get("timm_model", "repvit_m1")
    hidden = recognizer_cfg.get("gru_hidden", "auto")
    if hidden == "auto" or not hidden:
        hidden = BACKBONE_GRU_HIDDEN.get(backbone, 256)
    dropout = recognizer_cfg.get("dropout", 0.2)
    return TimmGRUStudent(backbone, num_classes, int(hidden), pretrained=True, dropout=dropout)
```

---

## Task 4: 修改 training/distillation.py（5 处改动）

**Files:**
- Modify: `training/distillation.py`

### 改动 1：替换 import（第 16 行）

- [ ] **Step 1: 替换 student import**

将：
```python
from models.student import RepViT_GRU
```
替换为：
```python
from models.student_factory import build_student
```

---

### 改动 2：新增 `_build_student_hooks` 辅助函数

- [ ] **Step 2: 在 FeatureProjector 类定义之后、run_distillation_training 之前，插入以下函数**

```python
def _build_student_hooks(backbone, n_hooks: int = 3) -> list:
    """
    用 timm feature_info 定位各 stage 模块并注册 hook。
    不同 backbone 层名不同，feature_info 是 timm 的统一接口。
    若 feature_info 层数不足或路径无效，回退到单 hook（feature loss 信号减少但不报错）。
    """
    if hasattr(backbone, "feature_info") and len(backbone.feature_info) >= n_hooks:
        try:
            return [
                FeatureHook(backbone.get_submodule(info["module"]))
                for info in backbone.feature_info[-n_hooks:]
            ]
        except (AttributeError, ModuleNotFoundError):
            pass
    print("[WARN] feature_info 不可用，回退到单 hook（feature loss 信号减少）")
    return [FeatureHook(backbone)]
```

---

### 改动 3：device 回退（第 55 行）

- [ ] **Step 3: 替换 device 定义**

将：
```python
    device = torch.device("cuda")
```
替换为：
```python
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

---

### 改动 4：替换 student 实例化 + hook 注册（第 93–101 行）

- [ ] **Step 4: 替换 student 实例化和 hook 注册**

将：
```python
    print("Building Student (RepViT-GRU)...")
    student = RepViT_GRU(num_classes=num_classes, hidden_dim=256, pretrained=True).to(device)
    model_ema = ModelEmaV2(student, decay=0.995)

    if hasattr(student.backbone, "stages"):
        student_stages = student.backbone.stages
        s_hooks = [FeatureHook(student_stages[idx]) for idx in [-3, -2, -1]]
    else:
        s_hooks = [FeatureHook(student.backbone)]
```
替换为：
```python
    backbone_name = recognizer_cfg.get("backbone") or recognizer_cfg.get("timm_model", "repvit_m1")
    print(f"Building Student ({backbone_name} + GRU)...")
    student = build_student(recognizer_cfg, num_classes).to(device)
    model_ema = ModelEmaV2(student, decay=0.995)

    s_hooks = _build_student_hooks(student.backbone, n_hooks=3)
```

---

### 改动 5：autocast 和 GradScaler 使用 device.type（第 123、146 行）

- [ ] **Step 5: 替换 autocast device**

将：
```python
    scaler = torch.amp.GradScaler("cuda")
```
替换为：
```python
    scaler = torch.amp.GradScaler(device.type)
```

将：
```python
                with torch.amp.autocast("cuda"):
```
替换为：
```python
                with torch.amp.autocast(device.type):
```

---

## Task 5: 更新 config.yaml（新增 backbone / gru_hidden 字段）

**Files:**
- Modify: `config.yaml`

- [ ] **Step 1: 在 normal 和 abnormal 两个 recognizer 块各新增两行**

在 `task_configs.normal.recognizer` 块中，`timm_model: repvit_m1` 下方新增：
```yaml
      backbone: repvit_m1
      gru_hidden: auto
```

在 `task_configs.abnormal.recognizer` 块中，`timm_model: repvit_m1` 下方新增：
```yaml
      backbone: repvit_m1
      gru_hidden: auto
```

（这两个字段不改变现有行为；只有 `run_all_backbones.py` 在运行时会动态覆盖它们。）

---

## Task 6: 验证 teacher checkpoint 存在

**Files:**
- 无文件改动

- [ ] **Step 1: 检查 checkpoint 路径**

运行：
```bash
ls -la "E:/ActionRecognition/behavionRecognition-src/outputs/rabbit/0518/teacher_finetuned_4cls_0518/"
```

预期：能看到 `rabbit_teacher_0515_ema.pth` 文件。

**如果文件不存在**：训练无法进行，需先运行 `python train_finetune_normal_3.py` 完成教师微调。停止后续 Task，先解决此问题。

---

## Task 7: 创建 smoke_test_backbones.py

**Files:**
- Create: `smoke_test_backbones.py`

- [ ] **Step 1: 创建文件**

```python
#!/usr/bin/env python
"""
快速验证所有 backbone 能正确初始化并运行 forward_seq。
在 CPU 上运行，用小 batch（2×8 帧）加速。
"""
import os
import sys

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)

import torch
from models.student_factory import build_student, BACKBONE_GRU_HIDDEN

NUM_CLASSES = 4
DEVICE = torch.device("cpu")

all_ok = True
for backbone in BACKBONE_GRU_HIDDEN:
    print(f"Testing {backbone}...", end=" ", flush=True)
    try:
        cfg = {"backbone": backbone, "gru_hidden": "auto", "dropout": 0.2}
        model = build_student(cfg, NUM_CLASSES).to(DEVICE)
        x = torch.randn(2, 8, 3, 224, 224)
        with torch.no_grad():
            logits = model.forward_seq(x)
        assert logits.shape == (2, NUM_CLASSES), f"shape error: {logits.shape}"
        # 打印实际特征维度（来自 dummy forward）
        feat_dim = model.gru.input_size
        print(f"OK  feat_dim={feat_dim}, gru_hidden={model.gru.hidden_size}, output={logits.shape}")
    except Exception as e:
        print(f"FAIL  {e}")
        all_ok = False

if all_ok:
    print("\nAll backbones OK!")
    sys.exit(0)
else:
    print("\nSome backbones FAILED. Fix before running training.")
    sys.exit(1)
```

- [ ] **Step 2: 运行 smoke test**

```bash
cd "E:/ActionRecognition/behavionRecognition-src" && python smoke_test_backbones.py
```

预期输出示例：
```
Testing repvit_m1... OK  feat_dim=384, gru_hidden=256, output=torch.Size([2, 4])
Testing mobilenetv4_conv_medium... OK  feat_dim=960, gru_hidden=512, output=torch.Size([2, 4])
Testing efficientnet_b2... OK  feat_dim=352, gru_hidden=256, output=torch.Size([2, 4])
Testing efficientnet_b4... OK  feat_dim=448, gru_hidden=384, output=torch.Size([2, 4])
Testing convnext_pico... OK  feat_dim=512, gru_hidden=256, output=torch.Size([2, 4])
Testing convnext_nano... OK  feat_dim=640, gru_hidden=384, output=torch.Size([2, 4])
All backbones OK!
```

**如果某个 backbone FAIL**：根据报错修正（常见：`features_only=True` 不支持该 backbone 名，需改名；或 timm 版本缺少该模型）。

---

## Task 8: 创建 run_all_backbones.py

**Files:**
- Create: `run_all_backbones.py`

- [ ] **Step 1: 创建文件**

```python
#!/usr/bin/env python
"""
顺序训练 5 个 backbone 变体的蒸馏模型。
结果各自保存到 outputs/rabbit/0518/distilled_<backbone>/ 。
运行方式（后台，日志写文件）：
  nohup python run_all_backbones.py > logs/all_backbones.log 2>&1 &
"""
import os
import sys
import time
import yaml

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)

from training.config import build_distill_config
from training.distillation import run_distillation_training
from training.distillation_losses import build_priority_composite_distill_loss

BACKBONES = [
    "mobilenetv4_conv_medium",
    "efficientnet_b2",
    "efficientnet_b4",
    "convnext_pico",
    "convnext_nano",
]

CONFIG_PATH = os.path.join(ROOT_DIR, "config.yaml")
LOG_DIR = os.path.join(ROOT_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)


def _resolve_anchor_idx(class_names):
    if "resting" in class_names:
        return class_names.index("resting")
    return 2 if len(class_names) > 2 else len(class_names) - 1


def train_backbone(backbone: str):
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        root_cfg = yaml.safe_load(f)

    # 覆盖 backbone 和输出目录（不修改磁盘上的 config.yaml）
    root_cfg["task_configs"]["normal"]["recognizer"]["backbone"] = backbone
    root_cfg["task_configs"]["normal"]["recognizer"]["gru_hidden"] = "auto"
    root_cfg["task_configs"]["normal"]["distillation"]["output_dir"] = (
        f"outputs/rabbit/0518/distilled_{backbone}"
    )

    runtime_cfg = build_distill_config(mode="normal", config_dict=root_cfg)

    anchor_idx = _resolve_anchor_idx(runtime_cfg["class_names"])
    criterion = build_priority_composite_distill_loss(
        anchor_idx=anchor_idx,
        alpha=runtime_cfg["loss_weights"]["feature"],
        beta=runtime_cfg["loss_weights"]["kd"],
        gamma=runtime_cfg["loss_weights"]["gt"],
        temperature=runtime_cfg["temperature"],
    )

    run_distillation_training(
        runtime_cfg=runtime_cfg,
        criterion=criterion,
        best_model_name=f"{backbone}_best.pth",
        best_ema_model_name=f"{backbone}_ema_best.pth",
    )


if __name__ == "__main__":
    results = {}
    total_start = time.time()

    for backbone in BACKBONES:
        print(f"\n{'='*60}", flush=True)
        print(f"  Training: {backbone}", flush=True)
        print(f"{'='*60}\n", flush=True)
        t0 = time.time()
        try:
            train_backbone(backbone)
            elapsed = time.time() - t0
            results[backbone] = f"OK ({elapsed / 60:.1f} min)"
            print(f"\n[OK] {backbone} done in {elapsed / 60:.1f} min", flush=True)
        except Exception as e:
            import traceback
            results[backbone] = f"FAILED: {e}"
            print(f"\n[FAIL] {backbone}: {e}", flush=True)
            traceback.print_exc()

    total_elapsed = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"Summary (total {total_elapsed / 60:.1f} min):")
    for backbone, status in results.items():
        print(f"  {backbone}: {status}")
```

---

## Task 9: 创建 logs 目录并后台启动训练

**Files:**
- 无文件改动（训练产物写入 `outputs/rabbit/0518/distilled_*/`）

- [ ] **Step 1: 确认 smoke test 全部通过（Task 7 Step 2 结果为 All backbones OK!）**

- [ ] **Step 2: 后台启动训练，日志写入文件**

```bash
cd "E:/ActionRecognition/behavionRecognition-src" && nohup python run_all_backbones.py > logs/all_backbones.log 2>&1 &
echo "Training PID: $!"
```

预期：输出一个 PID 数字（如 `Training PID: 12345`），说明进程已在后台运行。

- [ ] **Step 3: 确认进程已启动，查看日志头**

```bash
sleep 30 && head -50 "E:/ActionRecognition/behavionRecognition-src/logs/all_backbones.log"
```

预期：能看到第一个 backbone（mobilenetv4_conv_medium）的初始化和 epoch 1 训练输出，无 crash 信息。

- [ ] **Step 4: 检查输出目录已创建**

```bash
ls "E:/ActionRecognition/behavionRecognition-src/outputs/rabbit/0518/"
```

预期：能看到 `distilled_mobilenetv4_conv_medium/` 目录已创建。

---

## 结果说明

训练结束后，5 个 backbone 各自的最佳 checkpoint 保存在：

```
outputs/rabbit/0518/
  distilled_mobilenetv4_conv_medium/
    mobilenetv4_conv_medium_best.pth
    mobilenetv4_conv_medium_ema_best.pth
  distilled_efficientnet_b2/
    efficientnet_b2_best.pth
    efficientnet_b2_ema_best.pth
  distilled_efficientnet_b4/
    ...
  distilled_convnext_pico/
    ...
  distilled_convnext_nano/
    ...
```

完整日志见 `logs/all_backbones.log`。
