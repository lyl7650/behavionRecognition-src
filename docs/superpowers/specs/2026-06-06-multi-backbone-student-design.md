# Multi-Backbone Student Model Design

**Date:** 2026-06-06  
**Scope:** 为蒸馏训练流程新增多种 2D backbone 支持（MobileNetV4-Conv-M、EfficientNet-B2/B4、ConvNeXt-Pico/Nano），通过 config.yaml 切换，不破坏现有 RepViT_GRU 接口。

---

## 背景与约束

- 当前部署场景：PC CPU 实时 RTSP 推理，student 模型做流式逐帧推理（`forward` + GRU state），训练用 `forward_seq`（clip 输入）。
- 教师模型（SwinTeacher）不变。
- 3D 模型（MoViNet）暂不实现，待 pytorchvideo 兼容性确认后单独处理。
- 现有代码不是 git repo，不需要 commit。

---

## 文件改动清单

```
models/
  timm_gru_student.py   ← 新建：泛化后的 2D backbone + GRU
  student.py            ← 改为薄 shim（向后兼容）
  student_factory.py    ← 新建：build_student(recognizer_cfg, num_classes)

training/
  distillation.py       ← 改 3 处（见下文）

config.yaml             ← 每个 recognizer 块新增 backbone / gru_hidden 字段
```

---

## 1. `models/timm_gru_student.py`（新建）

从现有 `student.py` 泛化而来，将硬编码 `"repvit_m1"` 改为参数。

**构造函数：**
```python
class TimmGRUStudent(nn.Module):
    def __init__(self, backbone_name: str, num_classes: int,
                 hidden_dim: int = 256, pretrained: bool = True, dropout: float = 0.2):
        self.backbone = timm.create_model(
            backbone_name, pretrained=pretrained,
            features_only=True, out_indices=[-1]
        )
        # dummy forward 推断 in_channels（原逻辑保留）
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            in_channels = self.backbone(dummy)[0].shape[1]
        self.gap     = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.gru     = nn.GRU(input_size=in_channels, hidden_size=hidden_dim, batch_first=True)
        self.drop    = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_dim, num_classes)
        self._init_weights()
```

**两个 forward 方法原样保留：**
- `forward(x, hidden_state=None)` → 单帧 [B, C, H, W]，返回 `(logits, next_hidden)`，供部署推理使用。
- `forward_seq(x)` → clip [B, T, C, H, W]，返回 `logits`，供蒸馏训练使用。

---

## 2. `models/student.py`（改为薄 shim）

保持对外接口不变，防止任何直接 import `RepViT_GRU` 的地方报错。

```python
from models.timm_gru_student import TimmGRUStudent

class RepViT_GRU(TimmGRUStudent):
    def __init__(self, num_classes=5, hidden_dim=256, pretrained=True):
        super().__init__("repvit_m1", num_classes, hidden_dim, pretrained=pretrained)
```

---

## 3. `models/student_factory.py`（新建）

### GRU hidden size 查表

| backbone | 实测特征维度 | gru_hidden (auto) |
|---|---|---|
| repvit_m1 | 384 | 256 |
| mobilenetv4_conv_medium | 960 | 512 |
| efficientnet_b2 | 352 | 256 |
| efficientnet_b4 | ~448（待实测确认） | 384 |
| convnext_pico | 512 | 256 |
| convnext_nano | ~640（待实测确认） | 384 |

> 特征维度均通过 `features_only=True, out_indices=[-1]` + dummy forward 在本地实测，实现时以 dummy forward 结果为准，查表仅作默认初始值参考。

### `build_student` 函数

```python
BACKBONE_GRU_HIDDEN = {
    "repvit_m1": 256,
    "mobilenetv4_conv_medium": 512,
    "efficientnet_b2": 256,
    "efficientnet_b4": 384,
    "convnext_pico": 256,
    "convnext_nano": 384,
}

def build_student(recognizer_cfg: dict, num_classes: int) -> TimmGRUStudent:
    # backbone 字段优先于旧 timm_model 字段（向后兼容）
    backbone = recognizer_cfg.get("backbone") or recognizer_cfg.get("timm_model", "repvit_m1")
    hidden   = recognizer_cfg.get("gru_hidden", "auto")
    if hidden == "auto" or not hidden:
        hidden = BACKBONE_GRU_HIDDEN.get(backbone, 256)
    dropout  = recognizer_cfg.get("dropout", 0.2)
    return TimmGRUStudent(backbone, num_classes, int(hidden), pretrained=True, dropout=dropout)
```

---

## 4. `training/distillation.py`（改 3 处）

### 改动 1：import + student 实例化（L16, L93-94）

```python
# 删除：
from models.student import RepViT_GRU

# 新增：
from models.student_factory import build_student

# 删除：
print("Building Student (RepViT-GRU)...")
student = RepViT_GRU(num_classes=num_classes, hidden_dim=256, pretrained=True).to(device)

# 替换为：
backbone_name = recognizer_cfg.get("backbone") or recognizer_cfg.get("timm_model", "repvit_m1")
print(f"Building Student ({backbone_name} + GRU)...")
student = build_student(recognizer_cfg, num_classes).to(device)
```

### 改动 2：student feature hook 注册（L97-101）

现有代码用 `hasattr(student.backbone, "stages")` 判断，是 RepViT 专用 hack，且存在潜在 bug——
fallback 只注册 1 个 hook 但后续 `FeatureProjector` 按 3 个 teacher features 构建，维度不匹配会报错。

替换为基于 timm `feature_info` 的通用注册逻辑：

```python
# 删除（L97-101）：
if hasattr(student.backbone, "stages"):
    student_stages = student.backbone.stages
    s_hooks = [FeatureHook(student_stages[idx]) for idx in [-3, -2, -1]]
else:
    s_hooks = [FeatureHook(student.backbone)]

# 替换为（新增 helper 函数，放在 FeatureHook 类定义之后）：
def _build_student_hooks(backbone, n_hooks: int = 3) -> list:
    """
    用 timm feature_info 定位各 stage 模块并注册 hook。
    不同 backbone 的层名不同（stages / blocks 等），
    feature_info 是 timm features_only 模型的统一接口。
    若 feature_info 层数不足，回退到单 hook（训练仍可运行，feature loss 信号减少）。
    """
    if hasattr(backbone, "feature_info") and len(backbone.feature_info) >= n_hooks:
        return [
            FeatureHook(backbone.get_submodule(info["module"]))
            for info in backbone.feature_info[-n_hooks:]
        ]
    return [FeatureHook(backbone)]

# 使用：
s_hooks = _build_student_hooks(student.backbone, n_hooks=3)
```

> `teacher_feats = teacher_all[-len(s_hooks):]` 这一行现有代码已经按 `len(s_hooks)` 动态截取，
> 所以 hook 数量可变（1 或 3），后续 FeatureProjector 和 loss 计算均自动适配，不需要改。

### 改动 3：device 回退（L55）

现有代码硬编码 `device = torch.device("cuda")`，新场景可能在 CPU 上训练（如调试）。

```python
# 改前：
device = torch.device("cuda")

# 改后：
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

---

## 5. `config.yaml`（新增字段）

在两个 `recognizer` 块中各新增两个字段，旧字段 `timm_model` 和 `lstm_hidden` 保留（factory 读取时会 fallback）：

```yaml
recognizer:
  timm_model: repvit_m1      # 保留（向后兼容）
  backbone: repvit_m1        # 新增：factory 优先读此字段
  gru_hidden: auto           # 新增："auto" 查表，或直接填整数
  lstm_hidden: 256           # 保留（向后兼容）
  # 其余字段不变
```

切换到 EfficientNet-B4 时，只需改两行：
```yaml
backbone: efficientnet_b4
gru_hidden: auto             # → 自动选 384
```

---

## 数据流（不变）

```
Video Batch [B, T, C, H, W]
    │
    ├─ Teacher (SwinTeacher, frozen)
    │    └─ logits + [feat1, feat2, feat3]
    │
    └─ Student (TimmGRUStudent, trainable)
         ├─ backbone [B*T, C, H, W] → [B*T, in_channels, 7, 7]
         │    └─ feature hooks → student_raw_feats
         ├─ GAP + flatten → [B, T, in_channels]
         ├─ GRU → [B, T, hidden_dim]
         └─ FC → logits [B, num_classes]

Loss = γ·L_gt + β·L_kd + α(t)·L_feat
```

---

## 不在此次 scope 内

- MoViNet / 3D 模型（待 pytorchvideo 兼容性确认）
- 教师模型架构替换
- 后端推理代码（`action_recognizer.py`）适配新 backbone

---

## 开放问题（实现前需确认）

1. `efficientnet_b4` 和 `convnext_nano` 的实际特征维度需用 dummy forward 在本机实测，表中数值为估算。
2. `timm.create_model` 对 `mobilenetv4_conv_medium` 的 `feature_info` 层数需验证（是否 ≥ 3 个 stage）。
3. `_build_student_hooks` 用 `backbone.get_submodule(info["module"])` 时，不同 backbone 的 `feature_info['module']` 路径格式需在实现时逐一验证。
