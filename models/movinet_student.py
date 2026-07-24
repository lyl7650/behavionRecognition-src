import torch
import torch.nn as nn
from movinets import MoViNet
from movinets.config import _C

_VARIANTS = {
    "movinet_a2": _C.MODEL.MoViNetA2,
    "movinet_a3": _C.MODEL.MoViNetA3,
    "movinet_a4": _C.MODEL.MoViNetA4,
    "movinet_a5": _C.MODEL.MoViNetA5,
}


class MoViNetStudent(nn.Module):
    """
    MoViNet (A2/A3) 学生模型封装。
    原生 3D 模型，直接吃整段 clip，不使用 GRU state。
    部署时可切换 causal=True 做流式逐帧推理。

    is_3d = True：蒸馏训练时跳过 feature loss，只用 GT + KD logit loss。
    """

    is_3d = True

    def __init__(self, model_name: str, num_classes: int, pretrained: bool = True):
        super().__init__()
        if model_name not in _VARIANTS:
            raise ValueError(f"不支持的 MoViNet 变体: {model_name}，可选: {list(_VARIANTS)}")
        print(f"Initializing {model_name} (pretrained={pretrained})...")
        self.model = MoViNet(_VARIANTS[model_name], causal=False, pretrained=pretrained)

        # 替换分类头：最后一个 ConvBlock3D 里的 Conv3d
        last_block = self.model.classifier[-1]
        old_conv = last_block.conv_1.conv3d
        last_block.conv_1.conv3d = nn.Conv3d(
            old_conv.in_channels, num_classes,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )
        print(f"Classifier head replaced: {old_conv.in_channels} -> {num_classes} classes")

    def forward_seq(self, x: torch.Tensor) -> torch.Tensor:
        """
        训练接口：x [B, T, C, H, W] -> logits [B, num_classes]
        MoViNet 约定输入为 [B, C, T, H, W]，需 permute。
        """
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        return self.model(x)

    def forward(self, x: torch.Tensor, hidden_state=None):
        """
        推理接口（兼容 TimmGRUStudent 签名）。
        MoViNet 无 GRU state，hidden_state 参数忽略，返回 None 占位。
        x: [B, C, H, W]（单帧）或 [B, T, C, H, W]（clip）
        """
        if x.dim() == 4:
            # 单帧扩展为 T=1
            x = x.unsqueeze(1)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        logits = self.model(x)
        return logits, None
