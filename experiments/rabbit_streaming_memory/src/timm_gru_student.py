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

    def encode_frames(self, x):
        """Encode frames [N, C, H, W] into pooled features [N, D]."""
        features = self.backbone(x)[0]
        x = self.gap(features)
        return self.flatten(x)

    def forward_stream(self, x, hidden_state=None):
        """Stream a clip [B, T, C, H, W], returning per-step logits and state."""
        b, t, c, h, w = x.shape
        features = self.encode_frames(x.reshape(b * t, c, h, w))
        features = features.reshape(b, t, -1)
        out, next_hidden = self.gru(features, hidden_state)
        return self.fc(out), next_hidden

    def forward(self, x, hidden_state=None):
        """Single-frame inference: x [B, C, H, W] -> (logits, next_hidden)."""
        x = self.encode_frames(x)
        x = x.unsqueeze(1)
        out, next_hidden = self.gru(x, hidden_state)
        logits = self.fc(out[:, -1, :])
        return logits, next_hidden

    def forward_seq(self, x):
        """Clip 训练: x [B, T, C, H, W] → logits [B, num_classes]"""
        logits, _ = self.forward_stream(x)
        return logits[:, -1, :]
