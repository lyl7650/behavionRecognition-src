"""RKNN-oriented RepViT + GRU student; the original student remains unchanged."""

import timm
import torch
import torch.nn as nn


class TimmGRUStudentRKNN(nn.Module):
    def __init__(self, backbone_name, num_classes, hidden_dim=256, pretrained=True, dropout=0.2):
        super().__init__()
        print(f"Initializing RKNN {backbone_name} Backbone (Pretrained={pretrained})...")
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, features_only=True, out_indices=[-1])
        with torch.no_grad():
            in_channels = self.backbone(torch.randn(1, 3, 224, 224))[0].shape[1]
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.feat_norm = nn.LayerNorm(in_channels)
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

    @torch.no_grad()
    def reparameterize(self):
        """Irreversibly fold the timm backbone's Conv+BN branches for deployment."""
        from timm.utils.model import reparameterize_model

        self.backbone = reparameterize_model(self.backbone, inplace=True)
        return self

    def forward(self, x, hidden_state=None):
        x = self.flatten(self.gap(self.backbone(x)[0]))
        x = self.feat_norm(x).unsqueeze(1)
        out, next_hidden = self.gru(x, hidden_state)
        return self.fc(out[:, -1, :]), next_hidden

    def forward_seq(self, x):
        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w)
        x = self.feat_norm(self.flatten(self.gap(self.backbone(x)[0])))
        out, _ = self.gru(x.view(b, t, -1))
        return self.fc(out[:, -1, :])
