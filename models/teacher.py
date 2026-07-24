import os

import torch
import torch.nn as nn
from torchvision.models.video import Swin3D_B_Weights, swin3d_b


class SwinTeacher(nn.Module):
    def __init__(self, checkpoint_path=None, num_classes=400):
        super().__init__()
        print(f"Initializing Teacher: Video Swin-B (via Torchvision) -> Classes: {num_classes}")
        print("Loading/downloading Torchvision K400 weights...")
        try:
            weights = Swin3D_B_Weights.KINETICS400_V1
            self.model = swin3d_b(weights=weights)
        except Exception as e:
            print(f"Weight download failed, using random init: {e}")
            self.model = swin3d_b(weights=None)

        if num_classes != 400:
            old_head = self.model.head
            if isinstance(old_head, nn.Sequential):
                in_features = old_head[-1].in_features
                dropout_p = old_head[0].p
            else:
                in_features = old_head.in_features
                dropout_p = 0.5

            print(f"Replacing classifier head: {in_features} -> {num_classes}")
            self.model.head = nn.Sequential(
                nn.Dropout(p=dropout_p),
                nn.Linear(in_features, num_classes),
            )

        if checkpoint_path and isinstance(checkpoint_path, str) and checkpoint_path.endswith(".pth"):
            print(f"Loading finetuned checkpoint: {checkpoint_path}")
            if not os.path.exists(checkpoint_path):
                print(f"Warning: checkpoint not found {checkpoint_path}, skipping load")
            else:
                try:
                    state_dict = torch.load(checkpoint_path, map_location="cpu")
                    new_state_dict = {}
                    for key, value in state_dict.items():
                        new_key = key.replace("base.model.", "").replace("module.", "")
                        new_state_dict[new_key] = value

                    missing, unexpected = self.model.load_state_dict(new_state_dict, strict=False)
                    print(f"Checkpoint loaded (Missing: {len(missing)}, Unexpected: {len(unexpected)})")
                except Exception as e:
                    print(f"Checkpoint load error: {e}")

        self.model.eval()
        self.features = []

        def hook_fn(module, inputs, output):
            if output.dim() == 5:
                output = output.permute(0, 4, 1, 2, 3)
            self.features.append(output)

        for idx in [1, 2, 3, 4]:
            if idx < len(self.model.features):
                self.model.features[idx].register_forward_hook(hook_fn)

    def forward(self, x):
        self.features = []
        x = x.permute(0, 2, 1, 3, 4)
        logits = self.model(x)
        return logits, self.features
