from models.timm_gru_student import TimmGRUStudent


class RepViT_GRU(TimmGRUStudent):
    def __init__(self, num_classes=5, hidden_dim=256, pretrained=True):
        super().__init__("repvit_m1", num_classes, hidden_dim, pretrained=pretrained)
