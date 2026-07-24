from models.timm_gru_student import TimmGRUStudent

# GRU hidden size 查表（gru_hidden="auto" 时生效）
# 基于本机 features_only=True, out_indices=[-1] dummy forward 实测
BACKBONE_GRU_HIDDEN = {
    "repvit_m1": 256,               # 实测特征维度 384
    "repvit_m1_1": 256,             # 实测特征维度 512
    "repvit_m1_5": 256,             # 实测特征维度 512
    "repvit_m2_3": 384,             # 实测特征维度 640
    "mobilenetv4_conv_medium": 512,  # 实测特征维度 960
    "efficientnet_b2": 256,          # 实测特征维度 352
    "efficientnet_b4": 384,          # 实测特征维度 ~448
    "convnext_pico": 256,            # 实测特征维度 512
    "convnext_nano": 384,            # 实测特征维度 ~640
}

MOVINET_VARIANTS = {"movinet_a2", "movinet_a3", "movinet_a4", "movinet_a5"}


def build_student(recognizer_cfg: dict, num_classes: int):
    """
    根据 recognizer_cfg 构建学生模型。
    'backbone' 字段优先于旧 'timm_model' 字段（向后兼容）。
    backbone 为 movinet_a2/a3 时返回 MoViNetStudent，否则返回 TimmGRUStudent。
    """
    backbone = recognizer_cfg.get("backbone") or recognizer_cfg.get("timm_model", "repvit_m1")

    if backbone in MOVINET_VARIANTS:
        from models.movinet_student import MoViNetStudent
        return MoViNetStudent(backbone, num_classes, pretrained=True)

    hidden = recognizer_cfg.get("gru_hidden", "auto")
    if hidden == "auto" or not hidden:
        hidden = BACKBONE_GRU_HIDDEN.get(backbone, 256)
    dropout = recognizer_cfg.get("dropout", 0.2)
    return TimmGRUStudent(backbone, num_classes, int(hidden), pretrained=True, dropout=dropout)
