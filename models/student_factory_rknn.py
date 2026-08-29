"""Factory for the independent RKNN-friendly student training path."""

from models.student_factory import BACKBONE_GRU_HIDDEN, MOVINET_VARIANTS
from models.timm_gru_student_rknn import TimmGRUStudentRKNN


def build_student_rknn(recognizer_cfg, num_classes):
    backbone = recognizer_cfg.get("backbone") or recognizer_cfg.get("timm_model", "repvit_m1")
    if backbone in MOVINET_VARIANTS:
        raise ValueError("RKNN training currently supports the 2D timm + GRU student only")
    hidden = recognizer_cfg.get("gru_hidden", "auto")
    if hidden == "auto" or not hidden:
        hidden = BACKBONE_GRU_HIDDEN.get(backbone, 256)
    return TimmGRUStudentRKNN(
        backbone, num_classes, int(hidden), pretrained=True, dropout=recognizer_cfg.get("dropout", 0.2)
    )
