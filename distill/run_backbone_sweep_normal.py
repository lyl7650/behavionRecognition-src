import os
import sys
from datetime import datetime

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from training.config import build_distill_config
from training.distillation import run_distillation_training
from training.distillation_losses import build_priority_composite_distill_loss

BACKBONES = ["repvit_m1",
             #"repvit_m1_5", "repvit_m1_1", 
             ]

# repvit_m1 用默认 kd=8.0，更大的 backbone 初始化偏差更大，降低 KD 权重避免锁死
BACKBONE_KD_OVERRIDES = {
    "repvit_m1": 6.0,
    "repvit_m1_5": 6.0,
    "repvit_m1_1": 6.0,
    "repvit_m2_3": 6.0,
}

def _resolve_anchor_idx(class_names):
    try:
        return class_names.index("resting")
    except ValueError as exc:
        raise ValueError("normal distillation requires a 'resting' class") from exc


def main():
    base_cfg = build_distill_config("normal")
    date_str = datetime.now().strftime("%m-%d")
    base_output_dir = os.path.join(
        os.path.dirname(base_cfg["output_dir"]), date_str
    )

    for backbone in BACKBONES:
        recognizer_cfg = dict(base_cfg["recognizer"])
        recognizer_cfg["backbone"] = backbone
        recognizer_cfg["timm_model"] = backbone

        kd_weight = BACKBONE_KD_OVERRIDES.get(backbone, float(base_cfg["loss_weights"]["kd"]))

        runtime_cfg = {
            "dataset_path": base_cfg["dataset_path"],
            "class_names": list(base_cfg["class_names"]),
            "recognizer": recognizer_cfg,
            "teacher_checkpoint": base_cfg["teacher_checkpoint"],
            "temperature": float(base_cfg["temperature"]),
            "loss_weights": {
                "gt": float(base_cfg["loss_weights"]["gt"]),
                "kd": kd_weight,
                "feature": float(base_cfg["loss_weights"]["feature"]),
            },
            "learning_rate": float(base_cfg["learning_rate"]),
            "batch_size": int(base_cfg["batch_size"]),
            "epochs": int(base_cfg["epochs"]),
            "num_workers": int(base_cfg["num_workers"]),
            "output_dir": os.path.join(base_output_dir, backbone),
        }

        criterion = build_priority_composite_distill_loss(
            anchor_idx=_resolve_anchor_idx(runtime_cfg["class_names"]),
            alpha=runtime_cfg["loss_weights"]["feature"],
            beta=runtime_cfg["loss_weights"]["kd"],
            gamma=runtime_cfg["loss_weights"]["gt"],
            temperature=runtime_cfg["temperature"],
        )

        print(f"\n{'=' * 60}")
        print(f"Backbone: {backbone}  |  kd_weight={kd_weight}")
        print(f"Output:   {runtime_cfg['output_dir']}")
        print(f"{'=' * 60}\n")

        run_distillation_training(
            runtime_cfg=runtime_cfg,
            criterion=criterion,
            best_model_name="repvit_best_distilled.pth",
            best_ema_model_name="repvit_best_distilled_ema.pth",
        )


if __name__ == "__main__":
    main()
