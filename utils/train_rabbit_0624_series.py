"""
Distill 4 rabbit student models on 2026-06-24 dataset:
repvit_m2_3, repvit_m1_5, repvit_m1_1, repvit_m1_0

Using pre-trained teacher: outputs/rabbit/2026-07-06/teacher_finetuned_3cls_0706/rabbit_teacher_0706_ema-videos.pth
"""
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from training.config import build_distill_config
from distill.distill_normal_3 import train_distill


DATASET_PATH = "dataset/rabbit/2026-06-24"
DATE_TAG = "2026-06-24"
TEACHER_CHECKPOINT = "outputs/rabbit/2026-07-06/teacher_finetuned_3cls_0706/rabbit_teacher_0706_ema-videos.pth"
BACKBONES = ["repvit_m2_3", "repvit_m1_5", "repvit_m1_1", "repvit_m1_0"]


def train_student(backbone_name: str, step: int, teacher_checkpoint: str):
    """Train a single student model"""
    print("\n" + "="*80)
    print(f"[{step}/4] Distilling {backbone_name} on {DATASET_PATH}")
    print("="*80 + "\n")

    distill_cfg = build_distill_config("normal")
    distill_cfg["dataset_path"] = DATASET_PATH
    distill_cfg["teacher_checkpoint"] = teacher_checkpoint
    distill_cfg["recognizer"]["backbone"] = backbone_name
    distill_cfg["recognizer"]["timm_model"] = backbone_name
    distill_cfg["output_dir"] = f"outputs/rabbit/{DATE_TAG}/{backbone_name}"
    distill_cfg["output_model_name"] = f"{backbone_name}_best_{DATE_TAG}.pth"
    distill_cfg["output_ema_model_name"] = f"{backbone_name}_best_{DATE_TAG}_ema.pth"

    # Adjust batch size for larger models
    if backbone_name in ["repvit_m2_3", "repvit_m1_5"]:
        distill_cfg["batch_size"] = 4
        print(f"Using batch_size=4 for {backbone_name}")

    student_path = train_distill(config_override=distill_cfg)
    print(f"\n[{step}/4] Student trained: {student_path}\n")
    return student_path


def main():
    print("\n" + "="*80)
    print("Rabbit Model Training Series - 2026-06-24 Dataset")
    print("="*80)
    print(f"Dataset: {DATASET_PATH}")
    print(f"Teacher: {TEACHER_CHECKPOINT}")
    print(f"Backbones to train: {', '.join(BACKBONES)}")
    print("="*80 + "\n")

    # Verify teacher checkpoint exists
    if not Path(TEACHER_CHECKPOINT).exists():
        print(f"ERROR: Teacher checkpoint not found: {TEACHER_CHECKPOINT}")
        sys.exit(1)

    # Train each student
    results = {}
    for idx, backbone in enumerate(BACKBONES, start=1):
        student_path = train_student(backbone, idx, TEACHER_CHECKPOINT)
        results[backbone] = student_path

    # Summary
    print("\n" + "="*80)
    print("All training completed!")
    print("="*80)
    print(f"Teacher: {TEACHER_CHECKPOINT}")
    for backbone, path in results.items():
        print(f"{backbone}: {path}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
