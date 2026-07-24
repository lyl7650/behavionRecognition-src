import os
from scripts.pipeline import run_train_distill as rtd


def test_teacher_output_path_is_output_dir_plus_model_name():
    train_cfg = {
        "output_dir": os.path.normpath("outputs/rabbit/0715/teacher_finetuned_3cls_0715"),
        "output_model_name": "rabbit_teacher_0715_ema-videos.pth",
    }
    got = rtd.teacher_output_path(train_cfg)
    assert got == os.path.join(
        os.path.normpath("outputs/rabbit/0715/teacher_finetuned_3cls_0715"),
        "rabbit_teacher_0715_ema-videos.pth",
    )
    assert got.count("_ema") == 1
