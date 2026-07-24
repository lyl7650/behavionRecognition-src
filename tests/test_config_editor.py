import os
import yaml
import textwrap
import pytest
from scripts.pipeline import config_editor as ce
from scripts.pipeline import pipeline_config as pc


def test_clean_path_strips_raw_prefix():
    assert ce.clean_path('r"E:\\ActionRecognition\\x\\videos"') == r"E:\ActionRecognition\x\videos"


def test_clean_path_passes_through_plain():
    assert ce.clean_path(r"dataset\rabbit\2026-07-15\videos") == r"dataset\rabbit\2026-07-15\videos"


def test_yaml_safe_path_uses_forward_slashes():
    assert ce.yaml_safe_path(r"E:\new\videos") == "E:/new/videos"


def test_yaml_safe_path_roundtrips_through_yaml_and_sanitize():
    from training.config import sanitize_path
    safe = ce.yaml_safe_path(r"E:\new\videos")
    dumped = yaml.safe_dump({"p": safe})
    loaded = yaml.safe_load(dumped)["p"]
    assert sanitize_path(loaded) == os.path.normpath(r"E:\new\videos")


def _write_sample_config(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(textwrap.dedent('''
    task_configs:
      normal:
        dataset_path: r"E:\\old\\pet\\videos"
        class_names: ['eat','drink','other']
        finetune:
          loss_type: "ce"
          output_dir: "outputs/pet/{date}/teacher_finetuned_{date}"
          output_model_name: "pet_teacher_{date}_ema-videos.pth"
        distillation:
          output_dir: "outputs/pet/{date}"
    distillation:
      teacher_checkpoint: r"outputs\\pet\\old.pth"
    ''').lstrip(), encoding="utf-8")
    return cfg


def test_apply_switches_species_and_dataset(tmp_path):
    cfg = _write_sample_config(tmp_path)
    ce.apply_species_config(str(cfg), "rabbit", r"E:\new\rabbit\2026-07-15\videos",
                            now_fn=lambda: "20260715-2130")
    data = yaml.safe_load(cfg.read_text(encoding="utf-8"))
    assert data["task_configs"]["normal"]["class_names"] == ["resting", "active", "eating", "drinking"]
    assert "rabbit" in data["task_configs"]["normal"]["finetune"]["output_dir"]
    assert "rabbit" in data["task_configs"]["normal"]["finetune"]["output_model_name"]
    assert "rabbit" in data["task_configs"]["normal"]["distillation"]["output_dir"]
    assert "pet" not in data["task_configs"]["normal"]["finetune"]["output_dir"]
    assert data["task_configs"]["normal"]["finetune"]["loss_type"] == "ce"


def test_apply_writes_timestamped_backup(tmp_path):
    cfg = _write_sample_config(tmp_path)
    ce.apply_species_config(str(cfg), "rabbit", r"E:\new\rabbit\2026-07-15\videos",
                            now_fn=lambda: "20260715-2130")
    assert (tmp_path / "config" / "backups" / "config.yaml.20260715-2130.bak").is_file()


def test_apply_dataset_path_yaml_safe_and_parsable(tmp_path):
    cfg = _write_sample_config(tmp_path)
    ce.apply_species_config(str(cfg), "rabbit", r"E:\new\rabbit\2026-07-15\videos",
                            now_fn=lambda: "20260715-2130")
    from training.config import sanitize_path
    data = yaml.safe_load(cfg.read_text(encoding="utf-8"))
    got = sanitize_path(data["task_configs"]["normal"]["dataset_path"])
    assert got == os.path.normpath(r"E:\new\rabbit\2026-07-15\videos")


def test_apply_raises_if_token_missing(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(textwrap.dedent('''
    task_configs:
      normal:
        dataset_path: "x"
        class_names: ['eat']
        finetune:
          output_dir: "outputs/UNKNOWN/{date}"
          output_model_name: "UNKNOWN_{date}.pth"
        distillation:
          output_dir: "outputs/UNKNOWN/{date}"
    distillation:
      teacher_checkpoint: "y"
    ''').lstrip(), encoding="utf-8")
    with pytest.raises(ValueError):
        ce.apply_species_config(str(cfg), "rabbit", r"E:\a\b",
                                now_fn=lambda: "20260715-2130")


def test_set_teacher_checkpoint_yaml_safe(tmp_path):
    cfg = _write_sample_config(tmp_path)
    ce.set_teacher_checkpoint(str(cfg), r"outputs\rabbit\0715\t_ema.pth")
    data = yaml.safe_load(cfg.read_text(encoding="utf-8"))
    assert data["distillation"]["teacher_checkpoint"] == "outputs/rabbit/0715/t_ema.pth"
