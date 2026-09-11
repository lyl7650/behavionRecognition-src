import os
from types import SimpleNamespace
from unittest import mock

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


def test_merge_and_report_no_old_snapshot_returns_none(monkeypatch, capsys):
    monkeypatch.setattr(rtd.sm, "find_latest_old_snapshot",
                        lambda *_a, **_k: ("", []))
    report = rtd.merge_and_report("root", "2026-09-10", "snap", False, ["eat"],
                                  assume_yes=True)
    assert report is None
    assert "No old snapshot found" in capsys.readouterr().out


def test_merge_and_report_unattended_auto_selects_ambiguous(monkeypatch, capsys):
    monkeypatch.setattr(rtd.sm, "find_latest_old_snapshot",
                        lambda *_a, **_k: ("2026-08-01", ["x", "y"]))
    monkeypatch.setattr(rtd.sm, "resolve_old_layer",
                        lambda *_a, **_k: ("old_layer", False))
    merge = mock.Mock(return_value={"copied": 1, "skipped": 0, "filtered": 0})
    monkeypatch.setattr(rtd.sm, "merge_snapshot", merge)
    health = mock.Mock(return_value={
        "per_class": {"eat": 5}, "cross_split_leak": [], "source_leak": [],
        "unrecognized": [], "label_mismatch": []})
    monkeypatch.setattr(rtd.sm, "health_report", health)
    report = rtd.merge_and_report("root", "2026-09-10", "snap", False, ["eat"],
                                  assume_yes=True)
    merge.assert_called_once()
    assert report["per_class"] == {"eat": 5}
    out = capsys.readouterr().out
    assert "Unattended mode" in out


def test_merge_and_report_interactive_aborts_on_refusal(monkeypatch):
    monkeypatch.setattr(rtd.sm, "find_latest_old_snapshot",
                        lambda *_a, **_k: ("2026-08-01", ["x"]))
    with mock.patch("builtins.input", return_value="no"):
        try:
            rtd.merge_and_report("root", "2026-09-10", "snap", False, ["eat"],
                                 assume_yes=False)
            aborted = False
        except SystemExit:
            aborted = True
    assert aborted


def test_train_and_distill_ordering_and_teacher_backfill(monkeypatch, tmp_path):
    teacher_file = tmp_path / "teacher_ema-videos.pth"
    teacher_file.write_bytes(b"x")
    cfg = {"output_dir": str(tmp_path), "output_model_name": teacher_file.name}

    order = []
    monkeypatch.setattr(
        rtd.ce, "apply_species_config",
        lambda *a, **k: order.append("apply_species_config"))
    set_teacher = mock.Mock(
        side_effect=lambda *_a: order.append("set_teacher_checkpoint"))
    monkeypatch.setattr(rtd.ce, "set_teacher_checkpoint", set_teacher)
    fake_training_config = SimpleNamespace(
        build_train_config=lambda mode: order.append("build_train_config") or cfg)
    fake_teacher = SimpleNamespace(train=lambda **k: order.append("teacher_train"))
    fake_distill = SimpleNamespace(train_distill=lambda: order.append("distill"))

    monkeypatch.setitem(__import__("sys").modules, "training.config",
                        fake_training_config)
    monkeypatch.setitem(__import__("sys").modules, "train.train_finetune_normal_3",
                        fake_teacher)
    monkeypatch.setitem(__import__("sys").modules, "distill.distill_normal_3",
                        fake_distill)

    got = rtd.train_and_distill("pet", str(tmp_path / "snap"), frames=False)
    assert got == str(teacher_file)
    set_teacher.assert_called_once_with("config.yaml", str(teacher_file))
    assert order == ["apply_species_config", "build_train_config",
                     "teacher_train", "set_teacher_checkpoint", "distill"]


def test_train_and_distill_missing_teacher_aborts(monkeypatch, tmp_path):
    cfg = {"output_dir": str(tmp_path), "output_model_name": "missing.pth"}
    monkeypatch.setattr(rtd.ce, "apply_species_config", lambda *a, **k: None)
    monkeypatch.setattr(
        rtd.ce, "set_teacher_checkpoint", lambda *a, **k: None)
    fake_training_config = SimpleNamespace(build_train_config=lambda mode: cfg)
    fake_teacher = SimpleNamespace(train=lambda **k: None)
    fake_distill = SimpleNamespace(train_distill=lambda: None)
    monkeypatch.setitem(__import__("sys").modules, "training.config",
                        fake_training_config)
    monkeypatch.setitem(__import__("sys").modules, "train.train_finetune_normal_3",
                        fake_teacher)
    monkeypatch.setitem(__import__("sys").modules, "distill.distill_normal_3",
                        fake_distill)

    import pytest
    with pytest.raises(FileNotFoundError):
        rtd.train_and_distill("pet", str(tmp_path / "snap"))
