import importlib


def test_frames_default_config_has_label_map():
    m = importlib.import_module("utils.abnormal_crop")
    cfg = m.default_config()
    assert "label_map" in cfg
    assert "output_base_dir" in cfg
    assert "target_classes" in cfg
    assert "max_clips_per_video" in cfg


def test_frames_run_on_empty_sorted_is_noop(tmp_path):
    m = importlib.import_module("utils.abnormal_crop")
    src = tmp_path / "sorted-2026-07-15"
    (src / "eat").mkdir(parents=True)
    cfg = m.default_config()
    cfg["input_base_dir"] = str(src)
    cfg["output_base_dir"] = str(tmp_path / "out")
    cfg["class_names"] = ["eat", "drink", "active", "resting"]
    m.run(cfg, model_factory=lambda dev: object())


def test_frames_run_drives_video_calls_process_and_rebalance(tmp_path, monkeypatch):
    m = importlib.import_module("utils.abnormal_crop")
    src = tmp_path / "sorted-2026-07-15"
    (src / "eat").mkdir(parents=True)
    (src / "eat" / "v.mp4").write_text("x", encoding="utf-8")
    calls = {"process_video": 0, "rebalance": 0}
    def fake_process_video(video_path, class_name, class_counters, model, parent_dir_name, cfg):
        calls["process_video"] += 1
        return {"saved_folders": [f"{parent_dir_name}-{class_name}-v"], "failed_paths": []}
    def fake_rebalance(video_records, cfg):
        calls["rebalance"] += 1
        assert "eat" in video_records
    monkeypatch.setattr(m, "process_video", fake_process_video)
    monkeypatch.setattr(m, "rebalance_val", fake_rebalance)
    cfg = m.default_config()
    cfg["input_base_dir"] = str(src)
    cfg["output_base_dir"] = str(tmp_path / "out")
    cfg["class_names"] = ["eat", "drink", "active", "resting"]
    m.run(cfg, model_factory=lambda dev: object())
    assert calls["process_video"] == 1
    assert calls["rebalance"] == 1
