import importlib


class _FakeModel:
    def __init__(self):
        self.calls = []
    def __call__(self, frame, classes=None, verbose=False, max_det=None):
        self.calls.append({"classes": classes, "max_det": max_det})
        class _R:
            boxes = []
        return [_R()]
    def to(self, dev):
        return self


def test_default_config_has_expected_keys():
    m = importlib.import_module("utils.abnormal_crop_video")
    cfg = m.default_config()
    for k in ["input_base_dir", "output_base_dir", "class_names", "yolo_model_path",
              "target_classes", "max_det", "expand_ratio", "max_clip_duration",
              "max_clips_per_video"]:
        assert k in cfg


def test_get_clip_segments_accepts_cfg_override():
    m = importlib.import_module("utils.abnormal_crop_video")
    cfg = m.default_config()
    cfg["max_clip_duration"] = 5
    cfg["min_clip_duration"] = 2.0
    # 尾段 5.0-9.5 = 4.5s >= min(2.0) → 保留（与改造前语义一致）
    segments = m.get_clip_segments(9.5, cfg)
    assert segments == [(0.0, 5.0), (5.0, 9.5)]


def test_get_clip_segments_backward_compatible_without_cfg(monkeypatch):
    m = importlib.import_module("utils.abnormal_crop_video")
    monkeypatch.setattr(m, "MAX_CLIP_DURATION", 4)
    monkeypatch.setattr(m, "MIN_CLIP_DURATION", 2.0)
    assert m.get_clip_segments(9.5) == [(0.0, 4.0), (4.0, 8.0)]


def test_run_uses_config_on_empty_sorted(tmp_path):
    m = importlib.import_module("utils.abnormal_crop_video")
    src = tmp_path / "sorted-2026-07-15"
    (src / "eat").mkdir(parents=True)
    cfg = m.default_config()
    cfg["input_base_dir"] = str(src)
    cfg["output_base_dir"] = str(tmp_path / "out")
    cfg["class_names"] = ["eat", "drink", "other"]
    fake = _FakeModel()
    m.run(cfg, model_factory=lambda dev: fake)   # 空目录，不应报错
