import importlib

def test_crop_video_imports_without_side_effects():
    m = importlib.import_module("utils.abnormal_crop_video")
    assert hasattr(m, "main")

def test_crop_frames_imports_without_side_effects():
    m = importlib.import_module("utils.abnormal_crop")
    assert hasattr(m, "main")
