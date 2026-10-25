from unittest.mock import patch

import cv2
import numpy as np
import pytest
import torch

from utils.online_video_dataset import OnlineVideoDataset


class FakeCapture:
    def __init__(self, path):
        self.path = path
        self.index = 0
        self.opened = True

    def isOpened(self):
        return self.opened

    def get(self, property_id):
        if property_id == cv2.CAP_PROP_FPS:
            return 10
        return 10 if "short" in self.path else 100

    def set(self, property_id, value):
        assert property_id == cv2.CAP_PROP_POS_FRAMES
        self.index = int(value)
        return True

    def read(self):
        index = self.index
        self.index += 1
        return True, np.full((8, 8, 3), index * 2, dtype=np.uint8)

    def release(self):
        pass


def video(tmp_path, split, label, name):
    folder = tmp_path / split / label
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    path.touch()
    return path


def decoded_indices(sample):
    clip = sample["pixel_values"][:, :, 0, 0]
    return ((clip[:, 0] * .229 + .485) * 255 / 2).round().int().tolist()


def test_train_picks_new_windows_from_whole_video_and_val_is_fixed(tmp_path):
    video(tmp_path, "train", "eat", "long.avi")
    video(tmp_path, "val", "drink", "validation.avi")
    with patch("utils.online_video_dataset.cv2.VideoCapture", side_effect=FakeCapture), \
         patch("utils.online_video_dataset.torch.randint", side_effect=[torch.tensor(0), torch.tensor(20)]):
        train = OnlineVideoDataset(tmp_path, "train", ["eat", "drink"], window_seconds=2, temporal_window=4, imgsz=8)
        val = OnlineVideoDataset(tmp_path, "val", ["eat", "drink"], window_seconds=2, temporal_window=4, imgsz=8)
        first, second = train[0], train[0]
        fixed_a, fixed_b = val[0], val[0]

    assert decoded_indices(first) == [0, 6, 12, 19]
    assert decoded_indices(second) == [20, 26, 32, 39]
    assert decoded_indices(fixed_a) == decoded_indices(fixed_b) == [40, 46, 52, 59]
    assert first["pixel_values"].shape == (4, 3, 8, 8)
    assert first["labels"].item() == 0
    assert fixed_a["labels"].item() == 1


def test_rejects_video_shorter_than_window(tmp_path):
    video(tmp_path, "train", "eat", "short.avi")
    with patch("utils.online_video_dataset.cv2.VideoCapture", side_effect=FakeCapture):
        with pytest.raises(ValueError, match="video too short"):
            OnlineVideoDataset(tmp_path, "train", ["eat"], window_seconds=2)


def test_rejects_unknown_class_and_missing_split(tmp_path):
    video(tmp_path, "train", "other", "video.avi")
    with pytest.raises(ValueError, match="unexpected class"):
        OnlineVideoDataset(tmp_path, "train", ["eat"])
    with pytest.raises(FileNotFoundError, match="split directory"):
        OnlineVideoDataset(tmp_path, "val", ["eat"])


def test_rejects_empty_split(tmp_path):
    (tmp_path / "train").mkdir()
    with pytest.raises(ValueError, match="no train videos"):
        OnlineVideoDataset(tmp_path, "train", ["eat"])
