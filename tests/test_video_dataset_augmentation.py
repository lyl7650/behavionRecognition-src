import os
import tempfile
import unittest

import numpy as np
import torch
from PIL import Image

from utils.video_dataset import VideoDataset


class VideoDatasetAugmentationTests(unittest.TestCase):
    def test_train_dataset_applies_horizontal_flip_to_whole_clip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sample_dir = os.path.join(tmpdir, "train", "eat_001")
            os.makedirs(sample_dir)
            with open(os.path.join(sample_dir, "label.txt"), "w", encoding="utf-8") as handle:
                handle.write("0")

            for idx in range(4):
                frame = np.zeros((8, 8, 3), dtype=np.uint8)
                frame[:, :4, 0] = 255
                frame[:, 4:, 2] = 255
                Image.fromarray(frame).save(os.path.join(sample_dir, f"frame_{idx:04d}.jpg"))

            dataset = VideoDataset(
                tmpdir,
                split="train",
                cfg={
                    "class_names": ["eat"],
                    "imgsz": 8,
                    "temporal_window": 4,
                    "augment": {"fliplr": 1.0},
                },
                is_train=True,
                default_class_names=["eat"],
            )

            clip = dataset[0]["pixel_values"]

        red_left = clip[:, 0, :, :4].mean().item()
        red_right = clip[:, 0, :, 4:].mean().item()
        blue_left = clip[:, 2, :, :4].mean().item()
        blue_right = clip[:, 2, :, 4:].mean().item()

        self.assertLess(red_left, red_right)
        self.assertGreater(blue_left, blue_right)

    def test_val_dataset_ignores_random_augmentation_settings(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sample_dir = os.path.join(tmpdir, "val", "eat_001")
            os.makedirs(sample_dir)
            with open(os.path.join(sample_dir, "label.txt"), "w", encoding="utf-8") as handle:
                handle.write("0")

            frame = np.zeros((8, 8, 3), dtype=np.uint8)
            frame[:, :4, 0] = 255
            frame[:, 4:, 2] = 255
            Image.fromarray(frame).save(os.path.join(sample_dir, "frame_0000.jpg"))

            dataset = VideoDataset(
                tmpdir,
                split="val",
                cfg={
                    "class_names": ["eat"],
                    "imgsz": 8,
                    "temporal_window": 1,
                    "augment": {"fliplr": 1.0},
                },
                is_train=False,
                default_class_names=["eat"],
            )

            clip = dataset[0]["pixel_values"]

        red_left = clip[:, 0, :, :4].mean().item()
        red_right = clip[:, 0, :, 4:].mean().item()
        self.assertGreater(red_left, red_right)

    def test_train_transform_includes_configured_scale_rotate_and_translate(self):
        dataset = VideoDataset(
            tempfile.mkdtemp(),
            split="train",
            cfg={
                "class_names": ["eat"],
                "imgsz": 8,
                "temporal_window": 4,
                "augment": {
                    "scale_min": 0.8,
                    "scale_max": 1.0,
                    "rotate": 10,
                    "translate": 0.2,
                },
            },
            is_train=True,
            default_class_names=["eat"],
        )

        names = [type(transform).__name__ for transform in dataset.transform.transforms]

        self.assertIn("RandomResizedCrop", names)
        self.assertIn("RandomAffine", names)

    def test_error_fallback_uses_normalized_clip_scale(self):
        dataset = VideoDataset(
            tempfile.mkdtemp(),
            split="train",
            cfg={"class_names": ["eat"], "imgsz": 8, "temporal_window": 4},
            is_train=True,
            default_class_names=["eat"],
        )
        dataset.samples = [("missing", 0)]

        clip = dataset[0]["pixel_values"]
        expected = torch.tensor(
            [
                -0.485 / 0.229,
                -0.456 / 0.224,
                -0.406 / 0.225,
            ],
            dtype=torch.float32,
        )

        self.assertEqual(tuple(clip.shape), (4, 3, 8, 8))
        torch.testing.assert_close(clip[:, :, 0, 0], expected.repeat(4, 1))

    def test_train_dataset_can_reverse_clip_order(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sample_dir = os.path.join(tmpdir, "train", "eat_001")
            os.makedirs(sample_dir)
            with open(os.path.join(sample_dir, "label.txt"), "w", encoding="utf-8") as handle:
                handle.write("0")

            for idx, value in enumerate([0, 80, 160, 240]):
                frame = np.full((8, 8, 3), value, dtype=np.uint8)
                Image.fromarray(frame).save(os.path.join(sample_dir, f"frame_{idx:04d}.png"))

            dataset = VideoDataset(
                tmpdir,
                split="train",
                cfg={
                    "class_names": ["eat"],
                    "imgsz": 8,
                    "temporal_window": 4,
                    "augment": {"reverse": 1.0},
                },
                is_train=True,
                default_class_names=["eat"],
            )

            clip = dataset[0]["pixel_values"]

        frame_means = clip.mean(dim=(1, 2, 3))
        self.assertTrue(torch.all(frame_means[:-1] > frame_means[1:]))

    def test_val_dataset_ignores_reverse_augmentation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sample_dir = os.path.join(tmpdir, "val", "eat_001")
            os.makedirs(sample_dir)
            with open(os.path.join(sample_dir, "label.txt"), "w", encoding="utf-8") as handle:
                handle.write("0")

            for idx, value in enumerate([0, 80, 160, 240]):
                frame = np.full((8, 8, 3), value, dtype=np.uint8)
                Image.fromarray(frame).save(os.path.join(sample_dir, f"frame_{idx:04d}.png"))

            dataset = VideoDataset(
                tmpdir,
                split="val",
                cfg={
                    "class_names": ["eat"],
                    "imgsz": 8,
                    "temporal_window": 4,
                    "augment": {"reverse": 1.0},
                },
                is_train=False,
                default_class_names=["eat"],
            )

            clip = dataset[0]["pixel_values"]

        frame_means = clip.mean(dim=(1, 2, 3))
        self.assertTrue(torch.all(frame_means[:-1] < frame_means[1:]))

    def test_clip_with_variable_size_frames_loads_before_clip_augmentation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sample_dir = os.path.join(tmpdir, "train", "eat_001")
            os.makedirs(sample_dir)
            with open(os.path.join(sample_dir, "label.txt"), "w", encoding="utf-8") as handle:
                handle.write("0")

            for idx, shape in enumerate([(8, 10), (9, 11), (10, 12), (11, 13)]):
                height, width = shape
                frame = np.full((height, width, 3), idx * 60, dtype=np.uint8)
                Image.fromarray(frame).save(os.path.join(sample_dir, f"frame_{idx:04d}.png"))

            dataset = VideoDataset(
                tmpdir,
                split="train",
                cfg={
                    "class_names": ["eat"],
                    "imgsz": 8,
                    "temporal_window": 4,
                    "augment": {"fliplr": 0.0},
                },
                is_train=True,
                default_class_names=["eat"],
            )

            clip = dataset[0]["pixel_values"]

        self.assertEqual(tuple(clip.shape), (4, 3, 8, 8))
        frame_means = clip.mean(dim=(1, 2, 3))
        self.assertTrue(torch.all(frame_means[:-1] < frame_means[1:]))


if __name__ == "__main__":
    unittest.main()
