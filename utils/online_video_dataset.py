"""Sample windows from already-cropped, single-label long videos.

Layout: root/train/<class>/*.mp4 and root/val/<class>/*.mp4.
Training accesses choose a new window; validation uses the center window.
"""

import math
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".webm"}


class OnlineVideoDataset(Dataset):
    def __init__(self, root, split, class_names, window_seconds=3.0,
                 temporal_window=16, imgsz=224):
        if split not in ("train", "val"):
            raise ValueError(f"split must be train or val: {split}")
        if window_seconds <= 0 or temporal_window <= 0 or imgsz <= 0:
            raise ValueError("window_seconds, temporal_window and imgsz must be positive")
        self.split = split
        self.window_seconds = window_seconds
        self.temporal_window = temporal_window
        self.imgsz = imgsz
        self.samples = []
        if not class_names or len(set(class_names)) != len(class_names):
            raise ValueError("class_names must be unique")

        split_dir = Path(root) / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"split directory not found: {split_dir}")
        unexpected = [entry.name for entry in split_dir.iterdir()
                      if entry.is_dir() and entry.name not in class_names]
        if unexpected:
            raise ValueError(f"unexpected class directories in {split_dir}: {sorted(unexpected)}")

        for label, class_name in enumerate(class_names):
            class_dir = split_dir / class_name
            if not class_dir.is_dir():
                continue
            for path in sorted(class_dir.iterdir()):
                if not path.is_file() or path.suffix.lower() not in VIDEO_EXTENSIONS:
                    continue
                capture = cv2.VideoCapture(str(path))
                try:
                    if not capture.isOpened():
                        raise ValueError(f"cannot open cropped video: {path}")
                    fps = capture.get(cv2.CAP_PROP_FPS)
                    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                    if not math.isfinite(fps) or fps <= 0 or count <= 0:
                        raise ValueError(f"invalid fps or frame count: {path}")
                finally:
                    capture.release()
                window_frames = round(window_seconds * fps)
                if window_frames < temporal_window or count < window_frames:
                    raise ValueError(f"video too short for {window_seconds}s/{temporal_window} frames: {path}")
                self.samples.append((path, label, count, window_frames))

        if not self.samples:
            raise ValueError(f"no {split} videos in {split_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label, count, window_frames = self.samples[index]
        last_start = count - window_frames
        if self.split == "train":
            start = int(torch.randint(last_start + 1, (1,)).item())
        else:
            start = last_start // 2
        positions = np.linspace(start, start + window_frames - 1, self.temporal_window).astype(int)
        selected = set(positions)
        frames = []
        capture = cv2.VideoCapture(str(path))
        try:
            if not capture.isOpened() or not capture.set(cv2.CAP_PROP_POS_FRAMES, int(positions[0])):
                raise RuntimeError(f"cannot seek cropped video: {path}")
            for frame_index in range(int(positions[0]), int(positions[-1]) + 1):
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError(f"failed to decode frame {frame_index} from {path}")
                if frame_index in selected:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(cv2.resize(rgb, (self.imgsz, self.imgsz)))
        finally:
            capture.release()
        if len(frames) != self.temporal_window:
            raise RuntimeError(f"sampled {len(frames)}/{self.temporal_window} frames from {path}")
        clip = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float() / 255.0
        mean = clip.new_tensor((0.485, 0.456, 0.406))[None, :, None, None]
        std = clip.new_tensor((0.229, 0.224, 0.225))[None, :, None, None]
        return {"pixel_values": (clip - mean) / std,
                "labels": torch.tensor(label, dtype=torch.long)}
