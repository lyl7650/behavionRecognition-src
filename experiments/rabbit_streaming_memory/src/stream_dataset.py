"""Streaming dataset for long one-action videos stored by class directories.

Each video is treated as a temporally ordered sequence. Blocks are causal
12-second windows with 5-second burn-in; labels are the directory class for
every step, matching the project's current one-video-one-action labeling.
Read failures raise instead of silently returning zero clips.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".webm"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class VideoSpec:
    path: Path
    label: int
    fps: float
    frame_count: int


def inspect_video(path: Path) -> VideoSpec:
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"cannot open video: {path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0 or frame_count <= 0:
            raise RuntimeError(f"invalid video metadata: {path}, fps={fps}, frames={frame_count}")
    finally:
        capture.release()
    return VideoSpec(path=path, label=0, fps=fps, frame_count=frame_count)


def scan_videos(root: Path, class_names: list[str], exclude_paths: set[str] | None = None) -> list[VideoSpec]:
    specs: list[VideoSpec] = []
    excluded = {str(Path(p).resolve()).lower() for p in (exclude_paths or set())}
    split_dirs = [root]
    for split_dir in split_dirs:
        if not split_dir.is_dir():
            continue
        for class_name in class_names:
            class_dir = split_dir / class_name
            if not class_dir.is_dir():
                continue
            label = class_names.index(class_name)
            for path in sorted(class_dir.rglob("*")):
                if path.is_file() and path.suffix.lower() in VIDEO_EXTS:
                    if str(path.resolve()).lower() in excluded:
                        continue
                    spec = inspect_video(path)
                    specs.append(VideoSpec(path=spec.path, label=label, fps=spec.fps, frame_count=spec.frame_count))
    if not specs:
        raise RuntimeError(f"no readable videos under {root}")
    return specs


def target_to_source_frame(target_index: int, source_fps: float, target_fps: float) -> int:
    if target_index < 0 or source_fps <= 0 or target_fps <= 0:
        raise ValueError("target_index must be nonnegative and both frame rates positive")
    return int(round(target_index * source_fps / target_fps))


def effective_block_lengths(frame_count: int, block_frames: int, burn_in_frames: int) -> tuple[int, int]:
    if frame_count < 1 or block_frames < 1 or burn_in_frames < 0:
        raise ValueError("frame_count/block_frames must be positive and burn_in_frames nonnegative")
    read_frames = min(block_frames, frame_count)
    effective_burn_in = min(burn_in_frames, max(0, read_frames - 1))
    return read_frames, effective_burn_in


def _preprocess(frame_bgr: np.ndarray, imgsz: int, mean: np.ndarray, std: np.ndarray) -> torch.Tensor:
    rgb = cv2.cvtColor(cv2.resize(frame_bgr, (imgsz, imgsz), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    return (tensor - mean[:, None, None]) / std[:, None, None]


class StreamingVideoDataset(Dataset):
    """Yield a fixed-length contiguous block from one long video.

    Block tensors are [T, C, H, W]; labels are [T]. Empty or unreadable videos
    are rejected by scan_videos; decode failures here raise as required.
    """

    def __init__(
        self,
        specs: list[VideoSpec],
        block_seconds: float = 12.0,
        burn_in_seconds: float = 5.0,
        imgsz: int = 224,
        fps: float = 8.0,
        is_train: bool = False,
        fliplr: float = 0.0,
        seed: int = 42,
        pad_short: bool = False,
    ):
        if block_seconds <= burn_in_seconds:
            raise ValueError("block_seconds must exceed burn_in_seconds")
        self.specs = specs
        self.fps = float(fps)
        self.block_frames = int(round(block_seconds * self.fps))
        self.burn_in_frames = int(round(burn_in_seconds * self.fps))
        self.imgsz = imgsz
        self.mean = np.asarray(IMAGENET_MEAN, dtype=np.float32)
        self.std = np.asarray(IMAGENET_STD, dtype=np.float32)
        self.is_train = is_train
        self.fliplr = float(fliplr)
        self._rng = random.Random(seed)
        self.pad_short = bool(pad_short)
        if self.block_frames < 2:
            raise ValueError("block must contain at least 2 frames")

    def __len__(self) -> int:
        return len(self.specs)

    def _target_frame_count(self, spec: VideoSpec) -> int:
        duration_seconds = spec.frame_count / spec.fps
        return max(1, int(round(duration_seconds * self.fps)))

    def _random_start_frame(self, spec: VideoSpec) -> int:
        max_start = max(0, self._target_frame_count(spec) - self.block_frames)
        if self.is_train:
            return self._rng.randint(0, max_start)
        return max_start // 2

    def __getitem__(self, index: int):
        spec = self.specs[index]
        start = self._random_start_frame(spec)
        target_frame_count = self._target_frame_count(spec)
        available = max(1, target_frame_count - start)
        read_frames, burn_in_frames = effective_block_lengths(available, self.block_frames, self.burn_in_frames)
        frames: list[np.ndarray] = []
        capture = cv2.VideoCapture(str(spec.path))
        try:
            if not capture.isOpened():
                raise RuntimeError(f"cannot open video: {spec.path}")
            source_start = target_to_source_frame(start, spec.fps, self.fps)
            if not capture.set(cv2.CAP_PROP_POS_FRAMES, source_start):
                raise RuntimeError(f"cannot seek {spec.path} to source frame {source_start}")
            source_index = source_start
            for target_offset in range(read_frames):
                target_index = start + target_offset
                source_index_target = target_to_source_frame(target_index, spec.fps, self.fps)
                while source_index < source_index_target:
                    ok, skipped = capture.read()
                    if not ok or skipped is None:
                        raise RuntimeError(f"decode failed while advancing {spec.path} to source frame {source_index_target}")
                    source_index += 1
                ok, frame = capture.read()
                if not ok or frame is None:
                    raise RuntimeError(f"decode failed at {spec.path} source frame {source_index_target}")
                frames.append(frame)
                source_index = source_index_target + 1
        finally:
            capture.release()

        tensors = [_preprocess(frame, self.imgsz, self.mean, self.std) for frame in frames]
        block = torch.stack(tensors, dim=0)
        if self.pad_short and len(block) < self.block_frames:
            pad = block[:1].repeat(self.block_frames - len(block), 1, 1, 1)
            block = torch.cat((pad, block), dim=0)
        if self.is_train and self.fliplr > 0 and self._rng.random() < self.fliplr:
            block = torch.flip(block, dims=(3,))
        labels = torch.full((len(block),), spec.label, dtype=torch.long)
        return {
            "pixel_values": block,
            "labels": labels,
            "burn_in_frames": burn_in_frames,
            "video_path": str(spec.path),
        }
