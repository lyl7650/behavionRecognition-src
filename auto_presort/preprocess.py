"""把「帧列表 + 对应检测框」转成与训练一致的 (T,3,H,W) 张量。

抠图复用 build_smoothed_crop_boxes；采样/归一化复用 VideoDataset 的构造，
保证与训练 eval 路径逐位一致（parity test 守卫）。
"""
import cv2
import numpy as np
import torch
from torchvision.transforms import v2

from utils.abnormal_crop_video import build_smoothed_crop_boxes

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


def _eval_transform(imgsz):
    # 对齐 utils/video_dataset.py 的 eval 分支（is_train=False，无随机增强）
    return v2.Compose([
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])


def clip_from_frames(frames_bgr, boxes, imgsz, temporal_window, expand_ratio):
    """frames_bgr: 窗口内 BGR 帧列表；boxes: 与帧对齐的 xyxy 或 None（窗口内坐标）。"""
    if not frames_bgr:
        # 与 VideoDataset 一致：uint8 零帧再归一化
        zeros = torch.zeros(temporal_window, 3, imgsz, imgsz, dtype=torch.uint8)
        return _eval_transform(imgsz)(zeros)

    h, w = frames_bgr[0].shape[:2]
    crop_boxes = build_smoothed_crop_boxes(
        boxes[:len(frames_bgr)], frame_width=w, frame_height=h,
        fallback_size=(w, h), expand_ratio=expand_ratio,
    )
    resize = v2.Resize((imgsz, imgsz), antialias=True)
    idxs = np.linspace(0, len(frames_bgr) - 1, temporal_window).astype(int)
    tensors = []
    for i in idxs:
        x1, y1, x2, y2 = crop_boxes[i]
        crop = frames_bgr[i][y1:y2, x1:x2]
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(rgb).permute(2, 0, 1)
        tensors.append(resize(t))
    return _eval_transform(imgsz)(torch.stack(tensors))
