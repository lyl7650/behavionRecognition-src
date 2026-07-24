import importlib
import numpy as np
import pytest

pytest.importorskip("cv2")
torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")


def test_clip_from_frames_matches_videodataset_transform():
    pp = importlib.import_module("auto_presort.preprocess")
    from utils.video_dataset import VideoDataset

    # 32 帧移动色块，全画面框（None → fallback 全画面），确定性
    frames = []
    for i in range(32):
        f = np.zeros((120, 160, 3), dtype=np.uint8)
        f[:, :, 0] = (i * 5) % 256
        frames.append(f)

    imgsz, tw = 224, 16
    # 参考路径：构造一个空 VideoDataset 只为拿 transform / frame_resize
    ds = VideoDataset.__new__(VideoDataset)
    ds.cfg = {"imgsz": imgsz, "temporal_window": tw}
    ds.is_train = False
    from torchvision.transforms import v2
    ds.frame_resize = v2.Resize((imgsz, imgsz), antialias=True)
    ds.transform = ds._build_clip_transform(imgsz)

    # 参考：模拟 _load_frames 对 BGR 数组的处理（linspace 采样 + resize）→ transform
    idxs = np.linspace(0, len(frames) - 1, tw).astype(int)
    ref_frames = []
    import cv2
    for i in idxs:
        rgb = cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(rgb).permute(2, 0, 1)
        ref_frames.append(ds.frame_resize(t))
    ref = ds.transform(torch.stack(ref_frames))

    # 被测路径：全画面框
    got = pp.clip_from_frames(
        frames_bgr=frames, boxes=[None] * len(frames),
        imgsz=imgsz, temporal_window=tw, expand_ratio=1.25,
    )
    assert got.shape == (tw, 3, imgsz, imgsz)
    assert torch.allclose(got, ref, atol=1e-5)
