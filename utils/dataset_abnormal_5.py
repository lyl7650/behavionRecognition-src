from utils.video_dataset import VideoDataset as BaseVideoDataset, resolve_balance_mode

DEFAULT_CLASS_NAMES = ["normal", "convulsion", "limp", "sneeze", "vomit"]


class VideoDataset(BaseVideoDataset):
    def __init__(self, root, split="train", cfg=None, is_train=True):
        config = dict(cfg or {})
        effective_class_names = config.setdefault("class_names", list(DEFAULT_CLASS_NAMES))
        super().__init__(
            root,
            split=split,
            cfg=config,
            is_train=is_train,
            default_class_names=list(DEFAULT_CLASS_NAMES),
            balance_mode=resolve_balance_mode(effective_class_names),
        )


def get_dataset(data_dir, split, cfg, is_train=True):
    return VideoDataset(data_dir, split=split, cfg=cfg, is_train=is_train)
