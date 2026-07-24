import os

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2

IMAGE_SCAN_EXTENSIONS = (".jpg", ".png", ".jpeg")
FRAME_LOAD_EXTENSIONS = (".jpg", ".png")
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov")
NORMAL3_CLASS_NAMES = ("eat", "drink", "other")
ABNORMAL5_CLASS_NAMES = ("normal", "convulsion", "limp", "sneeze", "vomit")


def normalize_class_names(class_names):
    return tuple(str(name).lower() for name in class_names)


def resolve_balance_mode(class_names):
    normalized = normalize_class_names(class_names)
    if normalized == NORMAL3_CLASS_NAMES:
        return "normal3"
    if normalized == ABNORMAL5_CLASS_NAMES:
        return "abnormal5"
    return "none"


def get_sample_id(sample_tuple):
    path = sample_tuple[0]
    name = os.path.basename(path).split(".")[0]
    parts = name.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return int(parts[1])
    if name.isdigit():
        return int(name)
    return 0


class VideoDataset(Dataset):
    def __init__(
        self,
        root,
        split="train",
        cfg=None,
        is_train=True,
        default_class_names=None,
        balance_mode="none",
    ):
        self.root = os.path.join(root, split) if split else root
        self.cfg = cfg if cfg is not None else {}
        self.is_train = is_train
        self.default_class_names = list(default_class_names or [])
        self.balance_mode = balance_mode
        self.samples = []
        self.targets = []

        if "class_names" in self.cfg and self.cfg["class_names"]:
            self.keywords = self.cfg["class_names"]
        else:
            self.keywords = self.default_class_names
            if self.keywords:
                print(f"[Dataset] class_names not set, using defaults: {self.keywords}")

        img_size = self.cfg.get("imgsz", 224)
        self.frame_resize = v2.Resize((img_size, img_size), antialias=True)
        self.transform = self._build_clip_transform(img_size)

        print(f"[Dataset] Scan path: {self.root}")
        print(f"[Dataset] Target classes: {self.keywords}")

        if not os.path.exists(self.root):
            print(f"[Error] Path does not exist: {self.root}")
            return

        raw_samples = self._scan_samples()
        self.samples = self._apply_balance(raw_samples)
        self.targets = [sample[1] for sample in self.samples]

        print(f"[Dataset] Loaded {len(self.samples)} samples")

        if len(self.samples) > 0:
            counts = {}
            for target in self.targets:
                name = self.keywords[target]
                counts[name] = counts.get(name, 0) + 1
            print(f"[Dataset] Final distribution: {counts}")

    def _read_label_file(self, entry_path):
        """Read numeric label from label.txt inside a sample folder.

        Returns the integer class index if valid, or -1 if the file is
        absent, unreadable, or contains an out-of-range value.
        """
        label_file = os.path.join(entry_path, "label.txt")
        if os.path.isfile(label_file):
            try:
                idx = int(open(label_file).read().strip())
                if 0 <= idx < len(self.keywords):
                    return idx
            except (ValueError, IOError):
                pass
        return -1

    def _match_label(self, entry_name):
        name_lower = entry_name.lower()
        for idx, keyword in enumerate(self.keywords):
            if keyword.lower() in name_lower:
                return idx
        return -1

    def _scan_samples(self):
        raw_samples = []

        top_level_entries = sorted(os.listdir(self.root))
        for entry_name in top_level_entries:
            if entry_name.startswith("."):
                continue

            entry_path = os.path.join(self.root, entry_name)

            if os.path.isdir(entry_path):
                # For sample folders: prefer label.txt, fall back to name matching.
                label = self._read_label_file(entry_path)
                if label == -1:
                    label = self._match_label(entry_name)
                if label == -1:
                    continue

                contents = [f for f in os.listdir(entry_path) if not f.startswith(".")]
                has_images = any(
                    f.lower().endswith(IMAGE_SCAN_EXTENSIONS) for f in contents
                )
                has_subdirs = any(
                    os.path.isdir(os.path.join(entry_path, f)) for f in contents
                )

                if has_images and not has_subdirs:
                    raw_samples.append((entry_path, label))
                else:
                    for file_name in contents:
                        file_path = os.path.join(entry_path, file_name)
                        raw_samples.append((file_path, label))
            else:
                # For bare video files: use name matching only (no label.txt).
                label = self._match_label(entry_name)
                if label == -1:
                    continue
                if entry_name.lower().endswith(VIDEO_EXTENSIONS):
                    raw_samples.append((entry_path, label))

        return raw_samples

    def _apply_balance(self, raw_samples):
        if len(raw_samples) == 0:
            return raw_samples

        if self.balance_mode == "normal3":
            return self._apply_normal3_balance(raw_samples)
        if self.balance_mode == "abnormal5":
            return self._apply_abnormal5_balance(raw_samples)
        return raw_samples

    def _apply_normal3_balance(self, raw_samples):
        print("[Dataset] Applying hard balancing (Eat/Drink 1.5x strategy)...")

        samples_by_idx = {i: [] for i in range(len(self.keywords))}
        for sample in raw_samples:
            samples_by_idx[sample[1]].append(sample)

        idx_eat = -1
        idx_drink = -1
        idx_other = -1

        for idx, kw in enumerate(self.keywords):
            kw_lower = kw.lower()
            if "eat" in kw_lower:
                idx_eat = idx
            elif "drink" in kw_lower:
                idx_drink = idx
            elif "other" in kw_lower:
                idx_other = idx

        if idx_eat != -1 and idx_drink != -1 and idx_other != -1:
            n_eat = len(samples_by_idx[idx_eat])
            n_drink = len(samples_by_idx[idx_drink])

            target_base = max(n_eat, n_drink)
            if target_base == 0:
                target_base = 100

            target_other = int(target_base * 1.5)
            current_other = samples_by_idx[idx_other]

            if len(current_other) > target_other:
                current_other.sort(key=get_sample_id)
                samples_by_idx[idx_other] = current_other[-target_other:]
                print(
                    f"   [Trim] Truncate Other (keep higher ids): {len(current_other)} -> {target_other}"
                )
            else:
                print(
                    f"   [Keep] Other count ({len(current_other)}) within limit ({target_other}), no truncation"
                )

            samples = []
            for idx in samples_by_idx:
                samples.extend(samples_by_idx[idx])
            return samples

        print("   [Warn] Could not identify Eat/Drink/Other indices, skipping hard balance.")
        return raw_samples

    def _apply_abnormal5_balance(self, raw_samples):
        print("[Dataset] Applying hard balancing (abnormal 1.5x strategy)...")

        samples_by_idx = {i: [] for i in range(len(self.keywords))}
        for sample in raw_samples:
            samples_by_idx[sample[1]].append(sample)

        idx_normal = -1
        abnormal_indices = []

        for idx, kw in enumerate(self.keywords):
            if "normal" in kw.lower():
                idx_normal = idx
            else:
                abnormal_indices.append(idx)

        if idx_normal != -1 and len(abnormal_indices) > 0:
            abnormal_counts = [len(samples_by_idx[i]) for i in abnormal_indices]
            target_base = max(abnormal_counts)
            if target_base == 0:
                target_base = 100

            target_normal = int(target_base * 1.5)
            current_normal = samples_by_idx[idx_normal]

            if len(current_normal) > target_normal:
                current_normal.sort(key=get_sample_id)
                samples_by_idx[idx_normal] = current_normal[-target_normal:]
                print(
                    f"   [Trim] Truncate Normal (keep higher ids): {len(current_normal)} -> {target_normal}"
                )
            else:
                print(
                    f"   [Keep] Normal count ({len(current_normal)}) within limit ({target_normal}), no truncation"
                )

            samples = []
            for idx in samples_by_idx:
                samples.extend(samples_by_idx[idx])
            return samples

        print("   [Warn] Could not identify normal and abnormal indices, skipping hard balance.")
        return raw_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        try:
            frames = self._load_frames(video_path)
            pixel_values = torch.stack(frames)
            pixel_values = self.transform(pixel_values)
            pixel_values = self._maybe_reverse_clip(pixel_values)
            return {
                "pixel_values": pixel_values,
                "labels": torch.tensor(label, dtype=torch.long),
            }
        except Exception as e:
            print(f"Error loading {video_path}: {e}")
            img_size = self.cfg.get("imgsz", 224)
            seq_len = self.cfg.get("temporal_window", 16)
            pixel_values = torch.zeros(seq_len, 3, img_size, img_size, dtype=torch.uint8)
            pixel_values = self.transform(pixel_values)
            return {
                "pixel_values": pixel_values,
                "labels": torch.tensor(label, dtype=torch.long),
            }

    def _build_clip_transform(self, img_size):
        augment = self.cfg.get("augment") or {}
        transforms = []

        if self.is_train and augment:
            scale_min = float(augment.get("scale_min", 1.0) or 1.0)
            scale_max = float(augment.get("scale_max", 1.0) or 1.0)
            if scale_min < 1.0 or scale_max < 1.0:
                transforms.append(
                    v2.RandomResizedCrop(
                        size=(img_size, img_size),
                        scale=(scale_min, scale_max),
                        ratio=(1.0, 1.0),
                        antialias=True,
                    )
                )

            rotate = float(augment.get("rotate", 0.0) or 0.0)
            translate = float(augment.get("translate", 0.0) or 0.0)
            if rotate > 0 or translate > 0:
                transforms.append(
                    v2.RandomAffine(
                        degrees=(-rotate, rotate),
                        translate=(translate, translate) if translate > 0 else None,
                    )
                )

            fliplr = float(augment.get("fliplr", 0.0) or 0.0)
            if fliplr > 0:
                transforms.append(v2.RandomHorizontalFlip(p=fliplr))

            brightness = float(augment.get("brightness", 0.0) or 0.0)
            contrast = float(augment.get("contrast", 0.0) or 0.0)
            if brightness > 0 or contrast > 0:
                transforms.append(
                    v2.ColorJitter(
                        brightness=brightness if brightness > 0 else 0,
                        contrast=contrast if contrast > 0 else 0,
                    )
                )
        else:
            pass

        transforms.extend(
            [
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        return v2.Compose(transforms)

    def _maybe_reverse_clip(self, pixel_values):
        if not self.is_train:
            return pixel_values

        augment = self.cfg.get("augment") or {}
        reverse_prob = float(augment.get("reverse", 0.0) or 0.0)
        if reverse_prob <= 0:
            return pixel_values
        if torch.rand(()) >= reverse_prob:
            return pixel_values
        return torch.flip(pixel_values, dims=(0,))

    def _load_frames(self, path):
        frames = []
        seq_len = self.cfg.get("temporal_window", self.cfg.get("seq_len", 16))
        img_size = self.cfg.get("imgsz", 224)

        if os.path.isdir(path):
            all_images = sorted(
                [
                    os.path.join(path, f)
                    for f in os.listdir(path)
                    if f.lower().endswith(FRAME_LOAD_EXTENSIONS)
                ]
            )
            if len(all_images) > 0:
                indices = np.linspace(0, len(all_images) - 1, seq_len).astype(int)
                for i in indices:
                    try:
                        img = Image.open(all_images[i]).convert("RGB")
                        frame = v2.functional.pil_to_tensor(img)
                        frames.append(self.frame_resize(frame))
                    except Exception:
                        frames.append(torch.zeros(3, img_size, img_size, dtype=torch.uint8))
        else:
            cap = cv2.VideoCapture(path)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total > 0:
                indices = np.linspace(0, total - 1, seq_len).astype(int)
                for i in range(total):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    if i in indices:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frame = torch.from_numpy(frame).permute(2, 0, 1)
                        frames.append(self.frame_resize(frame))
            cap.release()

        while len(frames) < seq_len:
            frames.append(torch.zeros(3, img_size, img_size, dtype=torch.uint8))
        return frames[:seq_len]

    def get_class_weights(self, device="cpu"):
        if len(self.targets) == 0:
            return None
        counts = np.bincount(self.targets, minlength=len(self.keywords))
        weights = 1.0 / (counts + 1e-6)
        weights = weights / np.mean(weights)
        return torch.tensor(weights, dtype=torch.float, device=device)
