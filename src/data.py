"""PyTorch Dataset for the dumpsite/no_dumpsite detection task.

Each image has a matching ``.txt`` annotation file with zero or more lines of
``<label> <x_min> <y_min> <x_max> <y_max>`` (absolute pixel coordinates in the
resized image). An empty annotation file is a valid true negative - an image
with no dumpsite in it - and is kept, not dropped, so the model sees explicit
negatives during training.
"""

import glob
import os
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def preprocess_image(image: np.ndarray) -> torch.Tensor:
    """Convert an HWC float image in [0, 1] to a CHW float tensor."""
    return torch.tensor(image).permute(2, 0, 1).float()


class DumpsiteDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        transform=None,
        target_transform=None,
        common_size: Tuple[int, int] = (1024, 1024),
    ):
        """
        Args:
            root_dir: Directory containing an `images/` and `annotations/` subfolder.
            transform: Optional transform applied to the image array.
            target_transform: Optional transform applied to the target dict.
            common_size: (width, height) every image is resized to.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform
        self.common_size = common_size
        self.image_files = sorted(glob.glob(os.path.join(root_dir, "images", "*.jpeg")))

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path, txt_path = self._get_image_and_annotation_paths(idx)
        image = Image.open(img_path).convert("RGB")
        image = np.array(image.resize(self.common_size, resample=Image.BILINEAR)) / 255.0

        boxes, labels = self._load_annotations(txt_path)
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            boxes, labels = self.target_transform(boxes, labels)

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(labels, dtype=torch.int64),
        }
        return preprocess_image(image), target

    def _get_image_and_annotation_paths(self, idx: int) -> Tuple[str, str]:
        img_path = self.image_files[idx]
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        txt_path = os.path.join(self.root_dir, "annotations", img_name + ".txt")
        return img_path, txt_path

    @staticmethod
    def _load_annotations(txt_path: str) -> Tuple[List[List[float]], List[float]]:
        """Parse a ``label x_min y_min x_max y_max`` annotation file.

        An empty (or missing-lines) file yields ``([], [])`` — a true negative
        with zero ground-truth boxes, which is kept rather than dropped.
        """
        boxes: List[List[float]] = []
        labels: List[float] = []

        if not os.path.exists(txt_path):
            return boxes, labels

        with open(txt_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                label, x_min, y_min, x_max, y_max = map(float, parts)
                boxes.append([x_min, y_min, x_max, y_max])
                labels.append(label)

        return boxes, labels

    @staticmethod
    def collate_fn(batch):
        """Standard detection collate: keep images/targets as parallel tuples."""
        images, targets = zip(*batch)
        return images, targets
