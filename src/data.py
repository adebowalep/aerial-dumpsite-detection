"""PyTorch Dataset for the dumpsite/no_dumpsite detection task.

Each image has a matching ``.txt`` annotation file with zero or more lines of
``<label> <x_min> <y_min> <x_max> <y_max>`` (absolute pixel coordinates in the
resized image). An empty annotation file is a valid true negative - an image
with no dumpsite in it - and is kept, not dropped, so the model sees explicit
negatives during training.
"""

import glob
import os
from typing import Callable, List, Optional, Tuple

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
        bbox_transform: Optional[Callable] = None,
        common_size: Tuple[int, int] = (1024, 1024),
    ):
        """
        Args:
            root_dir: Directory containing an `images/` and `annotations/` subfolder.
            bbox_transform: Optional Albumentations-style ``Compose`` called as
                ``bbox_transform(image=image, bboxes=boxes, labels=labels)`` and
                returning a dict with the same three keys — see
                ``src/augmentation.py``. A plain image-only transform is not
                accepted here since it would desync images from their boxes.
            common_size: (width, height) every image is resized to.
        """
        self.root_dir = root_dir
        self.bbox_transform = bbox_transform
        self.common_size = common_size
        self.image_files = sorted(glob.glob(os.path.join(root_dir, "images", "*.jpeg")))

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path, txt_path = self._get_image_and_annotation_paths(idx)
        image = Image.open(img_path).convert("RGB")
        image = np.array(image.resize(self.common_size, resample=Image.BILINEAR)).astype(np.uint8)

        boxes, labels = self._load_annotations(txt_path)

        if self.bbox_transform is not None:
            transformed = self.bbox_transform(image=image, bboxes=boxes, labels=labels)
            image, boxes, labels = transformed["image"], transformed["bboxes"], transformed["labels"]

        image = image.astype(np.float32) / 255.0
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
