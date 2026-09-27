"""Convert this project's DumpsiteDataset-style directory layout into the
YOLO format ultralytics expects, so YOLOv8/v11/RT-DETR can train on the
same underlying data as the torchvision models in src/model.py.

Our format: <root>/images/*.jpeg + <root>/annotations/*.txt, one line per
box as "<label> <x_min> <y_min> <x_max> <y_max>" in absolute pixels (empty
file = true negative, zero boxes).

YOLO format: <out>/images/{train,val}/*.jpeg + <out>/labels/{train,val}/*.txt,
one line per box as "<class_id> <x_center> <y_center> <width> <height>",
all normalized to [0, 1] by the image's own width/height. An empty label
file is still the correct way to represent a true negative in YOLO format.
"""

import random
import shutil
from pathlib import Path
from typing import List, Tuple

from PIL import Image

CLASS_NAMES = ["dumpsite"]  # ultralytics has no separate background class


def _read_boxes(annotation_path: Path) -> List[Tuple[float, float, float, float]]:
    if not annotation_path.exists():
        return []
    boxes = []
    for line in annotation_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        _label, x_min, y_min, x_max, y_max = map(float, parts)
        boxes.append((x_min, y_min, x_max, y_max))
    return boxes


def _to_yolo_line(box: Tuple[float, float, float, float], image_width: int, image_height: int) -> str:
    x_min, y_min, x_max, y_max = box
    x_center = (x_min + x_max) / 2 / image_width
    y_center = (y_min + y_max) / 2 / image_height
    width = (x_max - x_min) / image_width
    height = (y_max - y_min) / image_height
    return f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"


def convert_split(source_root: str, out_root: str, split: str, image_paths: List[Path]) -> None:
    images_out = Path(out_root) / "images" / split
    labels_out = Path(out_root) / "labels" / split
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    for image_path in image_paths:
        annotation_path = Path(source_root) / "annotations" / (image_path.stem + ".txt")
        with Image.open(image_path) as img:
            width, height = img.size

        boxes = _read_boxes(annotation_path)
        lines = [_to_yolo_line(box, width, height) for box in boxes]

        shutil.copy(image_path, images_out / image_path.name)
        (labels_out / (image_path.stem + ".txt")).write_text("\n".join(lines))


def convert_dataset(source_root: str, out_root: str, train_val_split: float = 0.8, seed: int = 42) -> str:
    """Convert a DumpsiteDataset-style `source_root` into a YOLO-format
    dataset at `out_root`, writing train/val splits and a dataset.yaml.
    Returns the path to the written dataset.yaml.
    """
    image_paths = sorted(Path(source_root).glob("images/*.jpeg"))
    if not image_paths:
        raise FileNotFoundError(f"No images found under {source_root}/images/")

    rng = random.Random(seed)
    shuffled = image_paths[:]
    rng.shuffle(shuffled)
    split_idx = int(train_val_split * len(shuffled))
    train_paths, val_paths = shuffled[:split_idx], shuffled[split_idx:]

    convert_split(source_root, out_root, "train", train_paths)
    convert_split(source_root, out_root, "val", val_paths)

    yaml_path = Path(out_root) / "dataset.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {Path(out_root).resolve()}",
                "train: images/train",
                "val: images/val",
                f"names: {CLASS_NAMES}",
                "",
            ]
        )
    )
    return str(yaml_path)
