from pathlib import Path

import numpy as np
from PIL import Image

from src.yolo_format import convert_dataset


def _build_source(tmp_path):
    images_dir = tmp_path / "images"
    annotations_dir = tmp_path / "annotations"
    images_dir.mkdir()
    annotations_dir.mkdir()

    # Positive: one box exactly covering the left half of a 100x100 image.
    Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8)).save(images_dir / "dumpsite_0.jpeg")
    (annotations_dir / "dumpsite_0.txt").write_text("1 0 0 50 100\n")

    # Negative: true negative, zero boxes.
    Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8)).save(images_dir / "no_dumpsite_0.jpeg")
    (annotations_dir / "no_dumpsite_0.txt").write_text("")

    return tmp_path


def test_convert_dataset_writes_dataset_yaml_and_splits(tmp_path):
    source = _build_source(tmp_path)
    out_dir = tmp_path / "yolo_out"

    yaml_path = convert_dataset(str(source), str(out_dir), train_val_split=0.5, seed=0)

    assert Path(yaml_path).exists()
    assert (out_dir / "images" / "train").exists()
    assert (out_dir / "images" / "val").exists()
    total_images = len(list((out_dir / "images" / "train").glob("*.jpeg"))) + len(
        list((out_dir / "images" / "val").glob("*.jpeg"))
    )
    assert total_images == 2


def test_convert_dataset_normalizes_box_correctly(tmp_path):
    source = _build_source(tmp_path)
    out_dir = tmp_path / "yolo_out"
    convert_dataset(str(source), str(out_dir), train_val_split=1.0, seed=0)

    label_path = out_dir / "labels" / "train" / "dumpsite_0.txt"
    class_id, x_center, y_center, width, height = map(float, label_path.read_text().split())

    assert class_id == 0
    assert x_center == pytest_approx(0.25)  # box spans x=[0,50] of a 100-wide image
    assert y_center == pytest_approx(0.5)
    assert width == pytest_approx(0.5)
    assert height == pytest_approx(1.0)


def test_convert_dataset_writes_empty_label_for_true_negative(tmp_path):
    source = _build_source(tmp_path)
    out_dir = tmp_path / "yolo_out"
    convert_dataset(str(source), str(out_dir), train_val_split=1.0, seed=0)

    label_path = out_dir / "labels" / "train" / "no_dumpsite_0.txt"
    assert label_path.read_text() == ""


def pytest_approx(value):
    import pytest

    return pytest.approx(value, rel=1e-4)
