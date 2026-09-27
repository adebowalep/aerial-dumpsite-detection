"""Shared fixtures: build a tiny synthetic dumpsite-style dataset on disk."""

import numpy as np
import pytest
from PIL import Image


def _write_image(path, size=(64, 64), color=(120, 140, 100)):
    Image.fromarray(np.full((size[1], size[0], 3), color, dtype=np.uint8)).save(path, "JPEG")


@pytest.fixture
def dataset_dir(tmp_path):
    """A small dataset dir with one positive (boxed) and one negative (empty) sample."""
    images_dir = tmp_path / "images"
    annotations_dir = tmp_path / "annotations"
    images_dir.mkdir()
    annotations_dir.mkdir()

    # Positive: one dumpsite box.
    _write_image(images_dir / "dumpsite_0.jpeg")
    (annotations_dir / "dumpsite_0.txt").write_text("1 10 10 40 40\n")

    # Negative: true negative, zero ground-truth boxes.
    _write_image(images_dir / "no_dumpsite_0.jpeg")
    (annotations_dir / "no_dumpsite_0.txt").write_text("")

    return tmp_path
