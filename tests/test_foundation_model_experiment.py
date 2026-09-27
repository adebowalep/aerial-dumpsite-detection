"""Tests for the non-model parts of scripts/foundation_model_experiment.py
(dataset sampling and the label-efficiency curve logic) - no DINOv2/ResNet
download needed, so this runs fast and offline.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from foundation_model_experiment import label_efficiency_curve, list_labeled_images  # noqa: E402


def _build_dataset(tmp_path, num_positive=6, num_negative=6, num_mislabeled_positive=0):
    images_dir = tmp_path / "images"
    annotations_dir = tmp_path / "annotations"
    images_dir.mkdir()
    annotations_dir.mkdir()

    for i in range(num_positive):
        (images_dir / f"dumpsite_{i}.jpeg").touch()
        content = "" if i < num_mislabeled_positive else "1 0 0 10 10\n"
        (annotations_dir / f"dumpsite_{i}.txt").write_text(content)

    for i in range(num_negative):
        (images_dir / f"no_dumpsite_{i}.jpeg").touch()
        (annotations_dir / f"no_dumpsite_{i}.txt").write_text("")

    return tmp_path


def test_list_labeled_images_returns_balanced_sample(tmp_path):
    root = _build_dataset(tmp_path, num_positive=6, num_negative=6)
    paths, labels = list_labeled_images(str(root), num_images=8, seed=0)

    assert len(paths) == 8
    assert sum(labels) == 4  # 8 // 2 positives requested
    assert sum(1 - np.array(labels)) == 4


def test_list_labeled_images_trusts_annotation_over_filename(tmp_path):
    """A dumpsite_*.jpeg with an empty annotation file (found for real in
    the actual dataset: 10/3395 cases) must NOT be counted as a positive."""
    root = _build_dataset(tmp_path, num_positive=6, num_negative=6, num_mislabeled_positive=2)
    paths, labels = list_labeled_images(str(root), num_images=8, seed=0)

    for path, label in zip(paths, labels):
        if "dumpsite_0" in path.name or "dumpsite_1" in path.name:
            assert label == 0  # the two mislabeled ones must be excluded from positives


def test_label_efficiency_curve_reports_increasing_train_size():
    rng = np.random.RandomState(0)
    features = np.vstack([rng.normal(0, 1, (50, 4)), rng.normal(5, 1, (50, 4))])
    labels = np.array([0] * 50 + [1] * 50)

    curve = label_efficiency_curve(features, labels, fractions=[0.25, 1.0], seed=0)

    assert curve[0]["train_fraction"] == 0.25
    assert curve[1]["train_fraction"] == 1.0
    assert curve[1]["num_train_samples"] > curve[0]["num_train_samples"]
    # Well-separated synthetic classes should be trivially learnable.
    assert curve[1]["test_accuracy"] > 0.9
