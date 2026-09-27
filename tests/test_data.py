import torch

from src.data import DumpsiteDataset, preprocess_image


def _find(dataset, name_contains):
    idx = next(i for i, p in enumerate(dataset.image_files) if name_contains in p)
    return dataset[idx]


def test_dataset_length_matches_number_of_images(dataset_dir):
    dataset = DumpsiteDataset(str(dataset_dir))
    assert len(dataset) == 2


def test_positive_sample_has_correct_box_and_label(dataset_dir):
    dataset = DumpsiteDataset(str(dataset_dir), common_size=(64, 64))
    image, target = _find(dataset, "dumpsite_0")

    assert image.shape == (3, 64, 64)
    assert image.dtype == torch.float32
    assert target["boxes"].shape == (1, 4)
    assert target["boxes"].tolist() == [[10.0, 10.0, 40.0, 40.0]]
    assert target["labels"].tolist() == [1]


def test_negative_sample_is_kept_with_zero_boxes(dataset_dir):
    """A true negative (empty annotation file) must be kept as a zero-box
    sample, not dropped - this is the fix for the old full-image-class-0
    mislabeling bug (see README: Data Provenance & Fixes)."""
    dataset = DumpsiteDataset(str(dataset_dir))
    image, target = _find(dataset, "no_dumpsite_0")

    assert target["boxes"].shape == (0, 4)
    assert target["boxes"].dtype == torch.float32
    assert target["labels"].shape == (0,)
    assert target["labels"].dtype == torch.int64


def test_missing_annotation_file_yields_zero_boxes_not_a_crash(dataset_dir):
    (dataset_dir / "images" / "dumpsite_1.jpeg").write_bytes(
        (dataset_dir / "images" / "dumpsite_0.jpeg").read_bytes()
    )
    # No matching dumpsite_1.txt written on purpose.
    dataset = DumpsiteDataset(str(dataset_dir))
    _, target = _find(dataset, "dumpsite_1")
    assert target["boxes"].shape == (0, 4)


def test_collate_fn_keeps_batch_as_parallel_tuples(dataset_dir):
    dataset = DumpsiteDataset(str(dataset_dir))
    batch = [dataset[0], dataset[1]]

    images, targets = dataset.collate_fn(batch)

    assert len(images) == 2
    assert len(targets) == 2
    assert all(isinstance(t, dict) for t in targets)


def test_preprocess_image_converts_hwc_to_chw_tensor():
    import numpy as np

    array = np.zeros((8, 6, 3), dtype=np.float64)
    tensor = preprocess_image(array)

    assert tensor.shape == (3, 8, 6)
    assert tensor.dtype == torch.float32
