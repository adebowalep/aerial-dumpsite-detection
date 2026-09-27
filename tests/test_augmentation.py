import pytest

from src.augmentation import get_eval_transform, get_train_transform


def _sample():
    import numpy as np

    image = np.zeros((100, 100, 3), dtype=np.uint8)
    boxes = [[10, 10, 40, 40]]
    labels = [1]
    return image, boxes, labels


def test_eval_transform_is_identity_on_boxes():
    image, boxes, labels = _sample()
    transform = get_eval_transform()

    result = transform(image=image, bboxes=boxes, labels=labels)

    assert result["image"].shape == image.shape
    assert list(result["bboxes"][0]) == pytest.approx(boxes[0], abs=1e-3)
    assert list(result["labels"]) == labels


def test_train_transform_keeps_valid_boxes_and_labels_in_sync():
    image, boxes, labels = _sample()
    transform = get_train_transform()

    result = transform(image=image, bboxes=boxes, labels=labels)

    assert result["image"].shape == image.shape
    assert len(result["bboxes"]) == len(result["labels"])
    for box in result["bboxes"]:
        x_min, y_min, x_max, y_max = box
        assert x_max > x_min
        assert y_max > y_min


def test_train_transform_handles_zero_boxes_true_negative():
    image = _sample()[0]
    transform = get_train_transform()

    result = transform(image=image, bboxes=[], labels=[])

    assert result["bboxes"] == []
    assert result["labels"] == []
