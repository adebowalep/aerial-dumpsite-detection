"""Bbox-aware augmentation pipelines (Albumentations).

Plain image transforms (flip, crop, ...) silently desync an image from its
ground-truth boxes unless the boxes are transformed too. These pipelines
take `(image, bboxes, labels)` and return all three consistently, so they
can be applied directly inside DumpsiteDataset before boxes/labels are
converted to tensors.
"""

import albumentations as A

_BBOX_PARAMS = A.BboxParams(format="pascal_voc", label_fields=["labels"], min_visibility=0.2)


def get_train_transform() -> A.Compose:
    """Flips + color jitter + mild scale jitter, all bbox-aware."""
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.2),
            A.Affine(scale=(0.85, 1.15), translate_percent=(0.0, 0.05), p=0.3),
        ],
        bbox_params=_BBOX_PARAMS,
    )


def get_eval_transform() -> A.Compose:
    """Identity transform with the same interface, for validation/test."""
    return A.Compose([], bbox_params=_BBOX_PARAMS)
