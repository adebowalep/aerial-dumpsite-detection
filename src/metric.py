"""Detection evaluation metrics: IoU, precision/recall, AP, and mAP.

These are minimal, dependency-free reference implementations. For anything
beyond quick sanity checks, prefer a battle-tested implementation such as
``torchmetrics.detection.MeanAveragePrecision`` or ``pycocotools`` — they
handle edge cases (multiple GT matches, per-class/per-size breakdowns,
COCO-style IoU thresholds) that this module does not.
"""

from typing import List, Sequence, Tuple

import torch

BoundingBox = Sequence[float]


def calculate_iou(box_a: BoundingBox, box_b: BoundingBox, epsilon: float = 1e-5) -> float:
    """Intersection-over-union between two ``[x_min, y_min, x_max, y_max]`` boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    width = x2 - x1
    height = y2 - y1
    if width < 0 or height < 0:
        return 0.0
    area_overlap = width * height
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    area_combined = area_a + area_b - area_overlap
    return area_overlap / (area_combined + epsilon)


def calculate_ap(precision: torch.Tensor, recall: torch.Tensor) -> torch.Tensor:
    """11-point-style Average Precision from monotonic precision/recall curves."""
    mrec = torch.cat((torch.zeros((1,)), recall, torch.ones((1,))))
    mpre = torch.cat((torch.zeros((1,)), precision, torch.zeros((1,))))

    for i in range(mpre.size(0) - 1, 0, -1):
        mpre[i - 1] = torch.max(mpre[i - 1], mpre[i])

    i = torch.nonzero(mrec[1:] != mrec[:-1]) + 1
    return torch.sum((mrec[i] - mrec[i - 1]) * mpre[i])


def calculate_precision_recall(
    predictions: Sequence[BoundingBox],
    targets: Sequence[BoundingBox],
    iou_threshold: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Cumulative precision/recall for one image's predicted vs. ground-truth boxes.

    Each ground-truth box may be matched by at most one prediction; predictions
    are expected to already be sorted by descending confidence.
    """
    num_targets = len(targets)
    true_positives = torch.zeros(len(predictions))
    false_positives = torch.zeros(len(predictions))
    matched_targets = torch.zeros(num_targets, dtype=torch.bool)

    for i, pred in enumerate(predictions):
        best_iou = 0.0
        best_j = -1
        for j, target in enumerate(targets):
            if matched_targets[j]:
                continue
            iou = calculate_iou(pred, target)
            if iou > best_iou:
                best_iou = iou
                best_j = j

        if best_iou >= iou_threshold and best_j >= 0:
            true_positives[i] = 1
            matched_targets[best_j] = True
        else:
            false_positives[i] = 1

    cumulative_true_positives = torch.cumsum(true_positives, dim=0)
    cumulative_false_positives = torch.cumsum(false_positives, dim=0)

    precision = cumulative_true_positives / (
        cumulative_true_positives + cumulative_false_positives + 1e-16
    )
    # Unmatched ground-truth boxes are false negatives; recall is measured
    # against the fixed total number of targets, not a running count.
    recall = cumulative_true_positives / (num_targets + 1e-16)

    return precision, recall


def calculate_map(
    predictions: Sequence[Sequence[Sequence[BoundingBox]]],
    targets: Sequence[Sequence[Sequence[BoundingBox]]],
    iou_threshold: float = 0.5,
) -> float:
    """Mean Average Precision across classes, given per-class box lists."""
    ap_values: List[float] = []

    for c in range(len(predictions[0])):
        precision, recall = calculate_precision_recall(
            predictions[:, c, :], targets[:, c, :], iou_threshold
        )
        ap_values.append(calculate_ap(precision, recall).item())

    return sum(ap_values) / len(ap_values)
