import pytest
import torch

from src.metric import calculate_ap, calculate_iou, calculate_precision_recall


def test_iou_identical_boxes_is_one():
    box = [0, 0, 10, 10]
    assert calculate_iou(box, box) == pytest.approx(1.0, rel=1e-4)


def test_iou_disjoint_boxes_is_zero():
    assert calculate_iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0


def test_iou_partial_overlap():
    # Two 10x10 boxes overlapping in a 5x10 strip -> intersection 50, union 150.
    iou = calculate_iou([0, 0, 10, 10], [5, 0, 15, 10])
    assert iou == pytest.approx(50 / 150, rel=1e-4)


def test_precision_recall_all_true_positives():
    predictions = [[0, 0, 10, 10], [20, 20, 30, 30]]
    targets = [[0, 0, 10, 10], [20, 20, 30, 30]]

    precision, recall = calculate_precision_recall(predictions, targets, iou_threshold=0.5)

    assert torch.allclose(precision, torch.tensor([1.0, 1.0]))
    assert torch.allclose(recall, torch.tensor([0.5, 1.0]))


def test_precision_recall_false_negative_is_tracked():
    # One prediction matches; one ground-truth box is never detected -> a real
    # false negative. This is the exact bug that was silently swallowed before:
    # recall must NOT reach 1.0 when a target is missed.
    predictions = [[0, 0, 10, 10]]
    targets = [[0, 0, 10, 10], [50, 50, 60, 60]]

    _, recall = calculate_precision_recall(predictions, targets, iou_threshold=0.5)

    assert recall[-1].item() == pytest.approx(0.5, rel=1e-4)


def test_precision_recall_no_predictions_matches_nothing():
    predictions = [[100, 100, 110, 110]]
    targets = [[0, 0, 10, 10]]

    precision, recall = calculate_precision_recall(predictions, targets, iou_threshold=0.5)

    assert precision[-1].item() == 0.0
    assert recall[-1].item() == 0.0


def test_precision_recall_one_target_matched_at_most_once():
    # Two predictions both overlap the same single ground-truth box; only one
    # should count as a true positive.
    predictions = [[0, 0, 10, 10], [1, 1, 11, 11]]
    targets = [[0, 0, 10, 10]]

    precision, recall = calculate_precision_recall(predictions, targets, iou_threshold=0.3)

    assert recall[-1].item() == pytest.approx(1.0, rel=1e-4)
    assert precision[-1].item() == pytest.approx(0.5, rel=1e-4)


def test_ap_perfect_detector_is_one():
    precision = torch.tensor([1.0, 1.0])
    recall = torch.tensor([0.5, 1.0])
    assert calculate_ap(precision, recall).item() == pytest.approx(1.0, rel=1e-4)
