"""Tests for the parts of src/model.py that don't require downloading
pretrained weights or a GPU - the model factories themselves
(get_model, get_resnet50, ...) are exercised on the real GPU workspace
instead, not here.
"""

import types

import pytest
import torch

from src import model as model_module
from src.model import MODEL_REGISTRY, get_model_by_name


def test_model_registry_has_expected_variants():
    assert set(MODEL_REGISTRY) == {
        "resnet50_scratch_head",
        "resnet50_fpn",
        "resnet50_fpn_v2",
        "mobilenet_v3",
        "retinanet",
        "fcos",
    }


def test_get_model_by_name_rejects_unknown_name():
    with pytest.raises(ValueError):
        get_model_by_name("not_a_real_model")


def test_with_two_class_head_swaps_predictor_to_correct_num_classes():
    # A minimal stand-in for a FasterRCNN model: only the attribute path
    # `.roi_heads.box_predictor.cls_score.in_features` is actually read.
    fake_cls_score = types.SimpleNamespace(in_features=1024)
    fake_box_predictor = types.SimpleNamespace(cls_score=fake_cls_score)
    fake_roi_heads = types.SimpleNamespace(box_predictor=fake_box_predictor)
    fake_model = types.SimpleNamespace(roi_heads=fake_roi_heads)

    result = model_module._with_two_class_head(fake_model)

    assert result.roi_heads.box_predictor.cls_score.out_features == model_module.NUM_CLASSES
    assert result.roi_heads.box_predictor.bbox_pred.out_features == model_module.NUM_CLASSES * 4


class _FakeDetectionModel(torch.nn.Module):
    """Stands in for a torchvision detection model: in train() mode it
    returns a loss dict instead of predictions, like FasterRCNN does."""

    def __init__(self):
        super().__init__()
        self.param = torch.nn.Parameter(torch.zeros(1))
        self.received_images = None
        self.received_targets = None

    def forward(self, images, targets):
        self.received_images = images
        self.received_targets = targets
        return {"loss_classifier": self.param.sum() + 1.0, "loss_box_reg": self.param.sum() + 2.0}


class _FakeOptimizer:
    def __init__(self, params):
        self.zero_grad_calls = 0
        self.step_calls = 0

    def zero_grad(self):
        self.zero_grad_calls += 1

    def step(self):
        self.step_calls += 1


def _fake_batch():
    images = [torch.zeros(3, 4, 4)]
    targets = [{"boxes": torch.zeros(0, 4), "labels": torch.zeros(0, dtype=torch.int64)}]
    return images, targets


def test_train_batch_sums_losses_and_steps_optimizer():
    model = _FakeDetectionModel()
    optimizer = _FakeOptimizer(model.parameters())

    loss, losses = model_module.train_batch(_fake_batch(), model, optimizer, device="cpu")

    assert loss.item() == pytest.approx(3.0)
    assert set(losses) == {"loss_classifier", "loss_box_reg"}
    assert optimizer.step_calls == 1
    assert optimizer.zero_grad_calls == 1


def test_validate_batch_does_not_touch_gradients_or_an_optimizer():
    model = _FakeDetectionModel()

    loss, losses = model_module.validate_batch(_fake_batch(), model, device="cpu")

    assert loss.item() == pytest.approx(3.0)
    assert not loss.requires_grad
