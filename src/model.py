"""Faster R-CNN model factories and per-batch train/validate steps.

Several backbone variants are provided so accuracy/latency can be compared
across them (see the project README roadmap) — they all share the same
two-class (background, dumpsite) head.
"""

import warnings
from typing import Dict, List, Tuple

import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.fcos import FCOSClassificationHead
from torchvision.models.detection.retinanet import RetinaNetClassificationHead

# torchvision emits a benign UserWarning about internal API usage on some versions.
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.models._utils")

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 2  # 0 = background (implicit), 1 = dumpsite


def _with_two_class_head(model: FasterRCNN) -> FasterRCNN:
    """Swap in a fresh box predictor sized for (background, dumpsite)."""
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
    return model


def get_model() -> FasterRCNN:
    """Faster R-CNN with a pretrained ResNet50-FPN backbone, built from scratch."""
    backbone = resnet_fpn_backbone("resnet50", pretrained=True)
    return FasterRCNN(backbone, num_classes=NUM_CLASSES)


def get_resnet50() -> FasterRCNN:
    """Faster R-CNN starting from the full COCO-pretrained resnet50_fpn model."""
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    return _with_two_class_head(model)


def get_resnet50_fpn_v2() -> FasterRCNN:
    """Faster R-CNN starting from the improved resnet50_fpn_v2 recipe."""
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(pretrained=True)
    return _with_two_class_head(model)


def get_mobilenet_v3() -> FasterRCNN:
    """Lighter-weight Faster R-CNN (MobileNetV3-Large FPN) for edge/latency comparisons."""
    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)
    return _with_two_class_head(model)


def get_retinanet():
    """RetinaNet (ResNet50-FPN): anchor-based single-stage detector, for
    comparison against the two-stage Faster R-CNN family above."""
    model = torchvision.models.detection.retinanet_resnet50_fpn(pretrained=True)
    in_channels = model.head.classification_head.conv[0].in_channels
    num_anchors = model.head.classification_head.num_anchors
    model.head.classification_head = RetinaNetClassificationHead(in_channels, num_anchors, NUM_CLASSES)
    return model


def get_fcos():
    """FCOS (ResNet50-FPN): anchor-free single-stage detector."""
    model = torchvision.models.detection.fcos_resnet50_fpn(pretrained=True)
    in_channels = model.head.classification_head.conv[0].in_channels
    num_anchors = model.head.classification_head.num_anchors
    model.head.classification_head = FCOSClassificationHead(in_channels, num_anchors, NUM_CLASSES)
    return model


MODEL_REGISTRY = {
    "resnet50_scratch_head": get_model,
    "resnet50_fpn": get_resnet50,
    "resnet50_fpn_v2": get_resnet50_fpn_v2,
    "mobilenet_v3": get_mobilenet_v3,
    "retinanet": get_retinanet,
    "fcos": get_fcos,
}


def get_model_by_name(name: str) -> FasterRCNN:
    """Look up a model factory by name, e.g. for a config-driven training script."""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Options: {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name]()


Batch = Tuple[List[torch.Tensor], List[Dict[str, torch.Tensor]]]


def train_batch(
    inputs: Batch, model: FasterRCNN, optimizer: torch.optim.Optimizer, device: str = DEFAULT_DEVICE
):
    """Run one training step; returns (total_loss, per-component loss dict)."""
    model.train()
    images, targets = inputs
    images = [image.to(device) for image in images]
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

    optimizer.zero_grad()
    losses = model(images, targets)
    loss = sum(losses.values())
    loss.backward()
    optimizer.step()
    return loss, losses


@torch.no_grad()
def validate_batch(inputs: Batch, model: FasterRCNN, device: str = DEFAULT_DEVICE):
    """Run one validation step (loss only, no gradient update).

    Faster R-CNN only returns a loss dict in ``model.train()`` mode, so this
    intentionally still calls ``model.train()`` under ``torch.no_grad()`` to
    get comparable loss numbers without updating any weights.
    """
    model.train()
    images, targets = inputs
    images = [image.to(device) for image in images]
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

    losses = model(images, targets)
    loss = sum(losses.values())
    return loss, losses
