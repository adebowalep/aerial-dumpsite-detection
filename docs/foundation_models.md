# Remote-Sensing Foundation Models: Exploration Notes

This is a research direction, not a finished feature — it needs real
experimentation on the GPU workspace to know whether it actually helps.
This document lays out *why* it's worth trying, *which* models are the
best candidates, and *how* to run a fair first experiment.

## Why bother

The current pipeline fine-tunes an **ImageNet-pretrained** backbone
(ResNet50/MobileNetV3) on ~6.8k images, most of them small single-object
crops. ImageNet pretraining teaches ground-level, object-centric visual
priors (textures, shapes, natural-photo statistics) that transfer
imperfectly to nadir aerial/satellite imagery, which has different scale,
viewpoint, and color statistics. With labeled dumpsite data this scarce,
starting from a backbone that already understands *overhead* imagery
should need fewer labeled examples to reach the same accuracy — the
standard argument for self-supervised/foundation-model pretraining in a
low-label regime.

## Candidate approaches, ranked by effort-to-try

1. **SSL4EO / SatMAE / Prithvi (remote-sensing-specific self-supervised backbones).**
   Pretrained on millions of unlabeled satellite images (Sentinel-2,
   Landsat) with masked-autoencoder or contrastive objectives. Prithvi
   (IBM/NASA, on HuggingFace) and SatMAE both ship ViT backbones with
   loadable weights. Highest expected payoff since the pretraining domain
   matches the task domain most closely, but these are ViT-based, so
   plugging one in as a torchvision Faster R-CNN backbone needs a
   FPN-compatible feature-pyramid wrapper (more integration work than the
   other options).
2. **DINOv2 (general-purpose self-supervised ViT) as a frozen feature extractor.**
   Not remote-sensing-specific, but DINOv2 features are known to transfer
   well with just a linear probe or a small detection head on top,
   without any backbone fine-tuning — cheap to try first as a sanity
   check on whether self-supervised features help at all before investing
   in a remote-sensing-specific backbone.
3. **CLIP zero-shot / few-shot scoring** as an *auxiliary* signal (e.g.,
   re-ranking Faster R-CNN region proposals by CLIP similarity to a
   "waste dumpsite" text prompt), rather than a full detector replacement.
   Lowest integration effort, unlikely to be the best final answer, but a
   useful one-afternoon experiment.

**Recommendation: start with (2)** (fastest to validate the hypothesis
cheaply) and move to (1) only if it shows a real lift — don't build the
harder integration before knowing self-supervised features help at all
on this data.

## Suggested experiment plan

1. Freeze a pretrained DINOv2 ViT-S/14, extract patch features for the
   train/val split, train a **linear** classifier (single dumpsite/no_dumpsite
   presence label per image, not full detection) as the cheapest possible
   signal of "does this representation separate the two classes better
   than ImageNet features do." This needs no detection-specific code.
2. If (1) shows a clear separation improvement over an ImageNet-ResNet50
   baseline doing the same linear-probe task, invest in wrapping the ViT
   backbone into `torchvision.models.detection.FasterRCNN` (via a custom
   `BackboneWithFPN`-compatible module) and re-run the full benchmark in
   `benchmark.py` against the existing model variants.
3. Report AP@0.5 and **label-efficiency** curves (AP vs. % of training
   data used) — the actual claim of a foundation-model backbone is "fewer
   labels needed for the same accuracy," so that comparison matters more
   than a single full-data AP number.

## Minimal starting stub (feature extraction only, no detection head yet)

Not wired into the rest of the codebase yet — a starting point for step 1
above, meant to run standalone on the GPU workspace.

```python
import torch
from PIL import Image
from torchvision import transforms

# pip install: transformers (for DINOv2) or timm
from transformers import AutoImageProcessor, AutoModel

processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
model = AutoModel.from_pretrained("facebook/dinov2-small").eval()

def extract_features(image_path: str) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0]  # CLS token, shape (1, hidden_dim)
```

Run this over `processed/images/*.jpeg`, cache the resulting feature
vectors, and train `sklearn.linear_model.LogisticRegression` on top as
the cheapest possible probe — no GPU needed even, this could run on CPU
for the linear-probe stage since only the frozen backbone forward pass
benefits from a GPU.

## What this is *not*

This is not a promise that foundation-model pretraining will beat the
current ImageNet-pretrained baseline — it might not, especially since
Faster R-CNN's FPN already does reasonably well with limited data. Treat
step 1 above as a go/no-go gate before investing further.
