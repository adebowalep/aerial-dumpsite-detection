# Remote-Sensing Foundation Models: Exploration Notes

This is a research direction, not a finished feature. Step 1 of the
experiment plan below (a cheap go/no-go sanity check) has now actually
been run, for real, on a subset of this project's real dataset — see
[Step 1 Results](#step-1-results-real-run) below. Step 2 (full detection
integration) has **not** been run and needs the GPU workspace.

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

1. ~~Freeze a pretrained DINOv2 ViT-S/14, extract patch features for the
   train/val split, train a **linear** classifier (single dumpsite/no_dumpsite
   presence label per image, not full detection) as the cheapest possible
   signal of "does this representation separate the two classes better
   than ImageNet features do."~~ **Done — see results below.**
2. Since (1) shows a real, if modest, lead for DINOv2 — invest in wrapping
   the ViT backbone into `torchvision.models.detection.FasterRCNN` (via a
   custom `BackboneWithFPN`-compatible module) and re-run the full
   benchmark in `benchmark.py` against the existing model variants. **Not
   yet done** — needs the GPU workspace and real detection training, not
   just a linear probe.
3. Report AP@0.5 and **label-efficiency** curves (AP vs. % of training
   data used) for the actual detection task once (2) is done — the
   whole-image presence task in step 1 is a cheap proxy, not the real
   claim.

## Step 1 Results (real run)

Run via `python scripts/foundation_model_experiment.py --num-images 300`
against a balanced 300-image subset of the real `processed/` dataset
(150 dumpsite, 150 no_dumpsite, whole-image presence label, not full
detection), on CPU, no GPU used:

| Train fraction | n (train) | ImageNet-ResNet50 accuracy | DINOv2-small accuracy |
|---|---|---|---|
| 10% | 21 | 0.8111 | **0.8556** |
| 25% | 52 | 0.9111 | **0.9556** |
| 50% | 105 | **0.9556** | 0.9444 |
| 100% | 210 | 0.9667 | **0.9778** |

(Held-out test set: 90 images, stratified, single seed. Full numbers in
`foundation_model_results.json`.)

**Reading this**: DINOv2 features lead at 3 of 4 training sizes, most
clearly in the low-label regime (10%, 25%) — exactly the "fewer labels
needed" hypothesis this experiment was designed to test. The 50% point
where ImageNet edges ahead, and the generally small gaps throughout, are
a reminder this is a single run at n=300 with one train/test split, not
a statistically rigorous result — worth re-running with multiple seeds
and a larger sample before treating the lead as settled. It's a
promising enough signal to justify step 2, not a proof that step 2 will
pay off.

Also worth noting: building this experiment surfaced a real, previously
unknown **data-quality issue** — 10 of the 3,395 `dumpsite_*.jpeg` files
have an *empty* annotation file (likely source VOC2012 XML files with
zero `<object>` elements during the original conversion). The experiment
script trusts annotation content over filename prefix for exactly this
reason (see `list_labeled_images` in `scripts/foundation_model_experiment.py`
and its regression test). This doesn't affect the main detection
pipeline's correctness (Faster R-CNN et al. read the same annotation
files directly, so those 10 images correctly train as if they were
negatives) but is worth knowing about if you inspect the dataset by
filename.

## Reproducing / extending this experiment

```bash
python scripts/foundation_model_experiment.py --num-images 300
# larger sample, more fractions:
python scripts/foundation_model_experiment.py --num-images 1000 --fractions 0.05 0.1 0.25 0.5 1.0
```

`scripts/foundation_model_experiment.py` extracts frozen DINOv2-small (via
`transformers`) and ImageNet-ResNet50 (via `torchvision`) features for a
balanced sample, then trains an `sklearn` `LogisticRegression` at each
training-set fraction and reports test accuracy — all CPU-only, no GPU
needed for this linear-probe stage (only a full detection-head
integration, step 2 above, would benefit from one).

## What this is *not*

The step 1 result above is a real, if modest and preliminary, positive
signal — not proof that a foundation-model backbone will beat the
current ImageNet-pretrained detector once wired into the full detection
pipeline. Faster R-CNN's FPN already does reasonably well with limited
data, and a whole-image linear probe is a much easier task than
bounding-box regression. Treat step 2 as the real test.
