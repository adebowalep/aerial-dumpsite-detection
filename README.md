# Illegal Dumpsite Detection

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/pytorch-2.x-orange)
![torchvision](https://img.shields.io/badge/torchvision-detection-red)
![Tests](https://img.shields.io/badge/tests-pytest-brightgreen)
![License](https://img.shields.io/badge/license-MIT-blue)

> A Faster R-CNN pipeline for detecting and localizing illegal waste dumpsites in aerial/satellite imagery, built on top of two public remote-sensing datasets.

---

## Summary

This project fine-tunes a torchvision Faster R-CNN (ResNet50-FPN backbone) to detect illegal waste dumpsites in overhead imagery, framed as a binary object-detection task: **dumpsite** vs **no_dumpsite**. It combines two public sources — the [Global Dumpsite Test Data](https://www.scidb.cn/en/s/6bq2M3) (dumpsite images with bounding-box annotations from cities including Colombo, Dhaka, Guwahati, Kinshasa, Lagos, and New Delhi) and [AerialWaste](https://aerialwaste.org/) (used here as the negative/"no_dumpsite" class) — into a single training set, then trains a transfer-learned detector on top of it.

The codebase separates dataset loading, model construction, evaluation metrics, and visualization into their own modules (`src/`). The original exploratory workflow lives in `waste_project.ipynb`; there is now also a config-driven CLI (`train.py` / `infer.py` / `benchmark.py`) for reproducible runs — see [CLI Usage](#cli-usage).

---

## Why This Project Matters

Illegal dumping is a persistent environmental and public-health problem that is expensive to monitor manually across large areas. Automating detection from aerial or satellite imagery makes it possible to:

- Flag candidate dumpsites over wide areas without manual review of every image.
- Prioritize enforcement/cleanup resources toward the highest-confidence detections.
- Track dumpsite emergence over time as new imagery becomes available.

This project is a first, working baseline toward that goal — a transfer-learned two-stage detector trained on real-world aerial data — with a clear roadmap toward a benchmarked, deployable system (see [Roadmap](#roadmap)).

---

## Key Features

- **Transfer learning on real aerial imagery.** Fine-tunes a COCO-pretrained Faster R-CNN backbone rather than training from scratch, appropriate for a dataset of this size.
- **Multiple backbone options already scaffolded.** `src/model.py` includes `get_resnet50()`, `get_resnet50_fpn_v2()`, and a lightweight `get_mobilenet_v3()` variant for future accuracy/latency comparisons.
- **Custom `Dataset` implementation.** `DumpsiteDataset` loads images and paired bounding-box annotation files, resizes to a common size, and produces a torchvision-detection-compatible `(image, target)` format with a custom `collate_fn` that filters unlabeled samples.
- **IoU / precision / recall / mAP metric utilities.** `src/metric.py` implements detection metrics from first principles for evaluating model output against ground truth.
- **Visualization helpers.** `src/visualize.py` draws predicted/ground-truth boxes on images and reports train/test class-balance statistics.

---

## Architecture Overview

```
Global Dumpsite Test Data  +  AerialWaste (negatives)
                │
                ▼
        processed/{images,annotations}
        processed/test/{images,annotations}
                │
                ▼
        DumpsiteDataset (src/data.py)
        ├── resize to common size, normalize to [0,1]
        ├── parse per-image annotation .txt → boxes, labels
        └── collate_fn (drops unlabeled samples)
                │
                ▼
        DataLoader (train / val split)
                │
                ▼
        Faster R-CNN (src/model.py)
        ├── ResNet50-FPN backbone (pretrained, transfer-learned)
        ├── Region Proposal Network
        └── RoI heads → box_predictor (2 classes: dumpsite / no_dumpsite)
                │
                ▼
        train_batch / validate_batch loop (per-epoch checkpointing)
                │
                ▼
        Inference → decode_output (NMS) → src/metric.py (IoU, precision/recall, mAP)
                │
                ▼
        src/visualize.py (bounding-box + class-distribution plots)
```

---

## Project Structure

```
waste-project/
├── src/
│   ├── data.py           # DumpsiteDataset: loading, resizing, annotation parsing, collate_fn
│   ├── model.py           # Model factories (Faster R-CNN, RetinaNet, FCOS) + train/validate batch fns
│   ├── metric.py           # IoU, precision/recall, AP, mAP implementations
│   ├── augmentation.py    # Bbox-aware Albumentations train/eval transforms
│   ├── config.py          # YAML-backed TrainConfig + CLI-override loading
│   ├── yolo_format.py     # Convert processed/ into ultralytics' YOLO format
│   └── visualize.py       # Bounding-box drawing, class-distribution plots, output decoding
├── train.py               # CLI training entry point (see CLI Usage)
├── infer.py               # CLI inference entry point
├── benchmark.py           # Cross-model accuracy/latency/param-count comparison (torchvision models)
├── train_yolo.py          # YOLOv8/v11/RT-DETR training via ultralytics (separate path)
├── benchmark_ultralytics.py   # Same comparison, for ultralytics checkpoints
├── export_onnx.py         # ONNX export + CPU latency benchmark
├── app/
│   └── main.py            # FastAPI service + Gradio UI (see Demo App)
├── configs/
│   └── default.yaml       # Default hyperparameters, overridable from the CLI
├── docs/
│   └── foundation_models.md   # Remote-sensing foundation-model exploration plan + real step-1 results
├── scripts/
│   └── foundation_model_experiment.py   # DINOv2 vs. ImageNet-ResNet50 label-efficiency probe
├── tests/                 # pytest suite (CPU-only, no pretrained weights needed)
│   ├── conftest.py
│   ├── test_data.py
│   ├── test_metric.py
│   ├── test_model.py
│   ├── test_config.py
│   ├── test_augmentation.py
│   ├── test_app.py
│   ├── test_yolo_format.py
│   └── test_foundation_model_experiment.py
├── .github/workflows/tests.yml   # Runs pytest on every push/PR
├── data/                  # Raw source datasets (gitignored — see Setup)
├── processed/             # Preprocessed train/test images + annotations (gitignored)
│   ├── images/
│   ├── annotations/
│   └── test/
├── waste_project.ipynb    # Original exploratory training/evaluation notebook
├── generate_data.ipynb    # Dataset preprocessing / generation notebook
├── requirements.txt
├── requirements-dev.txt   # requirements.txt + pytest/pytest-cov
├── pytest.ini
├── LICENSE
├── .gitignore
└── README.md
```

> **Note on data:** `data/` and `processed/` are gitignored — the raw imagery is large and belongs to third-party datasets with their own licensing/access terms. Download and prepare the source datasets yourself following the links above, and point `generate_data.ipynb` / `root_dir` at them.

---

## Workflow

1. **Data preparation.** `generate_data.ipynb` aggregates and reformats the two source datasets into `processed/{images,annotations}` (train) and `processed/test/{images,annotations}`.
2. **Dataset loading.** `DumpsiteDataset` (`src/data.py`) loads each image + its annotation file, resizes to a common resolution, and returns `(image_tensor, target_dict)` pairs; `train_test_split` via `random_split` produces train/val loaders.
3. **Model training.** `waste_project.ipynb` instantiates a Faster R-CNN (`src/model.py::get_model`), trains with SGD over a configurable number of epochs, logs per-batch losses (classifier, box regression, objectness, RPN box regression) via `torch_snippets.Report`, and checkpoints the model after every epoch.
4. **Evaluation.** The trained model is reloaded and run over `processed/test/`; predictions are decoded with NMS (`src/visualize.py::decode_output`) and compared against ground truth using `src/metric.py`.
5. **Visualization.** Predicted and ground-truth boxes are drawn on sample images, and class-distribution tables/plots summarize the train/test split.

---

## Technologies

- **Python 3.9+**
- **PyTorch / torchvision** — Faster R-CNN, `FastRCNNPredictor`, ResNet-FPN backbones, NMS
- **NumPy / Pillow / OpenCV** — image loading and preprocessing
- **Matplotlib / Seaborn** — visualization and class-distribution plots
- **torch-snippets / torchsummary** — training-loop logging and model summaries
- **Jupyter** — experiment notebooks

---

## Setup

```bash
# 1. Clone the repo
git clone <your-repo-url>
cd waste-project

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Obtain the source datasets and place them as data1/VOC2012 and data1/Aerial
#    (the paths generate_data.ipynb expects) — see "Data Provenance & Fixes" below:
#    - Global Dumpsite Test Data: https://www.scidb.cn/en/s/6bq2M3
#    - AerialWaste: https://aerialwaste.org/
#    Then run generate_data.ipynb to produce processed/{images,annotations}.
```

---

## How to Run

```bash
jupyter notebook waste_project.ipynb
```

Run the cells in order: dataset loading → class-distribution check → model instantiation → training loop → inference → evaluation → visualization.

---

## CLI Usage

Hyperparameters used to live hardcoded in notebook cells; they now live in `configs/default.yaml` and can be overridden per run without editing any file.

```bash
# Train (writes checkpoints + a copy of the resolved config to outputs/)
python train.py --config configs/default.yaml

# Override anything from the CLI
python train.py --config configs/default.yaml --model-name retinanet --epochs 20 --lr 0.001

# Run inference with a trained checkpoint
python infer.py --checkpoint outputs/resnet50_fpn_best.pth --image path/to/image.jpeg

# Compare trained checkpoints across architectures: AP@0.5, latency, param count
python benchmark.py --checkpoint-dir outputs --models resnet50_fpn retinanet fcos mobilenet_v3
```

`--model-name` accepts anything in `src.model.MODEL_REGISTRY`: `resnet50_fpn`, `resnet50_fpn_v2`, `mobilenet_v3`, `retinanet`, `fcos` (RetinaNet and FCOS are anchor-based/anchor-free single-stage alternatives to the two-stage Faster R-CNN variants — see [Roadmap](#roadmap)). Training uses bbox-aware Albumentations augmentation (flips, color jitter, mild scale jitter) by default; disable with `--no-augmentation`.

All three scripts were smoke-tested end-to-end in this repo against a tiny synthetic dataset (not committed) to verify the training/inference/benchmark plumbing itself is correct — they have **not** been run against the real dataset, which needs a GPU. Do that first on your training workspace before trusting any numbers out of them.

---

## YOLOv8/v11 & RT-DETR (via ultralytics)

A separate training path from `train.py`, since `ultralytics` owns its own full train/val/loss loop and data format rather than plugging into `src/model.py`'s torchvision-based training loop.

```bash
# Converts processed/ into YOLO format under yolo_data/ the first time, then trains
python train_yolo.py --data-root processed --model yolov8n.pt --epochs 50
python train_yolo.py --data-root processed --model yolo11n.pt --epochs 50 --skip-conversion
python train_yolo.py --data-root processed --model rtdetr-l.pt --epochs 50 --skip-conversion

# Compare against each other and (separately) against benchmark.py's results
python benchmark_ultralytics.py --checkpoints outputs/yolo/yolov8n/weights/best.pt outputs/yolo/rtdetr-l/weights/best.pt
```

`src/yolo_format.py` converts this project's `images/` + `annotations/` layout into YOLO's normalized-bbox format, correctly writing an **empty** label file for true negatives (not a fake full-image box — the same fix applied to the torchvision data pipeline). Verified end-to-end in this repo: converted the synthetic mini dataset, ran one real YOLOv8n training epoch (loss decreased, checkpoint saved) and confirmed RT-DETR loads via the identical `ultralytics` API — real plumbing, not just written-and-assumed-to-work. Not run against the real dataset or for enough epochs to produce a meaningful accuracy number; that's for the GPU workspace.

---

## Edge Export (ONNX)

```bash
python export_onnx.py --checkpoint outputs/mobilenet_v3_best.pth --model-name mobilenet_v3 \
    --output outputs/mobilenet_v3.onnx --image-size 1024
```

torchvision's Faster R-CNN/RetinaNet/FCOS models export to ONNX directly (opset ≥ 11 handles their internal anchor-generation/NMS ops) — verified working end-to-end in this repo: exported a trained checkpoint and ran it through `onnxruntime`'s CPU execution provider, latency printed automatically after export. **TFLite is not implemented** — converting a detection model with in-graph NMS through onnx → tf → tflite is a substantial separate effort (see [Roadmap](#roadmap)); ONNX Runtime already covers most CPU/edge deployment targets on its own.

---

## Demo App (FastAPI + Gradio)

```bash
CHECKPOINT_PATH=outputs/resnet50_fpn_best.pth MODEL_NAME=resnet50_fpn \
    uvicorn app.main:app --host 0.0.0.0 --port 8000
```

- **`GET /health`** — readiness check.
- **`POST /predict`** — upload an image, get back JSON detections (`{box, score, label}`), with an optional `score_threshold` query param.
- **`GET /demo`** — a Gradio upload UI (image in, annotated image + text summary out), mounted directly on the FastAPI app.

The model checkpoint is loaded lazily on first request, so the app starts and serves `/health` even before a checkpoint exists. Smoke-tested end-to-end (including a real bug this caught: `score_threshold or SCORE_THRESHOLD` silently discarded an explicit `0.0` threshold since `0.0` is falsy in Python — fixed, and pinned with a regression test in `tests/test_app.py`).

---

## Foundation-Model Experiment

`docs/foundation_models.md` lays out the reasoning and a staged plan; step 1 (a cheap go/no-go check, not full detection) has actually been run on a real 300-image subset of `processed/`:

```bash
python scripts/foundation_model_experiment.py --num-images 300
```

DINOv2-small (frozen) features beat ImageNet-ResNet50 features on a whole-image dumpsite/no_dumpsite linear probe at 3 of 4 training-set sizes, most clearly in the low-label regime (10%: 0.856 vs 0.811, 25%: 0.956 vs 0.911) — a real, if preliminary and single-seed, signal in favor of trying a foundation-model backbone for the actual detector. Full table and caveats in the doc. This also surfaced a genuine data-quality finding: 10 of 3,395 `dumpsite_*.jpeg` files have an empty annotation file, which the experiment script now correctly treats as a negative (trusting annotation content over filename).

---

## How to Test

```bash
pip install -r requirements-dev.txt
pytest -v
```

The suite (`tests/`) covers `src/metric.py` (IoU, precision/recall, AP — including a regression test for the false-negative bug described below), `src/data.py` (positive/negative sample loading, the zero-box true-negative fix, missing-annotation handling, collation), and the orchestration logic in `src/model.py` (`train_batch`/`validate_batch`, the model registry) using fake models/optimizers so it runs in seconds on CPU with no GPU or downloaded weights required. It runs automatically on every push via GitHub Actions (`.github/workflows/tests.yml`).

The model *factories* themselves (`get_model`, `get_resnet50`, …) aren't unit-tested since they require downloading pretrained ImageNet/COCO weights — verify those by actually training on the GPU workspace.

---

## Current Status

- ✅ End-to-end training pipeline runs: data loading → Faster R-CNN fine-tuning → checkpointing.
- ✅ Training/validation loss curves are logged and plotted per epoch.
- ✅ Test-set evaluation loop (precision/recall/AP@0.5, `src/metric.py`) is now wired up end-to-end in the notebook (see [Data Provenance & Fixes](#data-provenance--fixes) — it was previously blocked by a data bug, not a code bug).
- ⚠️ Only the ResNet50-FPN backbone has actually been trained; the MobileNetV3 and FPN-v2 variants are scaffolded but unevaluated.
- ⚠️ No numbers have been reported yet from an actual training run against the corrected data — that's the immediate next step.

---

## Data Provenance & Fixes

`generate_data.ipynb` builds `processed/` from two raw sources that are **not included in this repo** (third-party data, gitignored — see [Setup](#setup)):

- **Dumpsites (positive class):** a VOC2012-style annotated dataset (`data1/VOC2012`, XML annotations) — this is the "dumpsite" class in the [Global Dumpsite Test Data](https://www.scidb.cn/en/s/6bq2M3).
- **Non-dumpsites (negative class):** [AerialWaste](https://aerialwaste.org/) images (`data1/Aerial`) filtered by `is_candidate_location == 0`, plus a separate held-out `testing_2.json` split used to build `processed/test/`.

Auditing the generation notebook against the actual files in `processed/` turned up two real data bugs, now fixed:

1. **Negative images were mislabeled as an explicit "class 0" box, not zero boxes.** `save_images_as_yolo_no_dumpsite` wrote every `no_dumpsite` annotation as a fake full-image box (`0 0 0 1024 1024`). torchvision's Faster R-CNN reserves class `0` for *implicit* background — giving it an explicit labeled box is non-standard and biases training. **Fixed**: negative annotation files are now empty (zero ground-truth boxes), `DumpsiteDataset` (`src/data.py`) was updated to keep zero-box images as true negatives instead of silently dropping them, and the 3,395 existing `no_dumpsite_*.txt` files in `processed/annotations/` were rewritten to match (original files preserved in `processed/annotations_original_backup/`).
2. **Test-set boxes were corrupted by a COCO bbox-format mismatch.** `save_testing_data` did `xmin, ymin, xmax, ymax = bbox`, but COCO's `bbox` field is `[x, y, width, height]`, not corner coordinates — every one of the 20 `processed/test/annotations/*.txt` files had `x_max < x_min` and `y_max < y_min`, so IoU against ground truth was always `0`. This is almost certainly why the notebook's evaluation cell was abandoned mid-way in the original version. **Fixed**: the generator now does `xmax, ymax = x + w, y + h`; the existing test annotations were repaired in place using the same arithmetic (the underlying width/height values were still intact, just mislabeled — original files preserved in `processed/test_original_backup/`).

Both backup folders are gitignored (they live under `processed/`) — delete them once you've spot-checked the fix, or keep them for reference.

---

## Roadmap

- [x] Add `pytest` unit tests for `src/metric.py`, `src/data.py`, and the `src/model.py` orchestration logic.
- [x] Add a CLI training/inference entry point and a config file (`train.py`, `infer.py`, `configs/default.yaml`) instead of hardcoded notebook hyperparameters.
- [x] Add bbox-aware data augmentation (flips, color jitter, scale) via Albumentations (`src/augmentation.py`).
- [x] Add RetinaNet and FCOS (anchor-based/anchor-free single-stage) alongside the Faster R-CNN variants, plus a `benchmark.py` harness for accuracy/latency/param-count comparison — **not yet run against real trained checkpoints**, see [CLI Usage](#cli-usage).
- [x] Export the best small model to ONNX and benchmark CPU inference latency (`export_onnx.py`) — TFLite deliberately deferred, see below.
- [x] Ship a small FastAPI + Gradio demo for uploading an image and viewing detections (`app/main.py`).
- [x] Extend the benchmark to YOLOv8/v11 and RT-DETR (via `ultralytics`) — `train_yolo.py` + `benchmark_ultralytics.py`, a separate training path from the torchvision models above. Plumbing verified (real 1-epoch YOLOv8n training run, RT-DETR load check); **no real accuracy numbers yet**.
- [x] Execute step 1 of the foundation-model experiment plan (`docs/foundation_models.md`) and report label-efficiency results — real run on 300 real images, DINOv2 leads ImageNet-ResNet50 especially in the low-label regime. Step 2 (full detection integration) still open.
- [ ] **Re-run training end-to-end (on a GPU workspace) against the corrected data and report actual precision/recall/AP@0.5 numbers on held-out data for each model — nothing above has real detection accuracy numbers yet.**
- [ ] Convert the ONNX export to TFLite for mobile/embedded targets (in-graph NMS makes onnx → tf → tflite non-trivial for detection models — treat as its own effort, deferred until a real trained checkpoint exists to convert).
- [ ] Step 2 of the foundation-model plan: wrap a DINOv2 (or remote-sensing-specific) backbone into a torchvision-compatible detector and re-run `benchmark.py` against it.
- [ ] Containerize the demo app (Dockerfile) and deploy it somewhere reachable by a URL (HuggingFace Spaces, Render, Fly.io).
