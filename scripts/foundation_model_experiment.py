#!/usr/bin/env python3
"""Step 1 of docs/foundation_models.md: does a frozen self-supervised
foundation-model backbone (DINOv2) separate dumpsite/no_dumpsite images
better than a frozen ImageNet-pretrained ResNet50, on the whole-image
presence task (not full detection - that's step 2, gated on this result)?

Runs entirely on CPU: extracts frozen features for a balanced subset of the
real dataset in processed/, trains a linear probe (logistic regression) at
several training-set sizes for each feature type, and reports a
label-efficiency comparison.

    python scripts/foundation_model_experiment.py --num-images 300
"""

import argparse
import json
import random
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torchvision
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from transformers import AutoImageProcessor, AutoModel


def list_labeled_images(data_root: str, num_images: int, seed: int) -> Tuple[List[Path], List[int]]:
    """Balanced sample of (image_path, label) where label=1 means the
    matching annotation file is non-empty (a real dumpsite box present).

    The label always comes from the annotation content, not the filename
    prefix: a handful of `dumpsite_*.jpeg` files in the real dataset (10 out
    of 3395, found while running this experiment) turn out to have an empty
    annotation file - likely source XML files with zero `<object>` elements
    during the original VOC2012 conversion. Trusting the filename there
    would silently mislabel those as positives.
    """
    images_dir = Path(data_root) / "images"
    annotations_dir = Path(data_root) / "annotations"

    def has_box(path: Path) -> bool:
        annotation_path = annotations_dir / (path.stem + ".txt")
        return annotation_path.exists() and annotation_path.read_text().strip() != ""

    positives = [p for p in sorted(images_dir.glob("dumpsite_*.jpeg")) if has_box(p)]
    negatives = [p for p in sorted(images_dir.glob("no_dumpsite_*.jpeg")) if not has_box(p)]

    rng = random.Random(seed)
    rng.shuffle(positives)
    rng.shuffle(negatives)

    n_per_class = num_images // 2
    selected = positives[:n_per_class] + negatives[:n_per_class]
    labels = [1] * min(n_per_class, len(positives)) + [0] * min(n_per_class, len(negatives))

    return selected, labels


class DinoV2FeatureExtractor:
    name = "dinov2_small"

    def __init__(self):
        self.processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
        self.model = AutoModel.from_pretrained("facebook/dinov2-small").eval()

    @torch.no_grad()
    def extract(self, image: Image.Image) -> np.ndarray:
        inputs = self.processor(images=image, return_tensors="pt")
        outputs = self.model(**inputs)
        return outputs.last_hidden_state[:, 0].squeeze(0).numpy()  # CLS token


class ImageNetResNet50FeatureExtractor:
    name = "imagenet_resnet50"

    def __init__(self):
        weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
        self.model = torchvision.models.resnet50(weights=weights)
        self.model.fc = torch.nn.Identity()  # expose the 2048-d pooled features
        self.model.eval()
        self.transform = weights.transforms()

    @torch.no_grad()
    def extract(self, image: Image.Image) -> np.ndarray:
        tensor = self.transform(image).unsqueeze(0)
        return self.model(tensor).squeeze(0).numpy()


def extract_all_features(extractor, image_paths: List[Path]) -> np.ndarray:
    features = []
    for path in image_paths:
        image = Image.open(path).convert("RGB")
        features.append(extractor.extract(image))
    return np.stack(features)


def label_efficiency_curve(
    features: np.ndarray, labels: np.ndarray, fractions: List[float], seed: int
) -> List[dict]:
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        features, labels, test_size=0.3, random_state=seed, stratify=labels
    )

    results = []
    for fraction in fractions:
        if fraction < 1.0:
            X_train, _, y_train, _ = train_test_split(
                X_train_full, y_train_full, train_size=fraction, random_state=seed, stratify=y_train_full
            )
        else:
            X_train, y_train = X_train_full, y_train_full

        clf = LogisticRegression(max_iter=2000)
        clf.fit(X_train, y_train)
        accuracy = clf.score(X_test, y_test)
        results.append({"train_fraction": fraction, "num_train_samples": len(X_train), "test_accuracy": accuracy})

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", dest="data_root", type=str, default="processed")
    parser.add_argument("--num-images", dest="num_images", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--fractions", nargs="+", type=float, default=[0.1, 0.25, 0.5, 1.0], help="Fractions of the training split to probe at"
    )
    parser.add_argument("--output-json", dest="output_json", type=str, default="foundation_model_results.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Sampling {args.num_images} balanced images from {args.data_root} ...")
    image_paths, labels = list_labeled_images(args.data_root, args.num_images, args.seed)
    labels = np.array(labels)
    print(f"Got {len(image_paths)} images ({labels.sum()} positive, {(1 - labels).sum()} negative)")

    extractors = [ImageNetResNet50FeatureExtractor(), DinoV2FeatureExtractor()]

    all_results = {}
    for extractor in extractors:
        print(f"\nExtracting features with {extractor.name} ...")
        start = time.perf_counter()
        features = extract_all_features(extractor, image_paths)
        elapsed = time.perf_counter() - start
        print(f"  {len(image_paths)} images in {elapsed:.1f}s ({elapsed / len(image_paths) * 1000:.1f} ms/image), feature dim={features.shape[1]}")

        curve = label_efficiency_curve(features, labels, args.fractions, args.seed)
        all_results[extractor.name] = curve
        for point in curve:
            print(f"    train_fraction={point['train_fraction']:<5} n={point['num_train_samples']:<4} test_accuracy={point['test_accuracy']:.4f}")

    Path(args.output_json).write_text(json.dumps(all_results, indent=2))
    print(f"\nWrote {args.output_json}")


if __name__ == "__main__":
    main()
