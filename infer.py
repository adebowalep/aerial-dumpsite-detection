#!/usr/bin/env python3
"""CLI inference entry point.

    python infer.py --checkpoint outputs/resnet50_fpn_best.pth --image path/to/image.jpeg
    python infer.py --checkpoint outputs/resnet50_fpn_best.pth --image-dir path/to/images/ --output-dir results/
"""

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import torch
from PIL import Image

from src.model import get_model_by_name

TARGET_TO_LABEL = {0: "no_dumpsite", 1: "dumpsite"}


def load_model(checkpoint_path: str, model_name: str, device: str):
    model = get_model_by_name(model_name).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model


def predict_image(model, image_path: str, device: str, score_threshold: float = 0.5) -> dict:
    image = Image.open(image_path).convert("RGB")
    array = np.array(image).astype(np.float32) / 255.0
    tensor = torch.tensor(array).permute(2, 0, 1).to(device)

    with torch.no_grad():
        output = model([tensor])[0]

    boxes = output["boxes"].cpu().numpy()
    scores = output["scores"].cpu().numpy()
    labels = output["labels"].cpu().numpy()

    keep = scores >= score_threshold
    detections = [
        {
            "box": boxes[i].tolist(),
            "score": float(scores[i]),
            "label": TARGET_TO_LABEL.get(int(labels[i]), str(labels[i])),
        }
        for i in np.nonzero(keep)[0]
    ]
    return {"image": str(image_path), "detections": detections}


def iter_image_paths(image: str = None, image_dir: str = None) -> List[str]:
    if image:
        return [image]
    paths = sorted(str(p) for p in Path(image_dir).glob("*.jpeg"))
    if not paths:
        raise FileNotFoundError(f"No .jpeg images found in {image_dir}")
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--model-name", dest="model_name", type=str, default="resnet50_fpn")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", type=str, default=None)
    group.add_argument("--image-dir", dest="image_dir", type=str, default=None)
    parser.add_argument("--score-threshold", dest="score_threshold", type=float, default=0.5)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-json", dest="output_json", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(args.checkpoint, args.model_name, device)
    image_paths = iter_image_paths(args.image, args.image_dir)

    results = [predict_image(model, path, device, args.score_threshold) for path in image_paths]

    for result in results:
        print(f"{result['image']}: {len(result['detections'])} detection(s)")
        for det in result["detections"]:
            print(f"  {det['label']} @ {det['score']:.2f} -> {det['box']}")

    if args.output_json:
        Path(args.output_json).write_text(json.dumps(results, indent=2))
        print(f"Wrote results to {args.output_json}")


if __name__ == "__main__":
    main()
