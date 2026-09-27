#!/usr/bin/env python3
"""Benchmark trained YOLOv8/v11/RT-DETR checkpoints (ultralytics) the same
way benchmark.py does for the torchvision models: AP@0.5, per-image CPU/GPU
latency, and parameter count on the same test split, so results can be
compared directly against benchmark_results.json.

    python benchmark_ultralytics.py --checkpoints outputs/yolo/yolov8n/weights/best.pt \
        outputs/yolo/rtdetr-l/weights/best.pt --test-root processed/test
"""

import argparse
import json
import time
from pathlib import Path
from typing import List

from ultralytics import RTDETR, YOLO

from src.data import DumpsiteDataset
from src.metric import calculate_ap, calculate_precision_recall


def load_model(checkpoint_path: str):
    # ultralytics stores the architecture family in the checkpoint itself;
    # RTDETR and YOLO both load via their own class but expose the same API.
    try:
        return YOLO(checkpoint_path)
    except Exception:
        return RTDETR(checkpoint_path)


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.model.parameters())


def measure_latency_ms(model, image_path: str, num_warmup=5, num_runs=20) -> float:
    for _ in range(num_warmup):
        model.predict(image_path, verbose=False)
    start = time.perf_counter()
    for _ in range(num_runs):
        model.predict(image_path, verbose=False)
    elapsed = time.perf_counter() - start
    return (elapsed / num_runs) * 1000


def evaluate_accuracy(model, test_root: str) -> float:
    test_dataset = DumpsiteDataset(test_root)
    ap_values = []

    for image_path in test_dataset.image_files:
        annotation_path = Path(test_root) / "annotations" / (Path(image_path).stem + ".txt")
        gt_boxes = []
        if annotation_path.exists():
            for line in annotation_path.read_text().splitlines():
                parts = line.strip().split()
                if len(parts) == 5:
                    _label, x_min, y_min, x_max, y_max = map(float, parts)
                    gt_boxes.append([x_min, y_min, x_max, y_max])
        if not gt_boxes:
            continue

        result = model.predict(image_path, verbose=False)[0]
        pred_boxes = result.boxes.xyxy.cpu().numpy().tolist()

        precision, recall = calculate_precision_recall(pred_boxes, gt_boxes, iou_threshold=0.5)
        ap_values.append(calculate_ap(precision, recall).item())

    return sum(ap_values) / len(ap_values) if ap_values else float("nan")


def benchmark_checkpoint(checkpoint_path: str, test_root: str) -> dict:
    model = load_model(checkpoint_path)
    sample_image = next(iter(DumpsiteDataset(test_root).image_files))

    return {
        "checkpoint": checkpoint_path,
        "ap_at_0.5": evaluate_accuracy(model, test_root),
        "latency_ms": measure_latency_ms(model, sample_image),
        "num_params": (n := count_parameters(model)),
        "num_params_millions": round(n / 1e6, 2),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoints", nargs="+", required=True, help="Path(s) to trained .pt checkpoints")
    parser.add_argument("--test-root", dest="test_root", type=str, default="processed/test")
    parser.add_argument("--output-json", dest="output_json", type=str, default="benchmark_ultralytics_results.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = [benchmark_checkpoint(ckpt, args.test_root) for ckpt in args.checkpoints]

    header = f"{'Checkpoint':<45}{'AP@0.5':>10}{'Latency (ms)':>15}{'Params (M)':>13}"
    print(header)
    print("-" * len(header))
    for r in results:
        ap_str = f"{r['ap_at_0.5']:.4f}" if r["ap_at_0.5"] == r["ap_at_0.5"] else "n/a"
        print(f"{Path(r['checkpoint']).name:<45}{ap_str:>10}{r['latency_ms']:>15.2f}{r['num_params_millions']:>13.2f}")

    Path(args.output_json).write_text(json.dumps(results, indent=2))
    print(f"\nWrote {args.output_json}")


if __name__ == "__main__":
    main()
