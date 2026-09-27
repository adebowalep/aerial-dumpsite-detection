#!/usr/bin/env python3
"""Benchmark trained checkpoints across model architectures: accuracy (AP@0.5),
CPU/GPU latency, and parameter count, on the same test split.

This does NOT train anything — it expects one checkpoint per model, already
trained with train.py (e.g. on the Udacity GPU workspace). Point it at a
directory of checkpoints named `<model_name>_best.pth`:

    python benchmark.py --checkpoint-dir outputs --models resnet50_fpn retinanet fcos

Results are printed as a table and written to `benchmark_results.json`.
"""

import argparse
import json
import time
from pathlib import Path
from typing import List

import torch
from torch.utils.data import DataLoader

from src.data import DumpsiteDataset
from src.metric import calculate_ap, calculate_precision_recall
from src.model import MODEL_REGISTRY, get_model_by_name
from src.visualize import decode_output

TARGET_TO_LABEL = {0: "no_dumpsite", 1: "dumpsite"}


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters())


def measure_latency_ms(model, sample_image: torch.Tensor, device: str, num_warmup=5, num_runs=20) -> float:
    model.eval()
    image = sample_image.to(device)
    with torch.no_grad():
        for _ in range(num_warmup):
            model([image])
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(num_runs):
            model([image])
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
    return (elapsed / num_runs) * 1000


def evaluate_accuracy(model, test_loader, device: str) -> float:
    model.eval()
    ap_values = []
    for images, targets in test_loader:
        images_on_device = [im.to(device) for im in images]
        with torch.no_grad():
            outputs = model(images_on_device)
        for output, target in zip(outputs, targets):
            bbs, confs, labels = decode_output(output, TARGET_TO_LABEL)
            gt_boxes = target["boxes"].cpu().numpy().tolist()
            if not gt_boxes:
                continue
            precision, recall = calculate_precision_recall(bbs, gt_boxes, iou_threshold=0.5)
            ap_values.append(calculate_ap(precision, recall).item())
    return sum(ap_values) / len(ap_values) if ap_values else float("nan")


def benchmark_model(name: str, checkpoint_path: str, test_loader, device: str) -> dict:
    model = get_model_by_name(name).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    sample_image, _ = next(iter(test_loader.dataset))
    latency_ms = measure_latency_ms(model, sample_image, device)
    ap50 = evaluate_accuracy(model, test_loader, device)
    num_params = count_parameters(model)

    return {
        "model": name,
        "ap_at_0.5": ap50,
        "latency_ms": latency_ms,
        "num_params": num_params,
        "num_params_millions": round(num_params / 1e6, 2),
    }


def print_table(results: List[dict]) -> None:
    header = f"{'Model':<20}{'AP@0.5':>10}{'Latency (ms)':>15}{'Params (M)':>13}"
    print(header)
    print("-" * len(header))
    for r in results:
        ap_str = f"{r['ap_at_0.5']:.4f}" if r["ap_at_0.5"] == r["ap_at_0.5"] else "n/a"
        print(f"{r['model']:<20}{ap_str:>10}{r['latency_ms']:>15.2f}{r['num_params_millions']:>13.2f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint-dir", dest="checkpoint_dir", type=str, default="outputs")
    parser.add_argument("--models", nargs="+", default=list(MODEL_REGISTRY), choices=list(MODEL_REGISTRY))
    parser.add_argument("--test-root", dest="test_root", type=str, default="processed/test")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-json", dest="output_json", type=str, default="benchmark_results.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    test_dataset = DumpsiteDataset(args.test_root)
    test_loader = DataLoader(test_dataset, batch_size=4, collate_fn=DumpsiteDataset.collate_fn)

    results = []
    for name in args.models:
        checkpoint_path = Path(args.checkpoint_dir) / f"{name}_best.pth"
        if not checkpoint_path.exists():
            print(f"Skipping '{name}': no checkpoint at {checkpoint_path}")
            continue
        print(f"Benchmarking {name}...")
        results.append(benchmark_model(name, str(checkpoint_path), test_loader, device))

    if not results:
        print("No checkpoints found - train models first with train.py.")
        return

    print()
    print_table(results)
    Path(args.output_json).write_text(json.dumps(results, indent=2))
    print(f"\nWrote {args.output_json}")


if __name__ == "__main__":
    main()
