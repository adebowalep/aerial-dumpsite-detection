#!/usr/bin/env python3
"""Export a trained detection checkpoint to ONNX and benchmark CPU latency.

    python export_onnx.py --checkpoint outputs/mobilenet_v3_best.pth \
        --model-name mobilenet_v3 --output outputs/mobilenet_v3.onnx

torchvision's Faster R-CNN/RetinaNet/FCOS models export to ONNX directly
(opset >= 11 handles their internal NMS/anchor-generation ops) - no need for
a separate tracing wrapper. TFLite is NOT covered here: converting a
detection model with in-graph NMS through onnx -> tf -> tflite is its own
substantial effort (see README roadmap) and is not attempted by this script.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import onnxruntime
import torch

from src.model import get_model_by_name


def export(checkpoint_path: str, model_name: str, output_path: str, image_size: int) -> None:
    model = get_model_by_name(model_name)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()

    dummy_input = [torch.rand(3, image_size, image_size)]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        (dummy_input,),
        output_path,
        input_names=["images"],
        output_names=["boxes", "labels", "scores"],
        dynamic_axes={"images": {0: "num_images"}},
        opset_version=11,
    )
    print(f"Exported to {output_path}")


def benchmark_onnx_latency(onnx_path: str, image_size: int, num_warmup=5, num_runs=20) -> float:
    session = onnxruntime.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    dummy_input = np.random.rand(3, image_size, image_size).astype(np.float32)
    input_name = session.get_inputs()[0].name

    for _ in range(num_warmup):
        session.run(None, {input_name: dummy_input})

    start = time.perf_counter()
    for _ in range(num_runs):
        session.run(None, {input_name: dummy_input})
    elapsed = time.perf_counter() - start

    return (elapsed / num_runs) * 1000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--model-name", dest="model_name", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--image-size", dest="image_size", type=int, default=1024)
    parser.add_argument("--skip-benchmark", dest="skip_benchmark", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    export(args.checkpoint, args.model_name, args.output, args.image_size)

    if not args.skip_benchmark:
        latency_ms = benchmark_onnx_latency(args.output, args.image_size)
        print(f"ONNX Runtime CPU latency: {latency_ms:.2f} ms/image (image_size={args.image_size})")


if __name__ == "__main__":
    main()
