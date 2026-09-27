#!/usr/bin/env python3
"""Train YOLOv8/v11 or RT-DETR via ultralytics on this project's dataset.

    python train_yolo.py --data-root processed --model yolov8n.pt --epochs 50
    python train_yolo.py --data-root processed --model yolo11n.pt --epochs 50
    python train_yolo.py --data-root processed --model rtdetr-l.pt --epochs 50

This is a SEPARATE training path from train.py: ultralytics owns its own
full train/val/loss loop and data format, unlike the torchvision models in
src/model.py. src/yolo_format.py converts processed/ (this project's
image+annotation-file format) into the YOLO images/+labels/ layout the
first time; pass --skip-conversion to reuse an existing converted copy.
"""

import argparse
from pathlib import Path

from ultralytics import RTDETR, YOLO

from src.yolo_format import convert_dataset


def build_model(model_name: str):
    if model_name.startswith("rtdetr"):
        return RTDETR(model_name)
    return YOLO(model_name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", dest="data_root", type=str, default="processed")
    parser.add_argument("--yolo-data-dir", dest="yolo_data_dir", type=str, default="yolo_data")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="e.g. yolov8n.pt, yolo11n.pt, rtdetr-l.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", type=str, default=None, help="e.g. 0 for first GPU, or cpu; omit for ultralytics' own auto-detect")
    parser.add_argument("--project", type=str, default="outputs/yolo")
    parser.add_argument(
        "--skip-conversion",
        dest="skip_conversion",
        action="store_true",
        help="Reuse an already-converted YOLO-format dataset at --yolo-data-dir",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset_yaml_path = Path(args.yolo_data_dir) / "dataset.yaml"
    if args.skip_conversion and dataset_yaml_path.exists():
        dataset_yaml = str(dataset_yaml_path)
    else:
        print(f"Converting {args.data_root} -> YOLO format at {args.yolo_data_dir} ...")
        dataset_yaml = convert_dataset(args.data_root, args.yolo_data_dir)

    model = build_model(args.model)
    train_kwargs = dict(
        data=dataset_yaml,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.model.replace(".pt", ""),
    )
    if args.device is not None:
        train_kwargs["device"] = args.device

    model.train(**train_kwargs)


if __name__ == "__main__":
    main()
