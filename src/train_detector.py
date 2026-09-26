# Code written by Rajesh Kumar
# Cambridge Institute of Technology
# Food Detection and Calorie Estimation System

"""Fine-tune a YOLOv8n detector on the Indian food dataset."""

import argparse
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default=str(ROOT / "data" / "data.yaml"))
    p.add_argument("--model", default="yolov8n.pt")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default="mps")
    p.add_argument("--project", default=str(ROOT / "models" / "runs"))
    p.add_argument("--name", default="yolo_food")
    p.add_argument(
        "--amp",
        action="store_true",
        help="Enable automatic mixed precision (unstable on MPS as of ultralytics "
        "8.4 / torch 2.8 - causes loss to go NaN partway through epoch 1). "
        "Off by default.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        amp=args.amp,
    )


if __name__ == "__main__":
    main()
