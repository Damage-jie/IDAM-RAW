"""Evaluate an IDAM-RAW checkpoint with COCO detection metrics."""

from __future__ import annotations

import argparse

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate IDAM-RAW.")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", default="configs/datasets/cr7_raw.yaml")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    YOLO(args.weights).val(
        data=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
