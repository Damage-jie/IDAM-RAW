"""Train IDAM-RAW with the controlled protocol described in the paper."""

from __future__ import annotations

import argparse

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train IDAM-RAW.")
    parser.add_argument("--model", default="configs/models/idam_raw_yolov8s.yaml")
    parser.add_argument("--data", default="configs/datasets/cr7_raw.yaml")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0", help="Device ID(s), e.g. 0 or 0,1,2,3.")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2024, choices=(2024, 2025, 2026))
    parser.add_argument("--project", default="runs/idam_raw")
    parser.add_argument("--name", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        optimizer="AdamW",
        lr0=1e-3,
        cos_lr=True,
        pretrained=False,
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
        fliplr=0.5,
        seed=args.seed,
        deterministic=True,
        project=args.project,
        name=args.name or f"idam_raw_seed{args.seed}",
    )


if __name__ == "__main__":
    main()
