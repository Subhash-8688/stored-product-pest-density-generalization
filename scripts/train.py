#!/usr/bin/env python3
"""Train a YOLOv11s chapter experiment with reproducible defaults."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolo11s.pt", help="Ultralytics checkpoint or model YAML.")
    parser.add_argument("--pretrained", help="Optional checkpoint loaded into a YAML-defined model.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--project", type=Path, default=Path("outputs/runs/train"))
    parser.add_argument("--name", required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bbox-loss", choices=["ciou", "eiou", "focal_eiou"], default="ciou")
    parser.add_argument("--focal-eiou-gamma", type=float, default=2.0)
    return parser.parse_args()


def main() -> None:
    """Train a model variant."""
    args = parse_args()
    model = YOLO(args.model)
    if args.pretrained:
        model.load(args.pretrained)
    model.train(
        data=str(args.data),
        project=str(args.project),
        name=args.name,
        device=args.device,
        epochs=args.epochs,
        patience=30,
        batch=args.batch,
        imgsz=args.imgsz,
        workers=args.workers,
        seed=args.seed,
        deterministic=True,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        cos_lr=True,
        close_mosaic=10,
        amp=True,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        hsv_h=0.02,
        hsv_s=0.5,
        hsv_v=0.4,
        degrees=5.0,
        translate=0.08,
        scale=0.2,
        shear=2.0,
        perspective=0.0005,
        fliplr=0.5,
        mosaic=0.3,
        mixup=0.1,
        auto_augment="randaugment",
        bbox_loss=args.bbox_loss,
        focal_eiou_gamma=args.focal_eiou_gamma,
    )


if __name__ == "__main__":
    main()
