#!/usr/bin/env python3
"""Evaluate a trained model on one independent test domain."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    """Run validation and report accuracy and throughput metrics."""
    args = parse_args()
    metrics = YOLO(str(args.weights)).val(
        data=str(args.data),
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
    )
    speed = metrics.speed
    total_ms = speed["preprocess"] + speed["inference"] + speed["postprocess"]
    report = {
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "mAP50": float(metrics.box.map50),
        "mAP50-95": float(metrics.box.map),
        "classwise_precision": [float(value) for value in metrics.box.p],
        "speed_ms_per_image": {key: float(value) for key, value in speed.items()},
        "inference_fps": float(1000.0 / speed["inference"]),
        "end_to_end_fps": float(1000.0 / total_ms),
    }
    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
