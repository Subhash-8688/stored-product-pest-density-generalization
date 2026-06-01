#!/usr/bin/env python3
"""Generate qualitative detection panels for chapter model variants."""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import cv2
import matplotlib.pyplot as plt
import numpy as np
import yaml
from ultralytics import YOLO

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
NAMES = {0: "TB", 1: "RZ", 2: "KP"}
COLORS = {0: (230, 57, 70), 1: (38, 126, 201), 2: (42, 157, 143)}


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=Path, default=Path("configs/experiments/models.example.yaml"))
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/detections"))
    parser.add_argument("--num-images", type=int, default=4)
    parser.add_argument("--seed", type=int, default=27)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.70)
    parser.add_argument("--device", default="0")
    return parser.parse_args()


def resolve_models(manifest: Path) -> dict[str, YOLO]:
    """Load model weights listed by the YAML manifest."""
    root = manifest.resolve().parents[2]
    config = yaml.safe_load(manifest.read_text())
    return {name: YOLO(str((root / values["weights"]).resolve())) for name, values in config["models"].items()}


def draw_boxes(image: np.ndarray, result) -> np.ndarray:
    """Draw predicted boxes and class labels."""
    canvas = image.copy()
    if result.boxes is None:
        return canvas
    for xyxy, cls_id, confidence in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.cls.cpu().numpy(), result.boxes.conf.cpu().numpy()):
        cls_id = int(cls_id)
        x1, y1, x2, y2 = np.round(xyxy).astype(int)
        color = COLORS.get(cls_id, (255, 255, 255))
        text = f"{NAMES.get(cls_id, cls_id)} {confidence:.2f}"
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        cv2.putText(canvas, text, (x1, max(14, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return canvas


def main() -> None:
    """Generate the panel."""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    models = resolve_models(args.models)
    paths = sorted(path for path in args.images.iterdir() if path.suffix.lower() in IMG_EXTS)
    selected = random.Random(args.seed).sample(paths, min(args.num_images, len(paths)))
    fig, axes = plt.subplots(len(selected), len(models) + 1, figsize=(2.7 * (len(models) + 1), 2.8 * len(selected)), squeeze=False)
    for row, path in enumerate(selected):
        bgr = cv2.imread(str(path))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        axes[row, 0].imshow(rgb)
        axes[row, 0].set_title("Image" if row == 0 else "")
        axes[row, 0].axis("off")
        for column, (name, model) in enumerate(models.items(), 1):
            result = model.predict(path, imgsz=args.imgsz, conf=args.conf, iou=args.iou, device=args.device, verbose=False)[0]
            axes[row, column].imshow(draw_boxes(rgb, result))
            axes[row, column].set_title(name if row == 0 else "")
            axes[row, column].axis("off")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.94, bottom=0.01, wspace=0.03, hspace=0.05)
    fig.savefig(args.output_dir / "detection_comparison.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
