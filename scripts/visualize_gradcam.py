#!/usr/bin/env python3
"""Generate Grad-CAM panels for chapter model variants."""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from ultralytics import YOLO

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=Path, default=Path("configs/experiments/models.example.yaml"))
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/gradcam"))
    parser.add_argument("--num-images", type=int, default=4)
    parser.add_argument("--seed", type=int, default=27)
    parser.add_argument("--imgsz", type=int, default=640)
    return parser.parse_args()


def resolve_models(manifest: Path) -> dict:
    """Load model weights and target layers from YAML."""
    root = manifest.resolve().parents[2]
    config = yaml.safe_load(manifest.read_text())
    return {
        name: {"weights": (root / values["weights"]).resolve(), "target_layer": int(values["target_layer"])}
        for name, values in config["models"].items()
    }


def load_tensor(path: Path, imgsz: int) -> tuple[np.ndarray, torch.Tensor]:
    """Load one image as RGB pixels and a normalized tensor."""
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise FileNotFoundError(path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
    return rgb, torch.from_numpy(rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0


def compute_gradcam(weights: Path, layer_index: int, tensor: torch.Tensor, imgsz: int) -> np.ndarray:
    """Calculate a Grad-CAM map for the strongest predicted class score."""
    model = YOLO(str(weights)).model.cpu().eval()
    activations, gradients = {}, {}

    def hook(_module, _inputs, output):
        activations["value"] = output
        output.register_hook(lambda grad: gradients.update(value=grad))

    handle = model.model[layer_index].register_forward_hook(hook)
    x = tensor.clone().requires_grad_(True)
    model.zero_grad(set_to_none=True)
    output = model(x)
    predictions = output[0] if isinstance(output, tuple) else output
    predictions[:, 4:, :].max().backward()
    handle.remove()
    activation, gradient = activations["value"].detach(), gradients["value"].detach()
    cam = torch.relu((gradient.mean(dim=(2, 3), keepdim=True) * activation).sum(dim=1, keepdim=True))
    cam = torch.nn.functional.interpolate(cam, size=(imgsz, imgsz), mode="bilinear", align_corners=False).squeeze()
    cam = cam.cpu().numpy()
    return (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)


def overlay(rgb: np.ndarray, cam: np.ndarray) -> np.ndarray:
    """Overlay a color-coded Grad-CAM map on an image."""
    heat = cv2.cvtColor(cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(rgb, 0.58, heat, 0.42, 0)


def main() -> None:
    """Generate a multi-image qualitative panel."""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    models = resolve_models(args.models)
    paths = sorted(path for path in args.images.iterdir() if path.suffix.lower() in IMG_EXTS)
    selected = random.Random(args.seed).sample(paths, min(args.num_images, len(paths)))
    fig, axes = plt.subplots(len(selected), len(models) + 1, figsize=(2.5 * (len(models) + 1), 2.6 * len(selected)), squeeze=False)
    rows = []
    for row, image_path in enumerate(selected):
        rgb, tensor = load_tensor(image_path, args.imgsz)
        axes[row, 0].imshow(rgb)
        axes[row, 0].set_title("Image" if row == 0 else "")
        axes[row, 0].axis("off")
        for column, (name, config) in enumerate(models.items(), 1):
            cam = compute_gradcam(config["weights"], config["target_layer"], tensor, args.imgsz)
            axes[row, column].imshow(overlay(rgb, cam))
            axes[row, column].set_title(name if row == 0 else "")
            axes[row, column].axis("off")
            rows.append({"image": image_path.name, "model": name, "target_layer": config["target_layer"]})
    fig.subplots_adjust(left=0.01, right=0.99, top=0.94, bottom=0.01, wspace=0.03, hspace=0.05)
    fig.savefig(args.output_dir / "gradcam_concentrated_comparison.png", dpi=600, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(rows).to_csv(args.output_dir / "gradcam_metadata.csv", index=False)


if __name__ == "__main__":
    main()
