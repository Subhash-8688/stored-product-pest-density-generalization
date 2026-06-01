#!/usr/bin/env python3
"""Characterize scene density and spatial concentration across arbitrary splits."""

from __future__ import annotations

import argparse
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
import numpy as np
import pandas as pd
import yaml
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True, help="Density-split YAML manifest.")
    parser.add_argument("--root", type=Path, help="Base directory for paths in the manifest. Defaults to the project root.")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_labels(path: Path) -> list[tuple[int, float, float, float, float]]:
    """Read and validate one YOLO label file."""
    if not path.exists():
        raise FileNotFoundError(f"missing label: {path}")
    boxes = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"expected five values at {path}:{line_number}")
        cls_id = int(float(parts[0]))
        x, y, width, height = map(float, parts[1:])
        if not all(0 <= value <= 1 for value in (x, y, width, height)) or width <= 0 or height <= 0:
            raise ValueError(f"invalid normalized box at {path}:{line_number}")
        boxes.append((cls_id, x, y, width, height))
    return boxes


def mean_nearest_neighbor(points: list[tuple[float, float]], diagonal: float) -> float:
    """Return normalized mean nearest-neighbor centroid distance."""
    if len(points) < 2:
        return np.nan
    values = np.asarray(points, dtype=np.float64)
    distances = np.sqrt(((values[:, None, :] - values[None, :, :]) ** 2).sum(axis=2))
    np.fill_diagonal(distances, np.inf)
    return float(distances.min(axis=1).mean() / diagonal)


def build_tables(config: dict, root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate image-level and instance-level density measurements."""
    image_rows, instance_rows = [], []
    names = {int(key): value for key, value in config["names"].items()}
    for split, split_cfg in config["splits"].items():
        image_dir = (root / split_cfg["images"]).resolve()
        label_dir = (root / split_cfg["labels"]).resolve()
        label = split_cfg.get("label", split.replace("_", " "))
        for image_path in sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMG_EXTS):
            with Image.open(image_path) as image:
                width, height = image.size
            area = width * height
            boxes = load_labels(label_dir / f"{image_path.stem}.txt")
            centers = [(x * width, y * height) for _, x, y, _, _ in boxes]
            species_counts = defaultdict(int)
            total_box_area = 0.0
            for cls_id, x, y, box_width, box_height in boxes:
                box_area = box_width * width * box_height * height
                total_box_area += box_area
                species_counts[cls_id] += 1
                instance_rows.append(
                    {"split": split, "label": label, "image": image_path.name, "class_id": cls_id, "species": names[cls_id], "x": x, "y": y}
                )
            count = len(boxes)
            row = {
                "split": split,
                "label": label,
                "image": image_path.name,
                "width": width,
                "height": height,
                "instances": count,
                "instances_per_mpx": count / (area / 1_000_000),
                "occupancy_percent": 100 * total_box_area / area,
                "nearest_neighbor_norm": mean_nearest_neighbor(centers, float(np.hypot(width, height))),
            }
            row.update({f"{species}_count": species_counts[cls_id] for cls_id, species in names.items()})
            image_rows.append(row)
    return pd.DataFrame(image_rows), pd.DataFrame(instance_rows)


def save_plots(images: pd.DataFrame, instances: pd.DataFrame, output: Path) -> None:
    """Save publication-oriented comparison plots."""
    labels = images[["split", "label"]].drop_duplicates().itertuples(index=False, name=None)
    style = {split: (label, COLORS[index % len(COLORS)]) for index, (split, label) in enumerate(labels)}
    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    for split, (label, color) in style.items():
        data = images[images["split"] == split]
        ax.scatter(data["instances_per_mpx"], data["occupancy_percent"], label=label, color=color, alpha=0.75, edgecolor="black", linewidth=0.35)
    ax.set(xlabel="Scene density (instances/MPx)", ylabel="Insect-area occupancy (%)")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output / "scatter_scene_density_vs_occupancy.png", dpi=600)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    for split, (label, color) in style.items():
        data = images.loc[images["split"] == split, "occupancy_percent"]
        ax.hist(data, bins=18, alpha=0.48, color=color, label=label)
        ax.axvline(data.mean(), color=color, linewidth=1.8)
        ax.axvline(data.median(), color=color, linewidth=1.5, linestyle="--")
    ax.set(xlabel="Insect-area occupancy (%)", ylabel="Images")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output / "hist_percent_insect_area_by_split.png", dpi=600)
    plt.close(fig)

    columns = min(2, len(style))
    rows = int(np.ceil(len(style) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(5.2 * columns, 4.2 * rows), squeeze=False)
    for ax, (split, (label, _color)) in zip(axes.flat, style.items()):
        points = instances[instances["split"] == split]
        heat = ax.hist2d(points["x"], points["y"], bins=36, range=[[0, 1], [0, 1]], cmap="inferno")
        fig.colorbar(heat[3], ax=ax, label="Instances")
        ax.set(title=label, xlabel="Normalized x-center", ylabel="Normalized y-center", xlim=(0, 1), ylim=(1, 0))
    for ax in axes.flat[len(style) :]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output / "spatial_heatmaps_by_split.png", dpi=600)
    plt.close(fig)


def main() -> None:
    """Run density characterization."""
    args = parse_args()
    config = yaml.safe_load(args.dataset.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    root = args.root.resolve() if args.root else args.dataset.resolve().parents[2]
    images, instances = build_tables(config, root)
    images.to_csv(args.output_dir / "image_level_summary.csv", index=False)
    instances.to_csv(args.output_dir / "instance_level_summary.csv", index=False)
    summary = (
        images.groupby(["split", "label"])
        .agg(
            images=("image", "count"),
            instances=("instances", "sum"),
            mean_instances_per_image=("instances", "mean"),
            mean_instances_per_mpx=("instances_per_mpx", "mean"),
            mean_occupancy_percent=("occupancy_percent", "mean"),
            mean_nearest_neighbor_norm=("nearest_neighbor_norm", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(args.output_dir / "split_summary.csv", index=False)
    save_plots(images, instances, args.output_dir)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
