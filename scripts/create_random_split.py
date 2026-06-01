#!/usr/bin/env python3
"""Combine train and validation data, then create a random 80:20 partition."""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

import yaml

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True, help="Dataset root containing images/{train,val}.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true", help="Replace an existing output directory.")
    return parser.parse_args()


def list_pairs(root: Path) -> list[tuple[Path, Path]]:
    """Return unique image-label pairs from train and validation."""
    pairs = []
    seen = set()
    for split in ("train", "val"):
        image_dir = root / "images" / split
        label_dir = root / "labels" / split
        for image in sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMG_EXTS):
            if image.stem in seen:
                raise ValueError(f"duplicate image stem while combining splits: {image.stem}")
            label = label_dir / f"{image.stem}.txt"
            if not label.exists():
                raise FileNotFoundError(f"missing label for {image}")
            seen.add(image.stem)
            pairs.append((image, label))
    return pairs


def main() -> None:
    """Write the random split dataset."""
    args = parse_args()
    if not 0 < args.train_ratio < 1:
        raise ValueError("--train-ratio must be between 0 and 1")
    if args.output.exists():
        if not args.force:
            raise FileExistsError(f"{args.output} already exists; pass --force to replace it")
        shutil.rmtree(args.output)
    pairs = list_pairs(args.source)
    random.Random(args.seed).shuffle(pairs)
    cutoff = round(len(pairs) * args.train_ratio)
    for split, rows in (("train", pairs[:cutoff]), ("val", pairs[cutoff:])):
        image_out = args.output / "images" / split
        label_out = args.output / "labels" / split
        image_out.mkdir(parents=True, exist_ok=True)
        label_out.mkdir(parents=True, exist_ok=True)
        for image, label in rows:
            shutil.copy2(image, image_out / image.name)
            shutil.copy2(label, label_out / label.name)
    dataset_yaml = {
        "path": str(args.output.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {0: "TB", 1: "RZ", 2: "KP"},
    }
    (args.output / "dataset.yaml").write_text(yaml.safe_dump(dataset_yaml, sort_keys=False), encoding="utf-8")
    print(f"Wrote {cutoff} train and {len(pairs) - cutoff} validation pairs to {args.output}")


if __name__ == "__main__":
    main()
