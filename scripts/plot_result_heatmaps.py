#!/usr/bin/env python3
"""Regenerate publication heatmaps from versioned result tables."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

METRICS = ["P_TB", "P_RZ", "P_KP", "precision", "recall", "mAP50", "mAP50_95"]
LABELS = ["P (TB)", "P (RZ)", "P (KP)", "P", "R", "mAP50", "mAP50-95"]


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/result_heatmaps"))
    return parser.parse_args()


def draw(path: Path, output: Path) -> None:
    """Draw one metric heatmap."""
    frame = pd.read_csv(path).set_index("model")
    values = frame[METRICS]
    fig, ax = plt.subplots(figsize=(8.4, max(2.3, 0.43 * len(frame) + 1.2)))
    sns.heatmap(values, annot=True, fmt=".2f", cmap="YlGnBu", vmin=0.0, vmax=1.0, linewidths=0.5, cbar_kws={"label": "Metric value"}, ax=ax)
    ax.set(xlabel="", ylabel="")
    ax.set_xticklabels(LABELS, rotation=0)
    ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    fig.savefig(output / f"{path.stem}.png", dpi=600, bbox_inches="tight")
    fig.savefig(output / f"{path.stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Render every result CSV."""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(args.results_dir.glob("*.csv")):
        draw(path, args.output_dir)
        print(f"Wrote {path.stem}")


if __name__ == "__main__":
    main()
