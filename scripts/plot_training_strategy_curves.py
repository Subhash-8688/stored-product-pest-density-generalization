#!/usr/bin/env python3
"""Compare training curves for density-stratified and random partitions."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec

WINDOW = 7
PALETTE = {"Density-stratified": "#2A6FBB", "Random 80:20": "#D95F02"}
LOSS_PANELS = [
    ("train/box_loss", "(a) Box loss", "Training loss"),
    ("train/cls_loss", "(b) Class loss", "Training loss"),
    ("train/dfl_loss", "(c) DFL loss", "Training loss"),
]
METRIC_PANELS = [
    ("metrics/precision(B)", "(d) Precision", "Validation score"),
    ("metrics/recall(B)", "(e) Recall", "Validation score"),
    ("metrics/mAP50(B)", "(f) mAP50", "Validation score"),
    ("metrics/mAP50-95(B)", "(g) mAP50-95", "Validation score"),
]


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--density-results", type=Path, required=True)
    parser.add_argument("--random-results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/training_strategy_curves"))
    return parser.parse_args()


def load(path: Path, strategy: str) -> pd.DataFrame:
    """Load one Ultralytics results CSV."""
    frame = pd.read_csv(path)
    frame.columns = [column.strip() for column in frame.columns]
    frame["strategy"] = strategy
    return frame


def draw_curve(ax, data: pd.DataFrame, column: str, title: str, ylabel: str, metric: bool) -> None:
    """Draw raw and seven-epoch-smoothed curves."""
    for strategy, group in data.groupby("strategy", sort=False):
        group = group.sort_values("epoch")
        color = PALETTE[strategy]
        smooth = group[column].rolling(window=WINDOW, min_periods=1, center=True).mean()
        ax.plot(group["epoch"], group[column], color=color, alpha=0.12, linewidth=0.8)
        ax.plot(group["epoch"], smooth, color=color, linewidth=2.0, label=strategy)
        ax.text(group["epoch"].iloc[-1] + 1.5, smooth.iloc[-1], f"{smooth.iloc[-1]:.2f}", va="center", fontsize=8, color=color)
        if metric:
            best = group[column].idxmax()
            ax.scatter(group.loc[best, "epoch"], group.loc[best, column], marker="*", s=52, color=color, edgecolor="#222222", linewidth=0.4)
    ax.set(title=title, xlabel="Epoch", ylabel=ylabel)
    ax.grid(color="#D9D9D9", linewidth=0.75)
    if metric:
        ax.set_ylim(0, 1.03)


def main() -> None:
    """Generate the comparison plot."""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = pd.concat([load(args.density_results, "Density-stratified"), load(args.random_results, "Random 80:20")])
    sns.set_theme(context="paper", style="whitegrid")
    fig = plt.figure(figsize=(15.2, 7.8))
    grid = GridSpec(2, 12, figure=fig, hspace=0.72, wspace=1.05)
    axes = [
        fig.add_subplot(grid[0, 0:4]),
        fig.add_subplot(grid[0, 4:8]),
        fig.add_subplot(grid[0, 8:12]),
        fig.add_subplot(grid[1, 0:3]),
        fig.add_subplot(grid[1, 3:6]),
        fig.add_subplot(grid[1, 6:9]),
        fig.add_subplot(grid[1, 9:12]),
    ]
    for ax, (column, title, ylabel) in zip(axes[:3], LOSS_PANELS):
        draw_curve(ax, data, column, title, ylabel, metric=False)
    for ax, (column, title, ylabel) in zip(axes[3:], METRIC_PANELS):
        draw_curve(ax, data, column, title, ylabel, metric=True)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.985), ncol=2, frameon=False)
    fig.subplots_adjust(top=0.89, bottom=0.08, left=0.055, right=0.985, hspace=0.72, wspace=0.95)
    fig.savefig(args.output_dir / "training_curves_density_vs_random.png", dpi=600, bbox_inches="tight")
    fig.savefig(args.output_dir / "training_curves_density_vs_random.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
