#!/usr/bin/env python3
"""Audit visual and annotation similarity across detection dataset splits."""

from pathlib import Path
import argparse
import hashlib
import html
import json
import math
import os
import re
from collections import Counter, defaultdict
from datetime import date, datetime

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image, ImageDraw, ImageOps
from scipy.fftpack import dct
from skimage.metrics import structural_similarity as ssim


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
DATASETS = {
    "Detect": DATA_ROOT / "Detect",
    "Detect-conc": DATA_ROOT / "Detect-conc",
}
OUT = PROJECT_ROOT / "outputs" / "leakage_analysis_detect_vs_detect_conc"

SPLITS = ["train", "val", "test"]
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
CLASS_NAMES = {0: "TB", 1: "RZ", 2: "KP"}
TIME_RE = re.compile(r"(\d{14})(\d{0,3})")
PREFIX_RE = re.compile(r"^([A-Za-z]+)")
SPLIT_COLORS = {"train": "#2f6db3", "val": "#e58b25", "test": "#2f9d58"}
CLASS_COLORS = {0: "#1f77b4", 1: "#ff7f0e", 2: "#2ca02c"}

PAIR_COLUMNS = [
    "dataset", "score", "split_pair", "a_path", "b_path", "a_data_relpath", "b_data_relpath",
    "a_prefix", "b_prefix", "a_dominant_class", "b_dominant_class", "a_class_signature", "b_class_signature",
    "phash_dist", "dhash_dist", "ahash_dist", "ssim_192", "mean_label_iou", "matched_iou50", "matched_iou75",
    "class_counts_same", "box_count_a", "box_count_b", "box_count_delta", "time_delta_s",
    "high_visual", "strong_annotation", "very_strong",
]
NEAREST_COLUMNS = PAIR_COLUMNS + ["hash_rank"]


def bits_to_int(bits):
    n = 0
    for bit in bits:
        n = (n << 1) | int(bool(bit))
    return n


def popcount(value):
    return int(value).bit_count()


def phash_from_image(im, hash_size=8, highfreq_factor=4):
    size = hash_size * highfreq_factor
    gray = ImageOps.grayscale(im).resize((size, size), Image.Resampling.LANCZOS)
    arr = np.asarray(gray, dtype=np.float32)
    coeff = dct(dct(arr, axis=0, norm="ortho"), axis=1, norm="ortho")[:hash_size, :hash_size].flatten()
    return bits_to_int(coeff > np.median(coeff[1:]))


def dhash_from_image(im, hash_size=8):
    gray = ImageOps.grayscale(im).resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    arr = np.asarray(gray, dtype=np.int16)
    return bits_to_int((arr[:, 1:] > arr[:, :-1]).flatten())


def ahash_from_image(im, hash_size=8):
    gray = ImageOps.grayscale(im).resize((hash_size, hash_size), Image.Resampling.LANCZOS)
    arr = np.asarray(gray, dtype=np.float32)
    return bits_to_int((arr > arr.mean()).flatten())


def thumb_array(im, size=192):
    gray = ImageOps.grayscale(im)
    gray.thumbnail((size, size), Image.Resampling.LANCZOS)
    canvas = Image.new("L", (size, size), 0)
    canvas.paste(gray, ((size - gray.width) // 2, (size - gray.height) // 2))
    return np.asarray(canvas, dtype=np.float32) / 255.0


def parse_filename(path):
    stem = Path(path).stem
    prefix_match = PREFIX_RE.match(stem)
    prefix = prefix_match.group(1).lower() if prefix_match else ""
    capture_ms = None
    match = TIME_RE.search(stem)
    if match:
        ymdhms, ms = match.groups()
        try:
            dt = datetime.strptime(ymdhms, "%Y%m%d%H%M%S")
            capture_ms = int(dt.timestamp() * 1000) + int((ms or "0").ljust(3, "0")[:3])
        except ValueError:
            capture_ms = None
    return stem, prefix, capture_ms


def label_for_image(dataset_root, split, image_path):
    return dataset_root / "labels" / split / f"{image_path.stem}.txt"


def parse_label(path):
    boxes = []
    bad = []
    if not path.exists():
        return boxes, ["missing"]
    text = path.read_text(errors="replace").strip()
    if not text:
        return boxes, bad
    for idx, line in enumerate(text.splitlines(), 1):
        parts = line.split()
        if len(parts) < 5:
            bad.append(f"{idx}:len")
            continue
        try:
            cls = int(float(parts[0]))
            x, y, w, h = map(float, parts[1:5])
        except ValueError:
            bad.append(f"{idx}:parse")
            continue
        boxes.append((cls, x, y, w, h))
        if cls not in CLASS_NAMES or not all(math.isfinite(v) and 0 <= v <= 1 for v in (x, y, w, h)):
            bad.append(f"{idx}:range")
    return boxes, bad


def xyxy(box):
    cls, x, y, w, h = box
    return cls, x - w / 2, y - h / 2, x + w / 2, y + h / 2


def iou(a, b):
    ca, ax1, ay1, ax2, ay2 = xyxy(a)
    cb, bx1, by1, bx2, by2 = xyxy(b)
    if ca != cb:
        return 0.0
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    bb = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    den = aa + bb - inter
    return inter / den if den > 0 else 0.0


def label_overlap(boxes_a, boxes_b):
    if not boxes_a and not boxes_b:
        return 1.0, 0, 0, True, 0
    if not boxes_a or not boxes_b:
        return 0.0, 0, 0, False, abs(len(boxes_a) - len(boxes_b))
    vals = []
    for a in boxes_a:
        vals.append(max((iou(a, b) for b in boxes_b if b[0] == a[0]), default=0.0))
    for b in boxes_b:
        vals.append(max((iou(b, a) for a in boxes_a if a[0] == b[0]), default=0.0))
    return (
        float(np.mean(vals)),
        int(sum(v >= 0.50 for v in vals)),
        int(sum(v >= 0.75 for v in vals)),
        Counter(x[0] for x in boxes_a) == Counter(x[0] for x in boxes_b),
        abs(len(boxes_a) - len(boxes_b)),
    )


def class_signature(boxes):
    counts = Counter(b[0] for b in boxes)
    return ";".join(f"{CLASS_NAMES[k]}:{counts[k]}" for k in sorted(CLASS_NAMES) if counts[k])


def dominant_class(boxes):
    if not boxes:
        return ""
    cls, _ = Counter(b[0] for b in boxes).most_common(1)[0]
    return CLASS_NAMES.get(cls, str(cls))


def index_dataset(name, root):
    records = []
    errors = []
    for split in SPLITS:
        image_dir = root / "images" / split
        for path in sorted(image_dir.iterdir()):
            if not path.is_file() or path.suffix.lower() not in EXTS:
                continue
            label_path = label_for_image(root, split, path)
            boxes, bad = parse_label(label_path)
            stem, prefix, capture_ms = parse_filename(path)
            rec = {
                "dataset": name,
                "id": len(records),
                "split": split,
                "path": path,
                "relpath": str(path.relative_to(root)),
                "data_relpath": str(path.relative_to(DATA_ROOT)),
                "label_relpath": str(label_path.relative_to(root)),
                "stem": stem,
                "prefix": prefix,
                "capture_ms": capture_ms,
                "boxes": boxes,
                "label_bad": bad,
                "box_count": len(boxes),
                "class_signature": class_signature(boxes),
                "dominant_class": dominant_class(boxes),
            }
            try:
                image_bytes = path.read_bytes()
                rec["sha256"] = hashlib.sha256(image_bytes).hexdigest()
                rec["bytes"] = len(image_bytes)
                rec["label_sha256"] = hashlib.sha256(label_path.read_bytes()).hexdigest() if label_path.exists() else None
                with Image.open(path) as im:
                    im = ImageOps.exif_transpose(im)
                    rec["width"], rec["height"] = im.size
                    rec["phash"] = phash_from_image(im)
                    rec["dhash"] = dhash_from_image(im)
                    rec["ahash"] = ahash_from_image(im)
                    rec["thumb"] = thumb_array(im)
            except Exception as exc:
                errors.append({"path": str(path), "error": repr(exc)})
                continue
            records.append(rec)
    return records, errors


def is_high_visual(pair):
    return ((pair["phash_dist"] <= 8 and pair["dhash_dist"] <= 8) or (pair["ssim_192"] >= 0.95 and pair["phash_dist"] <= 12))


def is_strong_annotation(pair):
    return pair["mean_label_iou"] >= 0.75 and bool(pair["class_counts_same"])


def compute_pairs(records, dataset_name):
    by_split = {split: [r for r in records if r["split"] == split] for split in SPLITS}
    pairs = []
    nearest = []
    for a_split, b_split in [("train", "val"), ("train", "test"), ("val", "test")]:
        for a in by_split[a_split]:
            for b in by_split[b_split]:
                ph = popcount(a["phash"] ^ b["phash"])
                dh = popcount(a["dhash"] ^ b["dhash"])
                ah = popcount(a["ahash"] ^ b["ahash"])
                candidate = ph <= 12 or dh <= 10 or (ph <= 16 and dh <= 14 and ah <= 14)
                hash_rank = ph + dh + ah / 2
                nearest_candidate = ph <= 24 or dh <= 24 or hash_rank <= 54
                if candidate or nearest_candidate:
                    ss = float(ssim(a["thumb"], b["thumb"], data_range=1.0))
                    mean_iou, m50, m75, same_counts, box_delta = label_overlap(a["boxes"], b["boxes"])
                    time_delta_s = None
                    if a["capture_ms"] is not None and b["capture_ms"] is not None:
                        time_delta_s = abs(a["capture_ms"] - b["capture_ms"]) / 1000.0
                    score = (
                        max(0, 1 - ph / 16) * 30
                        + max(0, 1 - dh / 16) * 20
                        + max(0, min(1, (ss - 0.70) / 0.30)) * 30
                        + mean_iou * 15
                    )
                    if time_delta_s is not None and time_delta_s <= 2:
                        score += 5
                    pair = {
                        "dataset": dataset_name,
                        "score": score,
                        "split_pair": f"{a_split}-{b_split}",
                        "a_path": a["relpath"],
                        "b_path": b["relpath"],
                        "a_data_relpath": a["data_relpath"],
                        "b_data_relpath": b["data_relpath"],
                        "a_prefix": a["prefix"],
                        "b_prefix": b["prefix"],
                        "a_dominant_class": a["dominant_class"],
                        "b_dominant_class": b["dominant_class"],
                        "a_class_signature": a["class_signature"],
                        "b_class_signature": b["class_signature"],
                        "phash_dist": ph,
                        "dhash_dist": dh,
                        "ahash_dist": ah,
                        "ssim_192": ss,
                        "mean_label_iou": mean_iou,
                        "matched_iou50": m50,
                        "matched_iou75": m75,
                        "class_counts_same": same_counts,
                        "box_count_a": a["box_count"],
                        "box_count_b": b["box_count"],
                        "box_count_delta": box_delta,
                        "time_delta_s": time_delta_s,
                    }
                    pair["high_visual"] = is_high_visual(pair)
                    pair["strong_annotation"] = is_strong_annotation(pair)
                    pair["very_strong"] = pair["high_visual"] and pair["strong_annotation"]
                    near_pair = dict(pair)
                    near_pair["hash_rank"] = hash_rank
                    nearest.append(near_pair)
                    if candidate:
                        pairs.append(pair)
    pairs.sort(key=lambda p: (p["high_visual"], p["very_strong"], p["score"], p["ssim_192"]), reverse=True)
    nearest.sort(key=lambda p: (p["hash_rank"], -p["ssim_192"], -p["score"]))
    return pairs, nearest[:1000]


def sequence_adjacency(records, dataset_name, max_delta_s=10.0):
    rows = []
    groups = defaultdict(list)
    for r in records:
        if r["capture_ms"] is not None:
            key = (r["prefix"], r["dominant_class"] or r["class_signature"])
            groups[key].append(r)
    for (prefix, cls), group in groups.items():
        group.sort(key=lambda r: r["capture_ms"])
        for i, a in enumerate(group):
            j = i + 1
            while j < len(group) and (group[j]["capture_ms"] - a["capture_ms"]) / 1000.0 <= max_delta_s:
                b = group[j]
                if a["split"] != b["split"]:
                    rows.append(
                        {
                            "dataset": dataset_name,
                            "prefix": prefix,
                            "class": cls,
                            "a_split": a["split"],
                            "b_split": b["split"],
                            "split_pair": "-".join(sorted([a["split"], b["split"]])),
                            "a_path": a["relpath"],
                            "b_path": b["relpath"],
                            "delta_s": (b["capture_ms"] - a["capture_ms"]) / 1000.0,
                        }
                    )
                j += 1
    rows.sort(key=lambda r: (r["split_pair"], r["prefix"], r["delta_s"]))
    return rows


def exact_duplicate_groups(records, key):
    grouped = defaultdict(list)
    for r in records:
        if r.get(key):
            grouped[r[key]].append(r)
    return [group for group in grouped.values() if len({r["split"] for r in group}) > 1]


def write_csv(path, rows):
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)


def records_public_df(records):
    rows = []
    for r in records:
        rows.append(
            {
                "dataset": r["dataset"],
                "split": r["split"],
                "path": r["relpath"],
                "sha256": r["sha256"],
                "label_sha256": r["label_sha256"],
                "prefix": r["prefix"],
                "capture_ms": r["capture_ms"],
                "dominant_class": r["dominant_class"],
                "class_signature": r["class_signature"],
                "box_count": r["box_count"],
                "width": r["width"],
                "height": r["height"],
            }
        )
    return pd.DataFrame(rows)


def compare_shared_splits(all_records):
    by_dataset = {name: [r for r in all_records if r["dataset"] == name] for name in DATASETS}
    rows = []
    for split in SPLITS:
        detect_hashes = {r["sha256"] for r in by_dataset["Detect"] if r["split"] == split}
        conc_hashes = {r["sha256"] for r in by_dataset["Detect-conc"] if r["split"] == split}
        rows.append(
            {
                "split": split,
                "detect_images": len(detect_hashes),
                "detect_conc_images": len(conc_hashes),
                "shared_image_hashes": len(detect_hashes & conc_hashes),
                "detect_only_hashes": len(detect_hashes - conc_hashes),
                "detect_conc_only_hashes": len(conc_hashes - detect_hashes),
                "same_set_by_hash": detect_hashes == conc_hashes,
            }
        )
    return pd.DataFrame(rows)


def label_counts(dataset_root):
    counts = {}
    for split in SPLITS:
        counts[split] = len(list((dataset_root / "labels" / split).glob("*.txt")))
    return counts


def summarize_dataset(name, root, records, pairs, adjacency, errors):
    split_counts = Counter(r["split"] for r in records)
    labels = label_counts(root)
    exact_images = exact_duplicate_groups(records, "sha256")
    exact_labels = exact_duplicate_groups(records, "label_sha256")
    bad_labels = sum(1 for r in records if r["label_bad"])
    high_pairs = [p for p in pairs if p["high_visual"]]
    very_pairs = [p for p in pairs if p["very_strong"]]
    return {
        "dataset": name,
        "root": str(root),
        "total_images": len(records),
        "split_counts": dict(split_counts),
        "label_counts": labels,
        "bad_label_files": bad_labels,
        "image_decode_errors": len(errors),
        "visual_candidate_pairs": len(pairs),
        "high_visual_pairs": len(high_pairs),
        "very_strong_pairs": len(very_pairs),
        "exact_cross_split_image_duplicate_groups": len(exact_images),
        "exact_cross_split_label_duplicate_groups": len(exact_labels),
        "sequence_adjacency_pairs_le_10s": len(adjacency),
        "high_visual_by_split_pair": dict(Counter(p["split_pair"] for p in high_pairs)),
        "very_strong_by_split_pair": dict(Counter(p["split_pair"] for p in very_pairs)),
        "sequence_adjacency_by_split_pair": dict(Counter(r["split_pair"] for r in adjacency)),
    }


def savefig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def make_plots(combined_pairs, summaries, shared):
    sns.set_theme(style="whitegrid")
    high = combined_pairs[combined_pairs["high_visual"].fillna(False).astype(bool)].copy()
    very = combined_pairs[combined_pairs["very_strong"].fillna(False).astype(bool)].copy()

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Detect vs Detect-conc Leakage Overview", fontsize=18, fontweight="bold")
    summary_df = pd.DataFrame(summaries)
    ax = axes[0, 0]
    count_rows = []
    for summary in summaries:
        for split in SPLITS:
            count_rows.append({"dataset": summary["dataset"], "split": split, "images": summary["split_counts"].get(split, 0)})
    sns.barplot(pd.DataFrame(count_rows), x="split", y="images", hue="dataset", ax=ax)
    ax.set_title("Split Sizes")

    ax = axes[0, 1]
    metric_rows = []
    for summary in summaries:
        metric_rows.extend(
            [
                {"dataset": summary["dataset"], "metric": "visual candidates", "pairs": summary["visual_candidate_pairs"]},
                {"dataset": summary["dataset"], "metric": "high visual", "pairs": summary["high_visual_pairs"]},
                {"dataset": summary["dataset"], "metric": "very strong", "pairs": summary["very_strong_pairs"]},
            ]
        )
    sns.barplot(pd.DataFrame(metric_rows), x="metric", y="pairs", hue="dataset", ax=ax)
    ax.set_title("Cross-Split Candidate Counts")
    ax.tick_params(axis="x", rotation=15)

    ax = axes[1, 0]
    matrix_rows = []
    for summary in summaries:
        for split_pair in ["train-val", "train-test", "val-test"]:
            matrix_rows.append(
                {
                    "dataset": summary["dataset"],
                    "split_pair": split_pair,
                    "high_visual": summary["high_visual_by_split_pair"].get(split_pair, 0),
                    "very_strong": summary["very_strong_by_split_pair"].get(split_pair, 0),
                }
            )
    matrix_df = pd.DataFrame(matrix_rows)
    sns.barplot(matrix_df, x="split_pair", y="high_visual", hue="dataset", ax=ax)
    ax.set_title("High-Visual Pairs By Split Pair")

    ax = axes[1, 1]
    shared_plot = shared.melt(id_vars="split", value_vars=["shared_image_hashes", "detect_only_hashes", "detect_conc_only_hashes"], var_name="set", value_name="images")
    sns.barplot(shared_plot, x="split", y="images", hue="set", ax=ax)
    ax.set_title("Detect vs Detect-conc Shared Images By Hash")
    ax.legend(fontsize=8)
    savefig(OUT / "01_leakage_overview.png")

    if not high.empty:
        plt.figure(figsize=(10, 7))
        sns.scatterplot(data=high, x="ssim_192", y="mean_label_iou", hue="dataset", style="split_pair", alpha=0.6, s=45)
        plt.axvline(0.95, color="black", linestyle="--", linewidth=1)
        plt.axhline(0.75, color="black", linestyle=":", linewidth=1)
        plt.title("High-Visual Cross-Split Pairs: SSIM vs Annotation IoU")
        plt.xlabel("SSIM on 192px grayscale thumbnail")
        plt.ylabel("Mean symmetric same-class bbox IoU")
        plt.xlim(max(0.60, high["ssim_192"].min() - 0.02), 1.005)
        plt.ylim(-0.02, 1.02)
        savefig(OUT / "02_high_visual_similarity_vs_iou.png")

    heat_rows = []
    for dataset, group in high.groupby("dataset"):
        counts = group.groupby(["split_pair", "a_dominant_class"]).size().reset_index(name="count")
        for _, row in counts.iterrows():
            heat_rows.append(row.to_dict() | {"dataset": dataset})
    if heat_rows:
        heat = pd.DataFrame(heat_rows)
        for dataset in sorted(heat["dataset"].unique()):
            pivot = heat[heat["dataset"] == dataset].pivot_table(index="a_dominant_class", columns="split_pair", values="count", aggfunc="sum", fill_value=0)
            plt.figure(figsize=(8, 4.5))
            sns.heatmap(pivot, annot=True, fmt=",d", cmap="YlOrRd", cbar_kws={"label": "High-visual pair count"})
            plt.title(f"{dataset}: High-Visual Pairs By Class And Split Pair")
            plt.xlabel("Split pair")
            plt.ylabel("Dominant class")
            savefig(OUT / f"03_{dataset.lower().replace('-', '_')}_high_visual_heatmap.png")

    if not high.empty:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        sns.histplot(data=high, x="ssim_192", hue="dataset", bins=25, element="step", ax=axes[0])
        axes[0].set_title("SSIM Distribution")
        sns.histplot(data=high, x="phash_dist", hue="dataset", bins=range(0, int(high["phash_dist"].max()) + 2), element="step", ax=axes[1])
        axes[1].set_title("pHash Distance Distribution")
        sns.histplot(data=high, x="mean_label_iou", hue="dataset", bins=25, element="step", ax=axes[2])
        axes[2].set_title("Annotation IoU Distribution")
        savefig(OUT / "04_high_visual_metric_distributions.png")


def plot_similarity_vs_annotation_overlap(pairs_df, output_name, title):
    if pairs_df.empty:
        return

    plot_df = pairs_df.copy()
    plot_df["high_visual_threshold"] = (
        ((plot_df["phash_dist"] <= 8) & (plot_df["dhash_dist"] <= 8))
        | ((plot_df["ssim_192"] >= 0.95) & (plot_df["phash_dist"] <= 12))
    )
    plot_df["strong_annotation_threshold"] = (plot_df["mean_label_iou"] >= 0.75) & (plot_df["class_counts_same"].fillna(False).astype(bool))

    plt.figure(figsize=(10.5, 7.2))
    ax = plt.gca()
    ax.axvspan(0.95, 1.0, ymin=0.75, ymax=1.0, color="#f5b7b1", alpha=0.28, label="High-risk region")
    sns.scatterplot(
        data=plot_df,
        x="ssim_192",
        y="mean_label_iou",
        hue="dataset",
        style="split_pair",
        size="hash_rank" if "hash_rank" in plot_df.columns else None,
        sizes=(25, 90),
        alpha=0.62,
        ax=ax,
    )
    ax.axvline(0.95, color="#222222", linestyle="--", linewidth=1.2)
    ax.axhline(0.75, color="#222222", linestyle=":", linewidth=1.2)
    ax.text(0.952, 0.78, "high visual + strong annotation", fontsize=9, color="#7f1d1d")
    ax.set_title(title)
    ax.set_xlabel("Visual similarity: SSIM on 192px grayscale thumbnail")
    ax.set_ylabel("Annotation overlap: mean symmetric same-class bbox IoU")
    ax.set_xlim(max(0.0, min(0.30, float(plot_df["ssim_192"].min()) - 0.04)), 1.005)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)
    savefig(OUT / output_name)


def load_and_annotate(dataset_name, rel_path, max_size=(360, 270)):
    root = DATASETS[dataset_name]
    image_path = root / rel_path
    split = Path(rel_path).parts[1]
    label_path = root / "labels" / split / f"{Path(rel_path).stem}.txt"
    im = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
    original_w, original_h = im.size
    im.thumbnail(max_size, Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(im)
    sx, sy = im.width / original_w, im.height / original_h
    for cls, x, y, w, h in parse_label(label_path)[0]:
        color = CLASS_COLORS.get(cls, "#ffffff")
        x1 = (x - w / 2) * original_w * sx
        y1 = (y - h / 2) * original_h * sy
        x2 = (x + w / 2) * original_w * sx
        y2 = (y + h / 2) * original_h * sy
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        draw.text((x1 + 2, y1 + 2), CLASS_NAMES.get(cls, str(cls)), fill=color)
    canvas = Image.new("RGB", max_size, (245, 245, 245))
    canvas.paste(im, ((max_size[0] - im.width) // 2, (max_size[1] - im.height) // 2))
    return canvas


def annotated_pair_sheet(pairs_df, path, title, max_pairs=16):
    rows = pairs_df.head(max_pairs).reset_index(drop=True)
    if rows.empty:
        return
    thumb_w, thumb_h = 360, 270
    margin, header_h, caption_h = 16, 28, 56
    pair_w = thumb_w * 2 + margin * 3
    pair_h = header_h + thumb_h + caption_h + margin
    sheet = Image.new("RGB", (pair_w, 58 + len(rows) * pair_h), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((margin, 14), title, fill=(0, 0, 0))
    for idx, row in rows.iterrows():
        y0 = 58 + idx * pair_h
        ax, bx = margin, margin * 2 + thumb_w
        img_y = y0 + header_h
        a = load_and_annotate(row["dataset"], row["a_path"], (thumb_w, thumb_h))
        b = load_and_annotate(row["dataset"], row["b_path"], (thumb_w, thumb_h))
        sheet.paste(a, (ax, img_y))
        sheet.paste(b, (bx, img_y))
        a_split, b_split = row["split_pair"].split("-")
        draw.text((ax, y0), f"{row['dataset']} A: {a_split} {Path(row['a_path']).name}", fill=(0, 0, 0))
        draw.text((bx, y0), f"B: {b_split} {Path(row['b_path']).name}", fill=(0, 0, 0))
        caption = (
            f"#{idx + 1} {row['split_pair']} | score {row['score']:.1f} | "
            f"pHash {int(row['phash_dist'])}, dHash {int(row['dhash_dist'])} | "
            f"SSIM {row['ssim_192']:.3f} | bbox IoU {row['mean_label_iou']:.3f} | "
            f"{row['a_class_signature']} vs {row['b_class_signature']}"
        )
        draw.text((ax, img_y + thumb_h + 8), caption, fill=(0, 0, 0))
    sheet.save(path, quality=94)


def md_table(headers, rows):
    lines = ["|" + "|".join(headers) + "|", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        vals = []
        for value in row:
            if isinstance(value, float):
                vals.append(f"{value:.3f}")
            else:
                vals.append(str(value).replace("|", "/"))
        lines.append("|" + "|".join(vals) + "|")
    return "\n".join(lines)


def link(path, label=None):
    label = label or path.name
    return f"[{label}](<{path.resolve()}>)"


def write_html_report(summary):
    figures = [
        ("01_leakage_overview.png", "Overview dashboard"),
        ("02_high_visual_similarity_vs_iou.png", "High-visual SSIM vs annotation IoU"),
        ("02_closest_similarity_vs_annotation_overlap.png", "Closest pairs: visual similarity vs annotation overlap"),
        ("03_detect_high_visual_heatmap.png", "Detect high-visual heatmap"),
        ("03_detect_conc_high_visual_heatmap.png", "Detect-conc high-visual heatmap"),
        ("04_high_visual_metric_distributions.png", "High-visual metric distributions"),
        ("05_top_high_visual_pairs_annotated.jpg", "Top annotated high-visual pairs"),
        ("05_closest_cross_split_pairs_annotated.jpg", "Closest cross-split pairs by hash distance"),
    ]
    sections = []
    for filename, title in figures:
        if (OUT / filename).exists():
            sections.append(f"<section><h2>{html.escape(title)}</h2><a href='{html.escape(filename)}'><img src='{html.escape(filename)}' alt='{html.escape(title)}'></a></section>")
    cards = "".join(
        f"<div class='card'><div class='num'>{s['high_visual_pairs']:,}</div><div>{html.escape(s['dataset'])} high-visual pairs</div></div>"
        for s in summary["datasets"]
    )
    doc = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Leakage Visual Report</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 32px; background: #fafafa; color: #222; }}
.summary {{ display: grid; grid-template-columns: repeat(4, minmax(160px, 1fr)); gap: 12px; margin: 24px 0; }}
.card, section {{ background: white; border: 1px solid #ddd; border-radius: 6px; padding: 14px; }}
.num {{ font-size: 28px; font-weight: 700; color: #b22222; }}
section {{ margin: 18px 0; }}
img {{ width: 100%; height: auto; border: 1px solid #ddd; }}
code {{ background: #eee; padding: 2px 4px; border-radius: 3px; }}
</style></head><body>
<h1>Detect / Detect-conc Leakage Visual Report</h1>
<p>Train and validation splits are compared by hash across both folders; test differs by design.</p>
<div class="summary">{cards}</div>
{''.join(sections)}
</body></html>
"""
    (OUT / "visual_report.html").write_text(doc, encoding="utf-8")


def write_markdown_report(summary, combined_high, combined_nearest, shared_df):
    dataset_rows = []
    for s in summary["datasets"]:
        dataset_rows.append(
            [
                s["dataset"],
                s["split_counts"].get("train", 0),
                s["split_counts"].get("val", 0),
                s["split_counts"].get("test", 0),
                s["visual_candidate_pairs"],
                s["high_visual_pairs"],
                s["very_strong_pairs"],
                s["sequence_adjacency_pairs_le_10s"],
                s["exact_cross_split_image_duplicate_groups"],
            ]
        )

    shared_rows = [
        [
            row.split,
            row.detect_images,
            row.detect_conc_images,
            row.shared_image_hashes,
            row.detect_only_hashes,
            row.detect_conc_only_hashes,
            row.same_set_by_hash,
        ]
        for row in shared_df.itertuples(index=False)
    ]

    split_rows = []
    for dataset, group in combined_high.groupby("dataset"):
        counts = group.groupby("split_pair").size()
        split_rows.append([dataset, counts.get("train-val", 0), counts.get("train-test", 0), counts.get("val-test", 0)])

    class_rows = []
    for dataset, group in combined_high.groupby("dataset"):
        counts = group.groupby("a_dominant_class").size().sort_values(ascending=False)
        for cls, count in counts.items():
            class_rows.append([dataset, cls or "unknown", int(count)])

    top_cols = ["dataset", "split_pair", "a_path", "b_path", "phash_dist", "dhash_dist", "ssim_192", "mean_label_iou", "time_delta_s", "a_class_signature", "b_class_signature"]
    top_rows = []
    for row in combined_high.sort_values(["score", "ssim_192"], ascending=False).head(30).itertuples(index=False):
        item = row._asdict()
        top_rows.append([item.get(c, "") for c in top_cols])

    nearest_cols = ["dataset", "split_pair", "a_path", "b_path", "hash_rank", "phash_dist", "dhash_dist", "ssim_192", "mean_label_iou", "time_delta_s", "a_class_signature", "b_class_signature"]
    nearest_rows = []
    if not combined_nearest.empty:
        for row in combined_nearest.sort_values(["hash_rank", "ssim_192"], ascending=[True, False]).head(20).itertuples(index=False):
            item = row._asdict()
            nearest_rows.append([item.get(c, "") for c in nearest_cols])

    train_val_shared = shared_df.loc[shared_df["split"].isin(["train", "val"]), "same_set_by_hash"].all()
    shared_statement = (
        "The train and validation sets are confirmed to be the same across the two folders by image hash, while the test sets differ."
        if train_val_shared
        else "The split-level hash comparison is reported below; inspect it before assuming that the two folders share identical train and validation sets."
    )

    if combined_high.empty:
        high_visual_note = "No pairs met the strict high-visual-similarity threshold in either dataset. I still wrote closest-neighbor evidence so this absence can be inspected visually."
        if not combined_nearest.empty:
            max_ssim = combined_nearest["ssim_192"].max()
            min_hash_rank = combined_nearest["hash_rank"].min()
            key_result = (
                "The key result is that **no cross-split pair in either dataset met the strict high-visual-similarity threshold**. "
                f"The closest-neighbor fallback found a best hash-rank of {min_hash_rank:.1f}, and the highest SSIM among closest-neighbor rows was {max_ssim:.3f}, "
                "well below the 0.95 SSIM high-visual cutoff."
            )
        else:
            key_result = "The key result is that **no cross-split visual candidates were found** under the configured perceptual-hash screening thresholds."
    else:
        high_visual_note = "The table below lists the strongest pairs that met the strict high-visual-similarity threshold."
        key_result = (
            "The key result is that both datasets contain high-visual-similarity cross-split pairs, meaning visually near-duplicate frames appear across train/val/test boundaries. "
            "Inspect the split-level hash comparison and annotated pairs before interpreting performance from a randomly partitioned dataset."
        )

    if combined_high.empty:
        interpretation = (
            "No cross-split pair met the configured strict high-visual-similarity threshold. Review the closest-neighbor evidence and timestamp-adjacency tables before drawing conclusions about leakage. "
            "If capture-session metadata is available, a session-level split remains the cleanest way to reduce leakage risk from sequential image capture."
        )
    else:
        interpretation = (
            "At least one cross-split pair met the configured strict high-visual-similarity threshold. These visually similar frames can make random-split validation metrics optimistic when related captures appear across dataset boundaries. "
            "Use the annotated pair sheets, split-pair counts, and timestamp-adjacency tables to identify whether a session-level partition is needed."
        )

    if combined_high.empty:
        visual_evidence = """- `visual_report.html`: browsable visual report.
- `01_leakage_overview.png`: overview of split sizes, leakage counts, and shared split hashes.
- `02_closest_similarity_vs_annotation_overlap.png`: visual similarity vs annotation overlap for the closest cross-split neighbors.
- `05_closest_cross_split_pairs_annotated.jpg`: closest visual-neighbor fallback with YOLO boxes overlaid."""
    else:
        visual_evidence = """- `visual_report.html`: browsable visual report.
- `01_leakage_overview.png`: overview of split sizes, leakage counts, and shared split hashes.
- `02_high_visual_similarity_vs_iou.png`: high-visual pairs plotted by visual similarity and annotation overlap.
- `03_detect_high_visual_heatmap.png`: Detect high-visual pairs by class and split pair.
- `03_detect_conc_high_visual_heatmap.png`: Detect-conc high-visual pairs by class and split pair.
- `04_high_visual_metric_distributions.png`: SSIM, pHash, and annotation IoU distributions.
- `05_top_high_visual_pairs_annotated.jpg`: top high-visual pairs with YOLO boxes overlaid."""

    report = f"""# Detect / Detect-conc Data Leakage Analysis

Generated: {date.today().isoformat()}
Datasets:

- `Detect`: `{DATASETS['Detect']}`
- `Detect-conc`: `{DATASETS['Detect-conc']}`

## Executive Summary

I ran the same cross-split leakage audit on both `Detect` and `Detect-conc`, with special attention to high-visual-similarity cross-split pairs. {shared_statement}

{key_result}

{md_table(['dataset','train imgs','val imgs','test imgs','visual candidates','high visual pairs','very strong pairs','sequence-adjacent <=10s','exact dup groups'], dataset_rows)}

## Detect vs Detect-conc Split Comparison

{md_table(['split','Detect images','Detect-conc images','shared hashes','Detect only','Detect-conc only','same set by hash'], shared_rows)}

## High-Visual-Similarity Cross-Split Pairs

High visual similarity is defined as `(pHash distance <= 8 and dHash distance <= 8)` or `(SSIM >= 0.95 and pHash distance <= 12)`.

{high_visual_note}

{md_table(['dataset','train-val','train-test','val-test'], split_rows)}

Class distribution among high-visual pairs:

{md_table(['dataset','dominant class','high-visual pair count'], class_rows)}

Top high-visual-similarity pairs:

{md_table(['dataset','split pair','image A','image B','pHash','dHash','SSIM','label IoU','time delta s','A classes','B classes'], top_rows)}

Closest cross-split visual neighbors by hash distance:

{md_table(['dataset','split pair','image A','image B','hash rank','pHash','dHash','SSIM','label IoU','time delta s','A classes','B classes'], nearest_rows)}

## Visual Evidence

{visual_evidence}

## Output Tables

- `Detect_cross_split_candidates.csv` and `Detect_high_visual_pairs.csv`
- `Detect-conc_cross_split_candidates.csv` and `Detect-conc_high_visual_pairs.csv`
- `combined_high_visual_pairs.csv`
- `combined_closest_cross_split_pairs.csv`
- `shared_split_comparison.csv`
- `summary.json`

## Interpretation

{interpretation}
"""
    (OUT / "leakage_analysis_report.md").write_text(report, encoding="utf-8")


def main():
    global DATA_ROOT, DATASETS, OUT
    parser = argparse.ArgumentParser(description="Audit cross-split similarity in sequentially captured detection data.")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT, help="Directory containing Detect and Detect-conc.")
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    DATA_ROOT = args.data_root.resolve()
    DATASETS = {
        "Detect": DATA_ROOT / "Detect",
        "Detect-conc": DATA_ROOT / "Detect-conc",
    }
    OUT = args.output_dir.resolve()
    OUT.mkdir(parents=True, exist_ok=True)

    all_records = []
    summaries = []
    all_pairs = []
    all_nearest = []
    all_adjacency = []
    errors_by_dataset = {}

    for name, root in DATASETS.items():
        print(f"Indexing {name}...", flush=True)
        records, errors = index_dataset(name, root)
        print(f"Computing cross-split pairs for {name}...", flush=True)
        pairs, nearest = compute_pairs(records, name)
        adjacency = sequence_adjacency(records, name)

        records_public_df(records).to_csv(OUT / f"{name}_image_index.csv", index=False)
        pd.DataFrame(pairs, columns=PAIR_COLUMNS).to_csv(OUT / f"{name}_cross_split_candidates.csv", index=False)
        pd.DataFrame([p for p in pairs if p["high_visual"]], columns=PAIR_COLUMNS).to_csv(OUT / f"{name}_high_visual_pairs.csv", index=False)
        pd.DataFrame(nearest, columns=NEAREST_COLUMNS).to_csv(OUT / f"{name}_closest_cross_split_pairs.csv", index=False)
        pd.DataFrame(adjacency).to_csv(OUT / f"{name}_sequence_adjacency_le_10s.csv", index=False)

        summaries.append(summarize_dataset(name, root, records, pairs, adjacency, errors))
        errors_by_dataset[name] = errors
        all_records.extend(records)
        all_pairs.extend(pairs)
        all_nearest.extend(nearest)
        all_adjacency.extend(adjacency)

    shared = compare_shared_splits(all_records)
    shared.to_csv(OUT / "shared_split_comparison.csv", index=False)

    combined_pairs = pd.DataFrame(all_pairs, columns=PAIR_COLUMNS)
    combined_high = combined_pairs[combined_pairs["high_visual"].fillna(False).astype(bool)].copy()
    combined_nearest = pd.DataFrame(all_nearest, columns=NEAREST_COLUMNS)
    combined_pairs.to_csv(OUT / "combined_cross_split_candidates.csv", index=False)
    combined_high.to_csv(OUT / "combined_high_visual_pairs.csv", index=False)
    combined_nearest.to_csv(OUT / "combined_closest_cross_split_pairs.csv", index=False)

    summary = {
        "datasets": summaries,
        "shared_split_comparison": shared.to_dict(orient="records"),
        "errors_by_dataset": errors_by_dataset,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Generating plots and annotated image sheets...", flush=True)
    make_plots(combined_pairs, summaries, shared)
    plot_similarity_vs_annotation_overlap(combined_nearest, "02_closest_similarity_vs_annotation_overlap.png", "Closest Cross-Split Pairs: Visual Similarity vs Annotation Overlap")
    annotated_pair_sheet(combined_high.sort_values(["score", "ssim_192"], ascending=False), OUT / "05_top_high_visual_pairs_annotated.jpg", "Top High-Visual Cross-Split Pairs", max_pairs=16)
    annotated_pair_sheet(combined_nearest.sort_values(["hash_rank", "ssim_192"], ascending=[True, False]), OUT / "05_closest_cross_split_pairs_annotated.jpg", "Closest Cross-Split Pairs By Hash Distance", max_pairs=16)
    write_markdown_report(summary, combined_high, combined_nearest, shared)
    write_html_report(summary)

    print(f"Wrote analysis to {OUT}", flush=True)
    print(json.dumps({s["dataset"]: {"high_visual_pairs": s["high_visual_pairs"], "very_strong_pairs": s["very_strong_pairs"], "by_split": s["high_visual_by_split_pair"]} for s in summaries}, indent=2), flush=True)


if __name__ == "__main__":
    main()
