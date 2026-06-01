# Density-Stratified Generalization for Stored-Product Pest Detection

This repository is the reproducibility companion for a YOLOv11s study of tiny stored-product pest detection under scene-density shift. It packages the study code without datasets, trained weights, or machine-specific paths.

The experiments address three questions:

1. Does density-stratified partitioning provide a more realistic estimate of detector generalization than a conventional random split for sequentially captured images?
2. How do Ghost Convolution (GC), DySample (DS), and Focal-EIoU (FE) affect detection across free and concentrated scene-density domains?
3. Does feature recalibration after the YOLOv11 C2PSA block improve robustness under density shift?

The target classes are:

| ID | Code | Species and life stage |
| --- | --- | --- |
| 0 | TB | *Tribolium castaneum*, adult |
| 1 | RZ | *Rhyzopertha dominica*, adult |
| 2 | KP | *Trogoderma granarium* (Khapra beetle), larva |

## Repository Contents

| Path | Purpose |
| --- | --- |
| `patches/ultralytics_chapter2.patch` | Clean Ultralytics integration for DySample, EMA, ECA, attention YAML parsing, and opt-in Focal-EIoU |
| `configs/models/` | YOLOv11s architectural variants used by the ablation and attention studies |
| `configs/datasets/` | Portable dataset YAML templates |
| `scripts/train.py` | Reproducible training entry point |
| `scripts/evaluate.py` | Evaluation metrics and timing in milliseconds per image and FPS |
| `scripts/infer_video.py` | Annotated video inference for qualitative demonstrations |
| `scripts/analyze_scene_density.py` | Scene density, occupancy, nearest-neighbor concentration, CSV summaries, and publication plots |
| `scripts/create_random_split.py` | Random 80:20 train-validation partition for the leakage comparison |
| `scripts/analyze_leakage.py` | Cross-split exact-hash, perceptual-hash, SSIM, and annotation-overlap audit |
| `scripts/visualize_gradcam.py` | Grad-CAM comparison panels from a model manifest |
| `scripts/visualize_detections.py` | Qualitative detection panels from a model manifest |

## Setup

The integration is pinned to Ultralytics commit `eec4148e7b976cbbe1378aeee03f52337c79479e`.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
bash scripts/apply_ultralytics_patch.sh
export PYTHONPATH="$PWD/vendor/ultralytics:$PYTHONPATH"
```

The setup script clones the pinned Ultralytics source into `vendor/ultralytics` and applies the chapter patch. The patch preserves baseline CIoU behavior. Focal-EIoU is enabled only when `--bbox-loss focal_eiou` is passed to training.

## Dataset Layout

Place local data under `data/` or pass explicit paths to the scripts. See [docs/dataset_layout.md](docs/dataset_layout.md). Dataset images and annotations are intentionally excluded from Git.

## Core Experiments

Train the density-stratified baseline:

```bash
python scripts/train.py \
  --model yolo11s.pt \
  --data configs/datasets/density_development.example.yaml \
  --name yolo11s_density_baseline
```

Train the modified GC + DS + FE model:

```bash
python scripts/train.py \
  --model configs/models/yolo11s_ghost_dysample.yaml \
  --data configs/datasets/density_development.example.yaml \
  --bbox-loss focal_eiou \
  --name yolo11s_ghost_dysample_focal_eiou
```

Evaluate on the two independent mixed-species domains:

```bash
python scripts/evaluate.py --weights weights/best.pt --data configs/datasets/test_free.example.yaml
python scripts/evaluate.py --weights weights/best.pt --data configs/datasets/test_conc.example.yaml
```

Characterize density shift:

```bash
python scripts/analyze_scene_density.py \
  --dataset configs/experiments/density_splits.example.yaml \
  --output-dir outputs/density_analysis
```

Regenerate metric heatmaps:

```bash
python scripts/plot_result_heatmaps.py
```

## Qualitative Demo

[View the YOLOv11s + CBAM mixed-scene detection video](docs/assets/yolo11s_cbam_mixed_scene_demo.mp4).

[![YOLOv11s + CBAM mixed-scene detections](docs/assets/yolo11s_cbam_mixed_scene_poster.jpg)](docs/assets/yolo11s_cbam_mixed_scene_demo.mp4)

The demo uses the standalone CBAM attention variant reported in the attention study. The video is a qualitative illustration, not a substitute for the independent test-set metrics.

Regenerate a video with a local checkpoint:

```bash
python scripts/infer_video.py \
  --weights weights/yolo11s_cbam.pt \
  --source data/demo/mixed_scene.mp4 \
  --output outputs/yolo11s_cbam_mixed_scene_demo.mp4
```

Create the random 80:20 comparison split:

```bash
python scripts/create_random_split.py \
  --source data/density_development \
  --output data/random_80_20 \
  --seed 42
```

## Reproducibility Notes

- Images are resized to `640 x 640`.
- Training defaults reproduce the chapter configuration: 300 epochs, batch size 16, AdamW, cosine learning-rate decay, and mosaic disabled during the final 10 epochs.
- Augmentation is applied to training images only. Validation and test images are not augmented.
- The reported FPS values are throughput estimates derived from Ultralytics milliseconds-per-image timings. Use repeated runs on the same hardware for comparative latency claims.
- The repository uses AGPL-3.0 because it distributes a patch against Ultralytics AGPL-3.0 source.

See [docs/reproducibility.md](docs/reproducibility.md) for the experiment matrix and metric definitions.
The locally inspected environment is recorded in [docs/environment.md](docs/environment.md); confirm it matches the final experiment machine before manuscript submission.
