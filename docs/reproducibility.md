# Reproducibility Guide

## Evaluation Protocol

The detector is developed using a species-specific density-stratified train-validation split and evaluated on two independent mixed-species test domains:

- `test free`: lower scene density and more dispersed insect locations.
- `test conc`: higher scene density and greater spatial concentration.

The random 80:20 split experiment is a controlled comparison of partition strategies. Architecture and training settings remain fixed.

## Density Characterization

For image `I` with width `W`, height `H`, and `N` annotated instances:

```text
count density       = N
scene density       = N / (W * H / 1,000,000)           instances/MPx
area occupancy      = 100 * sum(box areas) / (W * H)    percent
spatial concentration = mean nearest-neighbor centroid distance / image diagonal
```

Lower normalized nearest-neighbor distance indicates greater spatial concentration.

## Training Defaults

| Setting | Value |
| --- | --- |
| Input size | 640 x 640 |
| Epochs | 300 |
| Batch size | 16 |
| Optimizer | AdamW |
| Initial learning rate | 0.001 |
| Final LR fraction | 0.01 |
| Momentum / beta1 | 0.937 |
| Weight decay | 0.0005 |
| Warmup epochs | 3 |
| Cosine LR | enabled |
| Close mosaic | final 10 epochs |
| Seed | 0 |

Training augmentation: HSV perturbation, rotation, translation, scaling, shear, perspective, horizontal flipping, mosaic, MixUp, and RandAugment. Random erasing is not claimed as an applied detection augmentation.

## Model Matrix

| Experiment | Model YAML | Box regression option |
| --- | --- | --- |
| YOLOv11s baseline | upstream `yolo11s.pt` | `ciou` |
| GC | `yolo11s_ghost.yaml` | `ciou` |
| DS | `yolo11s_dysample.yaml` | `ciou` |
| FE | upstream `yolo11s.pt` | `focal_eiou` |
| GC + DS | `yolo11s_ghost_dysample.yaml` | `ciou` |
| GC + FE | `yolo11s_ghost.yaml` | `focal_eiou` |
| DS + FE | `yolo11s_dysample.yaml` | `focal_eiou` |
| GC + DS + FE | `yolo11s_ghost_dysample.yaml` | `focal_eiou` |
| EMA after C2PSA | `yolo11s_ema.yaml` | `ciou` |
| CBAM after C2PSA | `yolo11s_cbam.yaml` | `ciou` |
| ECA after C2PSA | `yolo11s_eca.yaml` | `ciou` |
| Channel attention after C2PSA | `yolo11s_channel_attention.yaml` | `ciou` |

The main architectural ablation is intentionally complete: it includes the YOLOv11s baseline, each individual component, every two-component combination, and the full three-component model. Focal-EIoU changes the training objective rather than the network topology, so FE variants reuse the relevant model YAML with `--bbox-loss focal_eiou`.

## Detection Metrics

- Precision: fraction of predicted detections that are correct.
- Recall: fraction of ground-truth instances that are detected.
- `mAP50`: mean average precision at IoU `0.50`.
- `mAP50-95`: mean average precision averaged over IoU thresholds `0.50:0.05:0.95`.
- Class-wise precision: per-species prediction reliability.

## Loss Behavior

The patched code preserves CIoU as the default baseline regression loss. For FE experiments, pass:

```bash
--bbox-loss focal_eiou --focal-eiou-gamma 2.0
```

The implemented regression term is:

```text
L_focal-EIoU = IoU^gamma * L_EIoU
```

This makes Focal-EIoU explicit in command history and prevents accidental mixing of baseline and FE runs.
