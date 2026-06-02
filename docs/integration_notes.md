# Ultralytics Integration Notes

The repository keeps the upstream Ultralytics checkout separate from the study code. Running `scripts/apply_ultralytics_patch.sh` clones the pinned upstream commit into the ignored `vendor/ultralytics/` directory and applies `patches/ultralytics.patch`.

## Patch Scope

| Patched upstream file | Purpose |
| --- | --- |
| `ultralytics/cfg/default.yaml` | Adds explicit `bbox_loss` and `focal_eiou_gamma` training options |
| `ultralytics/nn/modules/block.py` | Adds EMA, ECA, and DySample implementations |
| `ultralytics/nn/modules/__init__.py` | Exports the added modules |
| `ultralytics/nn/tasks.py` | Allows YAML model parsing for EMA, ECA, DySample, CBAM, and Channel Attention |
| `ultralytics/utils/metrics.py` | Extends `bbox_iou` with EIoU and the optional focal overlap-quality weight |
| `ultralytics/utils/loss.py` | Routes detection training through CIoU, EIoU, or Focal-EIoU according to the selected option |

## Baseline Preservation

The default bounding-box regression option remains `ciou`. This means the baseline YOLOv11s behavior is preserved unless training explicitly passes:

```bash
--bbox-loss eiou
```

or:

```bash
--bbox-loss focal_eiou --focal-eiou-gamma 2.0
```

## Architecture Selection

The model YAML files and regression-loss argument are intentionally separate:

- `configs/models/yolo11s_ghost.yaml` selects Ghost Convolution substitutions.
- `configs/models/yolo11s_dysample.yaml` selects DySample at both neck upsampling locations.
- `configs/models/yolo11s_ghost_dysample.yaml` combines Ghost Convolution and DySample.
- `configs/models/yolo11s_ema.yaml`, `yolo11s_cbam.yaml`, `yolo11s_eca.yaml`, and `yolo11s_channel_attention.yaml` define the separate attention study after C2PSA.

Focal-EIoU changes the optimization objective rather than the network topology. This is why FE ablations reuse the relevant model YAML and add the `--bbox-loss focal_eiou` argument.

## Method Attribution

The added modules are integrations of established methods, not newly claimed algorithms. Their original papers and ready-to-import BibTeX entries are listed in [component_citations.md](component_citations.md) and [references.bib](references.bib).
