# Dataset Layout

The scripts expect YOLO-format labels. Each non-empty label row is:

```text
class_id x_center y_center width height
```

Coordinates are normalized to `[0, 1]`. Each image must have a matching `.txt` annotation with the same stem.

## Density-Stratified Development Data

```text
data/density_development/
  images/
    train/
    val/
  labels/
    train/
    val/
```

Training images represent free-to-moderate species-specific scenes. Validation images represent concentrated species-specific scenes.

## Independent Mixed-Species Test Domains

```text
data/test_free/
  images/test/
  labels/test/

data/test_conc/
  images/test/
  labels/test/
```

`test_free` and `test_conc` contain different images and are evaluated as separate domains.

## Leakage Audit Layout

The leakage audit accepts the chapter-style comparison layout:

```text
data/leakage_audit/
  Detect/
    images/{train,val,test}/
    labels/{train,val,test}/
  Detect-conc/
    images/{train,val,test}/
    labels/{train,val,test}/
```

Run:

```bash
python scripts/analyze_leakage.py --data-root data/leakage_audit
```
