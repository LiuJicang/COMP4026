# COMP4026 Group Project - Part C (Expression Recognition)

Repository structure for team collaboration:
- face recognition/
- face anonymisation/
- facial expression recognition/

This implementation provides a practical baseline for Part C:
- Train expression classifier on original images.
- Evaluate on anonymized images.
- Report expression consistency rate on original-anonymized pairs.

## 1) Interface Contract with Part B (Anonymization)

Part C does not need to share model internals with Part B, but must share a strict data interface.

Required outputs from Part B:
1. An anonymized validation set with expression labels (same class schema as original set).
2. A paired manifest mapping each original image to its anonymized image.

Manifest formats:
- `facial expression recognition/data/manifests/val_anonymized.csv`
  - columns: `image_path,label`
- `facial expression recognition/data/manifests/val_paired.csv`
  - columns: `orig_path,anon_path,label`

All paths are relative to `data.image_root` from config unless absolute paths are used.

## 2) Setup

```bash
pip install -r requirements.txt
```

## 3) Configure Data

Edit `facial expression recognition/configs/baseline.yaml` to match your local data folders.

Fill manifests:
- `facial expression recognition/data/manifests/train_original.csv`
- `facial expression recognition/data/manifests/val_original.csv`
- `facial expression recognition/data/manifests/val_anonymized.csv`
- `facial expression recognition/data/manifests/val_paired.csv`

## 4) Train Baseline

```bash
python facial\ expression\ recognition/scripts/train_expression.py --config facial\ expression\ recognition/configs/baseline.yaml
```

Best checkpoint is saved to:
- `facial expression recognition/outputs/baseline/best_model.pt`

## 5) Evaluate Utility + Consistency

```bash
python facial\ expression\ recognition/scripts/evaluate_expression.py --config facial\ expression\ recognition/configs/baseline.yaml --checkpoint facial\ expression\ recognition/outputs/baseline/best_model.pt
```

Outputs include:
- Original validation accuracy / macro F1
- Anonymized validation accuracy / macro F1
- Paired evaluation:
  - accuracy on original side
  - accuracy on anonymized side
  - expression consistency rate

## 6) Suggested Next Upgrade

If anonymized performance drops too much, implement dual-branch consistency learning:
- Keep classification loss on original + anonymized images.
- Add consistency loss between features of paired images.
