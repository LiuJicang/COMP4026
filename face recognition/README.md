# Face Recognition - Part A

This module implements the Part A baseline for the COMP4026 group project:

- Face detection: `Viola-Jones` via Haar Cascade
- Face recognition: `Eigenfaces` via OpenCV
- Open-set rejection: confidence threshold mapped to `unknown`

## Dataset Layout

Put your data directly under `face recognition/data/`:

```text
face recognition/
  data/
    train/
      Alice/
        img_001.jpg
        img_002.jpg
      Bob/
        img_001.jpg
    test/
      Alice/
        img_101.jpg
      Bob/
        img_101.jpg
    test_anonymized/
      Alice/
        img_101.jpg
      Bob/
        img_101.jpg
```

Folder names are the identity labels. You do not need to write `label0`, `label1`, or prepare manifest CSV files.

## Train

```powershell
conda run -n comp4026 python "face recognition/scripts/train_recognition.py" --config "face recognition/configs/baseline.yaml"
```

## Evaluate

```powershell
conda run -n comp4026 python "face recognition/scripts/evaluate_recognition.py" --config "face recognition/configs/baseline.yaml" --model-dir "face recognition/outputs/baseline"
```

## Notes

- If Haar Cascade is unavailable in the environment, the preprocessor falls back to full-image or center-crop mode so development can continue.
- `threshold` controls when a prediction is rejected as `unknown`. Lower thresholds make rejection stricter.
- `test_anonymized/` is optional. If you later provide it with the same folder and file structure as `test/`, the evaluator will also report anonymized and paired privacy metrics.
