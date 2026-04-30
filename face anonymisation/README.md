# Identity-Expression Anonymizer

This folder contains your standalone face anonymisation system inside the shared COMP4026 repository.

- it only adds code under `face anonymisation/`
- it does not import or modify your teammates' modules
- it targets: "not the same person" + "same expression"

## Project Layout

- `configs/`: training and inference configs
- `scripts/`: runnable entry points
- `src/`: models, guidance modules, losses, data helpers
- `outputs/`: generated checkpoints and results

## Install

```powershell
pip install -r "face anonymisation/requirements.txt"
pip install facenet-pytorch --no-deps
```

## Prepare Datasets

If your aligned face archive is stored as `D:\4026\archive.zip`, you can build the anonymizer train/val folders with:

```powershell
python "face anonymisation/scripts/prepare_celebahq_from_zip.py"
```

This will auto-discover `archive.zip` and write:

```text
CelebA-HQ/
  train/
  val/
```

To build the expression dataset directly from a public FER-2013 mirror:

```powershell
python "face anonymisation/scripts/prepare_fer2013_from_hf.py"
```

This writes:

```text
FER-2013/
  train/
  test/
  val/
```

## Train The Expression Teacher

This is your local expression-preservation teacher model.

Expected dataset layout:

```text
FER-2013/
  train/
    angry/
    disgust/
    fear/
    happy/
    neutral/
    sad/
    surprise/
  test/
    angry/
    ...
```

Run:

```powershell
python "face anonymisation/scripts/train_expression_teacher.py" --config "face anonymisation/configs/expression_teacher.yaml"
```

The teacher checkpoint is saved to:

- `face anonymisation/outputs/expression_teacher/best_model.pt`
- the default config uses `mobilenet_v3_small` at `112x112`, which is much more practical on CPU

## Train The Identity + Expression Anonymizer

Expected face dataset layout:

```text
CelebA-HQ/
  train/
    000001.jpg
    000002.jpg
  val/
    030001.jpg
    030002.jpg
```

Run:

```powershell
python "face anonymisation/scripts/train_gan_anonymizer.py" --config "face anonymisation/configs/gan_identity_expression.yaml"
```

This training uses:

- `FaceNet` identity guidance to push the generated face away from the original identity
- your local expression teacher to preserve expression predictions
- adversarial and reconstruction losses to keep outputs realistic

If you are training on CPU, use the smaller config first:

```powershell
python "face anonymisation/scripts/train_gan_anonymizer.py" --config "face anonymisation/configs/gan_identity_expression_cpu.yaml"
```

To continue from that checkpoint with a stronger CPU run:

```powershell
python "face anonymisation/scripts/train_gan_anonymizer.py" --config "face anonymisation/configs/gan_identity_expression_cpu_stage2.yaml"
```

## Run Inference

```powershell
python "face anonymisation/scripts/infer_gan_anonymizer.py" --config "face anonymisation/configs/gan_identity_expression.yaml"
```

## Optional Classical Baselines

```powershell
python "face anonymisation/scripts/run_anonymization.py" --config "face anonymisation/configs/baseline_blur.yaml"
python "face anonymisation/scripts/run_anonymization.py" --config "face anonymisation/configs/baseline_pixelate.yaml"
```

## Notes

- The identity-aware training path depends on `facenet-pytorch`.
- The expression-aware path depends on the checkpoint produced by `train_expression_teacher.py`.
- If you do not have `CelebA-HQ`, you can point the GAN configs to another aligned face dataset by changing `data.train_dir` and `data.val_dir`.
