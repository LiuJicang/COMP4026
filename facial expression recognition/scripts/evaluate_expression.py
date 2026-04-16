import argparse
from pathlib import Path
import sys

import torch
import yaml
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.datasets import ExpressionDataset, PairedExpressionDataset
from src.data.transforms import build_transforms
from src.models.expression_model import build_expression_model
from src.utils.metrics import classification_metrics, expression_consistency_rate


def load_config(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    all_true, all_pred = [], []
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        preds = logits.argmax(dim=1)
        all_true.extend(labels.cpu().tolist())
        all_pred.extend(preds.cpu().tolist())
    return all_true, all_pred


@torch.no_grad()
def paired_predict(model, loader, device):
    model.eval()
    pred_orig, pred_anon, true_labels = [], [], []
    for orig, anon, label in loader:
        orig = orig.to(device)
        anon = anon.to(device)
        true_labels.extend(label.tolist())

        pred_orig.extend(model(orig).argmax(dim=1).cpu().tolist())
        pred_anon.extend(model(anon).argmax(dim=1).cpu().tolist())

    return true_labels, pred_orig, pred_anon


def main():
    parser = argparse.ArgumentParser(description="Evaluate expression model on original and anonymized data")
    parser.add_argument("--config", type=str, default="configs/baseline.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    transform = build_transforms(cfg["data"]["image_size"], is_train=False)

    model = build_expression_model(
        num_classes=cfg["model"]["num_classes"],
        backbone=cfg["model"].get("backbone", "resnet18"),
        pretrained=False,
    ).to(device)
    state_dict = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)

    val_original_ds = ExpressionDataset(
        csv_path=cfg["data"]["val_csv"],
        image_root=cfg["data"]["image_root"],
        transform=transform,
    )

    loader_kwargs = {
        "batch_size": cfg["eval"]["batch_size"],
        "shuffle": False,
        "num_workers": cfg["eval"]["num_workers"],
        "pin_memory": True,
    }

    original_loader = DataLoader(val_original_ds, **loader_kwargs)

    y_true_o, y_pred_o = predict(model, original_loader, device)
    original_metrics = classification_metrics(y_true_o, y_pred_o)

    print("\n=== Original Validation ===")
    print(original_metrics)

    anon_csv = cfg["data"].get("anon_val_csv")
    if anon_csv and Path(anon_csv).exists():
        val_anon_ds = ExpressionDataset(
            csv_path=anon_csv,
            image_root=cfg["data"]["image_root"],
            transform=transform,
        )
        anon_loader = DataLoader(val_anon_ds, **loader_kwargs)
        y_true_a, y_pred_a = predict(model, anon_loader, device)
        anon_metrics = classification_metrics(y_true_a, y_pred_a)

        print("\n=== Anonymized Validation ===")
        print(anon_metrics)
    else:
        print("\n=== Anonymized Validation ===")
        print("Skipped: anon_val_csv is missing or file does not exist.")

    paired_csv = cfg["data"].get("paired_val_csv")
    if paired_csv and Path(paired_csv).exists():
        paired_ds = PairedExpressionDataset(
            csv_path=paired_csv,
            image_root=cfg["data"]["image_root"],
            transform=transform,
        )
        paired_loader = DataLoader(paired_ds, **loader_kwargs)
        y_true_p, pred_orig, pred_anon = paired_predict(model, paired_loader, device)

        pair_orig_metrics = classification_metrics(y_true_p, pred_orig)
        pair_anon_metrics = classification_metrics(y_true_p, pred_anon)
        consistency = expression_consistency_rate(pred_orig, pred_anon)

        print("\n=== Paired Evaluation ===")
        print({
            "pair_orig": pair_orig_metrics,
            "pair_anon": pair_anon_metrics,
            "expression_consistency_rate": consistency,
        })
    else:
        print("\n=== Paired Evaluation ===")
        print("Skipped: paired_val_csv is missing or file does not exist.")


if __name__ == "__main__":
    main()
