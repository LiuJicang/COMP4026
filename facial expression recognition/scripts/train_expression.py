import argparse
from pathlib import Path
import sys

import torch
import yaml
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.datasets import ExpressionDataset
from src.data.transforms import build_transforms
from src.models.expression_model import build_expression_model
from src.train.trainer import train_model


def load_config(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Train expression recognition baseline")
    parser.add_argument("--config", type=str, default="configs/baseline.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    train_transform = build_transforms(cfg["data"]["image_size"], is_train=True)
    val_transform = build_transforms(cfg["data"]["image_size"], is_train=False)

    train_ds = ExpressionDataset(
        csv_path=cfg["data"]["train_csv"],
        image_root=cfg["data"]["image_root"],
        transform=train_transform,
    )
    val_ds = ExpressionDataset(
        csv_path=cfg["data"]["val_csv"],
        image_root=cfg["data"]["image_root"],
        transform=val_transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=True,
    )

    model = build_expression_model(
        num_classes=cfg["model"]["num_classes"],
        backbone=cfg["model"].get("backbone", "resnet18"),
        pretrained=cfg["model"].get("pretrained", True),
    ).to(device)

    output_dir = Path(cfg["train"]["output_dir"])
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=cfg["train"]["epochs"],
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
        device=device,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
