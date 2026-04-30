import argparse
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import yaml

MODULE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = MODULE_ROOT.parent
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from src.data.transforms import build_classifier_transform
from src.models.expression_model import build_expression_model
from src.train.expression_trainer import train_expression_teacher


def load_config(config_path: Path):
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def resolve_data_path(image_root: Path, value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = image_root / path
    return path.resolve()


def main():
    parser = argparse.ArgumentParser(description="Train the standalone expression teacher model")
    parser.add_argument("--config", type=str, default="face anonymisation/configs/expression_teacher.yaml")
    args = parser.parse_args()

    cfg = load_config(resolve_repo_path(args.config))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    pin_memory = device == "cuda"

    train_transform = build_classifier_transform(cfg["data"]["image_size"], is_train=True)
    val_transform = build_classifier_transform(cfg["data"]["image_size"], is_train=False)

    image_root = resolve_repo_path(cfg["data"].get("image_root", "."))
    train_dir = resolve_data_path(image_root, cfg["data"]["train_dir"])
    val_dir = resolve_data_path(image_root, cfg["data"]["val_dir"])

    train_dataset = ImageFolder(train_dir, transform=train_transform)
    val_dataset = ImageFolder(val_dir, transform=val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=pin_memory,
    )

    model = build_expression_model(
        num_classes=cfg["model"]["num_classes"],
        backbone=cfg["model"].get("backbone", "resnet18"),
        pretrained=cfg["model"].get("pretrained", True),
    ).to(device)

    output_dir = resolve_repo_path(cfg["train"]["output_dir"])
    train_expression_teacher(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        class_to_idx=train_dataset.class_to_idx,
        cfg=cfg,
        device=device,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
