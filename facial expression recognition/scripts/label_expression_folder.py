import argparse
from pathlib import Path
import sys

import pandas as pd
from PIL import Image
import torch
import yaml
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.transforms import build_transforms
from src.models.expression_model import build_expression_model

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
FER_LABELS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]


def load_config(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class InferenceFolderDataset(Dataset):
    def __init__(self, input_dir: str, transform, max_images: int = 0):
        self.input_dir = Path(input_dir)
        if not self.input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")

        paths = []
        for path in self.input_dir.rglob("*"):
            if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS:
                paths.append(path)

        paths.sort()
        if max_images > 0:
            paths = paths[:max_images]

        if not paths:
            raise ValueError(f"No image files found in: {input_dir}")

        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index: int):
        abs_path = self.paths[index]
        rel_path = abs_path.relative_to(self.input_dir)

        image = Image.open(abs_path).convert("RGB")
        image = self.transform(image)

        return image, str(abs_path), str(rel_path), abs_path.name


@torch.no_grad()
def infer_and_export(model, loader, device, out_csv: str, class_names):
    model.eval()
    rows = []

    for images, abs_paths, rel_paths, file_names in tqdm(loader, desc="Labeling", leave=False):
        images = images.to(device)
        logits = model(images)
        probs = torch.softmax(logits, dim=1)

        confs, pred_ids = probs.max(dim=1)

        for i in range(images.size(0)):
            pred_id = int(pred_ids[i].item())
            confidence = float(confs[i].item())
            pred_label = class_names[pred_id] if pred_id < len(class_names) else f"class_{pred_id}"

            rows.append(
                {
                    "abs_path": abs_paths[i],
                    "rel_path": rel_paths[i].replace("\\", "/"),
                    "file_name": file_names[i],
                    "pred_label_id": pred_id,
                    "pred_label_name": pred_label,
                    "confidence": confidence,
                }
            )

    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"Saved pseudo labels to: {out_path}")
    print(f"Total images labeled: {len(df)}")


def main():
    parser = argparse.ArgumentParser(
        description="Label all face images in a folder using a trained expression model"
    )
    parser.add_argument("--config", type=str, default="facial expression recognition/configs/baseline.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument(
        "--output-csv",
        type=str,
        default="facial expression recognition/data/manifests/pseudo_labels_original.csv",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-images", type=int, default=0)
    args = parser.parse_args()

    cfg = load_config(args.config)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    transform = build_transforms(cfg["data"]["image_size"], is_train=False)
    ds = InferenceFolderDataset(args.input_dir, transform=transform, max_images=args.max_images)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
    )

    model = build_expression_model(
        num_classes=cfg["model"]["num_classes"],
        backbone=cfg["model"].get("backbone", "resnet18"),
        pretrained=False,
    ).to(device)
    state_dict = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)

    class_names = FER_LABELS if cfg["model"]["num_classes"] == 7 else [
        f"class_{i}" for i in range(cfg["model"]["num_classes"])
    ]

    infer_and_export(model, loader, device, args.output_csv, class_names)


if __name__ == "__main__":
    main()
