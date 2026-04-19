import argparse
import json
from pathlib import Path
import sys

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.datasets import FaceFolderDataset
from src.models.eigenface_recognizer import build_face_recognizer_from_config
from src.preprocess.detect_and_crop import FacePreprocessor
from src.train.resnet_trainer import train_resnet18_from_config


def load_config(config_path: str):
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def main():
    parser = argparse.ArgumentParser(description="Train a face recognition model")
    parser.add_argument("--config", type=str, default="configs/baseline.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    recognizer_name = cfg["model"].get("recognizer", "eigenfaces").lower()

    if recognizer_name == "resnet18":
        output_dir, summary = train_resnet18_from_config(cfg)
        with (output_dir / "training_summary.json").open("w", encoding="utf-8") as file:
            json.dump(summary, file, indent=2, ensure_ascii=False)
        print(f"Training samples used: {summary['train_samples_used']} / {summary['train_samples_seen']}")
        print("Skipped samples: 0")
        print(f"Model saved to: {output_dir}")
        return

    dataset = FaceFolderDataset(
        image_root=cfg["data"]["image_root"],
        split_dir=cfg["data"]["train_dir"],
    )

    if len(dataset) == 0:
        raise ValueError("Training folder is empty. Please put images under data/train/<person_name>/ before training.")

    preprocessor = FacePreprocessor.from_config(cfg["preprocess"])
    samples = []
    labels = []
    skipped = []

    for record in dataset.records:
        image_path = record.image_path
        try:
            samples.append(preprocessor.preprocess_path(image_path))
            labels.append(record.label)
        except Exception as exc:
            skipped.append({"image_path": str(image_path), "error": str(exc)})

    if not samples:
        raise RuntimeError("No training samples were processed successfully.")

    model = build_face_recognizer_from_config(cfg)
    model.train(
        faces=samples,
        labels=labels,
        label_to_name=dataset.label_to_name,
    )

    output_dir = Path(cfg["train"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save(output_dir)

    summary = {
        "train_dir": str(dataset.split_dir),
        "train_samples_seen": len(dataset),
        "train_samples_used": len(samples),
        "train_samples_skipped": len(skipped),
        "recognizer": cfg["model"].get("recognizer", "eigenfaces"),
        "labels": model.label_to_name,
        "threshold": model.threshold,
        "image_size": list(model.image_size),
        "cascade_path": str(preprocessor.cascade_path) if preprocessor.cascade_path else "",
        "fallback_mode": preprocessor.fallback_mode,
        "skipped_examples": skipped[:20],
    }
    for key, value in cfg["model"].items():
        if key not in summary:
            summary[key] = value

    with (output_dir / "training_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    print(f"Training samples used: {len(samples)} / {len(dataset)}")
    print(f"Skipped samples: {len(skipped)}")
    print(f"Model saved to: {output_dir}")


if __name__ == "__main__":
    main()
