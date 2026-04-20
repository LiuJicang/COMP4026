import argparse
import json
from pathlib import Path
import sys

import torch
import yaml
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.datasets import FaceFolderDataset, build_paired_records
from src.models.eigenface_recognizer import load_face_recognizer_from_dir
from src.models.resnet_recognizer import ResNet18Recognizer
from src.preprocess.detect_and_crop import FacePreprocessor
from src.utils.metrics import classification_metrics, paired_privacy_metrics


def load_config(config_path: str):
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_model_metadata(model_dir: Path):
    with (model_dir / "model_metadata.json").open("r", encoding="utf-8") as file:
        return json.load(file)


def evaluate_single_dataset(model, dataset, preprocessor):
    y_true = []
    y_pred = []
    confidences = []
    skipped = []

    for record in tqdm(dataset.records, desc="Eval original", unit="image", leave=False):
        image_path = record.image_path
        try:
            face = preprocessor.preprocess_path(image_path)
            prediction = model.predict(face)
            y_true.append(record.person_name)
            y_pred.append(prediction.predicted_name)
            confidences.append(prediction.confidence)
        except Exception as exc:
            skipped.append({"image_path": str(image_path), "error": str(exc)})

    metrics = classification_metrics(y_true, y_pred, confidences)
    metrics["evaluated_samples"] = len(y_true)
    metrics["skipped_samples"] = len(skipped)
    metrics["skipped_examples"] = skipped[:20]
    return metrics


def evaluate_single_resnet_dataset(model, dataset, preprocessor, batch_size: int, num_workers: int):
    from torch.utils.data import DataLoader

    from src.data.torch_datasets import ResNetFaceDataset

    tensor_dataset = ResNetFaceDataset(dataset, preprocessor)
    loader = DataLoader(
        tensor_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    y_true = []
    y_pred = []
    confidences = []

    progress = tqdm(loader, total=len(loader), desc="Eval batches", unit="batch", leave=False)

    for images, _, indices in progress:
        predictions = model.infer_batch(images)
        for index, prediction in zip(indices.tolist(), predictions):
            record = dataset.records[index]
            y_true.append(record.person_name)
            y_pred.append(prediction.predicted_name)
            confidences.append(prediction.confidence)
        progress.set_postfix(samples=len(y_true))

    metrics = classification_metrics(y_true, y_pred, confidences)
    metrics["evaluated_samples"] = len(y_true)
    metrics["skipped_samples"] = 0
    metrics["skipped_examples"] = []
    return metrics


def evaluate_paired_records(model, paired_records, preprocessor):
    y_true = []
    pred_orig = []
    pred_anon = []
    conf_orig = []
    conf_anon = []
    skipped = []

    for record in tqdm(paired_records, desc="Paired eval", unit="pair", leave=False):
        try:
            orig_face = preprocessor.preprocess_path(record.orig_path)
            anon_face = preprocessor.preprocess_path(record.anon_path)
            orig_prediction = model.predict(orig_face)
            anon_prediction = model.predict(anon_face)

            y_true.append(record.person_name)
            pred_orig.append(orig_prediction.predicted_name)
            pred_anon.append(anon_prediction.predicted_name)
            conf_orig.append(orig_prediction.confidence)
            conf_anon.append(anon_prediction.confidence)
        except Exception as exc:
            skipped.append(
                {
                    "orig_path": str(record.orig_path),
                    "anon_path": str(record.anon_path),
                    "error": str(exc),
                }
            )

    result = {
        "paired_original": classification_metrics(y_true, pred_orig, conf_orig),
        "paired_anonymized": classification_metrics(y_true, pred_anon, conf_anon),
        "privacy_shift": paired_privacy_metrics(pred_orig, pred_anon),
        "evaluated_samples": len(y_true),
        "skipped_samples": len(skipped),
        "skipped_examples": skipped[:20],
    }
    return result


def evaluate_paired_resnet_records(model, paired_records, preprocessor):
    y_true = []
    pred_orig = []
    pred_anon = []
    conf_orig = []
    conf_anon = []
    skipped = []

    progress = tqdm(paired_records, desc="Paired eval", unit="pair", leave=False)

    for record in progress:
        try:
            orig_face = preprocessor.preprocess_color_path(record.orig_path)
            anon_face = preprocessor.preprocess_color_path(record.anon_path)

            orig_tensor = torch.from_numpy(orig_face).permute(2, 0, 1).float() / 255.0
            anon_tensor = torch.from_numpy(anon_face).permute(2, 0, 1).float() / 255.0
            mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)
            orig_tensor = (orig_tensor - mean) / std
            anon_tensor = (anon_tensor - mean) / std

            orig_prediction = model.infer_batch(orig_tensor.unsqueeze(0))[0]
            anon_prediction = model.infer_batch(anon_tensor.unsqueeze(0))[0]

            y_true.append(record.person_name)
            pred_orig.append(orig_prediction.predicted_name)
            pred_anon.append(anon_prediction.predicted_name)
            conf_orig.append(orig_prediction.confidence)
            conf_anon.append(anon_prediction.confidence)
            progress.set_postfix(pairs=len(y_true))
        except Exception as exc:
            skipped.append(
                {
                    "orig_path": str(record.orig_path),
                    "anon_path": str(record.anon_path),
                    "error": str(exc),
                }
            )

    return {
        "paired_original": classification_metrics(y_true, pred_orig, conf_orig),
        "paired_anonymized": classification_metrics(y_true, pred_anon, conf_anon),
        "privacy_shift": paired_privacy_metrics(pred_orig, pred_anon),
        "evaluated_samples": len(y_true),
        "skipped_samples": len(skipped),
        "skipped_examples": skipped[:20],
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate a face recognition model on original and anonymized images")
    parser.add_argument("--config", type=str, default="configs/baseline.yaml")
    parser.add_argument("--model-dir", type=str, default="")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_dir = Path(args.model_dir) if args.model_dir else Path(cfg["train"]["output_dir"])
    metadata = load_model_metadata(model_dir)
    model_name = metadata.get("model_name", "eigenfaces").lower()

    if model_name == "resnet18":
        device = cfg.get("train", {}).get("device", "auto")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        model = ResNet18Recognizer.load(model_dir, device=device)
    else:
        model = load_face_recognizer_from_dir(model_dir)
    preprocessor = FacePreprocessor.from_config(cfg["preprocess"])

    summary = {}

    test_dataset = FaceFolderDataset(
        image_root=cfg["data"]["image_root"],
        split_dir=cfg["data"]["test_dir"],
        label_to_name=model.label_to_name,
    )
    if len(test_dataset):
        if model_name == "resnet18":
            summary["original_test"] = evaluate_single_resnet_dataset(
                model,
                test_dataset,
                preprocessor,
                batch_size=cfg.get("eval", {}).get("batch_size", 32),
                num_workers=cfg.get("eval", {}).get("num_workers", 0),
            )
        else:
            summary["original_test"] = evaluate_single_dataset(model, test_dataset, preprocessor)
        print("\n=== Original Test ===")
        print(summary["original_test"])
    else:
        print("\n=== Original Test ===")
        print("Skipped: test folder is empty.")

    anonymized_test_dir = cfg["data"].get("anonymized_test_dir", "")
    if anonymized_test_dir:
        anonymized_dataset = FaceFolderDataset(
            image_root=cfg["data"]["image_root"],
            split_dir=anonymized_test_dir,
            label_to_name=model.label_to_name,
        )
    else:
        anonymized_dataset = None

    if anonymized_dataset and len(anonymized_dataset):
        if model_name == "resnet18":
            summary["anonymized_test"] = evaluate_single_resnet_dataset(
                model,
                anonymized_dataset,
                preprocessor,
                batch_size=cfg.get("eval", {}).get("batch_size", 32),
                num_workers=cfg.get("eval", {}).get("num_workers", 0),
            )
        else:
            summary["anonymized_test"] = evaluate_single_dataset(model, anonymized_dataset, preprocessor)
        print("\n=== Anonymized Test ===")
        print(summary["anonymized_test"])

        paired_records = build_paired_records(test_dataset, anonymized_dataset)
        if paired_records:
            if model_name == "resnet18":
                summary["paired_evaluation"] = evaluate_paired_resnet_records(model, paired_records, preprocessor)
            else:
                summary["paired_evaluation"] = evaluate_paired_records(model, paired_records, preprocessor)
            print("\n=== Paired Evaluation ===")
            print(summary["paired_evaluation"])
        else:
            print("\n=== Paired Evaluation ===")
            print("Skipped: no matching original/anonymized files were found.")
    else:
        print("\n=== Anonymized Test ===")
        print("Skipped: anonymized_test_dir is empty or not configured.")

        print("\n=== Paired Evaluation ===")
        print("Skipped: anonymized_test_dir is empty or not configured.")

    if cfg.get("eval", {}).get("save_json", True):
        output_filename = cfg.get("eval", {}).get("summary_json", "evaluation_summary.json")
        output_path = model_dir / output_filename
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(summary, file, indent=2, ensure_ascii=False)
        print(f"\nSaved evaluation summary to: {output_path}")


if __name__ == "__main__":
    main()
