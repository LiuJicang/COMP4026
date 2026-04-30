import argparse
import json
from pathlib import Path
import random
import sys

import cv2
import torch
from tqdm import tqdm
import yaml

MODULE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = MODULE_ROOT.parent
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from src.data.transforms import image_to_tensor, tensor_to_bgr_image
from src.detection import FaceRegionDetector
from src.io_utils import (
    ProcessedRecord,
    load_image_records,
    records_have_labels,
    write_anonymized_manifest,
    write_paired_manifest,
)
from src.landmarks import FaceLandmarkDetector
from src.models.generator import UNetAnonymizer
from src.region_aware import apply_region_aware_blend

def load_config(config_path: Path):
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def resolve_optional_repo_path(value: str) -> Path | None:
    return resolve_repo_path(value) if value else None


def resolve_data_path(image_root: Path, value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = image_root / path
    return path.resolve()


def load_generator(cfg: dict, checkpoint_path: Path, device: str):
    generator_cfg = cfg["model"]["generator"]
    model = UNetAnonymizer(
        in_channels=3,
        out_channels=3,
        base_channels=generator_cfg.get("base_channels", 64),
        noise_channels=generator_cfg.get("noise_channels", 1),
        bottleneck_dropout=generator_cfg.get("bottleneck_dropout", 0.0),
        bottleneck_blocks=generator_cfg.get("bottleneck_blocks", 0),
        residual_output=generator_cfg.get("residual_output", False),
        residual_scale=generator_cfg.get("residual_scale", 0.5),
    ).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model


@torch.no_grad()
def infer_region(model, image_bgr, image_size: int, noise_channels: int, noise_std: float, device: str):
    height, width = image_bgr.shape[:2]
    tensor = image_to_tensor(image_bgr, image_size).unsqueeze(0).to(device)
    noise = torch.randn((1, noise_channels, image_size, image_size), device=device) * noise_std
    output = model(tensor, noise)
    rendered = tensor_to_bgr_image(output[0])
    return cv2.resize(rendered, (width, height), interpolation=cv2.INTER_LINEAR)


def choose_region(image, apply_to: str, detector: FaceRegionDetector):
    if apply_to == "full_image":
        height, width = image.shape[:2]
        return (0, 0, width, height), "full_image"

    selection = detector.select_region(image)
    if selection.box is None:
        raise RuntimeError("No face detected and inference.detection_enabled with face-only mode could not find a region.")
    return selection.box, selection.mode


def main():
    parser = argparse.ArgumentParser(description="Run learned face anonymisation inference on a directory or manifest")
    parser.add_argument("--config", type=str, default="face anonymisation/configs/gan_baseline.yaml")
    parser.add_argument("--limit", type=int, default=0, help="Optional cap on the number of images to process")
    args = parser.parse_args()

    cfg = load_config(resolve_repo_path(args.config))
    inference_cfg = cfg["inference"]
    export_cfg = cfg.get("export", {})

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    seed = int(inference_cfg.get("seed", 42))
    random.seed(seed)
    torch.manual_seed(seed)

    image_root = resolve_repo_path(cfg["data"].get("image_root", "."))
    source_dir = inference_cfg.get("source_dir", "")
    source_dir = str(resolve_data_path(image_root, source_dir)) if source_dir else ""
    manifest_csv_value = inference_cfg.get("manifest_csv", "")
    manifest_csv = resolve_data_path(image_root, manifest_csv_value) if manifest_csv_value else None

    records = load_image_records(
        image_root=image_root,
        source_dir=source_dir,
        manifest_csv=manifest_csv,
        label_mode=inference_cfg.get("label_mode", "none"),
        supported_extensions=cfg["data"].get("supported_extensions", []),
    )
    if args.limit > 0:
        records = records[: args.limit]

    output_dir = resolve_repo_path(inference_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = resolve_repo_path(inference_cfg["checkpoint"])
    generator = load_generator(cfg, checkpoint_path, device)
    image_size = int(cfg["data"]["image_size"])
    noise_channels = int(cfg["model"]["generator"].get("noise_channels", 1))
    noise_std = float(inference_cfg.get("noise_std", 1.0))

    apply_to = inference_cfg.get("apply_to", "full_image")
    detection_cfg = dict(cfg.get("detection", {}))
    if "detection_enabled" in inference_cfg:
        detection_cfg["enabled"] = inference_cfg.get("detection_enabled", False)
    detection_cfg["fallback_mode"] = "full_image"
    detector = FaceRegionDetector.from_config(detection_cfg)
    landmark_detector = FaceLandmarkDetector.from_config(cfg.get("landmarks", {}), resolve_repo_path)
    region_aware_cfg = inference_cfg.get("region_aware", {})
    region_aware_enabled = bool(region_aware_cfg.get("enabled", False))
    landmark_enabled = bool(cfg.get("landmarks", {}).get("enabled", False))
    preserve_expression_enabled = bool(region_aware_cfg.get("preserve_expression", {}).get("enabled", False))
    abstract_enabled = bool(region_aware_cfg.get("abstract", {}).get("enabled", False))

    processed_records: list[ProcessedRecord] = []
    skipped_examples = []
    detected_face_count = 0
    full_image_count = 0
    landmark_detected_count = 0

    try:
        for record in tqdm(records, desc="Inferring", unit="image"):
            try:
                image = cv2.imread(str(record.source_path))
                if image is None:
                    raise FileNotFoundError(f"Failed to read image: {record.source_path}")

                region, region_mode = choose_region(image, apply_to, detector)
                x, y, width, height = region

                result = image.copy()
                roi = image[y : y + height, x : x + width]
                inferred_roi = infer_region(generator, roi, image_size, noise_channels, noise_std, device)
                landmarks = landmark_detector.detect(roi)
                if landmarks is not None:
                    landmark_detected_count += 1

                if region_aware_enabled and apply_to == "face":
                    result[y : y + height, x : x + width] = apply_region_aware_blend(
                        roi,
                        inferred_roi,
                        region_aware_cfg,
                        landmarks=landmarks,
                    )
                else:
                    result[y : y + height, x : x + width] = inferred_roi

                if region_mode == "detected_face":
                    detected_face_count += 1
                else:
                    full_image_count += 1

                output_path = output_dir / record.relative_path
                output_path.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(output_path), result):
                    raise IOError(f"Failed to write image: {output_path}")

                processed_records.append(
                    ProcessedRecord(
                        source_path=record.source_path,
                        output_path=output_path,
                        label=record.label,
                    )
                )
            except Exception as exc:
                skipped_examples.append(
                    {
                        "source_path": str(record.source_path),
                        "error": str(exc),
                    }
                )
    finally:
        landmark_detector.close()

    labels_available = records_have_labels(processed_records)
    anonymized_manifest_path = resolve_optional_repo_path(export_cfg.get("anonymized_manifest", ""))
    paired_manifest_path = resolve_optional_repo_path(export_cfg.get("paired_manifest", ""))
    summary_json_path = resolve_optional_repo_path(export_cfg.get("summary_json", ""))

    if labels_available and anonymized_manifest_path is not None:
        write_anonymized_manifest(anonymized_manifest_path, image_root, processed_records)
    elif anonymized_manifest_path is not None:
        print("Skipped anonymized_manifest export because labels are missing.")

    if labels_available and paired_manifest_path is not None:
        write_paired_manifest(paired_manifest_path, image_root, processed_records)
    elif paired_manifest_path is not None:
        print("Skipped paired_manifest export because labels are missing.")

    summary = {
        "checkpoint": str(checkpoint_path),
        "image_root": str(image_root),
        "input_mode": "manifest" if manifest_csv else "source_dir",
        "source_dir": source_dir,
        "manifest_csv": str(manifest_csv) if manifest_csv else "",
        "apply_to": apply_to,
        "region_aware_enabled": region_aware_enabled,
        "landmark_enabled": landmark_enabled,
        "landmark_detected_samples": landmark_detected_count,
        "preserve_expression_enabled": preserve_expression_enabled,
        "abstract_enabled": abstract_enabled,
        "records_discovered": len(records),
        "processed_samples": len(processed_records),
        "skipped_samples": len(skipped_examples),
        "detected_face_samples": detected_face_count,
        "full_image_samples": full_image_count,
        "labels_available": labels_available,
        "output_dir": str(output_dir),
        "anonymized_manifest": str(anonymized_manifest_path) if labels_available and anonymized_manifest_path else "",
        "paired_manifest": str(paired_manifest_path) if labels_available and paired_manifest_path else "",
        "skipped_examples": skipped_examples[:20],
    }

    if summary_json_path is not None:
        summary_json_path.parent.mkdir(parents=True, exist_ok=True)
        with summary_json_path.open("w", encoding="utf-8") as file:
            json.dump(summary, file, indent=2, ensure_ascii=False)

    print("\n=== Learned Face Anonymisation Summary ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
