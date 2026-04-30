import argparse
import json
from pathlib import Path
import sys

import cv2
from tqdm import tqdm
import yaml

MODULE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = MODULE_ROOT.parent
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from src.anonymizers import ImageAnonymizer
from src.detection import FaceRegionDetector
from src.io_utils import (
    ProcessedRecord,
    load_image_records,
    records_have_labels,
    write_anonymized_manifest,
    write_paired_manifest,
)


def load_config(config_path: Path):
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def resolve_repo_path(value: str | Path, repo_root: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def resolve_optional_repo_path(value: str, repo_root: Path) -> Path | None:
    return resolve_repo_path(value, repo_root) if value else None


def determine_region(image, apply_to: str, detector: FaceRegionDetector):
    if apply_to == "full_image":
        height, width = image.shape[:2]
        return (0, 0, width, height), "full_image"

    if apply_to != "face":
        raise ValueError(f"Unsupported anonymizer.apply_to: {apply_to}")

    selection = detector.select_region(image)
    if selection.box is None:
        raise RuntimeError("No face detected and detection.fallback_mode=skip.")
    return selection.box, selection.mode


def main():
    parser = argparse.ArgumentParser(
        description="Generate anonymized images and paired manifests for the face anonymisation baseline"
    )
    parser.add_argument("--config", type=str, default="face anonymisation/configs/baseline_blur.yaml")
    parser.add_argument("--limit", type=int, default=0, help="Optional cap on the number of images to process")
    args = parser.parse_args()

    config_path = resolve_repo_path(args.config, REPO_ROOT)
    cfg = load_config(config_path)

    data_cfg = cfg.get("data", {})
    export_cfg = cfg.get("export", {})
    output_dir = resolve_repo_path(data_cfg["output_dir"], REPO_ROOT)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_root = resolve_repo_path(data_cfg.get("image_root", "."), REPO_ROOT)
    manifest_csv = resolve_optional_repo_path(data_cfg.get("manifest_csv", ""), REPO_ROOT)
    source_dir = data_cfg.get("source_dir", "")
    if source_dir:
        source_dir = str(resolve_repo_path(source_dir, REPO_ROOT))

    records = load_image_records(
        image_root=image_root,
        source_dir=source_dir,
        manifest_csv=manifest_csv,
        label_mode=data_cfg.get("label_mode", "none"),
        supported_extensions=data_cfg.get("supported_extensions", []),
    )
    if args.limit > 0:
        records = records[: args.limit]

    detector = FaceRegionDetector.from_config(cfg.get("detection", {}))
    anonymizer = ImageAnonymizer.from_config(cfg["anonymizer"])
    apply_to = cfg["anonymizer"].get("apply_to", "face")

    processed_records: list[ProcessedRecord] = []
    skipped_examples = []
    detected_face_count = 0
    full_image_count = 0

    for record in tqdm(records, desc="Anonymizing", unit="image"):
        try:
            image = cv2.imread(str(record.source_path))
            if image is None:
                raise FileNotFoundError(f"Failed to read image: {record.source_path}")

            region, region_mode = determine_region(image, apply_to, detector)
            anonymized = anonymizer.apply(image, region)

            output_path = output_dir / record.relative_path
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(output_path), anonymized):
                raise IOError(f"Failed to write image: {output_path}")

            if region_mode == "detected_face":
                detected_face_count += 1
            else:
                full_image_count += 1

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

    labels_available = records_have_labels(processed_records)

    anonymized_manifest_path = resolve_optional_repo_path(export_cfg.get("anonymized_manifest", ""), REPO_ROOT)
    paired_manifest_path = resolve_optional_repo_path(export_cfg.get("paired_manifest", ""), REPO_ROOT)
    summary_json_path = resolve_optional_repo_path(export_cfg.get("summary_json", ""), REPO_ROOT)

    if labels_available and anonymized_manifest_path is not None:
        write_anonymized_manifest(anonymized_manifest_path, image_root, processed_records)
    elif anonymized_manifest_path is not None:
        print("Skipped anonymized_manifest export because labels are missing.")

    if labels_available and paired_manifest_path is not None:
        write_paired_manifest(paired_manifest_path, image_root, processed_records)
    elif paired_manifest_path is not None:
        print("Skipped paired_manifest export because labels are missing.")

    summary = {
        "config_path": str(config_path),
        "image_root": str(image_root),
        "input_mode": "manifest" if manifest_csv else "source_dir",
        "source_dir": source_dir,
        "manifest_csv": str(manifest_csv) if manifest_csv else "",
        "method": anonymizer.method,
        "apply_to": apply_to,
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

    print("\n=== Face Anonymisation Summary ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
