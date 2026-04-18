import csv
from dataclasses import dataclass
from pathlib import Path


DEFAULT_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class ImageRecord:
    source_path: Path
    relative_path: Path
    label: str | None = None


@dataclass(frozen=True)
class ProcessedRecord:
    source_path: Path
    output_path: Path
    label: str | None = None


def load_image_records(
    image_root: Path,
    source_dir: str = "",
    manifest_csv: Path | None = None,
    label_mode: str = "none",
    supported_extensions: list[str] | None = None,
) -> list[ImageRecord]:
    if bool(source_dir) == bool(manifest_csv):
        raise ValueError("Configure exactly one of data.source_dir or data.manifest_csv.")

    if manifest_csv is not None:
        return _load_from_manifest(image_root, manifest_csv)

    return _load_from_directory(
        image_root=image_root,
        source_dir=Path(source_dir),
        label_mode=label_mode,
        supported_extensions=supported_extensions,
    )


def records_have_labels(records: list[ProcessedRecord]) -> bool:
    return bool(records) and all(record.label not in (None, "") for record in records)


def write_anonymized_manifest(path: Path, image_root: Path, records: list[ProcessedRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["image_path", "label"])
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "image_path": _to_manifest_path(record.output_path, image_root),
                    "label": record.label,
                }
            )


def write_paired_manifest(path: Path, image_root: Path, records: list[ProcessedRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["orig_path", "anon_path", "label"])
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "orig_path": _to_manifest_path(record.source_path, image_root),
                    "anon_path": _to_manifest_path(record.output_path, image_root),
                    "label": record.label,
                }
            )


def _load_from_manifest(image_root: Path, manifest_csv: Path) -> list[ImageRecord]:
    with manifest_csv.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None or "image_path" not in reader.fieldnames:
            raise ValueError(f"Manifest must contain an image_path column: {manifest_csv}")

        records = []
        for row in reader:
            raw_path = (row.get("image_path") or "").strip()
            if not raw_path:
                continue
            source_path, relative_path = _resolve_source_and_relative_path(raw_path, image_root)
            raw_label = row.get("label")
            label = str(raw_label).strip() if raw_label not in (None, "") else None
            records.append(ImageRecord(source_path=source_path, relative_path=relative_path, label=label))
        return records


def _load_from_directory(
    image_root: Path,
    source_dir: Path,
    label_mode: str,
    supported_extensions: list[str] | None,
) -> list[ImageRecord]:
    normalized_extensions = _normalize_extensions(supported_extensions)
    absolute_source_dir = source_dir if source_dir.is_absolute() else (image_root / source_dir)
    absolute_source_dir = absolute_source_dir.resolve()

    if not absolute_source_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {absolute_source_dir}")

    if label_mode not in {"none", "parent_dir"}:
        raise ValueError(f"Unsupported data.label_mode: {label_mode}")

    records = []
    for path in sorted(absolute_source_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in normalized_extensions:
            continue

        relative_inside_source = path.relative_to(absolute_source_dir)
        relative_path = relative_inside_source
        label = relative_inside_source.parent.name if label_mode == "parent_dir" else None
        records.append(ImageRecord(source_path=path.resolve(), relative_path=relative_path, label=label))
    return records


def _normalize_extensions(values: list[str] | None) -> set[str]:
    if not values:
        return set(DEFAULT_IMAGE_EXTENSIONS)
    return {
        value.lower() if value.startswith(".") else f".{value.lower()}"
        for value in values
    }


def _resolve_source_and_relative_path(path_value: str, image_root: Path) -> tuple[Path, Path]:
    candidate = Path(path_value)
    if not candidate.is_absolute():
        return (image_root / candidate).resolve(), candidate

    try:
        relative_path = candidate.resolve().relative_to(image_root)
    except ValueError as exc:
        raise ValueError(
            f"Absolute manifest paths must live under image_root for this pipeline: {candidate}"
        ) from exc
    return candidate.resolve(), relative_path


def _to_manifest_path(path: Path, image_root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(image_root).as_posix()
    except ValueError:
        return str(resolved)
