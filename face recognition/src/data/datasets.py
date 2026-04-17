from dataclasses import dataclass
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class FaceRecord:
    image_path: Path
    person_name: str
    label: int
    relative_key: str


@dataclass(frozen=True)
class PairedFaceRecord:
    orig_path: Path
    anon_path: Path
    person_name: str
    label: int
    relative_key: str


class FaceFolderDataset:
    """Reads identities from split/person_name/image files."""

    def __init__(self, image_root: str, split_dir: str, label_to_name: dict[int, str] | None = None) -> None:
        self.image_root = Path(image_root)
        self.split_dir = self._resolve_split_dir(split_dir)

        if not self.split_dir.exists():
            raise FileNotFoundError(f"Split directory does not exist: {self.split_dir}")

        person_dirs = sorted([path for path in self.split_dir.iterdir() if path.is_dir()], key=lambda path: path.name.lower())
        if label_to_name is None:
            self.label_to_name = {index: path.name for index, path in enumerate(person_dirs)}
        else:
            self.label_to_name = {int(key): str(value) for key, value in label_to_name.items()}

        self.name_to_label = {name: label for label, name in self.label_to_name.items()}
        self.records = self._scan_records(person_dirs)

    def __len__(self) -> int:
        return len(self.records)

    def _resolve_split_dir(self, split_dir: str) -> Path:
        path = Path(split_dir)
        if not path.is_absolute():
            path = self.image_root / path
        return path

    def _scan_records(self, person_dirs: list[Path]) -> list[FaceRecord]:
        records = []
        for person_dir in person_dirs:
            person_name = person_dir.name
            label = self.name_to_label.get(person_name, -1)
            image_paths = sorted(
                [
                    path
                    for path in person_dir.rglob("*")
                    if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
                ],
                key=lambda path: str(path).lower(),
            )
            for image_path in image_paths:
                records.append(
                    FaceRecord(
                        image_path=image_path,
                        person_name=person_name,
                        label=label,
                        relative_key=image_path.relative_to(self.split_dir).as_posix(),
                    )
                )
        return records


def build_paired_records(original_dataset: FaceFolderDataset, anonymized_dataset: FaceFolderDataset) -> list[PairedFaceRecord]:
    anonymized_by_key = {record.relative_key: record for record in anonymized_dataset.records}
    paired_records = []

    for original_record in original_dataset.records:
        anonymized_record = anonymized_by_key.get(original_record.relative_key)
        if anonymized_record is None:
            continue
        paired_records.append(
            PairedFaceRecord(
                orig_path=original_record.image_path,
                anon_path=anonymized_record.image_path,
                person_name=original_record.person_name,
                label=original_record.label,
                relative_key=original_record.relative_key,
            )
        )
    return paired_records
