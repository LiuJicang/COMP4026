import argparse
import random
import shutil
import zipfile
from pathlib import Path


MODULE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = MODULE_ROOT.parent
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def discover_zip_path(explicit_path: str | None) -> Path:
    candidates = []
    if explicit_path:
        candidates.append(Path(explicit_path))
    else:
        candidates.extend(
            [
                REPO_ROOT / "105_classes_pins_dataset.zip",
                REPO_ROOT.parent / "105_classes_pins_dataset.zip",
            ]
        )

    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate.exists():
            return candidate

    checked = "\n".join(str(path.resolve()) for path in candidates)
    raise FileNotFoundError(f"Could not find 105_classes_pins_dataset.zip. Checked:\n{checked}")


def should_extract(entry: zipfile.ZipInfo) -> bool:
    if entry.is_dir():
        return False
    suffix = Path(entry.filename).suffix.lower()
    return suffix in IMAGE_EXTENSIONS


def extract_archive(zip_path: Path, output_root: Path) -> Path:
    dataset_root = output_root / "105_classes_pins_dataset"
    dataset_root.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path) as archive:
        for entry in archive.infolist():
            if not should_extract(entry):
                continue

            member_path = Path(entry.filename)
            target_path = output_root / member_path
            target_path.parent.mkdir(parents=True, exist_ok=True)

            if target_path.exists() and target_path.stat().st_size == entry.file_size:
                continue

            with archive.open(entry) as source, target_path.open("wb") as destination:
                shutil.copyfileobj(source, destination)

    return dataset_root


def copy_or_link(source_path: Path, target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.exists():
        return

    try:
        target_path.hardlink_to(source_path)
    except OSError:
        shutil.copy2(source_path, target_path)


def build_split(dataset_root: Path, split_root: Path, val_ratio: float, seed: int) -> dict:
    train_dir = split_root / "train"
    val_dir = split_root / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    class_dirs = [path for path in sorted(dataset_root.iterdir()) if path.is_dir()]
    if not class_dirs:
        raise ValueError(f"No class folders found under: {dataset_root}")

    train_count = 0
    val_count = 0
    class_count = 0

    for class_dir in class_dirs:
        images = [
            path
            for path in sorted(class_dir.rglob("*"))
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ]
        if not images:
            continue

        class_count += 1
        rng.shuffle(images)

        if len(images) == 1:
            val_size = 0
        else:
            proposed = int(round(len(images) * val_ratio))
            val_size = min(max(1, proposed), len(images) - 1)

        val_images = images[:val_size]
        train_images = images[val_size:]

        for image_path in train_images:
            relative_path = image_path.relative_to(dataset_root)
            copy_or_link(image_path, train_dir / relative_path)
            train_count += 1

        for image_path in val_images:
            relative_path = image_path.relative_to(dataset_root)
            copy_or_link(image_path, val_dir / relative_path)
            val_count += 1

    return {
        "dataset_root": str(dataset_root),
        "split_root": str(split_root),
        "classes": class_count,
        "train_images": train_count,
        "val_images": val_count,
        "val_ratio": val_ratio,
        "seed": seed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract 105_classes_pins_dataset.zip and create a train/val split for face anonymisation GAN training"
    )
    parser.add_argument("--zip-path", type=str, default=None, help="Path to 105_classes_pins_dataset.zip")
    parser.add_argument("--output-root", type=str, default=".", help="Where to extract the dataset")
    parser.add_argument("--split-root", type=str, default="pins_105_split", help="Where to write the train/val split")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio per class")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for the split")
    args = parser.parse_args()

    zip_path = discover_zip_path(args.zip_path)
    output_root = resolve_repo_path(args.output_root)
    split_root = resolve_repo_path(args.split_root)

    dataset_root = extract_archive(zip_path=zip_path, output_root=output_root)
    summary = build_split(
        dataset_root=dataset_root,
        split_root=split_root,
        val_ratio=float(args.val_ratio),
        seed=int(args.seed),
    )

    print(summary)


if __name__ == "__main__":
    main()
