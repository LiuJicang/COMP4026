import argparse
import zipfile
from pathlib import Path


MODULE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = MODULE_ROOT.parent


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
                REPO_ROOT / "archive.zip",
                REPO_ROOT.parent / "archive.zip",
            ]
        )

    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate.exists():
            return candidate

    checked = "\n".join(str(path.resolve()) for path in candidates)
    raise FileNotFoundError(f"Could not find archive.zip. Checked:\n{checked}")


def extract_split_archive(zip_path: Path, output_root: Path, val_count: int) -> None:
    train_dir = output_root / "train"
    val_dir = output_root / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path) as archive:
        entries = [
            entry
            for entry in archive.infolist()
            if not entry.is_dir() and entry.filename.lower().endswith(".jpg")
        ]
        total = len(entries)
        if total == 0:
            raise ValueError(f"No JPG files found in archive: {zip_path}")

        val_count = min(val_count, total)
        split_index = total - val_count

        for index, entry in enumerate(entries):
            target_dir = train_dir if index < split_index else val_dir
            target_path = target_dir / Path(entry.filename).name
            if target_path.exists() and target_path.stat().st_size == entry.file_size:
                continue
            with archive.open(entry) as source, target_path.open("wb") as destination:
                destination.write(source.read())

    print(
        {
            "zip_path": str(zip_path),
            "output_root": str(output_root),
            "train_count": split_index,
            "val_count": val_count,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare CelebA-HQ train/val folders from a local zip archive")
    parser.add_argument("--zip-path", type=str, default=None, help="Path to the downloaded archive.zip")
    parser.add_argument("--output-root", type=str, default="CelebA-HQ", help="Output folder under the repo root")
    parser.add_argument("--val-count", type=int, default=3000, help="How many images to reserve for validation")
    args = parser.parse_args()

    zip_path = discover_zip_path(args.zip_path)
    output_root = resolve_repo_path(args.output_root)
    extract_split_archive(zip_path=zip_path, output_root=output_root, val_count=args.val_count)


if __name__ == "__main__":
    main()
