import argparse
import csv
from pathlib import Path

from PIL import Image
from huggingface_hub import hf_hub_download


MODULE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = MODULE_ROOT.parent
CLASS_NAMES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
CSV_SPLITS = {
    "fer2013/train.csv": "train",
    "fer2013/test.csv": "test",
    "fer2013/val.csv": "val",
}


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def sanitize_usage(value: str | None, fallback: str) -> str:
    if not value:
        return fallback
    cleaned = "".join(character for character in value if character.isalnum())
    return cleaned or fallback


def download_csvs(repo_id: str, cache_dir: Path) -> dict[str, Path]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    downloaded = {}
    for remote_path, split_name in CSV_SPLITS.items():
        downloaded[split_name] = Path(
            hf_hub_download(
                repo_id=repo_id,
                repo_type="dataset",
                filename=remote_path,
                local_dir=str(cache_dir),
            )
        )
    return downloaded


def write_split(csv_path: Path, split_dir: Path) -> int:
    for class_name in CLASS_NAMES:
        (split_dir / class_name).mkdir(parents=True, exist_ok=True)

    written = 0
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for index, row in enumerate(reader):
            label = int(row["emotion"])
            class_name = CLASS_NAMES[label]
            usage = sanitize_usage(row.get("Usage"), split_dir.name.title())
            target_path = split_dir / class_name / f"{usage}_{index:05d}.jpg"
            if target_path.exists():
                written += 1
                continue

            pixels = [int(value) for value in row["pixels"].split()]
            if len(pixels) != 48 * 48:
                raise ValueError(f"Unexpected pixel count in {csv_path} row {index}: {len(pixels)}")

            image = Image.frombytes("L", (48, 48), bytes(pixels))
            image.save(target_path, format="JPEG", quality=95)
            written += 1

    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Download FER-2013 CSVs and convert them into ImageFolder directories")
    parser.add_argument("--repo-id", type=str, default="chitradrishti/fer2013", help="Hugging Face dataset repo id")
    parser.add_argument("--output-root", type=str, default="FER-2013", help="Output folder under the repo root")
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="face anonymisation/outputs/dataset_cache/fer2013_source",
        help="Where to keep downloaded CSV files",
    )
    args = parser.parse_args()

    output_root = resolve_repo_path(args.output_root)
    cache_dir = resolve_repo_path(args.cache_dir)
    downloaded = download_csvs(repo_id=args.repo_id, cache_dir=cache_dir)

    summary = {"repo_id": args.repo_id, "output_root": str(output_root), "splits": {}}
    for split_name, csv_path in downloaded.items():
        split_dir = output_root / split_name
        count = write_split(csv_path=csv_path, split_dir=split_dir)
        summary["splits"][split_name] = {"csv": str(csv_path), "images": count}

    print(summary)


if __name__ == "__main__":
    main()
