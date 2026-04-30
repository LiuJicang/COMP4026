from pathlib import Path
from typing import Callable

from PIL import Image
from torch.utils.data import Dataset

from src.io_utils import DEFAULT_IMAGE_EXTENSIONS


class UnlabeledFaceImageDataset(Dataset):
    def __init__(
        self,
        image_root: str | Path,
        source_dir: str | Path,
        transform: Callable,
        supported_extensions: list[str] | None = None,
    ) -> None:
        self.image_root = Path(image_root).resolve()
        self.source_dir = Path(source_dir)
        if not self.source_dir.is_absolute():
            self.source_dir = (self.image_root / self.source_dir).resolve()
        else:
            self.source_dir = self.source_dir.resolve()

        if not self.source_dir.exists():
            raise FileNotFoundError(f"Dataset directory does not exist: {self.source_dir}")

        self.transform = transform
        normalized_extensions = _normalize_extensions(supported_extensions)
        self.paths = sorted(
            path
            for path in self.source_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in normalized_extensions
        )

        if not self.paths:
            raise ValueError(f"No images found under: {self.source_dir}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        image_path = self.paths[index]
        image = Image.open(image_path).convert("RGB")
        return self.transform(image)


def _normalize_extensions(values: list[str] | None) -> set[str]:
    if not values:
        return set(DEFAULT_IMAGE_EXTENSIONS)
    return {
        value.lower() if value.startswith(".") else f".{value.lower()}"
        for value in values
    }
