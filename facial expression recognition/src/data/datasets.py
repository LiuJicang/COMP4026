from pathlib import Path
from typing import Callable, Optional

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class ExpressionDataset(Dataset):
    """Single-image expression dataset from a CSV manifest.

    Required CSV columns:
    - image_path: path to image relative to image_root or absolute path
    - label: integer class id
    """

    def __init__(
        self,
        csv_path: str,
        image_root: str,
        transform: Optional[Callable] = None,
    ) -> None:
        self.df = pd.read_csv(csv_path)
        required = {"image_path", "label"}
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"Missing required columns in {csv_path}: {sorted(missing)}")

        self.image_root = Path(image_root)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int):
        row = self.df.iloc[index]
        image_path = Path(row["image_path"])
        if not image_path.is_absolute():
            image_path = self.image_root / image_path

        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        label = int(row["label"])
        return image, label


class PairedExpressionDataset(Dataset):
    """Paired original/anonymized dataset for expression consistency evaluation.

    Required CSV columns:
    - orig_path: path to original image
    - anon_path: path to anonymized image
    - label: integer expression class id (same for both images)
    """

    def __init__(
        self,
        csv_path: str,
        image_root: str,
        transform: Optional[Callable] = None,
    ) -> None:
        self.df = pd.read_csv(csv_path)
        required = {"orig_path", "anon_path", "label"}
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"Missing required columns in {csv_path}: {sorted(missing)}")

        self.image_root = Path(image_root)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def _load_image(self, value: str):
        image_path = Path(value)
        if not image_path.is_absolute():
            image_path = self.image_root / image_path

        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image

    def __getitem__(self, index: int):
        row = self.df.iloc[index]
        orig = self._load_image(str(row["orig_path"]))
        anon = self._load_image(str(row["anon_path"]))
        label = int(row["label"])
        return orig, anon, label
