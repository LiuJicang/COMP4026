from pathlib import Path

import torch
from torch.utils.data import Dataset

from src.data.datasets import FaceFolderDataset
from src.preprocess.detect_and_crop import FacePreprocessor


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)


class ResNetFaceDataset(Dataset):
    """Loads face crops as normalized tensors for ResNet training and inference."""

    def __init__(self, folder_dataset: FaceFolderDataset, preprocessor: FacePreprocessor) -> None:
        self.folder_dataset = folder_dataset
        self.preprocessor = preprocessor

    def __len__(self) -> int:
        return len(self.folder_dataset.records)

    def __getitem__(self, index: int):
        record = self.folder_dataset.records[index]
        image = self.preprocessor.preprocess_color_path(record.image_path)
        tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        tensor = (tensor - IMAGENET_MEAN) / IMAGENET_STD
        return tensor, int(record.label), int(index)
