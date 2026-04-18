import cv2
import numpy as np
import torch
from torchvision import transforms


IMAGENET_MEAN = [0.5, 0.5, 0.5]
IMAGENET_STD = [0.5, 0.5, 0.5]


def build_image_transform(image_size: int, is_train: bool):
    ops = [
        transforms.Resize((image_size, image_size)),
    ]
    if is_train:
        ops.extend(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.02),
            ]
        )
    ops.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return transforms.Compose(ops)


def build_classifier_transform(image_size: int, is_train: bool):
    ops = [
        transforms.Resize((image_size, image_size)),
    ]
    if is_train:
        ops.extend(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
            ]
        )
    ops.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return transforms.Compose(ops)


def image_to_tensor(image_bgr: np.ndarray, image_size: int) -> torch.Tensor:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_ready = transforms.ToPILImage()(image_rgb)
    transform = build_image_transform(image_size, is_train=False)
    return transform(pil_ready)


def tensor_to_bgr_image(tensor: torch.Tensor) -> np.ndarray:
    tensor = tensor.detach().cpu().clamp(-1.0, 1.0)
    tensor = tensor * 0.5 + 0.5
    image_rgb = (tensor.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
