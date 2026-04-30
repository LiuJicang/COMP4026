import random
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

from src.detection import FaceRegionDetector


@dataclass(frozen=True)
class CropBox:
    left: int
    top: int
    right: int
    bottom: int


class FaceHeadCropper:
    def __init__(
        self,
        enabled: bool = False,
        crop_scale: float = 1.65,
        upward_shift: float = 0.12,
        jitter_ratio: float = 0.05,
        fallback_scale: float = 0.92,
        min_side_ratio: float = 0.55,
        is_train: bool = False,
        detection_cfg: dict | None = None,
    ) -> None:
        self.enabled = bool(enabled)
        self.crop_scale = float(crop_scale)
        self.upward_shift = float(upward_shift)
        self.jitter_ratio = float(jitter_ratio)
        self.fallback_scale = float(fallback_scale)
        self.min_side_ratio = float(min_side_ratio)
        self.is_train = bool(is_train)

        detector_cfg = dict(detection_cfg or {})
        detector_cfg["fallback_mode"] = "skip_no_face"
        self.detector = FaceRegionDetector.from_config(detector_cfg)

    @classmethod
    def from_config(
        cls,
        crop_cfg: dict | None,
        detection_cfg: dict | None,
        is_train: bool,
    ) -> "FaceHeadCropper | None":
        crop_cfg = dict(crop_cfg or {})
        if not crop_cfg.get("enabled", False):
            return None

        return cls(
            enabled=True,
            crop_scale=crop_cfg.get("crop_scale", 1.65),
            upward_shift=crop_cfg.get("upward_shift", 0.12),
            jitter_ratio=crop_cfg.get("jitter_ratio", 0.05),
            fallback_scale=crop_cfg.get("fallback_scale", 0.92),
            min_side_ratio=crop_cfg.get("min_side_ratio", 0.55),
            is_train=is_train,
            detection_cfg=detection_cfg,
        )

    def crop(self, image: Image.Image) -> Image.Image:
        if not self.enabled:
            return image

        image_rgb = np.array(image.convert("RGB"))
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        crop_box = self._detect_crop_box(image_bgr)
        if crop_box is None:
            crop_box = self._fallback_crop_box(image.width, image.height)

        return image.crop((crop_box.left, crop_box.top, crop_box.right, crop_box.bottom))

    def _detect_crop_box(self, image_bgr: np.ndarray) -> CropBox | None:
        selection = self.detector.select_region(image_bgr)
        if selection.box is None:
            return None

        image_height, image_width = image_bgr.shape[:2]
        x, y, width, height = selection.box

        center_x = x + width * 0.5
        center_y = y + height * (0.5 - self.upward_shift)
        side = max(width, height) * self.crop_scale
        side = max(side, min(image_width, image_height) * self.min_side_ratio)

        if self.is_train and self.jitter_ratio > 0.0:
            jitter = self.jitter_ratio * side
            center_x += random.uniform(-jitter, jitter)
            center_y += random.uniform(-jitter, jitter)
            side *= random.uniform(1.0 - self.jitter_ratio, 1.0 + self.jitter_ratio)

        return self._square_box(center_x, center_y, side, image_width, image_height)

    def _fallback_crop_box(self, image_width: int, image_height: int) -> CropBox:
        side = min(image_width, image_height) * self.fallback_scale
        center_x = image_width * 0.5
        center_y = image_height * 0.5
        return self._square_box(center_x, center_y, side, image_width, image_height)

    def _square_box(
        self,
        center_x: float,
        center_y: float,
        side: float,
        image_width: int,
        image_height: int,
    ) -> CropBox:
        max_side = float(min(image_width, image_height))
        side = max(8.0, min(float(side), max_side))

        left = center_x - side * 0.5
        top = center_y - side * 0.5

        left = min(max(0.0, left), image_width - side)
        top = min(max(0.0, top), image_height - side)

        right = left + side
        bottom = top + side

        return CropBox(
            left=int(round(left)),
            top=int(round(top)),
            right=int(round(right)),
            bottom=int(round(bottom)),
        )
