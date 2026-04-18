from dataclasses import dataclass
from pathlib import Path
import warnings

import cv2
import numpy as np


def _auto_find_cascade() -> Path | None:
    candidates = []

    if hasattr(cv2, "data") and hasattr(cv2.data, "haarcascades"):
        candidates.append(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml")

    cv2_path = Path(cv2.__file__).resolve()
    candidates.extend(
        [
            cv2_path.parent / "data" / "haarcascade_frontalface_default.xml",
            cv2_path.parent.parent / "Library" / "etc" / "haarcascades" / "haarcascade_frontalface_default.xml",
        ]
    )

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


@dataclass(frozen=True)
class RegionSelection:
    box: tuple[int, int, int, int] | None
    mode: str


class FaceRegionDetector:
    def __init__(
        self,
        enabled: bool = True,
        cascade_path: str = "",
        scale_factor: float = 1.1,
        min_neighbors: int = 5,
        min_size: tuple[int, int] = (48, 48),
        expand_ratio: float = 0.15,
        fallback_mode: str = "full_image",
    ) -> None:
        self.enabled = bool(enabled)
        self.scale_factor = float(scale_factor)
        self.min_neighbors = int(min_neighbors)
        self.min_size = tuple(int(value) for value in min_size)
        self.expand_ratio = float(expand_ratio)
        self.fallback_mode = fallback_mode
        self.detector = None

        if not self.enabled:
            return

        resolved_cascade = Path(cascade_path) if cascade_path else _auto_find_cascade()
        if resolved_cascade is None or not resolved_cascade.exists():
            warnings.warn("No Haar Cascade found for face anonymisation. Detection will fall back according to config.")
            return

        self.detector = cv2.CascadeClassifier(str(resolved_cascade))
        if self.detector.empty():
            self.detector = None
            warnings.warn(f"Failed to load Haar Cascade from {resolved_cascade}. Detection will fall back.")

    @classmethod
    def from_config(cls, cfg: dict) -> "FaceRegionDetector":
        return cls(
            enabled=cfg.get("enabled", True),
            cascade_path=cfg.get("cascade_path", ""),
            scale_factor=cfg.get("scale_factor", 1.1),
            min_neighbors=cfg.get("min_neighbors", 5),
            min_size=tuple(cfg.get("min_size", [48, 48])),
            expand_ratio=cfg.get("expand_ratio", 0.15),
            fallback_mode=cfg.get("fallback_mode", "full_image"),
        )

    def select_region(self, image: np.ndarray) -> RegionSelection:
        height, width = image.shape[:2]
        full_image_box = (0, 0, width, height)

        if self.detector is None:
            if self.fallback_mode == "full_image":
                return RegionSelection(full_image_box, "full_image")
            return RegionSelection(None, "skipped_no_face")

        gray = self._to_grayscale(image)
        faces = self.detector.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_size,
        )

        if len(faces):
            x, y, box_width, box_height = max(faces, key=lambda box: box[2] * box[3])
            expanded = self._expand_box((x, y, box_width, box_height), width, height)
            return RegionSelection(expanded, "detected_face")

        if self.fallback_mode == "full_image":
            return RegionSelection(full_image_box, "full_image")
        return RegionSelection(None, "skipped_no_face")

    def _to_grayscale(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            return image
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def _expand_box(
        self,
        box: tuple[int, int, int, int],
        image_width: int,
        image_height: int,
    ) -> tuple[int, int, int, int]:
        x, y, width, height = box
        pad_x = int(round(width * self.expand_ratio))
        pad_y = int(round(height * self.expand_ratio))

        x0 = max(0, x - pad_x)
        y0 = max(0, y - pad_y)
        x1 = min(image_width, x + width + pad_x)
        y1 = min(image_height, y + height + pad_y)
        return (x0, y0, x1 - x0, y1 - y0)
