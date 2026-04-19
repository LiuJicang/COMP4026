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
            Path(__file__).resolve().parents[2] / "assets" / "haarcascade_frontalface_default.xml",
        ]
    )

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


class FacePreprocessor:
    """Applies Viola-Jones detection and crops normalized face images."""

    def __init__(
        self,
        image_size: tuple[int, int],
        cascade_path: str = "",
        scale_factor: float = 1.1,
        min_neighbors: int = 5,
        min_size: tuple[int, int] = (48, 48),
        equalize_hist: bool = True,
        fallback_mode: str = "full_image",
    ) -> None:
        self.image_size = tuple(int(v) for v in image_size)
        self.scale_factor = float(scale_factor)
        self.min_neighbors = int(min_neighbors)
        self.min_size = tuple(int(v) for v in min_size)
        self.equalize_hist = bool(equalize_hist)
        self.fallback_mode = fallback_mode

        resolved_cascade = Path(cascade_path) if cascade_path else _auto_find_cascade()
        self.cascade_path = resolved_cascade if resolved_cascade and resolved_cascade.exists() else None
        self.detector = None

        if self.cascade_path is not None:
            self.detector = cv2.CascadeClassifier(str(self.cascade_path))
            if self.detector.empty():
                self.detector = None
                warnings.warn(f"Failed to load Haar Cascade from {self.cascade_path}. Falling back without detection.")
        else:
            warnings.warn("No Haar Cascade found. Falling back without Viola-Jones detection.")

    @classmethod
    def from_config(cls, cfg: dict) -> "FacePreprocessor":
        detector_cfg = cfg.get("detector", {})
        return cls(
            image_size=tuple(cfg["image_size"]),
            cascade_path=detector_cfg.get("cascade_path", ""),
            scale_factor=detector_cfg.get("scale_factor", 1.1),
            min_neighbors=detector_cfg.get("min_neighbors", 5),
            min_size=tuple(detector_cfg.get("min_size", [48, 48])),
            equalize_hist=cfg.get("equalize_hist", True),
            fallback_mode=cfg.get("fallback_mode", "full_image"),
        )

    def preprocess_path(self, image_path: str | Path) -> np.ndarray:
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Failed to read image: {image_path}")
        return self.preprocess_image(image)

    def preprocess_color_path(self, image_path: str | Path) -> np.ndarray:
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Failed to read image: {image_path}")
        return self.preprocess_color_image(image)

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        gray = self._to_grayscale(image)
        detection_gray = cv2.equalizeHist(gray) if self.equalize_hist else gray
        bbox = self._detect_face_box(detection_gray)
        face = self._crop_with_bbox(detection_gray, bbox)
        return cv2.resize(face, self.image_size, interpolation=cv2.INTER_AREA)

    def preprocess_color_image(self, image: np.ndarray) -> np.ndarray:
        gray = self._to_grayscale(image)
        detection_gray = cv2.equalizeHist(gray) if self.equalize_hist else gray
        bbox = self._detect_face_box(detection_gray)
        face = self._crop_with_bbox(image, bbox)
        face = cv2.resize(face, self.image_size, interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

    def _to_grayscale(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            return image
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def _detect_face_box(self, gray: np.ndarray) -> tuple[int, int, int, int] | None:
        if self.detector is not None:
            faces = self.detector.detectMultiScale(
                gray,
                scaleFactor=self.scale_factor,
                minNeighbors=self.min_neighbors,
                minSize=self.min_size,
            )
            if len(faces):
                return tuple(int(v) for v in max(faces, key=lambda box: box[2] * box[3]))

        return self._fallback_box(gray.shape[:2])

    def _fallback_box(self, shape: tuple[int, int]) -> tuple[int, int, int, int]:
        height, width = shape
        if self.fallback_mode == "center_crop":
            side = min(height, width)
            top = max((height - side) // 2, 0)
            left = max((width - side) // 2, 0)
            return left, top, side, side

        return 0, 0, width, height

    def _crop_with_bbox(self, image: np.ndarray, bbox: tuple[int, int, int, int] | None) -> np.ndarray:
        if bbox is None:
            return image
        x, y, w, h = bbox
        return image[y : y + h, x : x + w]
