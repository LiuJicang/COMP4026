import cv2
import numpy as np

from src.landmarks import FaceLandmarkSet
from src.region_aware import apply_region_aware_blend


def _fit_odd_kernel(requested: int, max_size: int) -> int:
    if max_size <= 1:
        return 1

    kernel = max(1, int(requested))
    kernel = min(kernel, max_size if max_size % 2 == 1 else max_size - 1)
    if kernel < 1:
        kernel = 1
    if kernel % 2 == 0:
        kernel = max(1, kernel - 1)
    return kernel


def _adaptive_kernel_size(height: int, width: int, cfg: dict) -> int:
    min_side = min(height, width)
    requested = int(cfg.get("kernel_size", 35))
    min_kernel = int(cfg.get("min_kernel_size", requested))
    adaptive_ratio = float(cfg.get("adaptive_ratio", 0.0))
    max_kernel_ratio = float(cfg.get("max_kernel_ratio", 0.95))

    adaptive_request = int(round(min_side * adaptive_ratio)) if adaptive_ratio > 0 else requested
    kernel_request = max(requested, min_kernel, adaptive_request)
    max_kernel = max(1, int(round(min_side * max_kernel_ratio)))
    return _fit_odd_kernel(kernel_request, max_kernel)


def _build_feather_mask(height: int, width: int, cfg: dict) -> np.ndarray:
    edge_ratio = float(cfg.get("feather_edge_ratio", 0.12))
    edge = max(1, int(round(min(height, width) * edge_ratio)))

    x_distance = np.minimum(np.arange(width, dtype=np.float32), np.arange(width - 1, -1, -1, dtype=np.float32))
    y_distance = np.minimum(np.arange(height, dtype=np.float32), np.arange(height - 1, -1, -1, dtype=np.float32))
    mask = np.minimum(y_distance[:, None], x_distance[None, :]) / float(edge)
    mask = np.clip(mask, 0.0, 1.0)

    blur_kernel = int(cfg.get("feather_blur_kernel", 0))
    if blur_kernel > 1:
        blur_kernel = _fit_odd_kernel(blur_kernel, min(height, width))
        if blur_kernel > 1:
            mask = cv2.GaussianBlur(mask, (blur_kernel, blur_kernel), sigmaX=0)

    return mask.astype(np.float32)


class ImageAnonymizer:
    def __init__(
        self,
        method: str,
        gaussian_blur_cfg: dict | None = None,
        pixelate_cfg: dict | None = None,
    ) -> None:
        self.method = method
        self.gaussian_blur_cfg = gaussian_blur_cfg or {}
        self.pixelate_cfg = pixelate_cfg or {}

        if self.method not in {"gaussian_blur", "pixelate"}:
            raise ValueError(f"Unsupported anonymization method: {self.method}")

    @classmethod
    def from_config(cls, cfg: dict) -> "ImageAnonymizer":
        return cls(
            method=cfg.get("method", "gaussian_blur"),
            gaussian_blur_cfg=cfg.get("gaussian_blur", {}),
            pixelate_cfg=cfg.get("pixelate", {}),
        )

    def apply(
        self,
        image: np.ndarray,
        region: tuple[int, int, int, int],
        landmarks: FaceLandmarkSet | None = None,
    ) -> np.ndarray:
        x, y, width, height = region
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid target region: {region}")

        result = image.copy()
        roi = result[y : y + height, x : x + width]
        if roi.size == 0:
            raise ValueError(f"Empty ROI extracted from region: {region}")

        if self.method == "gaussian_blur":
            transformed = self._apply_gaussian_blur(roi, landmarks=landmarks)
        else:
            transformed = self._apply_pixelate(roi)

        result[y : y + height, x : x + width] = transformed
        return result

    def _apply_gaussian_blur(self, roi: np.ndarray, landmarks: FaceLandmarkSet | None = None) -> np.ndarray:
        height, width = roi.shape[:2]
        kernel_size = _adaptive_kernel_size(height, width, self.gaussian_blur_cfg)
        sigma = float(self.gaussian_blur_cfg.get("sigma", 0))
        blurred = cv2.GaussianBlur(roi, (kernel_size, kernel_size), sigmaX=sigma)

        region_aware_cfg = self.gaussian_blur_cfg.get("region_aware", {})
        if region_aware_cfg.get("enabled", False):
            return apply_region_aware_blend(roi, blurred, region_aware_cfg, landmarks=landmarks)

        if not self.gaussian_blur_cfg.get("feather_blend", False):
            return blurred

        mask = _build_feather_mask(height, width, self.gaussian_blur_cfg)
        blended = blurred.astype(np.float32) * mask[..., None] + roi.astype(np.float32) * (1.0 - mask[..., None])
        return blended.clip(0, 255).round().astype(np.uint8)

    def _apply_pixelate(self, roi: np.ndarray) -> np.ndarray:
        pixel_size = max(1, int(self.pixelate_cfg.get("pixel_size", 12)))
        height, width = roi.shape[:2]
        target_width = max(1, width // pixel_size)
        target_height = max(1, height // pixel_size)
        reduced = cv2.resize(roi, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
        return cv2.resize(reduced, (width, height), interpolation=cv2.INTER_NEAREST)
