import cv2
import numpy as np


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

    def apply(self, image: np.ndarray, region: tuple[int, int, int, int]) -> np.ndarray:
        x, y, width, height = region
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid target region: {region}")

        result = image.copy()
        roi = result[y : y + height, x : x + width]
        if roi.size == 0:
            raise ValueError(f"Empty ROI extracted from region: {region}")

        if self.method == "gaussian_blur":
            transformed = self._apply_gaussian_blur(roi)
        else:
            transformed = self._apply_pixelate(roi)

        result[y : y + height, x : x + width] = transformed
        return result

    def _apply_gaussian_blur(self, roi: np.ndarray) -> np.ndarray:
        max_kernel = min(roi.shape[0], roi.shape[1])
        kernel_size = _fit_odd_kernel(self.gaussian_blur_cfg.get("kernel_size", 35), max_kernel)
        sigma = float(self.gaussian_blur_cfg.get("sigma", 0))
        return cv2.GaussianBlur(roi, (kernel_size, kernel_size), sigmaX=sigma)

    def _apply_pixelate(self, roi: np.ndarray) -> np.ndarray:
        pixel_size = max(1, int(self.pixelate_cfg.get("pixel_size", 12)))
        height, width = roi.shape[:2]
        target_width = max(1, width // pixel_size)
        target_height = max(1, height // pixel_size)
        reduced = cv2.resize(roi, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
        return cv2.resize(reduced, (width, height), interpolation=cv2.INTER_NEAREST)
