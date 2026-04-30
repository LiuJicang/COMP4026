from __future__ import annotations

import cv2
import numpy as np

from src.landmarks import FaceLandmarkSet


def fit_odd_kernel(requested: int, max_size: int) -> int:
    if max_size <= 1:
        return 1

    kernel = max(1, int(requested))
    kernel = min(kernel, max_size if max_size % 2 == 1 else max_size - 1)
    if kernel < 1:
        kernel = 1
    if kernel % 2 == 0:
        kernel = max(1, kernel - 1)
    return kernel


def _soft_box_mask(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    softness: float,
) -> np.ndarray:
    softness = max(float(softness), 1e-6)
    left = np.clip((x_grid - x0) / softness, 0.0, 1.0)
    right = np.clip((x1 - x_grid) / softness, 0.0, 1.0)
    top = np.clip((y_grid - y0) / softness, 0.0, 1.0)
    bottom = np.clip((y1 - y_grid) / softness, 0.0, 1.0)
    return np.minimum(np.minimum(left, right), np.minimum(top, bottom))


def _soft_ellipse_mask(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    center_x: float,
    center_y: float,
    radius_x: float,
    radius_y: float,
    exponent: float,
) -> np.ndarray:
    radius_x = max(float(radius_x), 1e-6)
    radius_y = max(float(radius_y), 1e-6)
    exponent = max(float(exponent), 1e-6)
    distance = ((x_grid - center_x) / radius_x) ** 2 + ((y_grid - center_y) / radius_y) ** 2
    support = np.clip(1.0 - distance, 0.0, 1.0)
    return support ** exponent


def _build_region_geometry(height: int, width: int, cfg: dict) -> dict[str, np.ndarray]:
    x_coords = np.linspace(0.0, 1.0, num=width, dtype=np.float32)
    y_coords = np.linspace(0.0, 1.0, num=height, dtype=np.float32)
    x_grid, y_grid = np.meshgrid(x_coords, y_coords)

    softness = float(cfg.get("softness", 0.08))
    ellipse_exponent = float(cfg.get("ellipse_exponent", 0.35))

    face_support = _soft_ellipse_mask(
        x_grid,
        y_grid,
        center_x=float(cfg.get("face_center_x", 0.5)),
        center_y=float(cfg.get("face_center_y", 0.52)),
        radius_x=float(cfg.get("face_radius_x", 0.5)),
        radius_y=float(cfg.get("face_radius_y", 0.54)),
        exponent=ellipse_exponent,
    )
    head_support = _soft_ellipse_mask(
        x_grid,
        y_grid,
        center_x=float(cfg.get("head_center_x", 0.5)),
        center_y=float(cfg.get("head_center_y", 0.48)),
        radius_x=float(cfg.get("head_radius_x", 0.64)),
        radius_y=float(cfg.get("head_radius_y", 0.72)),
        exponent=float(cfg.get("head_ellipse_exponent", max(ellipse_exponent, 0.25))),
    )

    eye_nose_mask = np.maximum(
        _soft_box_mask(x_grid, y_grid, 0.14, 0.20, 0.86, 0.56, softness),
        _soft_box_mask(x_grid, y_grid, 0.34, 0.30, 0.66, 0.82, softness),
    )
    contour_mask = np.maximum(
        _soft_box_mask(x_grid, y_grid, 0.00, 0.12, 0.22, 0.96, softness),
        _soft_box_mask(x_grid, y_grid, 0.78, 0.12, 1.00, 0.96, softness),
    )
    hair_mask = _soft_box_mask(x_grid, y_grid, 0.04, 0.00, 0.96, 0.28, softness)
    ear_mask = np.maximum(
        _soft_box_mask(x_grid, y_grid, 0.00, 0.16, 0.18, 0.84, softness),
        _soft_box_mask(x_grid, y_grid, 0.82, 0.16, 1.00, 0.84, softness),
    )
    eye_mask = np.maximum(
        _soft_box_mask(x_grid, y_grid, 0.14, 0.20, 0.42, 0.50, softness),
        _soft_box_mask(x_grid, y_grid, 0.58, 0.20, 0.86, 0.50, softness),
    )
    pupil_mask = np.maximum(
        _soft_box_mask(x_grid, y_grid, 0.22, 0.27, 0.36, 0.41, softness),
        _soft_box_mask(x_grid, y_grid, 0.64, 0.27, 0.78, 0.41, softness),
    )
    nose_mask = _soft_box_mask(x_grid, y_grid, 0.40, 0.28, 0.60, 0.78, softness)
    forehead_mask = _soft_box_mask(x_grid, y_grid, 0.14, 0.00, 0.86, 0.30, softness)
    brow_mask = _soft_box_mask(x_grid, y_grid, 0.20, 0.12, 0.80, 0.24, softness)
    brow_core_mask = np.maximum(
        _soft_box_mask(x_grid, y_grid, 0.20, 0.13, 0.40, 0.22, softness),
        _soft_box_mask(x_grid, y_grid, 0.60, 0.13, 0.80, 0.22, softness),
    )
    mouth_mask = _soft_box_mask(x_grid, y_grid, 0.30, 0.66, 0.70, 0.84, softness)
    lip_mask = _soft_box_mask(x_grid, y_grid, 0.36, 0.70, 0.64, 0.80, softness)
    chin_mask = _soft_box_mask(x_grid, y_grid, 0.18, 0.76, 0.82, 1.00, softness)
    cheek_mask = np.maximum(
        _soft_box_mask(x_grid, y_grid, 0.10, 0.44, 0.34, 0.82, softness),
        _soft_box_mask(x_grid, y_grid, 0.66, 0.44, 0.90, 0.82, softness),
    )

    return {
        "face_support": face_support,
        "head_support": head_support,
        "eye_nose_mask": eye_nose_mask,
        "contour_mask": contour_mask,
        "hair_mask": hair_mask,
        "ear_mask": ear_mask,
        "eye_mask": eye_mask,
        "pupil_mask": pupil_mask,
        "nose_mask": nose_mask,
        "forehead_mask": forehead_mask,
        "brow_mask": brow_mask,
        "brow_core_mask": brow_core_mask,
        "mouth_mask": mouth_mask,
        "lip_mask": lip_mask,
        "chin_mask": chin_mask,
        "cheek_mask": cheek_mask,
    }


def _pixelate_image(image: np.ndarray, pixel_size: int) -> np.ndarray:
    pixel_size = max(1, int(pixel_size))
    if pixel_size <= 1:
        return image.copy()
    height, width = image.shape[:2]
    target_width = max(1, width // pixel_size)
    target_height = max(1, height // pixel_size)
    reduced = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(reduced, (width, height), interpolation=cv2.INTER_NEAREST)


def _posterize_image(image: np.ndarray, levels: int) -> np.ndarray:
    levels = max(2, int(levels))
    image_f = image.astype(np.float32) / 255.0
    posterized = np.round(image_f * (levels - 1)) / float(levels - 1)
    return (posterized * 255.0).clip(0, 255).round().astype(np.uint8)


def _mask_from_points_box(
    points: np.ndarray,
    height: int,
    width: int,
    pad_ratio_x: float,
    pad_ratio_y: float,
    blur_kernel: int,
) -> np.ndarray:
    if points is None or len(points) == 0:
        return np.zeros((height, width), dtype=np.float32)

    x0, y0 = points.min(axis=0)
    x1, y1 = points.max(axis=0)
    box_width = max(float(x1 - x0), 1.0)
    box_height = max(float(y1 - y0), 1.0)
    pad_x = box_width * float(pad_ratio_x)
    pad_y = box_height * float(pad_ratio_y)

    x0 = max(0, int(round(x0 - pad_x)))
    y0 = max(0, int(round(y0 - pad_y)))
    x1 = min(width - 1, int(round(x1 + pad_x)))
    y1 = min(height - 1, int(round(y1 + pad_y)))

    mask = np.zeros((height, width), dtype=np.float32)
    cv2.rectangle(mask, (x0, y0), (x1, y1), color=1.0, thickness=-1)
    blur_kernel = fit_odd_kernel(blur_kernel, min(height, width))
    if blur_kernel > 1:
        mask = cv2.GaussianBlur(mask, (blur_kernel, blur_kernel), sigmaX=0)
    return np.clip(mask, 0.0, 1.0)


def _mask_from_points_hull(points: np.ndarray, height: int, width: int, blur_kernel: int) -> np.ndarray:
    if points is None or len(points) < 3:
        return np.zeros((height, width), dtype=np.float32)

    mask = np.zeros((height, width), dtype=np.float32)
    polygon = cv2.convexHull(np.round(points).astype(np.int32))
    cv2.fillConvexPoly(mask, polygon, color=1.0)
    blur_kernel = fit_odd_kernel(blur_kernel, min(height, width))
    if blur_kernel > 1:
        mask = cv2.GaussianBlur(mask, (blur_kernel, blur_kernel), sigmaX=0)
    return np.clip(mask, 0.0, 1.0)


def _shrink_box_mask(mask: np.ndarray, shrink_ratio: float) -> np.ndarray:
    shrink_ratio = float(np.clip(shrink_ratio, 0.0, 0.95))
    if shrink_ratio <= 0:
        return mask

    height, width = mask.shape[:2]
    kernel_size = int(round(min(height, width) * shrink_ratio))
    kernel_size = fit_odd_kernel(kernel_size, min(height, width))
    if kernel_size <= 1:
        return mask
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    eroded = cv2.erode((mask > 0.05).astype(np.uint8), kernel, iterations=1)
    return eroded.astype(np.float32)


def _build_landmark_masks(
    height: int,
    width: int,
    cfg: dict,
    landmarks: FaceLandmarkSet | None,
) -> dict[str, np.ndarray]:
    if landmarks is None:
        return {}

    blur_kernel = int(cfg.get("mask_blur", 9))
    eye_mask = np.maximum(
        _mask_from_points_box(landmarks.left_eye, height, width, 0.30, 0.40, blur_kernel),
        _mask_from_points_box(landmarks.right_eye, height, width, 0.30, 0.40, blur_kernel),
    )
    pupil_mask = np.maximum(
        _mask_from_points_box(landmarks.left_iris, height, width, 0.45, 0.55, blur_kernel),
        _mask_from_points_box(landmarks.right_iris, height, width, 0.45, 0.55, blur_kernel),
    )
    brow_mask = np.maximum(
        _mask_from_points_box(landmarks.left_brow, height, width, 0.18, 0.35, blur_kernel),
        _mask_from_points_box(landmarks.right_brow, height, width, 0.18, 0.35, blur_kernel),
    )
    brow_core_mask = np.maximum(
        _mask_from_points_box(landmarks.left_brow, height, width, 0.04, 0.10, blur_kernel),
        _mask_from_points_box(landmarks.right_brow, height, width, 0.04, 0.10, blur_kernel),
    )
    mouth_mask = _mask_from_points_box(landmarks.lips, height, width, 0.16, 0.22, blur_kernel)
    lip_mask = _shrink_box_mask(mouth_mask, 0.08)
    nose_mask = _mask_from_points_box(landmarks.nose, height, width, 0.18, 0.16, blur_kernel)
    face_support = _mask_from_points_hull(landmarks.face_oval, height, width, blur_kernel)

    face_x0, face_y0 = landmarks.face_oval.min(axis=0)
    face_x1, face_y1 = landmarks.face_oval.max(axis=0)
    brow_top = landmarks.brows[:, 1].min()
    lip_bottom = landmarks.lips[:, 1].max()
    forehead_points = np.array(
        [
            [face_x0 + (face_x1 - face_x0) * 0.12, face_y0],
            [face_x1 - (face_x1 - face_x0) * 0.12, face_y0],
            [face_x1 - (face_x1 - face_x0) * 0.06, brow_top],
            [face_x0 + (face_x1 - face_x0) * 0.06, brow_top],
        ],
        dtype=np.float32,
    )
    chin_points = np.array(
        [
            [face_x0 + (face_x1 - face_x0) * 0.10, lip_bottom],
            [face_x1 - (face_x1 - face_x0) * 0.10, lip_bottom],
            [face_x1, face_y1],
            [face_x0, face_y1],
        ],
        dtype=np.float32,
    )

    forehead_mask = _mask_from_points_hull(forehead_points, height, width, blur_kernel)
    chin_mask = _mask_from_points_hull(chin_points, height, width, blur_kernel)
    eye_preserve_mask = np.clip(eye_mask - pupil_mask, 0.0, 1.0)

    return {
        "face_support": face_support,
        "eye_mask": eye_mask,
        "pupil_mask": pupil_mask,
        "brow_mask": brow_mask,
        "brow_core_mask": brow_core_mask,
        "mouth_mask": mouth_mask,
        "lip_mask": lip_mask,
        "nose_mask": nose_mask,
        "forehead_mask": forehead_mask,
        "chin_mask": chin_mask,
        "eye_preserve_mask": eye_preserve_mask,
    }


def build_region_aware_mask(height: int, width: int, cfg: dict, landmarks: FaceLandmarkSet | None = None) -> np.ndarray:
    if height <= 0 or width <= 0:
        raise ValueError("Region-aware mask expects positive height and width.")

    min_side = min(height, width)
    base_strength = float(cfg.get("base_strength", 0.58))
    identity_strength = float(cfg.get("identity_strength", 0.92))
    contour_strength = float(cfg.get("contour_strength", 0.76))
    hair_strength = float(cfg.get("hair_strength", contour_strength))
    ear_strength = float(cfg.get("ear_strength", contour_strength))
    eye_strength = float(cfg.get("eye_strength", identity_strength))
    pupil_strength = float(cfg.get("pupil_strength", eye_strength))
    nose_strength = float(cfg.get("nose_strength", identity_strength))
    forehead_strength = float(cfg.get("forehead_strength", hair_strength))
    chin_strength = float(cfg.get("chin_strength", contour_strength))
    brow_strength = float(cfg.get("brow_strength", 0.38))
    cheek_strength = float(cfg.get("cheek_strength", base_strength))
    mouth_strength = float(cfg.get("mouth_strength", 0.16))
    mask_blur = int(cfg.get("mask_blur", 31))

    small_face_reference = float(cfg.get("small_face_reference", 0))
    if small_face_reference > 0:
        small_face_mix = np.clip((small_face_reference - float(min_side)) / small_face_reference, 0.0, 1.0)
        base_strength = np.clip(base_strength + small_face_mix * float(cfg.get("small_face_base_boost", 0.0)), 0.0, 1.0)
        identity_strength = np.clip(
            identity_strength + small_face_mix * float(cfg.get("small_face_identity_boost", 0.0)),
            0.0,
            1.0,
        )
        contour_strength = np.clip(
            contour_strength + small_face_mix * float(cfg.get("small_face_contour_boost", 0.0)),
            0.0,
            1.0,
        )
        brow_strength = np.clip(
            brow_strength - small_face_mix * float(cfg.get("small_face_brow_penalty", 0.0)),
            0.0,
            1.0,
        )
        mouth_strength = np.clip(
            mouth_strength - small_face_mix * float(cfg.get("small_face_mouth_penalty", 0.0)),
            0.0,
            1.0,
        )
        cheek_strength = np.clip(
            cheek_strength - small_face_mix * float(cfg.get("small_face_cheek_penalty", 0.0)),
            0.0,
            1.0,
        )

    geometry = _build_region_geometry(height, width, cfg)
    geometry.update(_build_landmark_masks(height, width, cfg, landmarks))
    face_support = geometry["face_support"]
    head_support = geometry["head_support"]
    mask = face_support * base_strength

    mask = np.maximum(mask, identity_strength * geometry["eye_nose_mask"])
    mask = np.maximum(mask, contour_strength * geometry["contour_mask"])
    mask = np.maximum(mask, hair_strength * geometry["hair_mask"] * head_support)
    mask = np.maximum(mask, forehead_strength * geometry["forehead_mask"] * head_support)
    mask = np.maximum(mask, ear_strength * geometry["ear_mask"] * head_support)
    mask = np.maximum(mask, eye_strength * geometry["eye_mask"] * face_support)
    mask = np.maximum(mask, pupil_strength * geometry["pupil_mask"] * face_support)
    mask = np.maximum(mask, nose_strength * geometry["nose_mask"] * face_support)
    mask = np.maximum(mask, chin_strength * geometry["chin_mask"] * face_support)

    mouth_target = np.minimum(mask, mouth_strength)
    mask = mask * (1.0 - geometry["mouth_mask"]) + mouth_target * geometry["mouth_mask"]

    cheek_target = np.minimum(mask, cheek_strength)
    mask = mask * (1.0 - geometry["cheek_mask"]) + cheek_target * geometry["cheek_mask"]

    brow_target = np.minimum(mask, brow_strength)
    mask = mask * (1.0 - geometry["brow_mask"]) + brow_target * geometry["brow_mask"]

    mask = np.clip(mask * head_support, 0.0, 1.0)

    if mask_blur > 1:
        mask_blur = fit_odd_kernel(mask_blur, min_side)
        if mask_blur > 1:
            mask = cv2.GaussianBlur(mask, (mask_blur, mask_blur), sigmaX=0)
            mask = np.clip(mask, 0.0, 1.0)

    return mask.astype(np.float32)


def build_expression_preserve_mask(
    height: int,
    width: int,
    cfg: dict,
    landmarks: FaceLandmarkSet | None = None,
) -> np.ndarray:
    preserve_cfg = dict(cfg.get("preserve_expression", {}))
    if not preserve_cfg.get("enabled", False):
        return np.zeros((height, width), dtype=np.float32)

    geometry = _build_region_geometry(height, width, cfg)
    geometry.update(_build_landmark_masks(height, width, cfg, landmarks))
    face_support = geometry["face_support"]

    lip_strength = float(preserve_cfg.get("lip_strength", 1.0))
    brow_strength = float(preserve_cfg.get("brow_strength", 0.9))
    cheek_strength = float(preserve_cfg.get("cheek_strength", 0.8))
    eye_strength = float(preserve_cfg.get("eye_strength", 0.0))
    nose_strength = float(preserve_cfg.get("nose_strength", 0.0))
    mask_blur = int(preserve_cfg.get("mask_blur", 15))

    mask = np.zeros((height, width), dtype=np.float32)
    mask = np.maximum(mask, lip_strength * geometry["lip_mask"] * face_support)
    mask = np.maximum(mask, brow_strength * geometry["brow_core_mask"] * face_support)
    mask = np.maximum(mask, cheek_strength * geometry["cheek_mask"] * face_support)
    if eye_strength > 0:
        eye_preserve_mask = geometry.get("eye_preserve_mask", np.clip(geometry["eye_mask"] - geometry["pupil_mask"], 0.0, 1.0))
        mask = np.maximum(mask, eye_strength * eye_preserve_mask * face_support)
    if nose_strength > 0:
        mask = np.maximum(mask, nose_strength * geometry["nose_mask"] * face_support)

    mask = np.clip(mask, 0.0, 1.0)
    if mask_blur > 1:
        mask_blur = fit_odd_kernel(mask_blur, min(height, width))
        if mask_blur > 1:
            mask = cv2.GaussianBlur(mask, (mask_blur, mask_blur), sigmaX=0)
            mask = np.clip(mask, 0.0, 1.0)
    return mask.astype(np.float32)


def build_abstract_altered_roi(original_roi: np.ndarray, altered_roi: np.ndarray, cfg: dict) -> np.ndarray:
    abstract_cfg = dict(cfg.get("abstract", {}))
    if not abstract_cfg.get("enabled", False):
        return altered_roi

    candidates = []
    weights = []

    generated_weight = float(abstract_cfg.get("generated_weight", 0.0))
    if generated_weight > 0:
        candidates.append(altered_roi.astype(np.float32))
        weights.append(generated_weight)

    blur_weight = float(abstract_cfg.get("blur_weight", 0.0))
    if blur_weight > 0:
        min_side = min(altered_roi.shape[0], altered_roi.shape[1])
        requested = max(
            int(abstract_cfg.get("blur_min_kernel", 0)),
            int(round(min_side * float(abstract_cfg.get("blur_ratio", 0.0)))),
        )
        kernel = fit_odd_kernel(requested, min_side)
        blurred = cv2.GaussianBlur(altered_roi, (kernel, kernel), sigmaX=0) if kernel > 1 else altered_roi
        candidates.append(blurred.astype(np.float32))
        weights.append(blur_weight)

    pixelate_weight = float(abstract_cfg.get("pixelate_weight", 0.0))
    if pixelate_weight > 0:
        pixelated = _pixelate_image(altered_roi, int(abstract_cfg.get("pixel_size", 10)))
        candidates.append(pixelated.astype(np.float32))
        weights.append(pixelate_weight)

    posterize_weight = float(abstract_cfg.get("posterize_weight", 0.0))
    if posterize_weight > 0:
        posterized = _posterize_image(altered_roi, int(abstract_cfg.get("posterize_levels", 6)))
        candidates.append(posterized.astype(np.float32))
        weights.append(posterize_weight)

    if not candidates:
        return altered_roi

    total_weight = max(sum(weights), 1e-6)
    merged = np.zeros_like(altered_roi, dtype=np.float32)
    for candidate, weight in zip(candidates, weights):
        merged += candidate * (weight / total_weight)
    return merged.clip(0, 255).round().astype(np.uint8)


def apply_region_aware_blend(
    original_roi: np.ndarray,
    altered_roi: np.ndarray,
    cfg: dict,
    landmarks: FaceLandmarkSet | None = None,
) -> np.ndarray:
    altered_roi = build_abstract_altered_roi(original_roi, altered_roi, cfg)
    mask = build_region_aware_mask(original_roi.shape[0], original_roi.shape[1], cfg, landmarks=landmarks)
    preserve_mask = build_expression_preserve_mask(original_roi.shape[0], original_roi.shape[1], cfg, landmarks=landmarks)
    effective_mask = np.clip(mask * (1.0 - preserve_mask), 0.0, 1.0)
    blended = altered_roi.astype(np.float32) * effective_mask[..., None] + original_roi.astype(np.float32) * (1.0 - effective_mask[..., None])
    blended = blended.clip(0, 255).round().astype(np.uint8)

    post_blur_ratio = float(cfg.get("post_blur_ratio", 0.0))
    if post_blur_ratio > 0:
        min_side = min(original_roi.shape[0], original_roi.shape[1])
        min_kernel = int(cfg.get("post_blur_min_kernel", 0))
        requested = max(min_kernel, int(round(min_side * post_blur_ratio)))
        kernel = fit_odd_kernel(requested, min_side)
        if kernel > 1:
            blended = cv2.GaussianBlur(blended, (kernel, kernel), sigmaX=0)

    return blended
