from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np


FACE_OVAL = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400,
    377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
]
LEFT_EYE = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7]
RIGHT_EYE = [362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382]
LEFT_IRIS = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]
LEFT_BROW = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
RIGHT_BROW = [336, 296, 334, 293, 300, 285, 295, 282, 283, 276]
LIPS = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409,
    270, 269, 267, 0, 37, 39, 40, 185,
]
NOSE = [6, 197, 195, 5, 4, 1, 19, 94, 2, 164, 98, 97, 326, 327]


@dataclass
class FaceLandmarkSet:
    all_points: np.ndarray
    face_oval: np.ndarray
    left_eye: np.ndarray
    right_eye: np.ndarray
    left_iris: np.ndarray
    right_iris: np.ndarray
    left_brow: np.ndarray
    right_brow: np.ndarray
    lips: np.ndarray
    nose: np.ndarray

    @property
    def brows(self) -> np.ndarray:
        return np.concatenate([self.left_brow, self.right_brow], axis=0)

    @property
    def eyes(self) -> np.ndarray:
        return np.concatenate([self.left_eye, self.right_eye], axis=0)

    @property
    def irises(self) -> np.ndarray:
        return np.concatenate([self.left_iris, self.right_iris], axis=0)


class FaceLandmarkDetector:
    def __init__(self, enabled: bool, model_path: str, num_faces: int = 1) -> None:
        self.enabled = bool(enabled)
        self.model_path = Path(model_path) if model_path else None
        self.num_faces = int(num_faces)
        self._landmarker = None

        if not self.enabled or self.model_path is None or not self.model_path.exists():
            self.enabled = False
            return

        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(self.model_path)),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_faces=self.num_faces,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self._landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)

    @classmethod
    def from_config(cls, cfg: dict, resolve_path) -> "FaceLandmarkDetector":
        model_path = cfg.get("model_path", "")
        resolved = str(resolve_path(model_path)) if model_path else ""
        return cls(
            enabled=cfg.get("enabled", False),
            model_path=resolved,
            num_faces=cfg.get("num_faces", 1),
        )

    def close(self) -> None:
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None

    def detect(self, image_bgr: np.ndarray) -> FaceLandmarkSet | None:
        if not self.enabled or self._landmarker is None:
            return None

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        result = self._landmarker.detect(mp_image)
        if not result.face_landmarks:
            return None

        height, width = image_bgr.shape[:2]
        raw_points = result.face_landmarks[0]
        points = np.array([[pt.x * width, pt.y * height] for pt in raw_points], dtype=np.float32)
        return FaceLandmarkSet(
            all_points=points,
            face_oval=points[FACE_OVAL],
            left_eye=points[LEFT_EYE],
            right_eye=points[RIGHT_EYE],
            left_iris=points[LEFT_IRIS],
            right_iris=points[RIGHT_IRIS],
            left_brow=points[LEFT_BROW],
            right_brow=points[RIGHT_BROW],
            lips=points[LIPS],
            nose=points[NOSE],
        )
